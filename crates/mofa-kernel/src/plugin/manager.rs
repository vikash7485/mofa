//! Plugin manager — owns the registry of loaded LLM providers and co-ordinates
//! the session-safe hot-reload drain protocol.
//!
//! ## Session Drain Protocol
//!
//! Hot-reloading a plugin while sessions are active can corrupt session state
//! if the session holds a *direct* reference to the old plugin instance.  This
//! manager solves the problem in five explicit steps that [`reload_plugin`]
//! executes:
//!
//! ```text
//! ── Session Drain Protocol ─────────────────────────────────────────────────
//! 1. Mark Draining — gate blocks new requests from starting.
//! 2. Wait up to MOFA_PLUGIN_DRAIN_TIMEOUT secs for active_sessions == 0.
//!    On timeout: tracing::warn! + graceful termination, no panic.
//! 3. Checkpoint — SessionRegistry persists conversation history.
//! 4. Swap — replace Arc<dyn LLMProvider>. Existing Arc clones stay valid;
//!    there are no dangling references.
//! 5. Restore — rebind live sessions to the new provider Arc, set Active.
//! ──────────────────────────────────────────────────────────────────────────
//! ```
//!
//! All public methods return `Result<_, PluginError>` — there are **zero**
//! bare `.unwrap()` or `.expect()` calls in production code paths.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

use crate::llm::provider::LLMProvider;
use crate::plugin::error::PluginError;
use crate::plugin::PluginState;
use crate::session::registry::SessionRegistry;

// ─────────────────────────────────────────────────────────────────────────────
// Internal entry stored per registered plugin
// ─────────────────────────────────────────────────────────────────────────────

/// One slot in the plugin registry.
struct PluginEntry {
    /// The live provider instance, shared via reference-counting.
    ///
    /// Sessions hold their own `Arc` clone; swapping this field during reload
    /// does **not** invalidate the clones already given to sessions.
    provider: Arc<dyn LLMProvider>,

    /// Lifecycle state used to gate new requests during drain / swap.
    state: PluginState,

    /// Number of in-flight `dispatch_chat` calls currently using this plugin.
    ///
    /// Incremented by [`crate::llm::capability::dispatch_chat`] before the
    /// LLM call and decremented by the RAII guard on any exit path (including
    /// cancellation).
    active_sessions: Arc<AtomicUsize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// PluginManager
// ─────────────────────────────────────────────────────────────────────────────

/// Owns the map of registered LLM providers and co-ordinates hot-reload.
///
/// All state is wrapped in `Arc<RwLock<…>>` so the manager can be cloned
/// cheaply and shared across async tasks.
#[derive(Clone, Default)]
pub struct PluginManager {
    plugins: Arc<RwLock<HashMap<String, PluginEntry>>>,
}

impl PluginManager {
    /// Create a new, empty plugin manager.
    pub fn new() -> Self {
        Self::default()
    }

    // ─────────────────────────────────────────────────────────────────────
    // Registration
    // ─────────────────────────────────────────────────────────────────────

    /// Register (or replace) an LLM provider under `id`.
    ///
    /// The entry starts in [`PluginState::Active`] with an active-session
    /// counter at zero.
    pub async fn register_plugin(&self, id: impl Into<String>, provider: Arc<dyn LLMProvider>) {
        let mut map = self.plugins.write().await;
        map.insert(
            id.into(),
            PluginEntry {
                provider,
                state: PluginState::Active,
                active_sessions: Arc::new(AtomicUsize::new(0)),
            },
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // Safe lookup (replaces the bare .unwrap() at manager.rs:214)
    // ─────────────────────────────────────────────────────────────────────

    /// Look up a registered provider by ID.
    ///
    /// # Errors
    ///
    /// * [`PluginError::NotFound`] — no provider is registered under `id`.
    /// * [`PluginError::Other`]    — the plugin is currently in a transient
    ///   state (`Draining` or `Swapping`) and cannot accept new requests.
    pub async fn get_plugin(&self, id: &str) -> Result<Arc<dyn LLMProvider>, PluginError> {
        let map = self.plugins.read().await;
        let entry = map
            .get(id)
            .ok_or_else(|| PluginError::NotFound(id.to_string()))?;

        match &entry.state {
            PluginState::Active => Ok(Arc::clone(&entry.provider)),
            PluginState::Draining => Err(PluginError::Other(format!(
                "plugin '{}' is draining — not accepting new requests",
                id
            ))),
            PluginState::Swapping => Err(PluginError::Other(format!(
                "plugin '{}' is being swapped — not accepting new requests",
                id
            ))),
            // Legacy lifecycle states (Unloaded, Loading, Loaded, Running,
            // Paused, Error) are not valid for LLM dispatch.  Treat them as
            // "not ready" with a descriptive error rather than panicking.
            _ => Err(PluginError::Other(format!(
                "plugin '{}' is not in an active state ({:?}) — cannot dispatch",
                id, entry.state
            ))),
        }
    }

    /// Return the `Arc<AtomicUsize>` session counter for `id`.
    ///
    /// Used by [`crate::llm::capability`] to create a RAII decrement guard.
    ///
    /// # Errors
    ///
    /// * [`PluginError::NotFound`] — no provider is registered under `id`.
    pub async fn get_counter(&self, id: &str) -> Result<Arc<AtomicUsize>, PluginError> {
        let map = self.plugins.read().await;
        map.get(id)
            .map(|e| Arc::clone(&e.active_sessions))
            .ok_or_else(|| PluginError::NotFound(id.to_string()))
    }

    // ─────────────────────────────────────────────────────────────────────
    // State management
    // ─────────────────────────────────────────────────────────────────────

    /// Atomically update the lifecycle state of a plugin entry.
    ///
    /// # Errors
    ///
    /// * [`PluginError::NotFound`] — no entry with `id` exists.
    pub async fn set_plugin_state(
        &self,
        id: &str,
        state: PluginState,
    ) -> Result<(), PluginError> {
        let mut map = self.plugins.write().await;
        map.get_mut(id)
            .ok_or_else(|| PluginError::NotFound(id.to_string()))
            .map(|e| e.state = state)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Drain protocol steps
    // ─────────────────────────────────────────────────────────────────────

    /// **Step 2** — Wait until `active_sessions == 0` or `timeout` expires.
    ///
    /// This function is cancel-safe: it holds the lock only to read the
    /// counter, then yields between polls so that other tasks can run.
    ///
    /// # Errors
    ///
    /// * [`PluginError::DrainTimeout`] — sessions did not finish within the
    ///   deadline.  The caller should warn and terminate remaining sessions
    ///   instead of panicking.
    /// * [`PluginError::NotFound`] — no entry with `id` exists.
    pub async fn drain_sessions(
        &self,
        id: &str,
        timeout: Duration,
    ) -> Result<(), PluginError> {
        let counter = self.get_counter(id).await?;

        let result = tokio::time::timeout(timeout, async {
            loop {
                // Read without holding the write-lock so other tasks can
                // decrement the counter while we sleep.
                if counter.load(Ordering::Acquire) == 0 {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await;

        result.map_err(|_| {
            PluginError::DrainTimeout(format!(
                "plugin '{}' still had active sessions after {:?}",
                id, timeout
            ))
        })
    }

    /// **Step 3** — Checkpoint all sessions belonging to `id`.
    ///
    /// Delegates to [`SessionRegistry::checkpoint_all_for_plugin`].
    pub async fn checkpoint_sessions(
        &self,
        id: &str,
        registry: &SessionRegistry,
    ) -> Result<(), PluginError> {
        registry.checkpoint_all_for_plugin(id).await;
        Ok(())
    }

    /// **Step 4** — Swap the provider `Arc` for a new version.
    ///
    /// Existing `Arc` clones held by sessions remain valid because `Arc`
    /// reference-counts the allocation; only *new* `Arc::clone` calls from
    /// `get_plugin` will return the new provider.
    ///
    /// # Errors
    ///
    /// * [`PluginError::NotFound`] — no entry with `id` exists.
    pub async fn swap_plugin(
        &self,
        id: &str,
        new_provider: Arc<dyn LLMProvider>,
    ) -> Result<(), PluginError> {
        let mut map = self.plugins.write().await;
        let entry = map
            .get_mut(id)
            .ok_or_else(|| PluginError::NotFound(id.to_string()))?;
        entry.provider = new_provider;
        entry.state = PluginState::Swapping;
        Ok(())
    }

    /// **Step 5** — Rebind sessions to the new provider and mark Active.
    ///
    /// Delegates to [`SessionRegistry::rebind_plugin`].
    pub async fn restore_sessions(
        &self,
        id: &str,
        registry: &SessionRegistry,
    ) -> Result<(), PluginError> {
        // Get the new provider Arc to hand to the registry.
        let map = self.plugins.read().await;
        let entry = map
            .get(id)
            .ok_or_else(|| PluginError::NotFound(id.to_string()))?;
        let new_provider = Arc::clone(&entry.provider);
        drop(map); // release read-lock before acquiring write-lock below

        registry.rebind_plugin(id, new_provider).await;

        // Transition back to Active.
        self.set_plugin_state(id, PluginState::Active).await
    }

    // ─────────────────────────────────────────────────────────────────────
    // Public hot-reload entry point
    // ─────────────────────────────────────────────────────────────────────

    // ── Session Drain Protocol ───────────────────────────────────────────
    // 1. Mark Draining — gate blocks new requests.
    // 2. Wait up to MOFA_PLUGIN_DRAIN_TIMEOUT secs for active_sessions == 0.
    //    On timeout: tracing::warn! + graceful termination, no panic.
    // 3. Checkpoint — SessionRegistry persists conversation history.
    // 4. Swap — replace Arc<dyn LLMProvider>. Existing Arc clones stay valid.
    // 5. Restore — rebind live sessions to new provider Arc, set Active.
    // ────────────────────────────────────────────────────────────────────

    /// Hot-reload `plugin_id` with `new_provider` while preserving session state.
    ///
    /// This is the **primary entry point** for the fix described in issue #897.
    /// It executes the five-step drain protocol documented in the module-level
    /// doc comment.
    ///
    /// All five phases return typed errors — there are no `.unwrap()` calls
    /// anywhere in the path.
    pub async fn reload_plugin(
        &self,
        plugin_id: &str,
        new_provider: Arc<dyn LLMProvider>,
        registry: &SessionRegistry,
    ) -> Result<(), PluginError> {
        let timeout = drain_timeout();

        // Step 1: Mark Draining
        self.set_plugin_state(plugin_id, PluginState::Draining)
            .await?;

        // Step 2: Drain (with graceful timeout fallback)
        match self.drain_sessions(plugin_id, timeout).await {
            Ok(()) => {}
            Err(PluginError::DrainTimeout(ref msg)) => {
                tracing::warn!(
                    plugin_id = plugin_id,
                    reason = %msg,
                    "drain timeout exceeded — gracefully terminating remaining sessions"
                );
                registry
                    .terminate_sessions_for_plugin(plugin_id, "drain timeout")
                    .await;
            }
            Err(e) => return Err(e),
        }

        // Step 3: Checkpoint
        self.checkpoint_sessions(plugin_id, registry).await?;

        // Step 4: Swap provider Arc
        self.swap_plugin(plugin_id, new_provider).await?;

        // Step 5: Restore
        self.restore_sessions(plugin_id, registry).await?;

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Read the drain timeout from the `MOFA_PLUGIN_DRAIN_TIMEOUT` environment
/// variable (in seconds).  Defaults to 30 seconds when absent or unparseable.
pub fn drain_timeout() -> Duration {
    std::env::var("MOFA_PLUGIN_DRAIN_TIMEOUT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(30))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{ChatCompletionRequest, ChatCompletionResponse, Choice};
    use crate::llm::types::{ChatMessage, FinishReason, MessageContent, Role};
    use crate::session::registry::SessionRegistry;
    use async_trait::async_trait;

    // ── Mock provider ────────────────────────────────────────────────────

    struct MockProvider {
        id: String,
        version: u32,
    }

    impl MockProvider {
        fn new(id: &str, version: u32) -> Arc<Self> {
            Arc::new(Self {
                id: id.to_string(),
                version,
            })
        }
    }

    #[async_trait]
    impl LLMProvider for MockProvider {
        fn name(&self) -> &str {
            &self.id
        }

        async fn chat(
            &self,
            _request: ChatCompletionRequest,
        ) -> crate::agent::AgentResult<ChatCompletionResponse> {
            Ok(ChatCompletionResponse {
                choices: vec![Choice {
                    index: 0,
                    message: ChatMessage::assistant(format!("v{} reply", self.version)),
                    finish_reason: Some(FinishReason::Stop),
                    logprobs: None,
                }],
            })
        }
    }

    // ── Helper ───────────────────────────────────────────────────────────

    fn make_entry(content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            content: Some(MessageContent::Text(content.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Test: missing plugin returns error, no panic
    // ────────────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn get_plugin_returns_error_when_absent_no_panic() {
        let mgr = PluginManager::new();
        let result = mgr.get_plugin("nonexistent").await;
        assert!(
            matches!(result, Err(PluginError::NotFound(ref id)) if id == "nonexistent"),
            "expected NotFound error"
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // Test: plugin lookup rejected while Draining
    // ────────────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn get_plugin_rejects_while_draining() {
        let mgr = PluginManager::new();
        mgr.register_plugin("llm-test", MockProvider::new("llm-test", 1))
            .await;
        mgr.set_plugin_state("llm-test", PluginState::Draining)
            .await
            .unwrap();

        let result = mgr.get_plugin("llm-test").await;
        assert!(
            matches!(result, Err(PluginError::Other(_))),
            "expected Other error while draining"
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // Test: drain timeout does NOT panic
    // ────────────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn drain_timeout_exceeded_does_not_panic() {
        let mgr = PluginManager::new();
        mgr.register_plugin("llm-slow", MockProvider::new("llm-slow", 1))
            .await;

        // Artificially pin the counter at 1 to simulate an in-flight session.
        let counter = mgr.get_counter("llm-slow").await.unwrap();
        counter.fetch_add(1, Ordering::Relaxed);

        let result = mgr
            .drain_sessions("llm-slow", Duration::from_millis(10))
            .await;

        assert!(
            matches!(result, Err(PluginError::DrainTimeout(_))),
            "expected DrainTimeout"
        );
        // Restore counter so the test is clean
        counter.fetch_sub(1, Ordering::Relaxed);
    }

    // ────────────────────────────────────────────────────────────────────
    // CORE regression test — session history survives plugin reload
    // ────────────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn test_session_state_preserved_across_plugin_reload() {
        // ── 1. Setup ─────────────────────────────────────────────────────
        let manager = PluginManager::new();
        let registry = SessionRegistry::new();

        let provider_v1: Arc<dyn LLMProvider> = MockProvider::new("openai", 1);
        manager
            .register_plugin("openai", Arc::clone(&provider_v1))
            .await;

        // ── 2. Create a session and build up conversation history ─────────
        let session_id = "session-42".to_string();
        registry
            .create_session(session_id.clone(), "openai".to_string(), Arc::clone(&provider_v1))
            .await;

        registry
            .push_message(&session_id, make_entry("Hello, world!"))
            .await
            .expect("push_message failed");

        registry
            .push_message(&session_id, make_entry("What is 2 + 2?"))
            .await
            .expect("push_message failed");

        // Sanity check: two messages before reload.
        let history_before = registry
            .get_history(&session_id)
            .await
            .expect("get_history failed");
        assert_eq!(
            history_before.len(),
            2,
            "expected 2 messages before reload, got {}",
            history_before.len()
        );

        // ── 3. Hot-reload the plugin ──────────────────────────────────────
        let provider_v2: Arc<dyn LLMProvider> = MockProvider::new("openai", 2);
        manager
            .reload_plugin("openai", Arc::clone(&provider_v2), &registry)
            .await
            .expect("reload_plugin failed");

        // ── 4. Assert history is intact after reload ──────────────────────
        let history_after = registry
            .get_history(&session_id)
            .await
            .expect("get_history after reload failed");

        assert_eq!(
            history_after.len(),
            2,
            "session history must survive a plugin reload: expected 2 messages, got {}",
            history_after.len()
        );
        assert_eq!(
            history_after[0].text_content(),
            Some("Hello, world!"),
            "first message content must be intact"
        );
        assert_eq!(
            history_after[1].text_content(),
            Some("What is 2 + 2?"),
            "second message content must be intact"
        );

        // ── 5. Assert session is now bound to v2 ─────────────────────────
        let snapshot = registry
            .get_session(&session_id)
            .await
            .expect("session must still exist after reload");

        // The new provider's name() is the same "openai", but we can verify
        // that a round-trip chat call uses the new provider without error.
        let response = snapshot.plugin.chat(ChatCompletionRequest::new("gpt-4")).await;
        assert!(
            response.is_ok(),
            "call through rebound provider must succeed"
        );
        let response = response.unwrap();
        // v2 returns "v2 reply"
        assert!(
            response.content().map(|c| c.contains("v2")).unwrap_or(false),
            "response must come from the new provider v2"
        );
    }
}
