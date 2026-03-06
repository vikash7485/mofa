//! LLM capability dispatch — the safe path that replaces the panicking
//! `.unwrap()` at `plugin/manager.rs:214`.
//!
//! ## What was broken (issue #897)
//!
//! ```text
//! // BEFORE (panic origin — line ~214 in the old manager.rs)
//! let plugin = registry.get(id).unwrap();   // panics when plugin absent after reload
//! ```
//!
//! ## What this module provides
//!
//! * [`dispatch_chat`] — the canonical entry point for all LLM calls.
//!   It looks up the session and the plugin using `?`-propagation (no
//!   `.unwrap()`), wraps the call in a RAII counter guard so the
//!   active-session counter is always decremented even on cancellation, then
//!   forwards the request to the provider.
//!
//! * [`ActiveSessionGuard`] — a `Drop` guard that decrements the plugin's
//!   `active_sessions` counter when it goes out of scope, making
//!   [`dispatch_chat`] cancel-safe.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::llm::provider::LLMProvider;
use crate::llm::types::{ChatCompletionRequest, ChatCompletionResponse};
use crate::plugin::error::PluginError;
use crate::plugin::manager::PluginManager;
use crate::session::registry::{SessionId, SessionRegistry};

// ─────────────────────────────────────────────────────────────────────────────
// RAII guard — ensures the active-session counter is always decremented
// ─────────────────────────────────────────────────────────────────────────────

/// Decrements the plugin's `active_sessions` counter when dropped.
///
/// Because `drop` is always called — including when a future is cancelled by
/// `tokio::select!` or a timeout — this guard makes [`dispatch_chat`]
/// cancel-safe with respect to the drain protocol counter.
pub struct ActiveSessionGuard {
    counter: Arc<AtomicUsize>,
}

impl ActiveSessionGuard {
    fn new(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self { counter }
    }
}

impl Drop for ActiveSessionGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Release);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// dispatch_chat — the safe replacement for the panicking dispatch path
// ─────────────────────────────────────────────────────────────────────────────

/// Forward a chat request to the correct LLM provider for `session_id`.
///
/// This function is the **safe replacement** for the bare `.unwrap()` that
/// previously panicked at `plugin/manager.rs:214`.
///
/// ## Steps
///
/// 1. Look up the session in `registry`. Returns [`PluginError::SessionNotFound`]
///    if absent — no panic, no silent data loss.
/// 2. Look up the plugin in `manager`. Returns [`PluginError::NotFound`] (or
///    [`PluginError::Other`] while draining) — no `.unwrap()`.
/// 3. Increment `active_sessions` counter via [`ActiveSessionGuard`]; the
///    guard decrements on any exit path including future cancellation.
/// 4. Invoke `provider.chat(request).await` and return the result verbatim.
///
/// ## Cancel safety
///
/// The function **is** cancel-safe: the `ActiveSessionGuard` is constructed
/// before the `.await` point, so even if the future is dropped mid-flight the
/// counter is correctly decremented.
pub async fn dispatch_chat(
    manager: &PluginManager,
    registry: &SessionRegistry,
    session_id: &SessionId,
    request: ChatCompletionRequest,
) -> Result<ChatCompletionResponse, PluginError> {
    // ── Step 1: look up the session (no .unwrap()) ────────────────────────────
    let snapshot = registry
        .get_session(session_id)
        .await
        .ok_or_else(|| PluginError::SessionNotFound(session_id.clone()))?;

    let plugin_id = &snapshot.plugin_id;

    // ── Step 2: look up the plugin (no .unwrap()) ─────────────────────────────
    //
    // BEFORE (panic origin):
    //   let plugin = registry.get(id).unwrap();   // line ~214
    //
    // AFTER (safe):
    let plugin: Arc<dyn LLMProvider> = manager
        .get_plugin(plugin_id)
        .await
        .map_err(|e| PluginError::Other(format!("plugin '{}' unavailable: {}", plugin_id, e)))?;

    // ── Step 3: increment counter, attach RAII guard ──────────────────────────
    let counter = manager
        .get_counter(plugin_id)
        .await
        .map_err(|e| PluginError::Other(format!("counter lookup failed: {}", e)))?;

    // Guard increments on construction, decrements on drop (including cancel).
    let _guard = ActiveSessionGuard::new(counter);

    // ── Step 4: call the provider ─────────────────────────────────────────────
    plugin
        .chat(request)
        .await
        .map_err(|e| PluginError::Other(format!("LLM provider error: {}", e)))
    // _guard dropped here → counter decremented
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{
        ChatCompletionResponse, Choice, FinishReason,
    };
    use crate::session::registry::SessionRegistry;
    use async_trait::async_trait;

    // ── Mock provider ────────────────────────────────────────────────────

    struct OkProvider;

    #[async_trait]
    impl LLMProvider for OkProvider {
        fn name(&self) -> &str {
            "ok-provider"
        }
        async fn chat(
            &self,
            _r: ChatCompletionRequest,
        ) -> crate::agent::AgentResult<ChatCompletionResponse> {
            Ok(ChatCompletionResponse {
                choices: vec![Choice {
                    index: 0,
                    message: crate::llm::types::ChatMessage::assistant("ok"),
                    finish_reason: Some(FinishReason::Stop),
                    logprobs: None,
                }],
            })
        }
    }

    fn make_manager_and_registry() -> (PluginManager, SessionRegistry) {
        (PluginManager::new(), SessionRegistry::new())
    }

    // ────────────────────────────────────────────────────────────────────
    // Test: missing session returns error, no panic
    // ────────────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn dispatch_returns_error_for_missing_session_no_panic() {
        let (mgr, reg) = make_manager_and_registry();
        let result = dispatch_chat(
            &mgr,
            &reg,
            &"ghost-session".to_string(),
            ChatCompletionRequest::new("gpt-4"),
        )
        .await;

        assert!(
            matches!(result, Err(PluginError::SessionNotFound(ref id)) if id == "ghost-session"),
            "expected SessionNotFound"
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // Test: missing plugin returns error, no panic
    // ────────────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn dispatch_returns_error_for_missing_plugin_no_panic() {
        let (mgr, reg) = make_manager_and_registry();

        // Create a session pointing to a plugin that is NOT registered.
        reg.create_session(
            "s1".to_string(),
            "ghost-plugin".to_string(),
            Arc::new(OkProvider),
        )
        .await;

        let result = dispatch_chat(
            &mgr,
            &reg,
            &"s1".to_string(),
            ChatCompletionRequest::new("gpt-4"),
        )
        .await;

        // get_plugin("ghost-plugin") returns NotFound wrapped in Other
        assert!(
            matches!(result, Err(PluginError::Other(_))),
            "expected Other error"
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // Test: successful dispatch decrements counter back to zero
    // ────────────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn dispatch_succeeds_and_counter_returns_to_zero() {
        let (mgr, reg) = make_manager_and_registry();

        let provider: Arc<dyn LLMProvider> = Arc::new(OkProvider);
        mgr.register_plugin("ok", Arc::clone(&provider)).await;
        reg.create_session("s1".to_string(), "ok".to_string(), provider)
            .await;

        // Counter must be 0 before dispatch
        let counter = mgr.get_counter("ok").await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        let result = dispatch_chat(
            &mgr,
            &reg,
            &"s1".to_string(),
            ChatCompletionRequest::new("gpt-4"),
        )
        .await;
        assert!(result.is_ok(), "dispatch must succeed");

        // Counter must be back to 0 after the guard drops
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "active_sessions counter must return to 0 after dispatch"
        );
    }
}
