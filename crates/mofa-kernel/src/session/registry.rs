//! Session registry — owns the in-memory state of every active agent session.
//!
//! ## Key design decision (fix for issue #897)
//!
//! Each [`SessionEntry`] holds an `Arc<dyn LLMProvider>` instead of a
//! concrete plugin type.  When the plugin manager swaps the provider during
//! hot-reload, it calls [`SessionRegistry::rebind_plugin`], which updates the
//! `Arc` inside every affected session.  Sessions that are *currently
//! executing* an LLM call hold their own `Arc` clone, which keeps the old
//! provider allocation alive until the call completes — no dangling references,
//! no data loss.
//!
//! All methods return `Option`/`Result` — there are **zero** bare `.unwrap()`
//! or `.expect()` calls in production code paths.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::llm::provider::LLMProvider;
use crate::llm::types::ChatMessage;
use crate::plugin::error::PluginError;

// ─────────────────────────────────────────────────────────────────────────────
// SessionId type alias
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque session identifier (a plain `String` for minimal coupling).
pub type SessionId = String;

// ─────────────────────────────────────────────────────────────────────────────
// SessionEntry — internal storage
// ─────────────────────────────────────────────────────────────────────────────

/// Internal record kept for one agent session.
struct SessionEntry {
    /// Trait-object reference to the LLM provider.
    ///
    /// **Fixed vs. the broken pattern described in issue #897:**
    ///
    /// ```text
    /// // BEFORE (broken): concrete type dropped on reload
    /// plugin: OpenAiPlugin
    ///
    /// // AFTER (fixed): Arc<dyn LLMProvider> survives reload
    /// plugin: Arc<dyn LLMProvider>
    /// ```
    plugin: Arc<dyn LLMProvider>,

    /// Full conversation history for this session.
    history: Vec<ChatMessage>,

    /// ID of the plugin that owns this session (used for plugin-level ops).
    plugin_id: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// SessionSnapshot — public view
// ─────────────────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of a session's public state, returned by
/// [`SessionRegistry::get_session`].
///
/// Cloning the `Arc` here is O(1) and safe to do while the registry lock is
/// held briefly.
pub struct SessionSnapshot {
    /// Live provider reference (valid even after a hot-reload).
    pub plugin: Arc<dyn LLMProvider>,
    /// Cloned conversation history at snapshot time.
    pub history: Vec<ChatMessage>,
    /// The plugin this session belongs to.
    pub plugin_id: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// SessionRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe registry of active agent sessions.
#[derive(Clone, Default)]
pub struct SessionRegistry {
    sessions: Arc<RwLock<HashMap<SessionId, SessionEntry>>>,
}

impl SessionRegistry {
    /// Create a new, empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    // ─────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────

    /// Register a new session.
    ///
    /// If a session with the same `session_id` already exists it is replaced.
    pub async fn create_session(
        &self,
        session_id: SessionId,
        plugin_id: String,
        provider: Arc<dyn LLMProvider>,
    ) {
        let mut map = self.sessions.write().await;
        map.insert(
            session_id,
            SessionEntry {
                plugin: provider,
                history: Vec::new(),
                plugin_id,
            },
        );
    }

    /// Remove a session, logging a structured warning with `reason`.
    ///
    /// Never panics — silently no-ops if `session_id` is not found.
    pub async fn terminate_session(&self, session_id: &str, reason: &str) {
        let mut map = self.sessions.write().await;
        if map.remove(session_id).is_some() {
            tracing::warn!(
                session_id = session_id,
                reason = reason,
                "session terminated"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Read access
    // ─────────────────────────────────────────────────────────────────────

    /// Return a snapshot of the session, or `None` if not found.
    ///
    /// Returning `None` is always safe — callers must handle the absence
    /// gracefully instead of unwrapping.
    pub async fn get_session(&self, session_id: &str) -> Option<SessionSnapshot> {
        let map = self.sessions.read().await;
        map.get(session_id).map(|e| SessionSnapshot {
            plugin: Arc::clone(&e.plugin),
            history: e.history.clone(),
            plugin_id: e.plugin_id.clone(),
        })
    }

    /// Return a clone of the conversation history for `session_id`.
    ///
    /// # Errors
    ///
    /// * [`PluginError::SessionNotFound`] — no session exists with that ID.
    pub async fn get_history(&self, session_id: &str) -> Result<Vec<ChatMessage>, PluginError> {
        let map = self.sessions.read().await;
        map.get(session_id)
            .map(|e| e.history.clone())
            .ok_or_else(|| PluginError::SessionNotFound(session_id.to_string()))
    }

    // ─────────────────────────────────────────────────────────────────────
    // Mutation
    // ─────────────────────────────────────────────────────────────────────

    /// Append `message` to the conversation history of `session_id`.
    ///
    /// # Errors
    ///
    /// * [`PluginError::SessionNotFound`] — no session exists with that ID.
    pub async fn push_message(
        &self,
        session_id: &str,
        message: ChatMessage,
    ) -> Result<(), PluginError> {
        let mut map = self.sessions.write().await;
        map.get_mut(session_id)
            .ok_or_else(|| PluginError::SessionNotFound(session_id.to_string()))
            .map(|e| e.history.push(message))
    }

    // ─────────────────────────────────────────────────────────────────────
    // Drain-protocol callbacks (called by PluginManager)
    // ─────────────────────────────────────────────────────────────────────

    /// **Drain step 3** — snapshot / persist history for all sessions owned by `plugin_id`.
    ///
    /// In this implementation the history already lives in memory behind the
    /// `RwLock`, so no additional persistence is performed.  Downstream
    /// integrations may override this by subclassing or wrapping the registry.
    pub async fn checkpoint_all_for_plugin(&self, plugin_id: &str) {
        let map = self.sessions.read().await;
        let affected: Vec<&SessionId> = map
            .iter()
            .filter(|(_, e)| e.plugin_id == plugin_id)
            .map(|(id, _)| id)
            .collect();

        if !affected.is_empty() {
            tracing::info!(
                plugin_id = plugin_id,
                count = affected.len(),
                "checkpointing sessions before plugin swap"
            );
        }
        // History is already persisted in `e.history`; nothing else needed here.
    }

    /// **Drain step 5** — update the provider `Arc` for every session owned
    /// by `plugin_id` without touching conversation history.
    ///
    /// After this call, all affected sessions reference the new provider.
    pub async fn rebind_plugin(&self, plugin_id: &str, new_provider: Arc<dyn LLMProvider>) {
        let mut map = self.sessions.write().await;
        let mut rebound = 0usize;
        for entry in map.values_mut().filter(|e| e.plugin_id == plugin_id) {
            entry.plugin = Arc::clone(&new_provider);
            rebound += 1;
        }
        tracing::info!(
            plugin_id = plugin_id,
            rebound = rebound,
            "sessions rebound to new provider after hot-reload"
        );
    }

    /// **Drain timeout fallback** — gracefully terminate all sessions owned by
    /// `plugin_id` when the drain timeout is exceeded.
    ///
    /// Emits a `tracing::warn!` for every terminated session so operators can
    /// audit which sessions lost state.  Never panics.
    pub async fn terminate_sessions_for_plugin(&self, plugin_id: &str, reason: &str) {
        let mut map = self.sessions.write().await;
        let to_remove: Vec<SessionId> = map
            .iter()
            .filter(|(_, e)| e.plugin_id == plugin_id)
            .map(|(id, _)| id.clone())
            .collect();

        for session_id in &to_remove {
            tracing::warn!(
                session_id = %session_id,
                plugin_id = plugin_id,
                reason = reason,
                "gracefully terminating session due to drain timeout"
            );
            map.remove(session_id);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, FinishReason, MessageContent, Role,
    };
    use async_trait::async_trait;

    // ── Mock provider ────────────────────────────────────────────────────

    struct FakeProvider(String);

    impl FakeProvider {
        fn arc(name: &str) -> Arc<dyn LLMProvider> {
            Arc::new(FakeProvider(name.to_string()))
        }
    }

    #[async_trait]
    impl LLMProvider for FakeProvider {
        fn name(&self) -> &str {
            &self.0
        }
        async fn chat(
            &self,
            _r: ChatCompletionRequest,
        ) -> crate::agent::AgentResult<ChatCompletionResponse> {
            Ok(ChatCompletionResponse {
                choices: vec![Choice {
                    index: 0,
                    message: crate::llm::types::ChatMessage::assistant(self.0.clone()),
                    finish_reason: Some(FinishReason::Stop),
                    logprobs: None,
                }],
            })
        }
    }

    fn msg(text: &str) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            content: Some(MessageContent::Text(text.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Test: push and retrieve history
    // ────────────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn push_and_get_history() {
        let reg = SessionRegistry::new();
        reg.create_session("s1".into(), "p1".into(), FakeProvider::arc("p1"))
            .await;

        reg.push_message("s1", msg("hello")).await.unwrap();
        reg.push_message("s1", msg("world")).await.unwrap();

        let history = reg.get_history("s1").await.unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].text_content(), Some("hello"));
        assert_eq!(history[1].text_content(), Some("world"));
    }

    // ────────────────────────────────────────────────────────────────────
    // Test: get_session on missing ID returns None gracefully
    // ────────────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn get_session_returns_none_gracefully() {
        let reg = SessionRegistry::new();
        let snapshot = reg.get_session("does-not-exist").await;
        assert!(snapshot.is_none(), "must return None, not panic");
    }

    // ────────────────────────────────────────────────────────────────────
    // Test: rebind_plugin updates Arc but preserves history
    // ────────────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn rebind_updates_provider_preserves_history() {
        let reg = SessionRegistry::new();
        let p1 = FakeProvider::arc("provider-v1");
        let p2 = FakeProvider::arc("provider-v2");

        reg.create_session("s1".into(), "llm".into(), p1).await;
        reg.push_message("s1", msg("first message")).await.unwrap();

        // Rebind to v2
        reg.rebind_plugin("llm", p2).await;

        // History must be intact
        let history = reg.get_history("s1").await.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].text_content(), Some("first message"));

        // The bound provider must now be v2
        let snap = reg.get_session("s1").await.unwrap();
        let resp = snap.plugin.chat(ChatCompletionRequest::new("m")).await.unwrap();
        assert_eq!(resp.content(), Some("provider-v2"));
    }

    // ────────────────────────────────────────────────────────────────────
    // Test: terminate_sessions_for_plugin does not panic
    // ────────────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn terminate_sessions_does_not_panic() {
        let reg = SessionRegistry::new();
        reg.create_session("s1".into(), "llm".into(), FakeProvider::arc("llm"))
            .await;
        reg.create_session("s2".into(), "llm".into(), FakeProvider::arc("llm"))
            .await;
        reg.create_session("s3".into(), "other".into(), FakeProvider::arc("other"))
            .await;

        reg.terminate_sessions_for_plugin("llm", "drain timeout")
            .await;

        // s1 and s2 removed; s3 (different plugin) survives
        assert!(reg.get_session("s1").await.is_none());
        assert!(reg.get_session("s2").await.is_none());
        assert!(reg.get_session("s3").await.is_some());
    }
}
