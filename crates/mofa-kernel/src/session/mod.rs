//! Session management for `mofa-kernel`.
//!
//! The primary type is [`registry::SessionRegistry`], which stores active
//! agent sessions and co-ordinates with [`crate::plugin::manager::PluginManager`]
//! during hot-reloads.

pub mod registry;

pub use registry::{SessionId, SessionRegistry, SessionSnapshot};
