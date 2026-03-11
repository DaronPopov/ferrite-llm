//! High-level builder SDK for daemon-native GPU compute applications.
//!
//! `ptx-app` provides a concise builder-pattern API for writing GPU compute
//! tasks that integrate with the Ferrite daemon job supervisor:
//!
//! ```no_run
//! use ptx_app::{FerApp, Restart, DType};
//!
//! FerApp::new("matrix-solver")
//!     .pool_fraction(0.4)
//!     .streams(8)
//!     .restart(Restart::on_failure(3))
//!     .run(|ctx| {
//!         let a = ctx.tensor(&[1024, 1024], DType::F32)?.randn()?;
//!         let b = ctx.tensor(&[1024, 1024], DType::F32)?.randn()?;
//!         ctx.emit("result", &"done");
//!         Ok(())
//!     })
//!     .expect("app failed");
//! ```

pub mod builder;
pub mod ctx;
pub mod restart;
pub mod resource;
pub mod error;
pub(crate) mod daemon_client;
pub(crate) mod checkpoint;
pub(crate) mod emit;
pub mod tensor_factory;

// Public re-exports for the top-level API.
pub use builder::FerApp;
pub use ctx::{Ctx, PoolStats};
pub use restart::Restart;
pub use resource::Priority;
pub use error::AppError;
pub use tensor_factory::TensorBuilder;

// Re-export DType from ptx-tensor so users don't need a separate dependency.
pub use ptx_tensor::DType;

// Re-export ProgressBar so users can call .inc(), .finish(), etc.
// without adding indicatif as a direct dependency.
pub use indicatif::ProgressBar;
