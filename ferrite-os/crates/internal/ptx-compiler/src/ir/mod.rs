//! Intermediate Representation for computation graphs.

pub mod node;
pub mod tensor;
pub mod graph;

pub use node::{Node, NodeId, OpCode, OpAttrs};
pub use tensor::{TensorId, TensorMeta};
pub use graph::Graph;
