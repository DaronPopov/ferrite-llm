pub mod editor;
pub mod fxscript;
pub mod layout;
pub mod profiling;
pub mod state;
pub mod style;
pub mod widgets;

pub mod agent;
mod app;
mod commands;
mod demo;
mod files;
mod workspace;

pub use app::run_tui;
