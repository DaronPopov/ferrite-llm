//! Platform-agnostic window events.

/// Mouse button identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

/// Key state modifier flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
}

/// A window event.
#[derive(Debug, Clone)]
pub enum Event {
    /// Window close requested (e.g. user clicked X).
    Close,

    /// Window was resized.
    Resize { width: u32, height: u32 },

    /// Key pressed. `keycode` is the platform-native scancode.
    KeyDown { keycode: u32, modifiers: Modifiers },

    /// Key released.
    KeyUp { keycode: u32, modifiers: Modifiers },

    /// Mouse moved to position within the window.
    MouseMove { x: i32, y: i32 },

    /// Mouse button pressed.
    MouseDown { button: MouseButton, x: i32, y: i32 },

    /// Mouse button released.
    MouseUp { button: MouseButton, x: i32, y: i32 },

    /// Window gained focus.
    FocusIn,

    /// Window lost focus.
    FocusOut,
}
