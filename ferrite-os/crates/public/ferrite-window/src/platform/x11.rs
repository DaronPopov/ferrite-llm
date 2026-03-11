//! X11 window backend using `x11rb` (pure Rust X11 protocol).
//!
//! Uses `PutImage` with ZPixmap format for software framebuffer blitting.
//! No OpenGL, no GPU driver — just the X11 display server.

use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    self, AtomEnum, ConnectionExt, CreateGCAux, CreateWindowAux, EventMask, Gcontext,
    ImageFormat, PropMode, Window, WindowClass,
};
use x11rb::protocol::Event as X11Event;
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as WrapperConnectionExt;
use x11rb::COPY_DEPTH_FROM_PARENT;

use crate::event::{Event, Modifiers, MouseButton};
use crate::{WindowConfig, WindowError};

pub struct X11Window {
    conn: RustConnection,
    window: Window,
    gc: Gcontext,
    width: u32,
    height: u32,
    depth: u8,
    open: bool,
    wm_delete: u32,
}

impl X11Window {
    pub fn new(config: WindowConfig) -> Result<Self, WindowError> {
        let (conn, screen_num) = RustConnection::connect(None)
            .map_err(|e| WindowError::ConnectionFailed(format!("X11: {}", e)))?;

        let screen = &conn.setup().roots[screen_num];
        let depth = screen.root_depth;
        let window = conn.generate_id()
            .map_err(|e| WindowError::CreationFailed(format!("generate_id: {}", e)))?;
        let gc = conn.generate_id()
            .map_err(|e| WindowError::CreationFailed(format!("generate_id: {}", e)))?;

        let event_mask = EventMask::EXPOSURE
            | EventMask::STRUCTURE_NOTIFY
            | EventMask::KEY_PRESS
            | EventMask::KEY_RELEASE
            | EventMask::BUTTON_PRESS
            | EventMask::BUTTON_RELEASE
            | EventMask::POINTER_MOTION
            | EventMask::FOCUS_CHANGE;

        let win_aux = CreateWindowAux::new()
            .event_mask(event_mask)
            .background_pixel(screen.black_pixel);

        conn.create_window(
            COPY_DEPTH_FROM_PARENT,
            window,
            screen.root,
            0,
            0,
            config.width as u16,
            config.height as u16,
            0,
            WindowClass::INPUT_OUTPUT,
            0,
            &win_aux,
        )
        .map_err(|e| WindowError::CreationFailed(format!("create_window: {}", e)))?;

        // Create graphics context for PutImage
        conn.create_gc(gc, window, &CreateGCAux::new())
            .map_err(|e| WindowError::CreationFailed(format!("create_gc: {}", e)))?;

        // Set window title
        conn.change_property8(
            PropMode::REPLACE,
            window,
            AtomEnum::WM_NAME,
            AtomEnum::STRING,
            config.title.as_bytes(),
        )
        .map_err(|e| WindowError::CreationFailed(format!("set title: {}", e)))?;

        // Register for WM_DELETE_WINDOW so we get a clean close event
        let wm_protocols = intern_atom(&conn, b"WM_PROTOCOLS")?;
        let wm_delete = intern_atom(&conn, b"WM_DELETE_WINDOW")?;

        conn.change_property32(
            PropMode::REPLACE,
            window,
            wm_protocols,
            AtomEnum::ATOM,
            &[wm_delete],
        )
        .map_err(|e| WindowError::CreationFailed(format!("set WM_DELETE_WINDOW: {}", e)))?;

        // Resizable hint via WM_NORMAL_HINTS
        if !config.resizable {
            set_fixed_size(&conn, window, config.width, config.height)?;
        }

        // Fullscreen via _NET_WM_STATE_FULLSCREEN (EWMH standard)
        if config.fullscreen {
            let net_wm_state = intern_atom(&conn, b"_NET_WM_STATE")?;
            let net_wm_state_fullscreen = intern_atom(&conn, b"_NET_WM_STATE_FULLSCREEN")?;

            conn.change_property32(
                PropMode::REPLACE,
                window,
                net_wm_state,
                AtomEnum::ATOM,
                &[net_wm_state_fullscreen],
            )
            .map_err(|e| WindowError::CreationFailed(format!("set fullscreen: {}", e)))?;
        }

        // Map (show) the window
        conn.map_window(window)
            .map_err(|e| WindowError::CreationFailed(format!("map_window: {}", e)))?;
        conn.flush()
            .map_err(|e| WindowError::CreationFailed(format!("flush: {}", e)))?;

        Ok(Self {
            conn,
            window,
            gc,
            width: config.width,
            height: config.height,
            depth,
            open: true,
            wm_delete,
        })
    }

    pub fn poll_events(&mut self) -> Vec<Event> {
        let mut events = Vec::new();

        while let Ok(Some(event)) = self.conn.poll_for_event() {
            match event {
                X11Event::ClientMessage(msg) => {
                    // WM_DELETE_WINDOW
                    if msg.data.as_data32()[0] == self.wm_delete {
                        self.open = false;
                        events.push(Event::Close);
                    }
                }
                X11Event::ConfigureNotify(cfg) => {
                    let w = cfg.width as u32;
                    let h = cfg.height as u32;
                    if w != self.width || h != self.height {
                        self.width = w;
                        self.height = h;
                        events.push(Event::Resize { width: w, height: h });
                    }
                }
                X11Event::KeyPress(key) => {
                    events.push(Event::KeyDown {
                        keycode: key.detail as u32,
                        modifiers: x11_modifiers(key.state),
                    });
                }
                X11Event::KeyRelease(key) => {
                    events.push(Event::KeyUp {
                        keycode: key.detail as u32,
                        modifiers: x11_modifiers(key.state),
                    });
                }
                X11Event::ButtonPress(btn) => {
                    if let Some(button) = x11_button(btn.detail) {
                        events.push(Event::MouseDown {
                            button,
                            x: btn.event_x as i32,
                            y: btn.event_y as i32,
                        });
                    }
                }
                X11Event::ButtonRelease(btn) => {
                    if let Some(button) = x11_button(btn.detail) {
                        events.push(Event::MouseUp {
                            button,
                            x: btn.event_x as i32,
                            y: btn.event_y as i32,
                        });
                    }
                }
                X11Event::MotionNotify(motion) => {
                    events.push(Event::MouseMove {
                        x: motion.event_x as i32,
                        y: motion.event_y as i32,
                    });
                }
                X11Event::FocusIn(_) => events.push(Event::FocusIn),
                X11Event::FocusOut(_) => events.push(Event::FocusOut),
                _ => {}
            }
        }

        events
    }

    pub fn present(
        &mut self,
        pixels: &[u32],
        width: u32,
        height: u32,
    ) -> Result<(), WindowError> {
        if pixels.len() < (width * height) as usize {
            return Err(WindowError::PresentFailed(format!(
                "pixel buffer too small: {} < {}",
                pixels.len(),
                width * height
            )));
        }

        // X11 ZPixmap expects BGRX (blue in low byte) for 24-bit depth on
        // little-endian. Convert RGBA → BGRX in-place into a byte buffer.
        let num_pixels = (width * height) as usize;
        let mut data = Vec::with_capacity(num_pixels * 4);
        for &px in &pixels[..num_pixels] {
            let r = ((px >> 24) & 0xFF) as u8;
            let g = ((px >> 16) & 0xFF) as u8;
            let b = ((px >> 8) & 0xFF) as u8;
            // BGRX (X11 little-endian 32bpp)
            data.push(b);
            data.push(g);
            data.push(r);
            data.push(0xFF); // padding
        }

        self.conn
            .put_image(
                ImageFormat::Z_PIXMAP,
                self.window,
                self.gc,
                width as u16,
                height as u16,
                0,
                0,
                0,
                self.depth,
                &data,
            )
            .map_err(|e| WindowError::PresentFailed(format!("put_image: {}", e)))?;

        self.conn
            .flush()
            .map_err(|e| WindowError::PresentFailed(format!("flush: {}", e)))?;

        Ok(())
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    pub fn set_title(&mut self, title: &str) {
        let _ = self.conn.change_property8(
            PropMode::REPLACE,
            self.window,
            AtomEnum::WM_NAME,
            AtomEnum::STRING,
            title.as_bytes(),
        );
        let _ = self.conn.flush();
    }
}

impl Drop for X11Window {
    fn drop(&mut self) {
        let _ = self.conn.free_gc(self.gc);
        let _ = self.conn.destroy_window(self.window);
        let _ = self.conn.flush();
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn intern_atom(conn: &RustConnection, name: &[u8]) -> Result<u32, WindowError> {
    conn.intern_atom(false, name)
        .map_err(|e| WindowError::CreationFailed(format!("intern_atom: {}", e)))?
        .reply()
        .map(|r| r.atom)
        .map_err(|e| WindowError::CreationFailed(format!("intern_atom reply: {}", e)))
}

fn set_fixed_size(
    conn: &RustConnection,
    window: Window,
    w: u32,
    h: u32,
) -> Result<(), WindowError> {
    // WM_NORMAL_HINTS: set min_size == max_size to prevent resizing.
    // The structure is defined by ICCCM:
    //   flags (4 bytes), then pad(4)*4, then min_width/min_height,
    //   max_width/max_height, ...
    // flags bit 4 = PMinSize, bit 5 = PMaxSize
    let flags: u32 = (1 << 4) | (1 << 5); // PMinSize | PMaxSize
    let mut data = [0u32; 18]; // WM_SIZE_HINTS has 18 u32 fields
    data[0] = flags;
    // fields 5,6 = min_width, min_height
    data[5] = w;
    data[6] = h;
    // fields 7,8 = max_width, max_height
    data[7] = w;
    data[8] = h;

    let wm_normal_hints = intern_atom(conn, b"WM_NORMAL_HINTS")?;
    let wm_size_hints = intern_atom(conn, b"WM_SIZE_HINTS")?;

    conn.change_property32(
        PropMode::REPLACE,
        window,
        wm_normal_hints,
        wm_size_hints,
        &data,
    )
    .map_err(|e| WindowError::CreationFailed(format!("set size hints: {}", e)))?;

    Ok(())
}

fn x11_modifiers(state: xproto::KeyButMask) -> Modifiers {
    Modifiers {
        shift: state.contains(xproto::KeyButMask::SHIFT),
        ctrl: state.contains(xproto::KeyButMask::CONTROL),
        alt: state.contains(xproto::KeyButMask::MOD1),
    }
}

fn x11_button(detail: u8) -> Option<MouseButton> {
    match detail {
        1 => Some(MouseButton::Left),
        2 => Some(MouseButton::Middle),
        3 => Some(MouseButton::Right),
        _ => None, // scroll wheel (4/5) etc.
    }
}
