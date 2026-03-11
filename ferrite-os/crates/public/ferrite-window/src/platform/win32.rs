//! Win32 window backend using `windows-sys` (raw Win32 API).
//!
//! Uses `StretchDIBits` for software framebuffer blitting.
//! No DirectX, OpenGL, or GPU driver — just GDI.

#![cfg(windows)]

use std::ffi::OsStr;
use std::mem;
use std::os::windows::ffi::OsStrExt;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{
    BeginPaint, EndPaint, GetDC, ReleaseDC, StretchDIBits, BITMAPINFO, BITMAPINFOHEADER,
    BI_RGB, DIB_RGB_COLORS, PAINTSTRUCT, SRCCOPY,
};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::WindowsAndMessaging::{
    AdjustWindowRectEx, CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW,
    GetClientRect, GetSystemMetrics, LoadCursorW, PeekMessageW, PostQuitMessage,
    RegisterClassExW, SetWindowTextW, ShowWindow, TranslateMessage, CS_HREDRAW, CS_VREDRAW,
    CW_USEDEFAULT, IDC_ARROW, MSG, PM_REMOVE, SM_CXSCREEN, SM_CYSCREEN, SW_SHOW,
    WM_CLOSE, WM_DESTROY, WM_KEYDOWN, WM_KEYUP, WM_LBUTTONDOWN, WM_LBUTTONUP,
    WM_MBUTTONDOWN, WM_MBUTTONUP, WM_MOUSEMOVE, WM_PAINT, WM_RBUTTONDOWN, WM_RBUTTONUP,
    WM_SETFOCUS, WM_KILLFOCUS, WM_SIZE, WNDCLASSEXW, WS_OVERLAPPEDWINDOW, WS_POPUP,
    WS_EX_APPWINDOW, WS_VISIBLE,
};

use crate::event::{Event, Modifiers, MouseButton};
use crate::{WindowConfig, WindowError};

// Thread-local storage for the event queue — Win32 wndproc is a C callback
// that can't carry Rust state, so we use a thread-local.
thread_local! {
    static EVENTS: std::cell::RefCell<Vec<Event>> = std::cell::RefCell::new(Vec::new());
    static WINDOW_OPEN: std::cell::Cell<bool> = std::cell::Cell::new(true);
    static WINDOW_SIZE: std::cell::Cell<(u32, u32)> = std::cell::Cell::new((0, 0));
}

pub struct Win32Window {
    hwnd: HWND,
    width: u32,
    height: u32,
}

impl Win32Window {
    pub fn new(config: WindowConfig) -> Result<Self, WindowError> {
        unsafe {
            let hinstance = GetModuleHandleW(ptr::null());
            if hinstance == 0 {
                return Err(WindowError::CreationFailed("GetModuleHandleW failed".into()));
            }

            let class_name = wide_string("FerritWindowClass");

            let wc = WNDCLASSEXW {
                cbSize: mem::size_of::<WNDCLASSEXW>() as u32,
                style: CS_HREDRAW | CS_VREDRAW,
                lpfnWndProc: Some(wnd_proc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: hinstance,
                hIcon: 0,
                hCursor: LoadCursorW(0, IDC_ARROW),
                hbrBackground: 0,
                lpszMenuName: ptr::null(),
                lpszClassName: class_name.as_ptr(),
                hIconSm: 0,
            };

            if RegisterClassExW(&wc) == 0 {
                // Class may already be registered from a previous window — that's fine.
            }

            let title = wide_string(&config.title);

            let (style, ex_style, x, y, w, h) = if config.fullscreen {
                // Borderless fullscreen: WS_POPUP covers the entire screen.
                let screen_w = GetSystemMetrics(SM_CXSCREEN);
                let screen_h = GetSystemMetrics(SM_CYSCREEN);
                (WS_POPUP | WS_VISIBLE, 0u32, 0, 0, screen_w, screen_h)
            } else {
                // Windowed: compute rect that gives the desired client area.
                let style = WS_OVERLAPPEDWINDOW;
                let ex_style = WS_EX_APPWINDOW;
                let mut rect = RECT {
                    left: 0,
                    top: 0,
                    right: config.width as i32,
                    bottom: config.height as i32,
                };
                AdjustWindowRectEx(&mut rect, style, 0, ex_style);
                (
                    style,
                    ex_style,
                    CW_USEDEFAULT,
                    CW_USEDEFAULT,
                    rect.right - rect.left,
                    rect.bottom - rect.top,
                )
            };

            let hwnd = CreateWindowExW(
                ex_style,
                class_name.as_ptr(),
                title.as_ptr(),
                style,
                x,
                y,
                w,
                h,
                0,
                0,
                hinstance,
                ptr::null(),
            );

            if hwnd == 0 {
                return Err(WindowError::CreationFailed("CreateWindowExW returned NULL".into()));
            }

            WINDOW_OPEN.with(|o| o.set(true));
            WINDOW_SIZE.with(|s| s.set((config.width, config.height)));

            ShowWindow(hwnd, SW_SHOW);

            Ok(Self {
                hwnd,
                width: config.width,
                height: config.height,
            })
        }
    }

    pub fn poll_events(&mut self) -> Vec<Event> {
        unsafe {
            let mut msg: MSG = mem::zeroed();
            while PeekMessageW(&mut msg, self.hwnd, 0, 0, PM_REMOVE) != 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }

        // Sync cached size from the thread-local.
        WINDOW_SIZE.with(|s| {
            let (w, h) = s.get();
            self.width = w;
            self.height = h;
        });

        EVENTS.with(|e| e.borrow_mut().drain(..).collect())
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

        unsafe {
            let hdc = GetDC(self.hwnd);
            if hdc == 0 {
                return Err(WindowError::PresentFailed("GetDC failed".into()));
            }

            // BITMAPINFO for a top-down 32bpp DIB.
            // Use negative biHeight for top-down scanline order.
            let bmi = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: width as i32,
                    biHeight: -(height as i32), // negative = top-down
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB as u32,
                    biSizeImage: 0,
                    biXPelsPerMeter: 0,
                    biYPelsPerMeter: 0,
                    biClrUsed: 0,
                    biClrImportant: 0,
                },
                bmiColors: [mem::zeroed()],
            };

            // Convert RGBA (0xRRGGBBAA) → BGRA (Win32 expects 0x00RRGGBB in BI_RGB).
            let num_pixels = (width * height) as usize;
            let mut bgra = Vec::with_capacity(num_pixels);
            for &px in &pixels[..num_pixels] {
                let r = (px >> 24) & 0xFF;
                let g = (px >> 16) & 0xFF;
                let b = (px >> 8) & 0xFF;
                bgra.push((b) | (g << 8) | (r << 16));
            }

            let mut client_rect: RECT = mem::zeroed();
            GetClientRect(self.hwnd, &mut client_rect);
            let dst_w = client_rect.right - client_rect.left;
            let dst_h = client_rect.bottom - client_rect.top;

            StretchDIBits(
                hdc,
                0,
                0,
                dst_w,
                dst_h,
                0,
                0,
                width as i32,
                height as i32,
                bgra.as_ptr() as *const _,
                &bmi,
                DIB_RGB_COLORS,
                SRCCOPY,
            );

            ReleaseDC(self.hwnd, hdc);
        }

        Ok(())
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn is_open(&self) -> bool {
        WINDOW_OPEN.with(|o| o.get())
    }

    pub fn set_title(&mut self, title: &str) {
        let wide = wide_string(title);
        unsafe {
            SetWindowTextW(self.hwnd, wide.as_ptr());
        }
    }
}

impl Drop for Win32Window {
    fn drop(&mut self) {
        unsafe {
            DestroyWindow(self.hwnd);
        }
    }
}

// ---------------------------------------------------------------------------
// Win32 window procedure (C callback)
// ---------------------------------------------------------------------------

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_CLOSE => {
            WINDOW_OPEN.with(|o| o.set(false));
            push_event(Event::Close);
            0
        }
        WM_DESTROY => {
            PostQuitMessage(0);
            0
        }
        WM_SIZE => {
            let w = (lparam & 0xFFFF) as u32;
            let h = ((lparam >> 16) & 0xFFFF) as u32;
            WINDOW_SIZE.with(|s| s.set((w, h)));
            push_event(Event::Resize { width: w, height: h });
            0
        }
        WM_KEYDOWN => {
            push_event(Event::KeyDown {
                keycode: wparam as u32,
                modifiers: win32_modifiers(),
            });
            0
        }
        WM_KEYUP => {
            push_event(Event::KeyUp {
                keycode: wparam as u32,
                modifiers: win32_modifiers(),
            });
            0
        }
        WM_LBUTTONDOWN => {
            push_mouse_down(MouseButton::Left, lparam);
            0
        }
        WM_LBUTTONUP => {
            push_mouse_up(MouseButton::Left, lparam);
            0
        }
        WM_RBUTTONDOWN => {
            push_mouse_down(MouseButton::Right, lparam);
            0
        }
        WM_RBUTTONUP => {
            push_mouse_up(MouseButton::Right, lparam);
            0
        }
        WM_MBUTTONDOWN => {
            push_mouse_down(MouseButton::Middle, lparam);
            0
        }
        WM_MBUTTONUP => {
            push_mouse_up(MouseButton::Middle, lparam);
            0
        }
        WM_MOUSEMOVE => {
            let x = (lparam & 0xFFFF) as i16 as i32;
            let y = ((lparam >> 16) & 0xFFFF) as i16 as i32;
            push_event(Event::MouseMove { x, y });
            0
        }
        WM_SETFOCUS => {
            push_event(Event::FocusIn);
            0
        }
        WM_KILLFOCUS => {
            push_event(Event::FocusOut);
            0
        }
        WM_PAINT => {
            let mut ps: PAINTSTRUCT = mem::zeroed();
            BeginPaint(hwnd, &mut ps);
            EndPaint(hwnd, &ps);
            0
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn push_event(event: Event) {
    EVENTS.with(|e| e.borrow_mut().push(event));
}

fn push_mouse_down(button: MouseButton, lparam: LPARAM) {
    let x = (lparam & 0xFFFF) as i16 as i32;
    let y = ((lparam >> 16) & 0xFFFF) as i16 as i32;
    push_event(Event::MouseDown { button, x, y });
}

fn push_mouse_up(button: MouseButton, lparam: LPARAM) {
    let x = (lparam & 0xFFFF) as i16 as i32;
    let y = ((lparam >> 16) & 0xFFFF) as i16 as i32;
    push_event(Event::MouseUp { button, x, y });
}

fn win32_modifiers() -> Modifiers {
    unsafe {
        use windows_sys::Win32::UI::WindowsAndMessaging::GetKeyState;
        Modifiers {
            shift: GetKeyState(0x10) < 0,  // VK_SHIFT
            ctrl: GetKeyState(0x11) < 0,   // VK_CONTROL
            alt: GetKeyState(0x12) < 0,    // VK_MENU
        }
    }
}

fn wide_string(s: &str) -> Vec<u16> {
    OsStr::new(s).encode_wide().chain(std::iter::once(0)).collect()
}
