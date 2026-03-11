#[cfg(feature = "cuda-gl-interop")]
use std::ffi::CString;

#[cfg(feature = "cuda-gl-interop")]
use gl::types::GLuint;

#[cfg(feature = "cuda-gl-interop")]
pub struct GlxInteropBuffer {
    display: *mut x11::xlib::Display,
    window: x11::xlib::Window,
    context: x11::glx::GLXContext,
    vbo: GLuint,
}

#[cfg(feature = "cuda-gl-interop")]
impl GlxInteropBuffer {
    pub fn new(size_bytes: usize) -> Result<Self, String> {
        unsafe {
            let display = x11::xlib::XOpenDisplay(std::ptr::null());
            if display.is_null() {
                return Err("XOpenDisplay failed".to_string());
            }

            let screen = x11::xlib::XDefaultScreen(display);
            let root = x11::xlib::XRootWindow(display, screen);
            let mut attrs = [
                x11::glx::GLX_RGBA,
                x11::glx::GLX_DOUBLEBUFFER,
                x11::glx::GLX_DEPTH_SIZE,
                24,
                0,
            ];
            let vi = x11::glx::glXChooseVisual(display, screen, attrs.as_mut_ptr());
            if vi.is_null() {
                x11::xlib::XCloseDisplay(display);
                return Err("glXChooseVisual failed".to_string());
            }

            let cmap = x11::xlib::XCreateColormap(
                display,
                root,
                (*vi).visual,
                x11::xlib::AllocNone,
            );
            let mut swa: x11::xlib::XSetWindowAttributes = std::mem::zeroed();
            swa.colormap = cmap;
            swa.event_mask = x11::xlib::ExposureMask;
            let window = x11::xlib::XCreateWindow(
                display,
                root,
                0,
                0,
                16,
                16,
                0,
                (*vi).depth,
                x11::xlib::InputOutput as u32,
                (*vi).visual,
                x11::xlib::CWColormap | x11::xlib::CWEventMask,
                &mut swa,
            );

            let context = x11::glx::glXCreateContext(display, vi, std::ptr::null_mut(), 1);
            x11::xlib::XFree(vi as *mut _);
            if context.is_null() {
                x11::xlib::XDestroyWindow(display, window);
                x11::xlib::XCloseDisplay(display);
                return Err("glXCreateContext failed".to_string());
            }
            if x11::glx::glXMakeCurrent(display, window, context) == 0 {
                x11::glx::glXDestroyContext(display, context);
                x11::xlib::XDestroyWindow(display, window);
                x11::xlib::XCloseDisplay(display);
                return Err("glXMakeCurrent failed".to_string());
            }

            gl::load_with(|name| {
                let cname = CString::new(name).ok();
                if let Some(cstr) = cname {
                    match x11::glx::glXGetProcAddress(cstr.as_ptr() as *const u8) {
                        Some(sym) => sym as *const () as *const _,
                        None => std::ptr::null(),
                    }
                } else {
                    std::ptr::null()
                }
            });

            let mut vbo: GLuint = 0;
            gl::GenBuffers(1, &mut vbo as *mut _);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                size_bytes as isize,
                std::ptr::null(),
                gl::DYNAMIC_DRAW,
            );
            gl::BindBuffer(gl::ARRAY_BUFFER, 0);

            Ok(Self {
                display,
                window,
                context,
                vbo,
            })
        }
    }

    pub fn buffer_id(&self) -> u32 {
        self.vbo
    }
}

#[cfg(feature = "cuda-gl-interop")]
impl Drop for GlxInteropBuffer {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &self.vbo as *const _);
            x11::glx::glXMakeCurrent(self.display, 0, std::ptr::null_mut());
            x11::glx::glXDestroyContext(self.display, self.context);
            x11::xlib::XDestroyWindow(self.display, self.window);
            x11::xlib::XCloseDisplay(self.display);
        }
    }
}
