mod cuda_gl;
#[cfg(feature = "cuda-gl-interop")]
mod glx_buffer;

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;
use x11rb::connection::Connection;
use x11rb::protocol::Event;
use x11rb::protocol::xproto::{
    AtomEnum, ChangeGCAux, ClientMessageEvent, ConfigureNotifyEvent, ConnectionExt, CoordMode,
    CreateGCAux, CreateWindowAux, EventMask, Gcontext, Point, PropMode, Rectangle, Window,
    WindowClass,
};
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as _;
use x11rb::COPY_DEPTH_FROM_PARENT;

#[derive(Debug, Clone)]
enum SceneKind {
    Wave,
    Surface,
    Tensor,
}

impl SceneKind {
    fn from_name(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "surface" | "surf" => Self::Surface,
            "tensor" | "cloud" => Self::Tensor,
            _ => Self::Wave,
        }
    }

}

#[derive(Debug, Clone)]
struct TensorPayload {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Debug)]
struct RenderState {
    scene: SceneKind,
    tensor: Option<TensorPayload>,
    tensor_dirty: bool,
    phase: f32,
}

#[derive(Debug)]
enum Control {
    SetScene(SceneKind),
    SetTensor(TensorPayload),
    #[cfg(feature = "cuda-gl-interop")]
    SetTensorIpc { handle_hex: String, len_f32: usize },
    Shutdown,
}

#[derive(Debug, Deserialize)]
struct CommandEnvelope {
    cmd: String,
    scene: Option<String>,
    shape: Option<Vec<usize>>,
    data: Option<Vec<f32>>,
    ipc_handle: Option<String>,
    len_f32: Option<usize>,
}

fn parse_socket_path() -> PathBuf {
    let mut args = std::env::args().skip(1);
    let mut socket = PathBuf::from("/tmp/ferrite-renderd.sock");
    while let Some(arg) = args.next() {
        if arg == "--socket" {
            if let Some(path) = args.next() {
                socket = PathBuf::from(path);
            }
        }
    }
    socket
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let socket_path = parse_socket_path();
    let running = Arc::new(AtomicBool::new(true));
    let (tx, rx) = mpsc::channel::<Control>();

    let running_listener = running.clone();
    let socket_path_listener = socket_path.clone();
    thread::spawn(move || {
        if let Err(e) = run_socket_listener(&socket_path_listener, tx, running_listener) {
            eprintln!("renderd socket error: {}", e);
        }
    });

    let render_result = run_x11_window(rx, running.clone());
    running.store(false, Ordering::Relaxed);
    let _ = std::fs::remove_file(&socket_path);
    render_result
}

fn run_socket_listener(
    socket_path: &Path,
    tx: Sender<Control>,
    running: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = std::fs::remove_file(socket_path);
    let listener = UnixListener::bind(socket_path)?;
    listener.set_nonblocking(true)?;

    while running.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((mut stream, _)) => {
                let mut reader = BufReader::new(stream.try_clone()?);
                let mut line = String::new();
                while reader.read_line(&mut line)? > 0 {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        let msg = handle_command(trimmed, &tx);
                        let _ = writeln!(stream, "{}", msg);
                        if msg == "ok:shutdown" {
                            running.store(false, Ordering::Relaxed);
                            break;
                        }
                    }
                    line.clear();
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(25));
            }
            Err(e) => return Err(Box::new(e)),
        }
    }

    Ok(())
}

fn handle_command(line: &str, tx: &Sender<Control>) -> String {
    let env: CommandEnvelope = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return format!("err:parse {}", e),
    };

    match env.cmd.as_str() {
        "ping" => "ok:pong".to_string(),
        "scene" => {
            let scene = SceneKind::from_name(env.scene.as_deref().unwrap_or("wave"));
            if tx.send(Control::SetScene(scene)).is_ok() {
                "ok:scene".to_string()
            } else {
                "err:channel".to_string()
            }
        }
        "tensor" => {
            let shape = env.shape.unwrap_or_default();
            let data = env.data.unwrap_or_default();
            if tx
                .send(Control::SetTensor(TensorPayload { shape, data }))
                .is_ok()
            {
                "ok:tensor".to_string()
            } else {
                "err:channel".to_string()
            }
        }
        "tensor_ipc" => {
            let Some(handle_hex) = env.ipc_handle else {
                return "err:missing ipc_handle".to_string();
            };
            let len_f32 = env.len_f32.unwrap_or(0);
            #[cfg(feature = "cuda-gl-interop")]
            {
                if tx
                    .send(Control::SetTensorIpc { handle_hex, len_f32 })
                    .is_ok()
                {
                    "ok:tensor_ipc".to_string()
                } else {
                    "err:channel".to_string()
                }
            }
            #[cfg(not(feature = "cuda-gl-interop"))]
            {
                let _ = (handle_hex, len_f32);
                "err:cuda_gl_interop_disabled".to_string()
            }
        }
        "shutdown" => {
            let _ = tx.send(Control::Shutdown);
            "ok:shutdown".to_string()
        }
        other => format!("err:unknown {}", other),
    }
}

fn run_x11_window(
    rx: Receiver<Control>,
    running: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (conn, screen_num) = x11rb::connect(None)?;
    let screen = &conn.setup().roots[screen_num];
    let win = conn.generate_id()?;
    let gc = conn.generate_id()?;

    conn.create_window(
        COPY_DEPTH_FROM_PARENT,
        win,
        screen.root,
        0,
        0,
        1024,
        640,
        0,
        WindowClass::INPUT_OUTPUT,
        screen.root_visual,
        &CreateWindowAux::new()
            .background_pixel(screen.black_pixel)
            .event_mask(
                EventMask::EXPOSURE
                    | EventMask::KEY_PRESS
                    | EventMask::STRUCTURE_NOTIFY
                    | EventMask::BUTTON_PRESS,
            ),
    )?;
    conn.create_gc(
        gc,
        win,
        &CreateGCAux::new()
            .foreground(screen.white_pixel)
            .background(screen.black_pixel),
    )?;
    conn.change_property8(
        PropMode::REPLACE,
        win,
        AtomEnum::WM_NAME,
        AtomEnum::STRING,
        b"Ferrite RenderD - 3D Scene",
    )?;

    let wm_protocols = intern_atom(&conn, false, b"WM_PROTOCOLS")?;
    let wm_delete = intern_atom(&conn, false, b"WM_DELETE_WINDOW")?;
    conn.change_property32(
        PropMode::REPLACE,
        win,
        wm_protocols,
        AtomEnum::ATOM,
        &[wm_delete],
    )?;
    conn.map_window(win)?;
    conn.flush()?;

    let palette = create_palette(&conn, screen.default_colormap)?;
    let mut w = 1024u16;
    let mut h = 640u16;
    let mut state = RenderState {
        scene: SceneKind::Wave,
        tensor: None,
        tensor_dirty: false,
        phase: 0.0,
    };
    let mut interop = match cuda_gl::InteropPipeline::new_mock(1) {
        Ok(p) => Some(p),
        Err(_) => None,
    };
    #[cfg(feature = "cuda-gl-interop")]
    let mut use_mock_interop = true;
    #[cfg(not(feature = "cuda-gl-interop"))]
    let use_mock_interop = true;
    #[cfg(feature = "cuda-gl-interop")]
    let mut zero_copy = {
        let glx = glx_buffer::GlxInteropBuffer::new(1024 * 1024);
        match glx {
            Ok(owner) => match cuda_gl::ZeroCopyInteropPipeline::new(owner.buffer_id(), std::ptr::null_mut()) {
                Ok(pipe) => {
                    use_mock_interop = false;
                    Some((owner, pipe))
                }
                Err(_) => None,
            },
            Err(_) => None,
        }
    };
    let mut last = Instant::now();

    while running.load(Ordering::Relaxed) {
        while let Ok(msg) = rx.try_recv() {
            match msg {
                Control::SetScene(scene) => state.scene = scene,
                Control::SetTensor(payload) => {
                    state.scene = SceneKind::Tensor;
                    state.tensor = Some(payload);
                    state.tensor_dirty = true;
                }
                #[cfg(feature = "cuda-gl-interop")]
                Control::SetTensorIpc { handle_hex, len_f32 } => {
                    #[cfg(feature = "cuda-gl-interop")]
                    if let Some((_owner, pipe)) = zero_copy.as_mut() {
                        let _ = pipe.upload_from_ipc_hex(&handle_hex, len_f32);
                    }
                    state.scene = SceneKind::Tensor;
                    state.tensor_dirty = false;
                }
                Control::Shutdown => {
                    running.store(false, Ordering::Relaxed);
                }
            }
        }

        while let Some(event) = conn.poll_for_event()? {
            if handle_x11_event(event, wm_delete, &mut w, &mut h) {
                running.store(false, Ordering::Relaxed);
            }
        }

        let dt = last.elapsed().as_secs_f32();
        last = Instant::now();
        state.phase += (dt * 2.2).max(0.01);
        if use_mock_interop {
            if let Some(pipe) = interop.as_mut() {
                let _ = pipe.upload_f32(&[state.phase, w as f32, h as f32]);
            }
        }
        #[cfg(feature = "cuda-gl-interop")]
        if let Some((_owner, pipe)) = zero_copy.as_mut() {
            if matches!(state.scene, SceneKind::Tensor) && state.tensor_dirty {
                if let Some(t) = state.tensor.as_ref() {
                    let _ = pipe.upload_host_via_device(&t.data);
                    state.tensor_dirty = false;
                }
            }
        }

        render_frame(&conn, win, gc, &palette, &state, w, h)?;
        conn.flush()?;
        thread::sleep(Duration::from_millis(16));
    }

    Ok(())
}

fn intern_atom(
    conn: &RustConnection,
    only_if_exists: bool,
    name: &[u8],
) -> Result<u32, Box<dyn std::error::Error>> {
    Ok(conn.intern_atom(only_if_exists, name)?.reply()?.atom)
}

fn create_palette(
    conn: &RustConnection,
    colormap: u32,
) -> Result<[u32; 5], Box<dyn std::error::Error>> {
    let colors = [
        (30, 40, 56),
        (82, 139, 255),
        (98, 210, 160),
        (255, 192, 96),
        (255, 110, 110),
    ];
    let mut out = [0u32; 5];
    for (i, (r, g, b)) in colors.into_iter().enumerate() {
        let reply = conn
            .alloc_color(
                colormap,
                (r as u16) << 8,
                (g as u16) << 8,
                (b as u16) << 8,
            )?
            .reply()?;
        out[i] = reply.pixel;
    }
    Ok(out)
}

fn handle_x11_event(
    event: Event,
    wm_delete: u32,
    width: &mut u16,
    height: &mut u16,
) -> bool {
    match event {
        Event::ClientMessage(ClientMessageEvent { data, .. }) => data.as_data32()[0] == wm_delete,
        Event::ConfigureNotify(ConfigureNotifyEvent {
            width: w,
            height: h,
            ..
        }) => {
            *width = w.max(320);
            *height = h.max(240);
            false
        }
        Event::KeyPress(_) => true,
        _ => false,
    }
}

fn render_frame(
    conn: &RustConnection,
    window: Window,
    gc: Gcontext,
    palette: &[u32; 5],
    state: &RenderState,
    width: u16,
    height: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    conn.change_gc(gc, &ChangeGCAux::new().foreground(palette[0]))?;
    conn.poly_fill_rectangle(
        window,
        gc,
        &[Rectangle {
            x: 0,
            y: 0,
            width,
            height,
        }],
    )?;

    let mut pts = scene_points(state);
    pts.sort_by(|a, b| a.z.partial_cmp(&b.z).unwrap_or(std::cmp::Ordering::Equal));

    let mut buckets: [Vec<Point>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for p in pts {
        if let Some((sx, sy, depth, weight)) = project_point(p, width, height, state.phase) {
            let bucket = ((depth * 3.0) as usize).min(3).max((weight * 0.5) as usize);
            buckets[bucket].push(Point { x: sx, y: sy });
        }
    }

    for (i, points) in buckets.iter().enumerate() {
        if points.is_empty() {
            continue;
        }
        conn.change_gc(gc, &ChangeGCAux::new().foreground(palette[i + 1]))?;
        conn.poly_point(CoordMode::ORIGIN, window, gc, points)?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct P3 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

fn scene_points(state: &RenderState) -> Vec<P3> {
    match state.scene {
        SceneKind::Wave => {
            let mut out = Vec::with_capacity(44 * 44);
            for iz in 0..44 {
                let z = (iz as f32 / 43.0) * 2.0 - 1.0;
                for ix in 0..44 {
                    let x = (ix as f32 / 43.0) * 2.0 - 1.0;
                    let r = (x * x + z * z).sqrt();
                    let y = (r * 8.0 - state.phase * 1.7).sin() * 0.22;
                    out.push(P3 { x, y, z, w: (1.0 - r).clamp(0.0, 1.0) });
                }
            }
            out
        }
        SceneKind::Surface => {
            let mut out = Vec::with_capacity(52 * 52);
            for iz in 0..52 {
                let z = (iz as f32 / 51.0) * 2.3 - 1.15;
                for ix in 0..52 {
                    let x = (ix as f32 / 51.0) * 2.3 - 1.15;
                    let r = (x * x + z * z).sqrt().max(0.05);
                    let y = (r * 9.0 - state.phase * 0.9).sin() / (r * 4.0);
                    out.push(P3 { x, y: y.clamp(-0.8, 0.8), z, w: 0.6 });
                }
            }
            out
        }
        SceneKind::Tensor => tensor_points(state.tensor.as_ref()),
    }
}

fn tensor_points(tensor: Option<&TensorPayload>) -> Vec<P3> {
    let Some(t) = tensor else {
        return Vec::new();
    };
    if t.data.is_empty() {
        return Vec::new();
    }
    let rows = t.shape.first().copied().unwrap_or(1).max(1);
    let cols = t.shape.get(1).copied().unwrap_or_else(|| {
        ((t.data.len() as f32).sqrt() as usize).max(1)
    });
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in &t.data {
        min = min.min(v);
        max = max.max(v);
    }
    let range = (max - min).abs().max(1e-6);
    let step = (t.data.len() / 3000).max(1);
    let mut out = Vec::with_capacity((t.data.len() / step).max(1));
    for i in (0..t.data.len()).step_by(step) {
        let v = t.data[i];
        let r = i / cols;
        let c = i % cols;
        let x = (c as f32 / cols.max(1) as f32) * 2.0 - 1.0;
        let z = (r as f32 / rows.max(1) as f32) * 2.0 - 1.0;
        let y = ((v - min) / range) * 1.6 - 0.8;
        out.push(P3 {
            x,
            y,
            z,
            w: ((v - min) / range).clamp(0.0, 1.0),
        });
    }
    out
}

fn project_point(
    p: P3,
    width: u16,
    height: u16,
    phase: f32,
) -> Option<(i16, i16, f32, f32)> {
    let rot_y = phase * 0.35;
    let rot_x: f32 = 0.65;
    let cy = rot_y.cos();
    let sy = rot_y.sin();
    let cx = rot_x.cos();
    let sx = rot_x.sin();

    let xr = p.x * cy - p.z * sy;
    let zr = p.x * sy + p.z * cy;
    let yr = p.y * cx - zr * sx;
    let zz = p.y * sx + zr * cx + 3.2;
    if zz <= 0.1 {
        return None;
    }

    let scale = width.min(height) as f32 * 0.42;
    let sxp = (xr / zz) * scale + width as f32 * 0.5;
    let syp = (-yr / zz) * scale + height as f32 * 0.5;
    if sxp < 0.0 || syp < 0.0 || sxp >= width as f32 || syp >= height as f32 {
        return None;
    }
    let depth = (1.0 - ((zz - 2.0) / 3.0)).clamp(0.0, 1.0);
    Some((sxp as i16, syp as i16, depth, p.w))
}
