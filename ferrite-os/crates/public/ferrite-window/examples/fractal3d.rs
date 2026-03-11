//! Mandelbulb 3D Fractal — Ferrite-OS Graphics Demo
//!
//! Renders a power-8 Mandelbulb fractal via sphere tracing (ray marching)
//! with Phong lighting, ambient occlusion, and orbit-trap coloring.
//! All rendering runs on CPU threads and blits to a ferrite-window
//! software framebuffer. No OpenGL, no Vulkan, no GPU driver — just math.
//!
//! Run:  cargo run -p ferrite-window --example fractal3d --release
//!
//! The camera orbits the fractal automatically. Press ESC or close the window
//! to exit.

use ferrite_window::{Event, Window, WindowConfig};
use std::time::Instant;

// ── Display ──────────────────────────────────────────────────────────────────

const DISPLAY_W: u32 = 1920;
const DISPLAY_H: u32 = 1080;

// Render at lower resolution and upscale for high FPS.
// 480x270 = 1/4 of 1920x1080, clean 4x integer upscale.
const RENDER_W: u32 = 480;
const RENDER_H: u32 = 270;
const SCALE: u32 = DISPLAY_W / RENDER_W; // 4

// ── Ray marcher tuning ───────────────────────────────────────────────────────

const MAX_STEPS: u32 = 48;
const MAX_ITER: u32 = 5;
const POWER: f32 = 8.0;
const BAILOUT: f32 = 2.0;
const MIN_DIST: f32 = 0.001;
const MAX_DIST: f32 = 10.0;
const NORMAL_EPS: f32 = 0.001;

// Bounding sphere: the Mandelbulb fits inside radius ~1.2.
const BOUND_RADIUS: f32 = 1.25;

fn main() {
    let mut window = Window::new(WindowConfig {
        title: "Ferrite-OS  |  Mandelbulb Fractal".into(),
        width: DISPLAY_W,
        height: DISPLAY_H,
        resizable: false,
        fullscreen: true,
    })
    .expect("failed to create window");

    let mut render_buf = vec![0u32; (RENDER_W * RENDER_H) as usize];
    let mut display_buf = vec![0u32; (DISPLAY_W * DISPLAY_H) as usize];
    let start = Instant::now();
    let mut frame_count: u64 = 0;
    let mut fps_timer = Instant::now();

    while window.is_open() {
        for event in window.poll_events() {
            match event {
                Event::Close => return,
                // ESC key (X11 keycode 9, Win32 VK_ESCAPE 0x1B)
                Event::KeyDown { keycode, .. } if keycode == 9 || keycode == 0x1B => return,
                _ => {}
            }
        }

        let t = start.elapsed().as_secs_f32();

        render_frame(&mut render_buf, RENDER_W, RENDER_H, t);
        upscale(&render_buf, &mut display_buf, RENDER_W, RENDER_H, SCALE);

        window
            .present(&display_buf, DISPLAY_W, DISPLAY_H)
            .expect("present failed");

        frame_count += 1;
        let elapsed = fps_timer.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            let fps = frame_count as f32 / elapsed;
            window.set_title(&format!(
                "Ferrite-OS  |  Mandelbulb  |  {:.1} FPS  |  {}x{} @ {}x",
                fps, RENDER_W, RENDER_H, SCALE
            ));
            frame_count = 0;
            fps_timer = Instant::now();
        }
    }
}

// ── Multi-threaded frame dispatch ────────────────────────────────────────────

fn render_frame(pixels: &mut [u32], w: u32, h: u32, time: f32) {
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let rows_per_chunk = (h as usize + num_threads - 1) / num_threads;

    std::thread::scope(|scope| {
        for (chunk_idx, chunk) in pixels.chunks_mut(rows_per_chunk * w as usize).enumerate() {
            let start_y = chunk_idx * rows_per_chunk;
            scope.spawn(move || {
                for (local_y, row) in chunk.chunks_mut(w as usize).enumerate() {
                    let y = (start_y + local_y) as u32;
                    if y >= h {
                        break;
                    }
                    for x in 0..w {
                        row[x as usize] = render_pixel(x, y, w, h, time);
                    }
                }
            });
        }
    });
}

// ── Integer upscale (nearest-neighbor) ───────────────────────────────────────

fn upscale(src: &[u32], dst: &mut [u32], src_w: u32, src_h: u32, scale: u32) {
    let dst_w = (src_w * scale) as usize;
    for sy in 0..src_h {
        for sx in 0..src_w {
            let color = src[(sy * src_w + sx) as usize];
            let dx = (sx * scale) as usize;
            let dy = (sy * scale) as usize;
            for row in 0..scale as usize {
                let base = (dy + row) * dst_w + dx;
                for col in 0..scale as usize {
                    dst[base + col] = color;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Per-pixel ray march + shade
// ═══════════════════════════════════════════════════════════════════════════════

fn render_pixel(x: u32, y: u32, w: u32, h: u32, time: f32) -> u32 {
    let aspect = w as f32 / h as f32;
    let uv_x = (2.0 * x as f32 / w as f32 - 1.0) * aspect;
    let uv_y = 1.0 - 2.0 * y as f32 / h as f32;

    // Camera orbits the origin
    let angle = time * 0.2;
    let cam_dist = 2.6;
    let cam_y = 0.3 + 0.5 * (time * 0.12).sin();
    let eye = [cam_dist * angle.cos(), cam_y, cam_dist * angle.sin()];
    let target = [0.0, -0.05, 0.0];

    let (ro, rd) = camera_ray(eye, target, uv_x, uv_y, 1.8);

    // ── Bounding sphere test: skip rays that miss entirely ───────────
    if !ray_hits_sphere(ro, rd, [0.0, 0.0, 0.0], BOUND_RADIUS) {
        return background(uv_y);
    }

    // Advance ray to bounding sphere entry point
    let t_enter = ray_sphere_enter(ro, rd, [0.0, 0.0, 0.0], BOUND_RADIUS);
    let mut t_dist = t_enter.max(0.0);

    // ── Sphere-trace ─────────────────────────────────────────────────
    let mut steps_taken = 0u32;
    let mut trap = 1.0f32;
    let mut hit = false;

    for step in 0..MAX_STEPS {
        let p = vadd(ro, vscale(rd, t_dist));
        let (de, tr) = mandelbulb_de(p);
        trap = trap.min(tr);
        steps_taken = step;

        if de < MIN_DIST {
            hit = true;
            break;
        }
        if t_dist > MAX_DIST {
            break;
        }

        // Slightly aggressive stepping (0.9x DE) — minor artifact risk
        // but big speed gain.
        t_dist += de * 0.9;
    }

    if !hit {
        return background(uv_y);
    }

    // ── Surface shading ──────────────────────────────────────────────
    let p = vadd(ro, vscale(rd, t_dist));
    let n = calc_normal(p);

    // Directional light
    let light_dir = vnorm([0.6, 0.8, -0.5]);
    let ndotl = vdot(n, light_dir).max(0.0);

    // Blinn-Phong specular
    let view_dir = vnorm(vsub(eye, p));
    let half_v = vnorm(vadd(light_dir, view_dir));
    let spec = vdot(n, half_v).max(0.0).powf(24.0) * 0.35;

    // Ambient occlusion from march steps
    let ao = 1.0 - (steps_taken as f32 / MAX_STEPS as f32).powf(0.5);

    // Orbit-trap coloring
    let base = cosine_palette(trap * 2.5 + 0.15);

    let ambient = 0.10;
    let diffuse = ndotl * 0.8;
    let intensity = ambient + diffuse;

    let mut r = (base[0] * intensity + spec) * ao;
    let mut g = (base[1] * intensity + spec) * ao;
    let mut b = (base[2] * intensity + spec) * ao;

    // Distance fog
    let fog = (-t_dist * 0.2).exp();
    r *= fog;
    g *= fog;
    b *= fog;

    // Gamma correction
    r = r.sqrt();
    g = g.sqrt();
    b = b.sqrt();

    pack_rgb(r, g, b)
}

fn background(uv_y: f32) -> u32 {
    let v = 0.015 + 0.025 * (0.5 + 0.5 * uv_y);
    pack_rgb(v, v, v + 0.01)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Mandelbulb distance estimator
// ═══════════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn mandelbulb_de(pos: [f32; 3]) -> (f32, f32) {
    let mut z = pos;
    let mut dr = 1.0f32;
    let mut r;
    let mut trap = f32::MAX;

    for _ in 0..MAX_ITER {
        r = vlen(z);
        if r > BAILOUT {
            break;
        }
        trap = trap.min(r);

        let theta = fast_acos(z[2] / r);
        let phi = fast_atan2(z[1], z[0]);

        // r^(power-1) via integer exponentiation: r^7 = (r^2)^2 * r^2 * r
        let r2 = r * r;
        let r4 = r2 * r2;
        dr = r4 * r2 * r * POWER * dr + 1.0;

        // r^power = (r^4)^2
        let r_pow = r4 * r4;
        let tp = theta * POWER;
        let pp = phi * POWER;

        let st = tp.sin();
        z = [
            r_pow * st * pp.cos() + pos[0],
            r_pow * st * pp.sin() + pos[1],
            r_pow * tp.cos() + pos[2],
        ];
    }

    r = vlen(z);
    let de = if r > 1e-10 && dr > 1e-10 {
        0.5 * r * r.ln() / dr
    } else {
        MIN_DIST
    };

    (de, trap)
}

/// Tetrahedral normal estimation — 4 DE evals instead of 6 (central diff).
fn calc_normal(p: [f32; 3]) -> [f32; 3] {
    let e = NORMAL_EPS;
    // Tetrahedral offsets: (1,1,-1), (1,-1,1), (-1,1,1), (-1,-1,-1)
    let a = mandelbulb_de([p[0] + e, p[1] + e, p[2] - e]).0;
    let b = mandelbulb_de([p[0] + e, p[1] - e, p[2] + e]).0;
    let c = mandelbulb_de([p[0] - e, p[1] + e, p[2] + e]).0;
    let d = mandelbulb_de([p[0] - e, p[1] - e, p[2] - e]).0;

    vnorm([a + b - c - d, a - b + c - d, -a + b + c - d])
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bounding sphere acceleration
// ═══════════════════════════════════════════════════════════════════════════════

/// Does the ray (ro + t*rd) intersect a sphere at `center` with `radius`?
#[inline(always)]
fn ray_hits_sphere(ro: [f32; 3], rd: [f32; 3], center: [f32; 3], radius: f32) -> bool {
    let oc = vsub(ro, center);
    let b = vdot(oc, rd);
    let c = vdot(oc, oc) - radius * radius;
    let disc = b * b - c;
    disc >= 0.0
}

/// Entry distance of the ray into a bounding sphere (may be negative if inside).
#[inline(always)]
fn ray_sphere_enter(ro: [f32; 3], rd: [f32; 3], center: [f32; 3], radius: f32) -> f32 {
    let oc = vsub(ro, center);
    let b = vdot(oc, rd);
    let c = vdot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return MAX_DIST;
    }
    -b - disc.sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Camera
// ═══════════════════════════════════════════════════════════════════════════════

fn camera_ray(
    eye: [f32; 3],
    target: [f32; 3],
    uv_x: f32,
    uv_y: f32,
    fov_factor: f32,
) -> ([f32; 3], [f32; 3]) {
    let forward = vnorm(vsub(target, eye));
    let right = vnorm(vcross(forward, [0.0, 1.0, 0.0]));
    let up = vcross(right, forward);

    let rd = vnorm([
        right[0] * uv_x + up[0] * uv_y + forward[0] * fov_factor,
        right[1] * uv_x + up[1] * uv_y + forward[1] * fov_factor,
        right[2] * uv_x + up[2] * uv_y + forward[2] * fov_factor,
    ]);

    (eye, rd)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cosine palette (Inigo Quilez)
// ═══════════════════════════════════════════════════════════════════════════════

fn cosine_palette(t: f32) -> [f32; 3] {
    let tau = std::f32::consts::TAU;
    [
        0.5 + 0.5 * (tau * (1.0 * t + 0.00)).cos(),
        0.5 + 0.5 * (tau * (0.7 * t + 0.15)).cos(),
        0.5 + 0.5 * (tau * (0.4 * t + 0.20)).cos(),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fast math approximations
// ═══════════════════════════════════════════════════════════════════════════════

/// Fast acos approximation — max error ~0.01 rad (adequate for fractal DE).
#[inline(always)]
fn fast_acos(x: f32) -> f32 {
    let x = x.clamp(-1.0, 1.0);
    // Polynomial approximation: Handbook of Mathematical Functions (Abramowitz)
    let abs_x = x.abs();
    let r = (-0.0187293 * abs_x + 0.0742610) * abs_x - 0.2121144;
    let r = (r * abs_x + 1.5707288) * (1.0 - abs_x).max(0.0).sqrt();
    if x >= 0.0 { r } else { std::f32::consts::PI - r }
}

/// Fast atan2 approximation — max error ~0.01 rad.
#[inline(always)]
fn fast_atan2(y: f32, x: f32) -> f32 {
    let abs_x = x.abs();
    let abs_y = y.abs();

    let (a, b) = if abs_x > abs_y {
        (abs_y / (abs_x + 1e-20), false)
    } else {
        (abs_x / (abs_y + 1e-20), true)
    };

    // Polynomial approximation of atan(a) for a in [0, 1]
    let s = a * a;
    let mut r = -0.0464964 * s + 0.1593220;
    r = r * s - 0.3276420;
    r = r * s + 1.0;
    r *= a;

    if b {
        r = std::f32::consts::FRAC_PI_2 - r;
    }
    if x < 0.0 {
        r = std::f32::consts::PI - r;
    }
    if y < 0.0 {
        r = -r;
    }
    r
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pixel packing
// ═══════════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn pack_rgb(r: f32, g: f32, b: f32) -> u32 {
    let r = (r.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    let g = (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    let b = (b.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    (r << 24) | (g << 16) | (b << 8) | 0xFF
}

// ═══════════════════════════════════════════════════════════════════════════════
// Vector math — all inlined for hot loop performance
// ═══════════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn vadd(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline(always)]
fn vsub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline(always)]
fn vscale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline(always)]
fn vdot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
fn vcross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline(always)]
fn vlen(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline(always)]
fn vnorm(v: [f32; 3]) -> [f32; 3] {
    let len = vlen(v);
    if len < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    let inv = 1.0 / len;
    [v[0] * inv, v[1] * inv, v[2] * inv]
}
