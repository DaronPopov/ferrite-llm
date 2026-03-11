use rhai::{Array, Dynamic, Engine, FLOAT, INT, ImmutableString, Map};
use tachyonfx::{CellFilter, Effect, Interpolation, Motion, fx};

use crate::tui::style;

pub const DEFAULT_SCRIPT: &str = r##"
#{
  pipeline: #{
    compose: "sequence",
    items: [
      #{ op: "fade_from_fg", color: "accent", ms: 70, curve: "sine_out" },
      #{
        compose: "parallel",
        items: [
          #{ op: "hsl_shift_fg", hsl: [16.0, 12.0, 8.0], ms: 180, curve: "sine_in_out" },
          #{ op: "sweep_in", color: "accent", ms: 150, curve: "quad_out", direction: "left_to_right", gradient: 9, randomness: 1 },
          #{ op: "slide_in", color: "fg_bright", ms: 120, curve: "sine_out", direction: "left_to_right", gradient: 6, randomness: 0 }
        ]
      },
      #{
        compose: "parallel",
        items: [
          #{ op: "coalesce", ms: 90, curve: "quad_out" },
          #{ op: "hsl_shift_fg", hsl: [-10.0, 7.0, -3.0], ms: 110, curve: "sine_in" }
        ]
      },
      #{ op: "fade_to_fg", color: "fg", ms: 120, curve: "sine_in" }
    ]
  },
  tensor: #{
    compose: "sequence",
    items: [
      #{
        compose: "parallel",
        items: [
          #{ op: "fade_from_fg", color: "accent", ms: 90, curve: "sine_out" },
          #{ op: "hsl_shift_fg", hsl: [24.0, 16.0, 10.0], ms: 210, curve: "sine_in_out" },
          #{ op: "sweep_in", color: "fg_bright", ms: 150, curve: "quad_out", direction: "right_to_left", gradient: 10, randomness: 2 }
        ]
      },
      #{
        compose: "parallel",
        items: [
          #{ op: "coalesce", ms: 100, curve: "quad_out" },
          #{ op: "slide_out", color: "accent", ms: 110, curve: "sine_in_out", direction: "left_to_right", gradient: 8, randomness: 0 }
        ]
      },
      #{ op: "fade_to_fg", color: "fg", ms: 130, curve: "sine_in" }
    ]
  }
}
"##;

#[derive(Debug, Clone)]
pub struct FxScriptConfig {
    pub pipeline: Option<FxNode>,
    pub tensor: Option<FxNode>,
}

#[derive(Debug, Clone)]
pub enum FxNode {
    Sequence(Vec<FxNode>),
    Parallel(Vec<FxNode>),
    Op(FxOp),
}

#[derive(Debug, Clone)]
pub struct FxOp {
    pub kind: FxKind,
    pub ms: u32,
    pub curve: Interpolation,
    pub color: FxColorRef,
    pub hsl: [f32; 3],
    pub direction: Motion,
    pub gradient: u16,
    pub randomness: u16,
}

#[derive(Debug, Clone, Copy)]
pub enum FxKind {
    FadeFromFg,
    FadeToFg,
    HslShiftFg,
    Dissolve,
    Coalesce,
    SweepIn,
    SweepOut,
    SlideIn,
    SlideOut,
}

#[derive(Debug, Clone, Copy)]
pub enum FxColorRef {
    Accent,
    Fg,
    FgBright,
    Info,
    Good,
    Warn,
    Bad,
}

impl FxScriptConfig {
    pub fn build_pipeline_effect(&self, accent: ratatui::style::Color) -> Option<Effect> {
        self.pipeline.as_ref().map(|n| compile_node(n, accent))
    }

    pub fn build_tensor_effect(&self, accent: ratatui::style::Color) -> Option<Effect> {
        self.tensor.as_ref().map(|n| compile_node(n, accent))
    }
}

pub fn parse_script(src: &str) -> Result<FxScriptConfig, String> {
    let engine = Engine::new();
    let root = engine
        .eval::<Dynamic>(src)
        .map_err(|e| format!("rhai eval error: {}", e))?;
    parse_root(root)
}

fn parse_root(root: Dynamic) -> Result<FxScriptConfig, String> {
    let Some(map) = root.try_cast::<Map>() else {
        return Err("script root must be a map".to_string());
    };

    let pipeline = map.get("pipeline").map(parse_node).transpose()?;
    let tensor = map.get("tensor").map(parse_node).transpose()?;
    if pipeline.is_none() && tensor.is_none() {
        return Err("script must define at least one of: pipeline, tensor".to_string());
    }
    Ok(FxScriptConfig { pipeline, tensor })
}

fn parse_node(d: &Dynamic) -> Result<FxNode, String> {
    if let Some(arr) = d.clone().try_cast::<Array>() {
        let items = arr
            .iter()
            .map(parse_node)
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(FxNode::Sequence(items));
    }

    let Some(map) = d.clone().try_cast::<Map>() else {
        return Err("node must be a map or array".to_string());
    };

    if let Some(compose) = get_str(&map, "compose") {
        let items = get_array(&map, "items")?
            .iter()
            .map(parse_node)
            .collect::<Result<Vec<_>, _>>()?;
        return match compose.as_str() {
            "sequence" | "seq" => Ok(FxNode::Sequence(items)),
            "parallel" | "par" => Ok(FxNode::Parallel(items)),
            _ => Err(format!("unknown compose mode '{}'", compose)),
        };
    }

    Ok(FxNode::Op(parse_op(&map)?))
}

fn parse_op(map: &Map) -> Result<FxOp, String> {
    let kind = parse_kind(
        &get_str(map, "op").ok_or_else(|| "missing op in effect map".to_string())?,
    )?;
    Ok(FxOp {
        kind,
        ms: get_u32(map, "ms").unwrap_or(160).clamp(20, 3000),
        curve: parse_curve(get_str(map, "curve").as_deref().unwrap_or("sine_in_out"))?,
        color: parse_color(get_str(map, "color").as_deref().unwrap_or("accent"))?,
        hsl: get_hsl(map).unwrap_or([18.0, 10.0, 6.0]),
        direction: parse_direction(get_str(map, "direction").as_deref().unwrap_or("left_to_right"))?,
        gradient: get_u16(map, "gradient").unwrap_or(8).clamp(1, 48),
        randomness: get_u16(map, "randomness").unwrap_or(0).clamp(0, 16),
    })
}

fn compile_node(node: &FxNode, accent: ratatui::style::Color) -> Effect {
    match node {
        FxNode::Sequence(nodes) => {
            let effects: Vec<Effect> = nodes.iter().map(|n| compile_node(n, accent)).collect();
            fx::sequence(&effects)
        }
        FxNode::Parallel(nodes) => {
            let effects: Vec<Effect> = nodes.iter().map(|n| compile_node(n, accent)).collect();
            fx::parallel(&effects)
        }
        FxNode::Op(op) => compile_op(op, accent),
    }
}

fn compile_op(op: &FxOp, accent: ratatui::style::Color) -> Effect {
    let color = resolve_color(op.color, accent);
    let t = (op.ms, op.curve);
    match op.kind {
        FxKind::FadeFromFg => fx::fade_from_fg(color, t).with_filter(CellFilter::Text),
        FxKind::FadeToFg => fx::fade_to_fg(color, t).with_filter(CellFilter::Text),
        FxKind::HslShiftFg => fx::hsl_shift_fg(op.hsl, t).with_filter(CellFilter::Text),
        FxKind::Dissolve => fx::dissolve(t).with_filter(CellFilter::Text),
        FxKind::Coalesce => fx::coalesce(t).with_filter(CellFilter::Text),
        FxKind::SweepIn => fx::sweep_in(op.direction, op.gradient, op.randomness, color, t)
            .with_filter(CellFilter::Text),
        FxKind::SweepOut => fx::sweep_out(op.direction, op.gradient, op.randomness, color, t)
            .with_filter(CellFilter::Text),
        FxKind::SlideIn => fx::slide_in(op.direction, op.gradient, op.randomness, color, t)
            .with_filter(CellFilter::Text),
        FxKind::SlideOut => fx::slide_out(op.direction, op.gradient, op.randomness, color, t)
            .with_filter(CellFilter::Text),
    }
}

fn resolve_color(c: FxColorRef, accent: ratatui::style::Color) -> ratatui::style::Color {
    match c {
        FxColorRef::Accent => accent,
        FxColorRef::Fg => style::fg(),
        FxColorRef::FgBright => style::fg_bright(),
        FxColorRef::Info => style::info(),
        FxColorRef::Good => style::good(),
        FxColorRef::Warn => style::warn(),
        FxColorRef::Bad => style::bad(),
    }
}

fn get_str(map: &Map, key: &str) -> Option<String> {
    map.get(key).and_then(|v| {
        v.clone()
            .try_cast::<ImmutableString>()
            .map(|s| s.to_string())
            .or_else(|| v.clone().try_cast::<String>())
    })
}

fn get_u32(map: &Map, key: &str) -> Option<u32> {
    map.get(key)
        .and_then(|v| v.clone().try_cast::<INT>())
        .map(|n| n as u32)
}

fn get_u16(map: &Map, key: &str) -> Option<u16> {
    get_u32(map, key).map(|n| n as u16)
}

fn get_hsl(map: &Map) -> Option<[f32; 3]> {
    let arr = map.get("hsl")?.clone().try_cast::<Array>()?;
    if arr.len() != 3 {
        return None;
    }
    let mut out = [0.0f32; 3];
    for (i, v) in arr.iter().enumerate() {
        if let Some(x) = v.clone().try_cast::<FLOAT>() {
            out[i] = x as f32;
        } else if let Some(x) = v.clone().try_cast::<INT>() {
            out[i] = x as f32;
        } else {
            return None;
        }
    }
    Some(out)
}

fn get_array(map: &Map, key: &str) -> Result<Array, String> {
    map.get(key)
        .ok_or_else(|| format!("missing '{}' array", key))?
        .clone()
        .try_cast::<Array>()
        .ok_or_else(|| format!("'{}' must be an array", key))
}

fn parse_kind(s: &str) -> Result<FxKind, String> {
    match s {
        "fade_from_fg" => Ok(FxKind::FadeFromFg),
        "fade_to_fg" => Ok(FxKind::FadeToFg),
        "hsl_shift_fg" => Ok(FxKind::HslShiftFg),
        "dissolve" => Ok(FxKind::Dissolve),
        "coalesce" => Ok(FxKind::Coalesce),
        "sweep_in" => Ok(FxKind::SweepIn),
        "sweep_out" => Ok(FxKind::SweepOut),
        "slide_in" => Ok(FxKind::SlideIn),
        "slide_out" => Ok(FxKind::SlideOut),
        _ => Err(format!("unknown op '{}'", s)),
    }
}

fn parse_color(s: &str) -> Result<FxColorRef, String> {
    match s {
        "accent" => Ok(FxColorRef::Accent),
        "fg" => Ok(FxColorRef::Fg),
        "fg_bright" | "bright" => Ok(FxColorRef::FgBright),
        "info" => Ok(FxColorRef::Info),
        "good" => Ok(FxColorRef::Good),
        "warn" => Ok(FxColorRef::Warn),
        "bad" => Ok(FxColorRef::Bad),
        _ => Err(format!("unknown color '{}'", s)),
    }
}

fn parse_curve(s: &str) -> Result<Interpolation, String> {
    match s {
        "linear" => Ok(Interpolation::Linear),
        "quad_in" => Ok(Interpolation::QuadIn),
        "quad_out" => Ok(Interpolation::QuadOut),
        "quad_in_out" => Ok(Interpolation::QuadInOut),
        "sine_in" => Ok(Interpolation::SineIn),
        "sine_out" => Ok(Interpolation::SineOut),
        "sine_in_out" => Ok(Interpolation::SineInOut),
        "circ_in" => Ok(Interpolation::CircIn),
        "circ_out" => Ok(Interpolation::CircOut),
        "circ_in_out" => Ok(Interpolation::CircInOut),
        "expo_in" => Ok(Interpolation::ExpoIn),
        "expo_out" => Ok(Interpolation::ExpoOut),
        "expo_in_out" => Ok(Interpolation::ExpoInOut),
        "bounce_out" => Ok(Interpolation::BounceOut),
        "elastic_out" => Ok(Interpolation::ElasticOut),
        _ => Err(format!("unknown interpolation '{}'", s)),
    }
}

fn parse_direction(s: &str) -> Result<Motion, String> {
    match s {
        "left_to_right" | "ltr" => Ok(Motion::LeftToRight),
        "right_to_left" | "rtl" => Ok(Motion::RightToLeft),
        "up_to_down" | "utd" => Ok(Motion::UpToDown),
        "down_to_up" | "dtu" => Ok(Motion::DownToUp),
        _ => Err(format!("unknown motion '{}'", s)),
    }
}
