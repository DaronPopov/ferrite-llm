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
