use crate::lang::{Program, Step, SemanticOp};

pub struct Torch {
    // State like device or default precision could go here
}

impl Torch {
    pub fn new() -> Self {
        Self {}
    }

    pub fn sigmoid(&self, program: &mut Program, input: &str) -> String {
        let output = format!("{}_sig", input);
        program.add_step(Step::Op(
            SemanticOp::TorchUnary("sigmoid".to_string()),
            vec![input.to_string()],
            output.clone()
        ));
        output
    }

    pub fn add(&self, program: &mut Program, a: &str, b: &str) -> String {
        let output = format!("{}_{}_sum", a, b);
        program.add_step(Step::Op(
            SemanticOp::TorchBinary("add".to_string()),
            vec![a.to_string(), b.to_string()],
            output.clone()
        ));
        output
    }
}
