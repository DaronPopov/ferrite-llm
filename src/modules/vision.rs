use crate::lang::{Program, Step, SemanticOp};

pub struct Vision {
    // In a py-like world, we might store state here
}

impl Vision {
    pub fn new() -> Self {
        Self {}
    }

    /// Semantically applies a blur operation to the program graph
    pub fn blur(&self, program: &mut Program, input: &str) -> String {
        let output = format!("{}_blurred", input);
        program.add_step(Step::Op(
            SemanticOp::Vision("blur".to_string()),
            vec![input.to_string()],
            output.clone()
        ));
        output
    }

    pub fn edges(&self, program: &mut Program, input: &str) -> String {
        let output = format!("{}_edges", input);
        program.add_step(Step::Op(
            SemanticOp::Vision("edges".to_string()),
            vec![input.to_string()],
            output.clone()
        ));
        output
    }
}
