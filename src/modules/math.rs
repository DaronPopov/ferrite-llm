use crate::lang::{Program, Step, SemanticOp};

pub struct MathExpert {
    // Reusable math logic state
}

impl MathExpert {
    pub fn new() -> Self {
        Self {}
    }

    pub fn sin(&self, program: &mut Program, input: &str) -> String {
        let output = format!("{}_sin", input);
        program.add_step(Step::Op(
            SemanticOp::Math("sin".to_string()),
            vec![input.to_string()],
            output.clone()
        ));
        output
    }

    pub fn fuse(&self, program: &mut Program, ops: Vec<SemanticOp>, input: &str) -> String {
        let output = format!("{}_fused", input);
        program.add_step(Step::Op(
            SemanticOp::Fused(ops),
            vec![input.to_string()],
            output.clone()
        ));
        output
    }
}
