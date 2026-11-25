use super::*;

/// Evaluates a parsed [`Program`] against `ctx` using `config`.
pub fn evaluate_program(
    program: &Program,
    ctx: &dyn ExprContext,
    config: &EvalConfig,
) -> Result<Value, ExprError> {
    evaluate_program_with_extended(program, ctx, config, is_extended(program))
}

/// Evaluates a parsed [`Program`] with a precomputed `program_extended` flag, avoiding
/// redundant full-AST analysis walks on repeated [`crate::CompiledExpr::evaluate`] calls.
pub(crate) fn evaluate_program_with_extended(
    program: &Program,
    ctx: &dyn ExprContext,
    config: &EvalConfig,
    program_extended: bool,
) -> Result<Value, ExprError> {
    let mut evaluator = Evaluator::new(ctx, *config, program_extended);
    let root_env = evaluator.push_env(None)?;
    evaluator.hoist_decls(&program.body, root_env)?;

    let mut completion = RuntimeValue::Undefined;
    for stmt in &program.body {
        let flow = match evaluator.exec_stmt(stmt, root_env) {
            Ok(flow) => flow,
            Err(ExprError::Thrown(msg)) => {
                if let Some(v) = evaluator.pending_throw.take() {
                    let rendered = evaluator.runtime_to_js_string(&v)?;
                    return Err(ExprError::Thrown(rendered));
                }
                return Err(ExprError::Thrown(msg));
            }
            Err(err) => return Err(err),
        };
        match flow {
            Flow::Normal(v) => {
                if !matches!(v, RuntimeValue::Empty) {
                    completion = v;
                }
            }
            Flow::Return(v) => {
                completion = v;
                break;
            }
            Flow::Throw(v) => {
                let msg = evaluator.runtime_to_js_string(&v)?;
                return Err(ExprError::Thrown(msg));
            }
            Flow::Break => {
                return Err(ExprError::SyntaxError {
                    offset: 0,
                    message: "illegal 'break' statement".to_string(),
                });
            }
            Flow::Continue => {
                return Err(ExprError::SyntaxError {
                    offset: 0,
                    message: "illegal 'continue' statement".to_string(),
                });
            }
        }
    }

    evaluator.export_root_value(&completion)
}
