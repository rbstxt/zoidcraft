//! Recursive-descent / precedence-climbing parser for Vanilla CM3 and
//! Zoidium Expressions+.

use crate::ast::{
    ArrayElem, ArrayPatternElem, AssignOp, AssignTarget, BinaryOp, CallArg, ChainOp, Expr,
    ForInStmt, ForInit, ForOfStmt, ForStmt, FunctionBody, FunctionDecl, FunctionExpr, Literal,
    LogicalOp, ObjectKey, ObjectPatternProp, ObjectProp, Param, Pattern, Program, Stmt, SwitchCase,
    TemplatePart, TryStmt, UnaryOp, UpdateOp, VarDeclarator, VarKind, validate_program,
};
use crate::lexer::{Lexer, Token, TokenKind};
use crate::{ExprError, ExprMode};

const MAX_PARSE_DEPTH: usize = 128;

/// Parses and validates an expression source string in the requested [`ExprMode`].
pub fn parse_source(source: &str, mode: ExprMode) -> Result<Program, ExprError> {
    let tokens = Lexer::new(source).tokenize()?;
    let mut parser = Parser {
        tokens,
        pos: 0,
        mode,
        depth: 0,
    };
    let program = parser.parse_program()?;
    validate_program(&program, mode)?;
    Ok(program)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    mode: ExprMode,
    depth: usize,
}

impl Parser {
    fn peek(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn peek_kind(&self) -> &TokenKind {
        &self.peek().kind
    }

    fn peek_offset(&self, offset: usize) -> &Token {
        &self.tokens[(self.pos + offset).min(self.tokens.len() - 1)]
    }

    fn bump(&mut self) -> Token {
        let idx = self.pos.min(self.tokens.len() - 1);
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        self.tokens[idx].clone()
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::Eof)
    }

    fn check(&self, expected: &TokenKind) -> bool {
        self.peek_kind() == expected
    }

    fn eat(&mut self, expected: &TokenKind) -> bool {
        if self.check(expected) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, expected: &TokenKind, label: &str) -> Result<Token, ExprError> {
        if self.check(expected) {
            Ok(self.bump())
        } else {
            Err(ExprError::SyntaxError {
                offset: self.peek().span.start,
                message: format!("expected {label}"),
            })
        }
    }

    fn eat_semicolon(&mut self) -> Result<(), ExprError> {
        if self.eat(&TokenKind::Semicolon) {
            return Ok(());
        }
        if self.at_eof() || self.check(&TokenKind::RBrace) || self.peek().line_before {
            return Ok(());
        }
        Err(ExprError::SyntaxError {
            offset: self.peek().span.start,
            message: "expected ';' or newline".to_string(),
        })
    }

    fn require_expressions_plus(&self, feature: &str) -> Result<(), ExprError> {
        if self.mode == ExprMode::VanillaCm3 {
            Err(ExprError::ModeError {
                mode: self.mode,
                feature: feature.to_string(),
            })
        } else {
            Ok(())
        }
    }

    fn enter_depth(&mut self) -> Result<(), ExprError> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            return Err(ExprError::RecursionLimitExceeded {
                limit: MAX_PARSE_DEPTH,
            });
        }
        Ok(())
    }

    fn leave_depth(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    fn parse_program(&mut self) -> Result<Program, ExprError> {
        let mut body = Vec::new();
        while !self.at_eof() {
            body.push(self.parse_stmt()?);
        }
        Ok(Program { body })
    }

    // -----------------------------------------------------------------------
    // Statements
    // -----------------------------------------------------------------------

    fn parse_stmt(&mut self) -> Result<Stmt, ExprError> {
        self.enter_depth()?;
        let res = self.parse_stmt_inner();
        self.leave_depth();
        res
    }

    fn parse_stmt_inner(&mut self) -> Result<Stmt, ExprError> {
        match self.peek_kind() {
            TokenKind::Class => Err(ExprError::Forbidden("class".to_string())),
            TokenKind::Import => Err(ExprError::Forbidden("import".to_string())),
            TokenKind::Export => Err(ExprError::Forbidden("export".to_string())),
            TokenKind::Async => Err(ExprError::Forbidden("async".to_string())),
            TokenKind::Await => Err(ExprError::Forbidden("await".to_string())),
            TokenKind::Yield => Err(ExprError::Forbidden("yield".to_string())),
            TokenKind::With => Err(ExprError::Forbidden("with".to_string())),
            TokenKind::Debugger => Err(ExprError::Forbidden("debugger".to_string())),

            TokenKind::Semicolon => {
                self.bump();
                Ok(Stmt::Empty)
            }
            TokenKind::LBrace => Ok(Stmt::Block(self.parse_block()?)),
            TokenKind::Var => self.parse_var_decl_stmt(VarKind::Var),
            TokenKind::Let => {
                self.require_expressions_plus("let declaration")?;
                self.parse_var_decl_stmt(VarKind::Let)
            }
            TokenKind::Const => {
                self.require_expressions_plus("const declaration")?;
                self.parse_var_decl_stmt(VarKind::Const)
            }
            TokenKind::Function => {
                self.require_expressions_plus("function declaration")?;
                self.parse_function_decl()
            }
            TokenKind::If => self.parse_if_stmt(),
            TokenKind::While => self.parse_while_stmt(),
            TokenKind::Do => self.parse_do_while_stmt(),
            TokenKind::For => self.parse_for_stmt(),
            TokenKind::Switch => self.parse_switch_stmt(),
            TokenKind::Break => {
                self.bump();
                self.eat_semicolon()?;
                Ok(Stmt::Break)
            }
            TokenKind::Continue => {
                self.bump();
                self.eat_semicolon()?;
                Ok(Stmt::Continue)
            }
            TokenKind::Return => self.parse_return_stmt(),
            TokenKind::Throw => {
                self.require_expressions_plus("throw statement")?;
                self.bump();
                if self.peek().line_before || self.at_eof() || self.check(&TokenKind::Semicolon) {
                    return Err(ExprError::SyntaxError {
                        offset: self.peek().span.start,
                        message: "expected expression after 'throw'".to_string(),
                    });
                }
                let expr = self.parse_expr()?;
                self.eat_semicolon()?;
                Ok(Stmt::Throw(expr))
            }
            TokenKind::Try => {
                self.require_expressions_plus("try/catch statement")?;
                self.parse_try_stmt()
            }
            _ => {
                let expr = self.parse_expr()?;
                self.eat_semicolon()?;
                Ok(Stmt::Expr(expr))
            }
        }
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, ExprError> {
        self.expect(&TokenKind::LBrace, "'{'")?;
        let mut stmts = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.at_eof() {
            stmts.push(self.parse_stmt()?);
        }
        self.expect(&TokenKind::RBrace, "'}'")?;
        Ok(stmts)
    }

    fn parse_var_decl_stmt(&mut self, kind: VarKind) -> Result<Stmt, ExprError> {
        self.bump(); // consume var/let/const
        let decls = self.parse_var_declarators(kind)?;
        self.eat_semicolon()?;
        Ok(Stmt::VarDecl { kind, decls })
    }

    fn parse_var_declarators(&mut self, kind: VarKind) -> Result<Vec<VarDeclarator>, ExprError> {
        let mut decls = Vec::new();
        loop {
            let pattern = self.parse_pattern()?;
            let init = if self.eat(&TokenKind::Eq) {
                Some(self.parse_assignment_expr()?)
            } else {
                if kind == VarKind::Const {
                    return Err(ExprError::SyntaxError {
                        offset: self.peek().span.start,
                        message: "'const' declaration requires an initializer".to_string(),
                    });
                }
                None
            };
            decls.push(VarDeclarator { pattern, init });
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        Ok(decls)
    }

    fn parse_pattern(&mut self) -> Result<Pattern, ExprError> {
        self.enter_depth()?;
        let res = self.parse_pattern_inner();
        self.leave_depth();
        res
    }

    fn parse_pattern_inner(&mut self) -> Result<Pattern, ExprError> {
        match self.peek_kind() {
            TokenKind::Ident(_) => {
                let Token {
                    kind: TokenKind::Ident(name),
                    ..
                } = self.bump()
                else {
                    unreachable!();
                };
                Ok(Pattern::Ident(name))
            }
            TokenKind::LBracket => {
                self.require_expressions_plus("array destructuring")?;
                self.bump();
                let mut elems = Vec::new();
                while !self.check(&TokenKind::RBracket) && !self.at_eof() {
                    if self.eat(&TokenKind::Comma) {
                        elems.push(ArrayPatternElem::Hole);
                        continue;
                    }
                    if self.eat(&TokenKind::Ellipsis) {
                        let name = self.expect_ident("rest binding name")?;
                        elems.push(ArrayPatternElem::Rest(name));
                        break;
                    }
                    let pat = self.parse_pattern()?;
                    let default = if self.eat(&TokenKind::Eq) {
                        Some(self.parse_assignment_expr()?)
                    } else {
                        None
                    };
                    elems.push(ArrayPatternElem::Binding {
                        pattern: pat,
                        default,
                    });
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(&TokenKind::RBracket, "']'")?;
                Ok(Pattern::Array(elems))
            }
            TokenKind::LBrace => {
                self.require_expressions_plus("object destructuring")?;
                self.bump();
                let mut props = Vec::new();
                while !self.check(&TokenKind::RBrace) && !self.at_eof() {
                    if self.eat(&TokenKind::Ellipsis) {
                        let name = self.expect_ident("rest binding name")?;
                        props.push(ObjectPatternProp::Rest(name));
                        break;
                    }
                    let is_ident = matches!(self.peek_kind(), TokenKind::Ident(_));
                    let key = self.parse_static_prop_name()?;
                    let pattern = if self.eat(&TokenKind::Colon) {
                        self.parse_pattern()?
                    } else if is_ident {
                        Pattern::Ident(key.clone())
                    } else {
                        return Err(ExprError::SyntaxError {
                            offset: self.peek().span.start,
                            message: "expected ':' after non-identifier destructuring key"
                                .to_string(),
                        });
                    };
                    let default = if self.eat(&TokenKind::Eq) {
                        Some(self.parse_assignment_expr()?)
                    } else {
                        None
                    };
                    props.push(ObjectPatternProp::Prop {
                        key,
                        pattern,
                        default,
                    });
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(&TokenKind::RBrace, "'}'")?;
                Ok(Pattern::Object(props))
            }
            _ => Err(ExprError::SyntaxError {
                offset: self.peek().span.start,
                message: "expected binding identifier or destructuring pattern".to_string(),
            }),
        }
    }

    fn parse_function_decl(&mut self) -> Result<Stmt, ExprError> {
        self.bump(); // consume `function`
        if self.check(&TokenKind::Star) {
            return Err(ExprError::Forbidden("generator function*".to_string()));
        }
        let name = self.expect_ident("function name")?;
        let params = self.parse_param_list()?;
        let body = self.parse_block()?;
        Ok(Stmt::FunctionDecl(Box::new(FunctionDecl {
            name,
            params,
            body,
        })))
    }

    fn parse_param_list(&mut self) -> Result<Vec<Param>, ExprError> {
        self.expect(&TokenKind::LParen, "'('")?;
        let mut params = Vec::new();
        while !self.check(&TokenKind::RParen) && !self.at_eof() {
            let is_rest = self.eat(&TokenKind::Ellipsis);
            let pattern = if is_rest {
                Pattern::Ident(self.expect_ident("rest parameter name")?)
            } else {
                self.parse_pattern()?
            };
            let default = if !is_rest && self.eat(&TokenKind::Eq) {
                Some(self.parse_assignment_expr()?)
            } else {
                None
            };
            params.push(Param {
                pattern,
                default,
                is_rest,
            });
            if is_rest {
                break;
            }
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RParen, "')'")?;
        Ok(params)
    }

    fn parse_if_stmt(&mut self) -> Result<Stmt, ExprError> {
        self.bump(); // `if`
        self.expect(&TokenKind::LParen, "'('")?;
        let test = self.parse_expr()?;
        self.expect(&TokenKind::RParen, "')'")?;
        let consequent = Box::new(self.parse_stmt()?);
        let alternate = if self.eat(&TokenKind::Else) {
            Some(Box::new(self.parse_stmt()?))
        } else {
            None
        };
        Ok(Stmt::If {
            test,
            consequent,
            alternate,
        })
    }

    fn parse_while_stmt(&mut self) -> Result<Stmt, ExprError> {
        self.bump(); // `while`
        self.expect(&TokenKind::LParen, "'('")?;
        let test = self.parse_expr()?;
        self.expect(&TokenKind::RParen, "')'")?;
        let body = Box::new(self.parse_stmt()?);
        Ok(Stmt::While { test, body })
    }

    fn parse_do_while_stmt(&mut self) -> Result<Stmt, ExprError> {
        self.bump(); // `do`
        let body = Box::new(self.parse_stmt()?);
        self.expect(&TokenKind::While, "'while'")?;
        self.expect(&TokenKind::LParen, "'('")?;
        let test = self.parse_expr()?;
        self.expect(&TokenKind::RParen, "')'")?;
        self.eat_semicolon()?;
        Ok(Stmt::DoWhile { body, test })
    }

    fn parse_for_stmt(&mut self) -> Result<Stmt, ExprError> {
        self.bump(); // `for`
        self.expect(&TokenKind::LParen, "'('")?;

        // Check for `for (var/let/const <pat> of/in ...)`
        if matches!(
            self.peek_kind(),
            TokenKind::Var | TokenKind::Let | TokenKind::Const
        ) {
            let saved = self.pos;
            let kind = match self.bump().kind {
                TokenKind::Var => VarKind::Var,
                TokenKind::Let => {
                    self.require_expressions_plus("let declaration")?;
                    VarKind::Let
                }
                TokenKind::Const => {
                    self.require_expressions_plus("const declaration")?;
                    VarKind::Const
                }
                _ => unreachable!(),
            };
            if let Ok(pat) = self.parse_pattern() {
                if self.is_contextual_of() {
                    self.require_expressions_plus("for..of loop")?;
                    self.bump(); // `of`
                    let iterable = self.parse_expr()?;
                    self.expect(&TokenKind::RParen, "')'")?;
                    let body = self.parse_stmt()?;
                    return Ok(Stmt::ForOf(Box::new(ForOfStmt {
                        kind: Some(kind),
                        target: pat,
                        iterable,
                        body,
                    })));
                }
                if self.eat(&TokenKind::In) {
                    let object = self.parse_expr()?;
                    self.expect(&TokenKind::RParen, "')'")?;
                    let body = self.parse_stmt()?;
                    return Ok(Stmt::ForIn(Box::new(ForInStmt {
                        kind: Some(kind),
                        target: pat,
                        object,
                        body,
                    })));
                }
            }
            // Rewind and parse classic `for (var ...; ...; ...)`
            self.pos = saved;
            let kind = match self.bump().kind {
                TokenKind::Var => VarKind::Var,
                TokenKind::Let => VarKind::Let,
                TokenKind::Const => VarKind::Const,
                _ => unreachable!(),
            };
            let decls = self.parse_var_declarators(kind)?;
            self.expect(&TokenKind::Semicolon, "';'")?;
            return self.finish_classic_for(Some(ForInit::VarDecl { kind, decls }));
        }

        // Check for `for (<ident> of/in ...)` without declaration keyword
        if let TokenKind::Ident(_) = self.peek_kind() {
            let is_of = matches!(self.peek_offset(1).kind, TokenKind::Ident(ref s) if s == "of");
            let is_in = matches!(self.peek_offset(1).kind, TokenKind::In);
            if is_of {
                self.require_expressions_plus("for..of loop")?;
                let name = self.expect_ident("loop variable")?;
                self.bump(); // `of`
                let iterable = self.parse_expr()?;
                self.expect(&TokenKind::RParen, "')'")?;
                let body = self.parse_stmt()?;
                return Ok(Stmt::ForOf(Box::new(ForOfStmt {
                    kind: None,
                    target: Pattern::Ident(name),
                    iterable,
                    body,
                })));
            }
            if is_in {
                let name = self.expect_ident("loop variable")?;
                self.bump(); // `in`
                let object = self.parse_expr()?;
                self.expect(&TokenKind::RParen, "')'")?;
                let body = self.parse_stmt()?;
                return Ok(Stmt::ForIn(Box::new(ForInStmt {
                    kind: None,
                    target: Pattern::Ident(name),
                    object,
                    body,
                })));
            }
        }

        let init = if self.check(&TokenKind::Semicolon) {
            None
        } else {
            Some(ForInit::Expr(self.parse_expr()?))
        };
        self.expect(&TokenKind::Semicolon, "';'")?;
        self.finish_classic_for(init)
    }

    fn finish_classic_for(&mut self, init: Option<ForInit>) -> Result<Stmt, ExprError> {
        let test = if self.check(&TokenKind::Semicolon) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        self.expect(&TokenKind::Semicolon, "';'")?;
        let update = if self.check(&TokenKind::RParen) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        self.expect(&TokenKind::RParen, "')'")?;
        let body = self.parse_stmt()?;
        Ok(Stmt::For(Box::new(ForStmt {
            init,
            test,
            update,
            body,
        })))
    }

    fn parse_switch_stmt(&mut self) -> Result<Stmt, ExprError> {
        self.bump(); // `switch`
        self.expect(&TokenKind::LParen, "'('")?;
        let discriminant = self.parse_expr()?;
        self.expect(&TokenKind::RParen, "')'")?;
        self.expect(&TokenKind::LBrace, "'{'")?;

        let mut cases = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.at_eof() {
            let test = if self.eat(&TokenKind::Case) {
                let expr = self.parse_expr()?;
                self.expect(&TokenKind::Colon, "':'")?;
                Some(expr)
            } else if self.eat(&TokenKind::Default) {
                self.expect(&TokenKind::Colon, "':'")?;
                None
            } else {
                return Err(ExprError::SyntaxError {
                    offset: self.peek().span.start,
                    message: "expected 'case' or 'default' in switch".to_string(),
                });
            };
            let mut consequent = Vec::new();
            while !matches!(
                self.peek_kind(),
                TokenKind::Case | TokenKind::Default | TokenKind::RBrace | TokenKind::Eof
            ) {
                consequent.push(self.parse_stmt()?);
            }
            cases.push(SwitchCase { test, consequent });
        }
        self.expect(&TokenKind::RBrace, "'}'")?;
        Ok(Stmt::Switch {
            discriminant,
            cases,
        })
    }

    fn parse_return_stmt(&mut self) -> Result<Stmt, ExprError> {
        self.bump(); // `return`
        if self.at_eof()
            || self.check(&TokenKind::Semicolon)
            || self.check(&TokenKind::RBrace)
            || self.peek().line_before
        {
            let _ = self.eat(&TokenKind::Semicolon);
            return Ok(Stmt::Return(None));
        }
        let expr = self.parse_expr()?;
        self.eat_semicolon()?;
        Ok(Stmt::Return(Some(expr)))
    }

    fn parse_try_stmt(&mut self) -> Result<Stmt, ExprError> {
        self.bump(); // `try`
        let block = self.parse_block()?;
        let (catch_param, catch_body) = if self.eat(&TokenKind::Catch) {
            let param = if self.eat(&TokenKind::LParen) {
                let p = self.parse_pattern()?;
                self.expect(&TokenKind::RParen, "')'")?;
                Some(p)
            } else {
                None
            };
            let body = self.parse_block()?;
            (param, Some(body))
        } else {
            (None, None)
        };
        let finally_body = if self.eat(&TokenKind::Finally) {
            Some(self.parse_block()?)
        } else {
            None
        };
        if catch_body.is_none() && finally_body.is_none() {
            return Err(ExprError::SyntaxError {
                offset: self.peek().span.start,
                message: "'try' requires 'catch' or 'finally'".to_string(),
            });
        }
        Ok(Stmt::Try(Box::new(TryStmt {
            block,
            catch_param,
            catch_body,
            finally_body,
        })))
    }

    fn is_contextual_of(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::Ident(s) if s == "of")
    }

    // -----------------------------------------------------------------------
    // Expressions
    // -----------------------------------------------------------------------

    fn parse_expr(&mut self) -> Result<Expr, ExprError> {
        self.enter_depth()?;
        let res = self.parse_sequence_expr();
        self.leave_depth();
        res
    }

    fn parse_sequence_expr(&mut self) -> Result<Expr, ExprError> {
        let first = self.parse_assignment_expr()?;
        if !self.check(&TokenKind::Comma) {
            return Ok(first);
        }
        let mut items = vec![first];
        while self.eat(&TokenKind::Comma) {
            items.push(self.parse_assignment_expr()?);
        }
        Ok(Expr::Sequence(items))
    }

    fn parse_assignment_expr(&mut self) -> Result<Expr, ExprError> {
        self.enter_depth()?;
        let res = self.parse_assignment_expr_inner();
        self.leave_depth();
        res
    }

    fn parse_assignment_expr_inner(&mut self) -> Result<Expr, ExprError> {
        // Single-parameter arrow function: `x => body`
        if let TokenKind::Ident(_) = self.peek_kind()
            && !self.peek_offset(1).line_before
            && matches!(self.peek_offset(1).kind, TokenKind::FatArrow)
        {
            self.require_expressions_plus("arrow function")?;
            let name = self.expect_ident("parameter name")?;
            self.bump(); // `=>`
            let body = self.parse_arrow_body()?;
            return Ok(Expr::Function(Box::new(FunctionExpr {
                name: None,
                params: vec![Param {
                    pattern: Pattern::Ident(name),
                    default: None,
                    is_rest: false,
                }],
                body,
                is_arrow: true,
            })));
        }

        // Parenthesized arrow function: `(a, b = 1, ...rest) => body`
        if self.check(&TokenKind::LParen) && self.is_arrow_param_list_ahead() {
            self.require_expressions_plus("arrow function")?;
            let params = self.parse_param_list()?;
            self.expect(&TokenKind::FatArrow, "'=>'")?;
            let body = self.parse_arrow_body()?;
            return Ok(Expr::Function(Box::new(FunctionExpr {
                name: None,
                params,
                body,
                is_arrow: true,
            })));
        }

        let left = self.parse_conditional_expr()?;
        let assign_op = match self.peek_kind() {
            TokenKind::Eq => Some(AssignOp::Assign),
            TokenKind::PlusEq => Some(AssignOp::AddAssign),
            TokenKind::MinusEq => Some(AssignOp::SubAssign),
            TokenKind::StarEq => Some(AssignOp::MulAssign),
            TokenKind::SlashEq => Some(AssignOp::DivAssign),
            TokenKind::PercentEq => Some(AssignOp::RemAssign),
            TokenKind::StarStarEq => Some(AssignOp::PowAssign),
            _ => None,
        };
        if let Some(op) = assign_op {
            let op_offset = self.peek().span.start;
            self.bump();
            let target = expr_to_assign_target(left, op_offset)?;
            let value = self.parse_assignment_expr()?;
            return Ok(Expr::Assign {
                op,
                target: Box::new(target),
                value: Box::new(value),
            });
        }
        Ok(left)
    }

    /// Linearly scans from `(` to its matching `)` and returns `true` iff `=>`
    /// immediately follows on the same line, avoiding exponential backtracking.
    fn is_arrow_param_list_ahead(&self) -> bool {
        if !self.check(&TokenKind::LParen) {
            return false;
        }
        let mut paren_depth: usize = 0;
        let mut bracket_depth: usize = 0;
        let mut brace_depth: usize = 0;
        let mut template_depth: usize = 0;
        let mut i = self.pos;
        while i < self.tokens.len() {
            match self.tokens[i].kind {
                TokenKind::LParen => paren_depth += 1,
                TokenKind::RParen => {
                    if paren_depth == 0 {
                        return false;
                    }
                    paren_depth -= 1;
                    if paren_depth == 0 {
                        if bracket_depth != 0 || brace_depth != 0 || template_depth != 0 {
                            return false;
                        }
                        return self.tokens.get(i + 1).is_some_and(|tok| {
                            !tok.line_before && tok.kind == TokenKind::FatArrow
                        });
                    }
                }
                TokenKind::LBracket => bracket_depth += 1,
                TokenKind::RBracket => bracket_depth = bracket_depth.saturating_sub(1),
                TokenKind::LBrace => brace_depth += 1,
                TokenKind::RBrace => brace_depth = brace_depth.saturating_sub(1),
                TokenKind::TemplateHead(_) => template_depth += 1,
                TokenKind::TemplateTail(_) => template_depth = template_depth.saturating_sub(1),
                TokenKind::Eof => return false,
                _ => {}
            }
            i += 1;
        }
        false
    }

    fn parse_arrow_body(&mut self) -> Result<FunctionBody, ExprError> {
        if self.check(&TokenKind::LBrace) {
            Ok(FunctionBody::Block(self.parse_block()?))
        } else {
            Ok(FunctionBody::Expr(Box::new(self.parse_assignment_expr()?)))
        }
    }

    fn parse_conditional_expr(&mut self) -> Result<Expr, ExprError> {
        let test = self.parse_nullish_expr()?;
        if !self.eat(&TokenKind::Question) {
            return Ok(test);
        }
        let consequent = self.parse_assignment_expr()?;
        self.expect(&TokenKind::Colon, "':' in ternary expression")?;
        let alternate = self.parse_assignment_expr()?;
        Ok(Expr::Conditional {
            test: Box::new(test),
            consequent: Box::new(consequent),
            alternate: Box::new(alternate),
        })
    }

    fn parse_nullish_expr(&mut self) -> Result<Expr, ExprError> {
        let mut left = self.parse_logical_or()?;
        let mut chain = 0usize;
        while self.check(&TokenKind::Nullish) {
            self.require_expressions_plus("nullish coalescing '??'")?;
            self.bump();
            self.enter_depth()?;
            chain += 1;
            let right = match self.parse_logical_or() {
                Ok(r) => r,
                Err(e) => {
                    for _ in 0..chain {
                        self.leave_depth();
                    }
                    return Err(e);
                }
            };
            left = Expr::Logical {
                op: LogicalOp::Nullish,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        for _ in 0..chain {
            self.leave_depth();
        }
        Ok(left)
    }

    fn parse_logical_or(&mut self) -> Result<Expr, ExprError> {
        let mut left = self.parse_logical_and()?;
        let mut chain = 0usize;
        while self.eat(&TokenKind::PipePipe) {
            self.enter_depth()?;
            chain += 1;
            let right = match self.parse_logical_and() {
                Ok(r) => r,
                Err(e) => {
                    for _ in 0..chain {
                        self.leave_depth();
                    }
                    return Err(e);
                }
            };
            left = Expr::Logical {
                op: LogicalOp::Or,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        for _ in 0..chain {
            self.leave_depth();
        }
        Ok(left)
    }

    fn parse_logical_and(&mut self) -> Result<Expr, ExprError> {
        let mut left = self.parse_bit_or()?;
        let mut chain = 0usize;
        while self.eat(&TokenKind::AmpAmp) {
            self.enter_depth()?;
            chain += 1;
            let right = match self.parse_bit_or() {
                Ok(r) => r,
                Err(e) => {
                    for _ in 0..chain {
                        self.leave_depth();
                    }
                    return Err(e);
                }
            };
            left = Expr::Logical {
                op: LogicalOp::And,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        for _ in 0..chain {
            self.leave_depth();
        }
        Ok(left)
    }

    fn parse_bit_or(&mut self) -> Result<Expr, ExprError> {
        let mut left = self.parse_bit_xor()?;
        let mut chain = 0usize;
        while self.eat(&TokenKind::Pipe) {
            self.enter_depth()?;
            chain += 1;
            let right = match self.parse_bit_xor() {
                Ok(r) => r,
                Err(e) => {
                    for _ in 0..chain {
                        self.leave_depth();
                    }
                    return Err(e);
                }
            };
            left = Expr::Binary {
                op: BinaryOp::BitOr,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        for _ in 0..chain {
            self.leave_depth();
        }
        Ok(left)
    }

    fn parse_bit_xor(&mut self) -> Result<Expr, ExprError> {
        let mut left = self.parse_bit_and()?;
        let mut chain = 0usize;
        while self.eat(&TokenKind::Caret) {
            self.enter_depth()?;
            chain += 1;
            let right = match self.parse_bit_and() {
                Ok(r) => r,
                Err(e) => {
                    for _ in 0..chain {
                        self.leave_depth();
                    }
                    return Err(e);
                }
            };
            left = Expr::Binary {
                op: BinaryOp::BitXor,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        for _ in 0..chain {
            self.leave_depth();
        }
        Ok(left)
    }

    fn parse_bit_and(&mut self) -> Result<Expr, ExprError> {
        let mut left = self.parse_equality()?;
        let mut chain = 0usize;
        while self.eat(&TokenKind::Amp) {
            self.enter_depth()?;
            chain += 1;
            let right = match self.parse_equality() {
                Ok(r) => r,
                Err(e) => {
                    for _ in 0..chain {
                        self.leave_depth();
                    }
                    return Err(e);
                }
            };
            left = Expr::Binary {
                op: BinaryOp::BitAnd,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        for _ in 0..chain {
            self.leave_depth();
        }
        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expr, ExprError> {
        let mut left = self.parse_relational()?;
        let mut chain = 0usize;
        loop {
            let op = match self.peek_kind() {
                TokenKind::EqEq => BinaryOp::Eq,
                TokenKind::BangEq => BinaryOp::Ne,
                TokenKind::EqEqEq => BinaryOp::StrictEq,
                TokenKind::BangEqEq => BinaryOp::StrictNe,
                _ => break,
            };
            self.bump();
            self.enter_depth()?;
            chain += 1;
            let right = match self.parse_relational() {
                Ok(r) => r,
                Err(e) => {
                    for _ in 0..chain {
                        self.leave_depth();
                    }
                    return Err(e);
                }
            };
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        for _ in 0..chain {
            self.leave_depth();
        }
        Ok(left)
    }

    fn parse_relational(&mut self) -> Result<Expr, ExprError> {
        let mut left = self.parse_shift()?;
        let mut chain = 0usize;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Lt => BinaryOp::Lt,
                TokenKind::LtEq => BinaryOp::Le,
                TokenKind::Gt => BinaryOp::Gt,
                TokenKind::GtEq => BinaryOp::Ge,
                _ => break,
            };
            self.bump();
            self.enter_depth()?;
            chain += 1;
            let right = match self.parse_shift() {
                Ok(r) => r,
                Err(e) => {
                    for _ in 0..chain {
                        self.leave_depth();
                    }
                    return Err(e);
                }
            };
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        for _ in 0..chain {
            self.leave_depth();
        }
        Ok(left)
    }

    fn parse_shift(&mut self) -> Result<Expr, ExprError> {
        let mut left = self.parse_additive()?;
        let mut chain = 0usize;
        loop {
            let op = match self.peek_kind() {
                TokenKind::LtLt => BinaryOp::Shl,
                TokenKind::GtGt => BinaryOp::Shr,
                TokenKind::GtGtGt => BinaryOp::UShr,
                _ => break,
            };
            self.bump();
            self.enter_depth()?;
            chain += 1;
            let right = match self.parse_additive() {
                Ok(r) => r,
                Err(e) => {
                    for _ in 0..chain {
                        self.leave_depth();
                    }
                    return Err(e);
                }
            };
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        for _ in 0..chain {
            self.leave_depth();
        }
        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Expr, ExprError> {
        let mut left = self.parse_multiplicative()?;
        let mut chain = 0usize;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.bump();
            self.enter_depth()?;
            chain += 1;
            let right = match self.parse_multiplicative() {
                Ok(r) => r,
                Err(e) => {
                    for _ in 0..chain {
                        self.leave_depth();
                    }
                    return Err(e);
                }
            };
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        for _ in 0..chain {
            self.leave_depth();
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, ExprError> {
        let mut left = self.parse_exponentiation()?;
        let mut chain = 0usize;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::Percent => BinaryOp::Rem,
                _ => break,
            };
            self.bump();
            self.enter_depth()?;
            chain += 1;
            let right = match self.parse_exponentiation() {
                Ok(r) => r,
                Err(e) => {
                    for _ in 0..chain {
                        self.leave_depth();
                    }
                    return Err(e);
                }
            };
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        for _ in 0..chain {
            self.leave_depth();
        }
        Ok(left)
    }

    fn parse_exponentiation(&mut self) -> Result<Expr, ExprError> {
        self.enter_depth()?;
        let res = (|| {
            let is_unary_prefix = matches!(
                self.peek_kind(),
                TokenKind::Bang
                    | TokenKind::Minus
                    | TokenKind::Plus
                    | TokenKind::Tilde
                    | TokenKind::TypeOf
                    | TokenKind::Void
                    | TokenKind::Delete
                    | TokenKind::Await
                    | TokenKind::Yield
                    | TokenKind::New
            );
            let left = self.parse_unary()?;
            if self.check(&TokenKind::StarStar) {
                if is_unary_prefix {
                    return Err(ExprError::SyntaxError {
                        offset: self.peek().span.start,
                        message: "unary operator cannot be the unparenthesized base of '**'"
                            .to_string(),
                    });
                }
                self.bump();
                // Right-associative: `2 ** 3 ** 2 == 2 ** (3 ** 2)`.
                let right = self.parse_exponentiation()?;
                return Ok(Expr::Binary {
                    op: BinaryOp::Pow,
                    left: Box::new(left),
                    right: Box::new(right),
                });
            }
            Ok(left)
        })();
        self.leave_depth();
        res
    }

    fn parse_unary(&mut self) -> Result<Expr, ExprError> {
        self.enter_depth()?;
        let res = match self.peek_kind() {
            TokenKind::Bang => {
                self.bump();
                Ok(Expr::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(self.parse_unary()?),
                })
            }
            TokenKind::Minus => {
                self.bump();
                Ok(Expr::Unary {
                    op: UnaryOp::Neg,
                    expr: Box::new(self.parse_unary()?),
                })
            }
            TokenKind::Plus => {
                self.bump();
                Ok(Expr::Unary {
                    op: UnaryOp::Pos,
                    expr: Box::new(self.parse_unary()?),
                })
            }
            TokenKind::Tilde => {
                self.bump();
                Ok(Expr::Unary {
                    op: UnaryOp::BitNot,
                    expr: Box::new(self.parse_unary()?),
                })
            }
            TokenKind::TypeOf => {
                self.bump();
                Ok(Expr::Unary {
                    op: UnaryOp::TypeOf,
                    expr: Box::new(self.parse_unary()?),
                })
            }
            TokenKind::Void => {
                self.bump();
                Ok(Expr::Unary {
                    op: UnaryOp::Void,
                    expr: Box::new(self.parse_unary()?),
                })
            }
            TokenKind::Delete => {
                self.bump();
                Ok(Expr::Unary {
                    op: UnaryOp::Delete,
                    expr: Box::new(self.parse_unary()?),
                })
            }
            TokenKind::PlusPlus | TokenKind::MinusMinus => {
                let tok = self.bump();
                let op = if tok.kind == TokenKind::PlusPlus {
                    UpdateOp::Inc
                } else {
                    UpdateOp::Dec
                };
                let operand = self.parse_unary()?;
                let target = expr_to_assign_target(operand, tok.span.start)?;
                Ok(Expr::Update {
                    op,
                    prefix: true,
                    target: Box::new(target),
                })
            }
            TokenKind::Await => Err(ExprError::Forbidden("await".to_string())),
            TokenKind::Yield => Err(ExprError::Forbidden("yield".to_string())),
            TokenKind::New => {
                self.bump();
                if let TokenKind::Ident(name) = self.peek_kind() {
                    return Err(ExprError::Forbidden(format!("new {name}")));
                }
                Err(ExprError::Forbidden("new".to_string()))
            }
            _ => self.parse_postfix(),
        };
        self.leave_depth();
        res
    }

    fn parse_postfix(&mut self) -> Result<Expr, ExprError> {
        let expr = self.parse_call_and_member()?;
        if !self.peek().line_before
            && matches!(
                self.peek_kind(),
                TokenKind::PlusPlus | TokenKind::MinusMinus
            )
        {
            let tok = self.bump();
            let op = if tok.kind == TokenKind::PlusPlus {
                UpdateOp::Inc
            } else {
                UpdateOp::Dec
            };
            let target = expr_to_assign_target(expr, tok.span.start)?;
            return Ok(Expr::Update {
                op,
                prefix: false,
                target: Box::new(target),
            });
        }
        Ok(expr)
    }

    fn parse_call_and_member(&mut self) -> Result<Expr, ExprError> {
        let mut expr = self.parse_primary()?;
        let mut chain_ops: Option<Vec<ChainOp>> = None;
        let mut member_chain = 0usize;

        loop {
            match self.peek_kind() {
                TokenKind::OptionalChain => {
                    self.require_expressions_plus("optional chaining '?.'")?;
                    self.bump();
                    let op = if self.eat(&TokenKind::LBracket) {
                        let index = self.parse_expr()?;
                        self.expect(&TokenKind::RBracket, "']'")?;
                        ChainOp::Index {
                            index,
                            optional: true,
                        }
                    } else if self.check(&TokenKind::LParen) {
                        let args = self.parse_call_args()?;
                        ChainOp::Call {
                            args,
                            optional: true,
                        }
                    } else {
                        let prop = self.expect_property_name()?;
                        ChainOp::Member {
                            prop,
                            optional: true,
                        }
                    };
                    chain_ops.get_or_insert_with(Vec::new).push(op);
                }
                TokenKind::Dot => {
                    self.bump();
                    let prop = self.expect_property_name()?;
                    if let Some(ops) = &mut chain_ops {
                        ops.push(ChainOp::Member {
                            prop,
                            optional: false,
                        });
                    } else {
                        self.enter_depth()?;
                        member_chain += 1;
                        expr = Expr::Member {
                            object: Box::new(expr),
                            prop,
                        };
                    }
                }
                TokenKind::LBracket => {
                    self.bump();
                    let index = self.parse_expr()?;
                    self.expect(&TokenKind::RBracket, "']'")?;
                    if let Some(ops) = &mut chain_ops {
                        ops.push(ChainOp::Index {
                            index,
                            optional: false,
                        });
                    } else {
                        self.enter_depth()?;
                        member_chain += 1;
                        expr = Expr::Index {
                            object: Box::new(expr),
                            index: Box::new(index),
                        };
                    }
                }
                TokenKind::LParen => {
                    let args = self.parse_call_args()?;
                    if let Some(ops) = &mut chain_ops {
                        ops.push(ChainOp::Call {
                            args,
                            optional: false,
                        });
                    } else {
                        self.enter_depth()?;
                        member_chain += 1;
                        expr = Expr::Call {
                            callee: Box::new(expr),
                            args,
                        };
                    }
                }
                _ => break,
            }
        }

        for _ in 0..member_chain {
            self.leave_depth();
        }

        if let Some(ops) = chain_ops {
            Ok(Expr::OptionalChain {
                base: Box::new(expr),
                ops,
            })
        } else {
            Ok(expr)
        }
    }

    fn parse_call_args(&mut self) -> Result<Vec<CallArg>, ExprError> {
        self.expect(&TokenKind::LParen, "'('")?;
        let mut args = Vec::new();
        while !self.check(&TokenKind::RParen) && !self.at_eof() {
            if self.eat(&TokenKind::Ellipsis) {
                self.require_expressions_plus("spread arguments")?;
                args.push(CallArg::Spread(self.parse_assignment_expr()?));
            } else {
                args.push(CallArg::Expr(self.parse_assignment_expr()?));
            }
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RParen, "')'")?;
        Ok(args)
    }

    fn parse_primary(&mut self) -> Result<Expr, ExprError> {
        match self.peek_kind().clone() {
            TokenKind::Class => Err(ExprError::Forbidden("class".to_string())),
            TokenKind::Import => Err(ExprError::Forbidden("import".to_string())),
            TokenKind::Export => Err(ExprError::Forbidden("export".to_string())),
            TokenKind::Async => Err(ExprError::Forbidden("async".to_string())),
            TokenKind::Await => Err(ExprError::Forbidden("await".to_string())),
            TokenKind::Yield => Err(ExprError::Forbidden("yield".to_string())),
            TokenKind::With => Err(ExprError::Forbidden("with".to_string())),
            TokenKind::Debugger => Err(ExprError::Forbidden("debugger".to_string())),
            TokenKind::This => Err(ExprError::Forbidden("this".to_string())),
            TokenKind::Super => Err(ExprError::Forbidden("super".to_string())),

            TokenKind::Number(n) => {
                self.bump();
                Ok(Expr::Literal(Literal::Number(n)))
            }
            TokenKind::String(s) => {
                self.bump();
                Ok(Expr::Literal(Literal::String(s)))
            }
            TokenKind::True => {
                self.bump();
                Ok(Expr::Literal(Literal::Bool(true)))
            }
            TokenKind::False => {
                self.bump();
                Ok(Expr::Literal(Literal::Bool(false)))
            }
            TokenKind::Null => {
                self.bump();
                Ok(Expr::Literal(Literal::Null))
            }
            TokenKind::Undefined => {
                self.bump();
                Ok(Expr::Literal(Literal::Undefined))
            }
            TokenKind::Ident(name) => {
                self.bump();
                Ok(Expr::Ident(name))
            }
            TokenKind::TemplateNoSub(text) => {
                self.require_expressions_plus("template literal")?;
                self.bump();
                Ok(Expr::Template(vec![TemplatePart::Text(text)]))
            }
            TokenKind::TemplateHead(head) => {
                self.require_expressions_plus("template literal")?;
                self.bump();
                self.parse_template_rest(head)
            }
            TokenKind::LParen => {
                self.bump();
                let expr = self.parse_expr()?;
                self.expect(&TokenKind::RParen, "')'")?;
                Ok(expr)
            }
            TokenKind::LBracket => self.parse_array_literal(),
            TokenKind::LBrace => self.parse_object_literal(),
            TokenKind::Function => {
                self.require_expressions_plus("function expression")?;
                self.parse_function_expr()
            }
            _ => Err(ExprError::SyntaxError {
                offset: self.peek().span.start,
                message: format!("unexpected token {:?}", self.peek_kind()),
            }),
        }
    }

    fn parse_template_rest(&mut self, head: String) -> Result<Expr, ExprError> {
        let mut parts = Vec::new();
        if !head.is_empty() {
            parts.push(TemplatePart::Text(head));
        }
        loop {
            let expr = self.parse_expr()?;
            parts.push(TemplatePart::Expr(expr));
            match self.peek_kind().clone() {
                TokenKind::TemplateMiddle(mid) => {
                    self.bump();
                    if !mid.is_empty() {
                        parts.push(TemplatePart::Text(mid));
                    }
                }
                TokenKind::TemplateTail(tail) => {
                    self.bump();
                    if !tail.is_empty() {
                        parts.push(TemplatePart::Text(tail));
                    }
                    break;
                }
                _ => {
                    return Err(ExprError::SyntaxError {
                        offset: self.peek().span.start,
                        message: "unterminated template expression".to_string(),
                    });
                }
            }
        }
        Ok(Expr::Template(parts))
    }

    fn parse_array_literal(&mut self) -> Result<Expr, ExprError> {
        self.bump(); // `[`
        let mut elems = Vec::new();
        while !self.check(&TokenKind::RBracket) && !self.at_eof() {
            if self.eat(&TokenKind::Ellipsis) {
                self.require_expressions_plus("spread element")?;
                elems.push(ArrayElem::Spread(self.parse_assignment_expr()?));
            } else {
                elems.push(ArrayElem::Expr(self.parse_assignment_expr()?));
            }
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RBracket, "']'")?;
        Ok(Expr::Array(elems))
    }

    fn parse_object_literal(&mut self) -> Result<Expr, ExprError> {
        self.bump(); // `{`
        let mut props = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.at_eof() {
            if self.eat(&TokenKind::Ellipsis) {
                self.require_expressions_plus("object spread")?;
                props.push(ObjectProp::Spread(self.parse_assignment_expr()?));
            } else if self.eat(&TokenKind::LBracket) {
                self.require_expressions_plus("computed property key")?;
                let key_expr = self.parse_assignment_expr()?;
                self.expect(&TokenKind::RBracket, "']'")?;
                self.expect(&TokenKind::Colon, "':'")?;
                let value = self.parse_assignment_expr()?;
                props.push(ObjectProp::KeyValue {
                    key: ObjectKey::Computed(key_expr),
                    value,
                });
            } else {
                let is_ident = matches!(self.peek_kind(), TokenKind::Ident(_));
                let key = self.parse_static_prop_name()?;
                if self.eat(&TokenKind::Colon) {
                    let value = self.parse_assignment_expr()?;
                    props.push(ObjectProp::KeyValue {
                        key: ObjectKey::Static(key),
                        value,
                    });
                } else if is_ident {
                    self.require_expressions_plus("shorthand object property")?;
                    props.push(ObjectProp::KeyValue {
                        key: ObjectKey::Static(key.clone()),
                        value: Expr::Ident(key),
                    });
                } else {
                    return Err(ExprError::SyntaxError {
                        offset: self.peek().span.start,
                        message: "expected ':' after property key".to_string(),
                    });
                }
            }
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RBrace, "'}'")?;
        Ok(Expr::Object(props))
    }

    fn parse_function_expr(&mut self) -> Result<Expr, ExprError> {
        self.bump(); // `function`
        if self.check(&TokenKind::Star) {
            return Err(ExprError::Forbidden("generator function*".to_string()));
        }
        let name = if let TokenKind::Ident(_) = self.peek_kind() {
            Some(self.expect_ident("function name")?)
        } else {
            None
        };
        let params = self.parse_param_list()?;
        let body = self.parse_block()?;
        Ok(Expr::Function(Box::new(FunctionExpr {
            name,
            params,
            body: FunctionBody::Block(body),
            is_arrow: false,
        })))
    }

    fn expect_ident(&mut self, label: &str) -> Result<String, ExprError> {
        match self.peek_kind().clone() {
            TokenKind::Ident(s) => {
                self.bump();
                Ok(s)
            }
            _ => Err(ExprError::SyntaxError {
                offset: self.peek().span.start,
                message: format!("expected {label}"),
            }),
        }
    }

    fn expect_property_name(&mut self) -> Result<String, ExprError> {
        if let Some(name) = self.peek_kind().as_property_name() {
            let owned = name.to_string();
            self.bump();
            Ok(owned)
        } else {
            Err(ExprError::SyntaxError {
                offset: self.peek().span.start,
                message: "expected property name".to_string(),
            })
        }
    }

    fn parse_static_prop_name(&mut self) -> Result<String, ExprError> {
        match self.peek_kind().clone() {
            TokenKind::String(s) => {
                self.bump();
                Ok(s)
            }
            TokenKind::Number(n) => {
                self.bump();
                Ok(crate::builtins::format_js_number(n))
            }
            _ => self.expect_property_name(),
        }
    }
}

fn expr_to_assign_target(expr: Expr, offset: usize) -> Result<AssignTarget, ExprError> {
    match expr {
        Expr::Ident(name) => Ok(AssignTarget::Ident(name)),
        Expr::Member { object, prop } => Ok(AssignTarget::Member {
            object: *object,
            prop,
        }),
        Expr::Index { object, index } => Ok(AssignTarget::Index {
            object: *object,
            index: *index,
        }),
        _ => Err(ExprError::SyntaxError {
            offset,
            message: "invalid assignment target".to_string(),
        }),
    }
}
