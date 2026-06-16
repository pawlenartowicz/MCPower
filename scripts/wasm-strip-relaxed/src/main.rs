// Build-time WebKit-compatibility strip: rewrite the dead relaxed-SIMD opcodes
// out of a wasm-pack-built engine_wasm_bg.wasm so the (base-simd128) module
// validates on WebKit, which has not shipped the float relaxed ops at any iOS
// version (the whole module otherwise fails: "invalid extended simd op 271").
// The relaxed ops are faer's never-executed `RelaxedSimd` kernel monomorphizations
// — the runtime feature flag is always false, so the `Simd128` copies actually
// run — so removing them is perf-neutral on every platform; the live `Simd128`
// copies contain none of these ops and are left byte-identical. All three present
// ops have a semantics-preserving 1->1 baseline replacement, so a single in-place
// visitor pass suffices — no sequence expansion, no per-function scratch locals:
//   f64x2.relaxed_madd(a,b,c) = a*b+c  ->  call $fma  (helper computes (a*b)+c)
//   f64x2.relaxed_min         ->  f64x2.min
//   f64x2.relaxed_max         ->  f64x2.max
//
// Usage: strip <input.wasm> <output.wasm>
//        strip --count <input.wasm>     (report relaxed-op counts only, no rewrite)

use anyhow::{bail, Context, Result};
use walrus::ir::{BinaryOp, Binop, Call, Instr, InstrLocId, LocalGet, TernaryOp, VisitorMut};
use walrus::{FunctionId, FunctionKind, Module, ValType};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 2 && args[1] == "--count" {
        let mut module = Module::from_file(&args[2]).context("parse")?;
        let (madd, min, max) = scan(&mut module);
        println!("relaxed_madd={madd} relaxed_min={min} relaxed_max={max} total={}", madd + min + max);
        return Ok(());
    }
    if args.len() < 3 {
        bail!("usage: strip <input.wasm> <output.wasm>  |  strip --count <input.wasm>");
    }
    let input = &args[1];
    let output = &args[2];

    let mut module = Module::from_file(input).context("parse input")?;
    let (before_madd, before_min, before_max) = scan(&mut module);
    println!(
        "before: relaxed_madd={before_madd} relaxed_min={before_min} relaxed_max={before_max} total={}",
        before_madd + before_min + before_max
    );

    // Helper $fma(a,b,c) = (a*b)+c. One function, replaces every relaxed_madd by a call.
    let a = module.locals.add(ValType::V128);
    let b = module.locals.add(ValType::V128);
    let c = module.locals.add(ValType::V128);
    let mut fb = walrus::FunctionBuilder::new(
        &mut module.types,
        &[ValType::V128, ValType::V128, ValType::V128],
        &[ValType::V128],
    );
    fb.func_body()
        .instr(LocalGet { local: a })
        .instr(LocalGet { local: b })
        .instr(Binop { op: BinaryOp::F64x2Mul })
        .instr(LocalGet { local: c })
        .instr(Binop { op: BinaryOp::F64x2Add });
    let fma: FunctionId = fb.finish(vec![a, b, c], &mut module.funcs);

    // Rewrite pass.
    struct Strip {
        fma: FunctionId,
        madd: u64,
        min: u64,
        max: u64,
    }
    impl VisitorMut for Strip {
        fn visit_instr_mut(&mut self, instr: &mut Instr, _: &mut InstrLocId) {
            match instr {
                Instr::TernOp(t) if matches!(t.op, TernaryOp::F64x2RelaxedMadd) => {
                    *instr = Instr::Call(Call { func: self.fma });
                    self.madd += 1;
                }
                Instr::Binop(bin) => match bin.op {
                    BinaryOp::F64x2RelaxedMin => {
                        bin.op = BinaryOp::F64x2Min;
                        self.min += 1;
                    }
                    BinaryOp::F64x2RelaxedMax => {
                        bin.op = BinaryOp::F64x2Max;
                        self.max += 1;
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }
    let mut strip = Strip { fma, madd: 0, min: 0, max: 0 };
    let ids: Vec<FunctionId> = module.funcs.iter().map(|f| f.id()).collect();
    for id in ids {
        if id == fma {
            continue;
        }
        if let FunctionKind::Local(_) = module.funcs.get(id).kind {
            let lf = module.funcs.get_mut(id).kind.unwrap_local_mut();
            let entry = lf.entry_block();
            walrus::ir::dfs_pre_order_mut(&mut strip, lf, entry);
        }
    }
    println!(
        "rewrote: relaxed_madd={} relaxed_min={} relaxed_max={}",
        strip.madd, strip.min, strip.max
    );

    // Drop the `target_features` custom section so the module no longer advertises
    // +relaxed-simd (it's informational; consumers infer features from opcodes).
    let tf: Vec<_> = module
        .customs
        .iter()
        .filter(|(_, s)| s.name() == "target_features")
        .map(|(id, _)| id)
        .collect();
    for id in tf {
        module.customs.delete(id);
        println!("dropped target_features custom section");
    }

    module.emit_wasm_file(output).context("emit output")?;

    // Re-parse the output and recount as an independent check.
    let mut out = Module::from_file(output).context("re-parse output")?;
    let (am, an, ax) = scan(&mut out);
    println!("after (re-parsed): relaxed_madd={am} relaxed_min={an} relaxed_max={ax} total={}", am + an + ax);
    if am + an + ax != 0 {
        bail!("output still contains relaxed ops");
    }
    Ok(())
}

// Mutating scan (dfs needs &mut). Counts without changing anything.
fn scan(module: &mut Module) -> (u64, u64, u64) {
    struct C {
        madd: u64,
        min: u64,
        max: u64,
    }
    impl VisitorMut for C {
        fn visit_instr_mut(&mut self, instr: &mut Instr, _: &mut InstrLocId) {
            match instr {
                Instr::TernOp(t) if matches!(t.op, TernaryOp::F64x2RelaxedMadd) => self.madd += 1,
                Instr::Binop(b) => match b.op {
                    BinaryOp::F64x2RelaxedMin => self.min += 1,
                    BinaryOp::F64x2RelaxedMax => self.max += 1,
                    _ => {}
                },
                _ => {}
            }
        }
    }
    let mut c = C { madd: 0, min: 0, max: 0 };
    let ids: Vec<FunctionId> = module.funcs.iter().map(|f| f.id()).collect();
    for id in ids {
        if let FunctionKind::Local(_) = module.funcs.get(id).kind {
            let lf = module.funcs.get_mut(id).kind.unwrap_local_mut();
            let entry = lf.entry_block();
            walrus::ir::dfs_pre_order_mut(&mut c, lf, entry);
        }
    }
    (c.madd, c.min, c.max)
}
