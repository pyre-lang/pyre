//! Exercise `blackhole.py _done_with_this_frame` without a merge-point exit.
//! A Python loop-exit fixture can return ContinueRunningNormally before its
//! RETURN_VALUE, so construct the actual one-frame resume stream here.

use std::rc::Rc;
use std::sync::Arc;

use majit_ir::{Const, GcRef, Type};
use majit_metainterp::jitcode::JitCodeBuilder;
use majit_metainterp::resume::{NumberingState, TAGBOX, TAGCONST, tag};
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{Mode, PyExecutionContext, compile_source_with_filename};
use pyre_jit::call_jit::{BlackholeResult, blackhole_resume_via_rd_numb};

#[test]
fn terminal_ref_return_finishes_the_frame_and_leaves_its_execution_scope() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            // As a launcher does: the module GC types join the collector
            // `init_jit_hooks` builds.
            pyre_module::register();
            pyre_jit::eval::init_jit_hooks();
            let vinfo = pyre_jit::eval::driver_pair().1.clone();
            let ec = Rc::new(PyExecutionContext::default());
            let ec_ptr = Rc::as_ptr(&ec) as *mut PyExecutionContext;
            pyre_interpreter::call::set_last_exec_ctx(ec_ptr);

            let live = pyre_jit_trace::state::intern_liveness(&[], &[0], &[]).unwrap();
            let op_live = pyre_jit_trace::state::op_live();
            let op_return = pyre_jit_trace::jitcode_runtime::insns_opname_to_byte()["ref_return/r"];
            let mut code = JitCodeBuilder::default().finish();
            code.body_mut().code = vec![op_live, live as u8, (live >> 8) as u8, op_return, 0];
            code.body_mut().c_num_regs_r = 1;
            code.body_mut().startpoints = Some([0_usize, 3].into_iter().collect());
            let payload = pyre_jit_trace::PyJitCode::from_core_degenerate(
                Arc::new(code),
                std::ptr::null(),
                false,
            );
            let index = pyre_jit_trace::state::install_codeless_jitcode(Arc::new(payload));

            // Run twice through the same builder, including its release/reuse
            // path, with a distinct live application frame each time.
            for _ in 0..2 {
                let code =
                    compile_source_with_filename("pass\n", Mode::Exec, "terminal.py").unwrap();
                let mut frame = PyFrame::new_with_context(code, ec.clone()).unwrap();
                let frame_ptr = frame.as_mut_ptr();
                assert!(!frame.frame_finished_execution());
                unsafe { (*ec_ptr).enter(frame_ptr) };

                // `opencoder.Trace._list_of_boxes_virtualizable`: identity
                // first, then the static and array payload read from the frame.
                let (values, lengths) = unsafe { vinfo.load_list_of_boxes(frame_ptr.cast()) };
                let mut kinds: Vec<Type> =
                    vinfo.static_fields.iter().map(|f| f.field_type).collect();
                for (field, length) in vinfo.array_fields.iter().zip(lengths) {
                    kinds.extend(std::iter::repeat_n(field.item_type, length));
                }
                assert_eq!(values.len(), kinds.len());
                let constants: Vec<Const> = values
                    .into_iter()
                    .zip(kinds)
                    .map(|(value, kind)| match kind {
                        Type::Int => Const::Int(value),
                        Type::Ref => Const::from_gcref(GcRef(value as usize)),
                        Type::Float => Const::Float(f64::from_bits(value as u64)),
                        Type::Void => unreachable!(),
                    })
                    .collect();
                let mut writer = NumberingState::new(constants.len() + 9);
                writer.append_int(0); // patched section size
                writer.append_int(1); // one failarg: the live frame
                writer.append_int((constants.len() + 1) as i64);
                writer.append_int(tag(0, TAGBOX).unwrap() as i64);
                for slot in 0..constants.len() {
                    writer.append_int(tag(slot as i32, TAGCONST).unwrap() as i64);
                }
                writer.append_int(0); // no virtualrefs
                writer.append_int(index as i64);
                writer.append_int(0); // -live- position (`ResumeDataLoopMemo.number`)
                writer.append_int(tag(0, TAGBOX).unwrap() as i64); // ref register 0
                writer.patch_current_size(0);

                let result = blackhole_resume_via_rd_numb(
                    &writer.create_numbering(),
                    &constants,
                    majit_backend::FailArgSource::from(&[frame_ptr as i64][..]),
                    None,
                    None,
                    Some(&[Type::Ref]),
                    0,
                    false,
                    None,
                    None,
                );
                let BlackholeResult::DoneWithThisFrameRef(returned) = result else {
                    panic!("one ref_return must complete, not resume at a merge point");
                };
                assert_eq!(returned, frame_ptr.cast());
                assert!(frame.frame_finished_execution());
                assert!(unsafe { (*ec_ptr).topframeref.is_null() });
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
