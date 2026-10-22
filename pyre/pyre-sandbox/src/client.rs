//! In-process round-trips of `majit_rlib::rsandbox` against [`crate::sandlib::SandboxPolicy`].
//!
//! The client runtime (`sandboxed_io`, `encode_request`, `load_result`,
//! `reraise_error`, `syscall`) lives in `majit_rlib::rsandbox`; these tests
//! keep the controller harness that cannot move with it.

#[cfg(test)]
mod tests {
    use crate::sandlib::{Console, SandboxPolicy};
    use crate::vfs::{Dir, File, FsNode};
    use indexmap::IndexMap;
    use majit_rlib::rmarshal::{Loader, MarshalValue, load_int};
    use majit_rlib::rsandbox::{
        ResultKind, SandboxError, SandboxResult, SyscallResult, encode_request, load_result,
        reraise_error,
    };
    use std::rc::Rc;

    fn files_root() -> FsNode {
        let mut map: IndexMap<String, FsNode> = IndexMap::new();
        map.insert("hi.txt".into(), Rc::new(File::new("Hello, world!\n")));
        Rc::new(Dir::new(map))
    }

    fn serve(policy: &mut SandboxPolicy, request: &[u8]) -> Vec<u8> {
        let mut replies = Vec::new();
        let (i, mut o, mut e) = (Vec::new(), Vec::new(), Vec::new());
        let mut console = Console {
            input: &mut i.as_slice(),
            output: &mut o,
            error: &mut e,
            input_isatty: false,
        };
        policy
            .handle_until_return(request, &mut replies, &mut console)
            .unwrap();
        replies
    }

    fn decode(replies: Vec<u8>, kind: ResultKind) -> SandboxResult<SyscallResult> {
        let mut loader = Loader::from_bytes(replies);
        let error = load_int(&mut loader)?;
        if error != 0 {
            return Err(reraise_error(error, &mut loader));
        }
        let result = load_result(&mut loader, kind)?;
        loader.check_finished()?;
        Ok(result)
    }

    #[test]
    fn client_getcwd_roundtrip() {
        let mut p = SandboxPolicy::new(files_root(), "/tmp", vec![], false);
        let req = encode_request("ll_os.ll_os_getcwd", &[]);
        let reply = serve(&mut p, &req);
        assert_eq!(
            decode(reply, ResultKind::Str).unwrap(),
            SyscallResult::Str(b"/tmp".to_vec())
        );
    }

    #[test]
    fn client_open_read_roundtrip() {
        let mut p = SandboxPolicy::new(files_root(), "/", vec![], false);
        let open = encode_request(
            "ll_os.ll_os_open",
            &[
                MarshalValue::Str(b"/hi.txt".to_vec()),
                MarshalValue::Int(libc::O_RDONLY as i64),
                MarshalValue::Int(0o777),
            ],
        );
        let fd = match decode(serve(&mut p, &open), ResultKind::Int).unwrap() {
            SyscallResult::Int(fd) => fd,
            other => panic!("fd: {other:?}"),
        };
        let read = encode_request(
            "ll_os.ll_os_read",
            &[MarshalValue::Int(fd), MarshalValue::Int(100)],
        );
        assert_eq!(
            decode(serve(&mut p, &read), ResultKind::Str).unwrap(),
            SyscallResult::Str(b"Hello, world!\n".to_vec())
        );
    }

    #[test]
    fn client_decodes_oserror_with_errno() {
        let mut p = SandboxPolicy::new(files_root(), "/", vec![], false);
        let req = encode_request(
            "ll_os.ll_os_open",
            &[
                MarshalValue::Str(b"/hi.txt".to_vec()),
                MarshalValue::Int(libc::O_WRONLY as i64),
                MarshalValue::Int(0o666),
            ],
        );
        let err = decode(serve(&mut p, &req), ResultKind::Int).unwrap_err();
        assert_eq!(err, SandboxError::Os(libc::EPERM));
    }

    #[test]
    fn client_lseek_longlong_roundtrip() {
        let mut p = SandboxPolicy::new(files_root(), "/", vec![], false);
        let open = encode_request(
            "ll_os.ll_os_open",
            &[
                MarshalValue::Str(b"/hi.txt".to_vec()),
                MarshalValue::Int(libc::O_RDONLY as i64),
                MarshalValue::Int(0o777),
            ],
        );
        let fd = match decode(serve(&mut p, &open), ResultKind::Int).unwrap() {
            SyscallResult::Int(fd) => fd,
            other => panic!("fd: {other:?}"),
        };
        let req = encode_request(
            "ll_os.ll_os_lseek",
            &[
                MarshalValue::Int(fd),
                MarshalValue::Int(0),
                MarshalValue::Int(libc::SEEK_END as i64),
            ],
        );
        assert_eq!(
            decode(serve(&mut p, &req), ResultKind::LongLong).unwrap(),
            SyscallResult::Int(14)
        );
    }

    #[test]
    fn client_decodes_statresult() {
        let mut p = SandboxPolicy::new(files_root(), "/", vec![], false);
        let req = encode_request(
            "ll_os.ll_os_stat",
            &[MarshalValue::Str(b"/hi.txt".to_vec())],
        );
        match decode(serve(&mut p, &req), ResultKind::StatResult).unwrap() {
            SyscallResult::Stat(st) => assert_eq!(st.st_size, 14),
            other => panic!("stat: {other:?}"),
        }
    }
}
