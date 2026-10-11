//! posix implementation — PyPy: pypy/module/posix/interp_posix.py
//!
//! Verbatim move of the inline block previously in importing.rs.  The
//! shared `stat_result_type` helper is carried in here too; `init_posix`
//! is renamed to `register_module`.

#[cfg(not(feature = "sandbox"))]
use crate::importing::host::fs as host_fs;
use crate::importing::host::os as host_os;
use parking_lot::Mutex;
use pyre_object::PyObjectRef;
// Under sandbox, name libc through the seam facade so any direct syscall call
// in this module is a compile error (only types/constants/pure fns resolve).
#[cfg(feature = "sandbox")]
use crate::host_seam::sys as libc;

#[cfg(windows)]
bitflags::bitflags! {
    /// `WIN32_FIND_DATA.dwFileAttributes` bits this module decodes.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct FileAttributes: u32 {
        const READONLY = 0x1;
        const DIRECTORY = 0x10;
        const REPARSE_POINT = 0x400;
    }
}

/// PyPy `ApplevelForkCallbacks`, cached on the object space.
///
/// The callback collections are RPython lists, so use insertion-ordered Vecs;
/// this is process/interpreter state, never TLS.
#[derive(Default)]
struct ApplevelForkCallbacks {
    before_w: Vec<usize>,
    parent_w: Vec<usize>,
    child_w: Vec<usize>,
}

impl ApplevelForkCallbacks {
    const fn new_empty() -> Self {
        Self {
            before_w: Vec::new(),
            parent_w: Vec::new(),
            child_w: Vec::new(),
        }
    }
}

/// `posix.DirEntry` — native layout `[PyObject | w_name | w_path | w_stat |
/// w_lstat | dir_fd | enum_ino | enum_type]`, matching `interp_scandir.py
/// W_DirEntry`: the name and full path plus the cached `stat`
/// (`follow_symlinks=True`) and `lstat` (`follow_symlinks=False`) results.
/// `w_stat`/`w_lstat` are `PY_NULL` until first requested, so `entry.stat()`
/// re-fetches once and then returns the same object, and `is_dir`/`is_file`
/// share the same on-demand stat.  `dir_fd` is the descriptor a `scandir(fd)`
/// handed the entry (`-1` for a name), which its own stat resolves the bare
/// `name` against — the native counterpart of
/// `self.scandir_iterator.orig_fd`.  `enum_ino` is the inode `readdir` reported
/// at enumeration (`descr_inode`'s `self.inode`), so `inode()` answers from it
/// without a stat; it is `-1` when unavailable (non-unix hosts), which falls
/// back to a stat.  `enum_type` is the `d_type` `readdir` reported (the
/// `known_type` half of `self.flags`), so `is_dir`/`is_file`/`is_symlink`
/// answer from it without a stat when it is not `DT_UNKNOWN`; it defaults to
/// `DT_UNKNOWN` (`0`) — the value for a host or filesystem that reports no
/// type — which falls through to the stat.  `enum_tag` is the reparse tag the
/// same enumeration reported (`0` for a name that is no reparse point), which
/// is what `is_junction` reads; only a Windows directory walk fills it, and
/// `DirEntry_is_junction` reads the same tag off `win32_lstat` there.
/// `win32_lstat` is that walk's whole find record, kept so the first `stat()`
/// builds the `stat_result` from it instead of returning to the name.
/// The layout carries no instance dict; `name`/`path` are read-only getset
/// descriptors, so the type is not instantiable and not acceptable as a base.
#[crate::pyre_class("posix.DirEntry", cpython_heaptype)]
#[derive(Default)]
pub struct W_DirEntry {
    pub w_name: PyObjectRef,
    pub w_path: PyObjectRef,
    pub w_stat: PyObjectRef,
    pub w_lstat: PyObjectRef,
    pub dir_fd: i32,
    pub enum_ino: i64,
    pub enum_type: i32,
    pub enum_tag: i64,
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    pub win32_lstat: Option<WinFindData>,
}

/// The `WIN32_FIND_DATAW` members `_Py_attribute_data_to_stat` reads, in the
/// place `DirEntry.win32_lstat` keeps them: an entry carries the record its
/// enumeration reported and turns it into a `stat_result` only when asked.
/// Building that object at enumeration time instead costs one `os.stat_result`
/// -- ten sequence slots and thirteen named extras -- for every name in the
/// directory, which is most of what listing one used to cost.
///
/// Stored as plain integers rather than the `WIN32_FIND_DATAW` itself: the
/// find record is around 600 bytes, nearly all of it the two name buffers the
/// entry has already turned into its `name` and `path`.
#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
#[derive(Clone, Copy)]
pub struct WinFindData {
    /// `dwFileAttributes`.
    pub file_attributes: u32,
    /// `dwReserved0`, which carries the reparse tag where the attribute word
    /// says the name is a reparse point and is undefined otherwise.
    pub reserved0: u32,
    /// `nFileSizeHigh` and `nFileSizeLow` joined.
    pub file_size: u64,
    /// `ftCreationTime`, `ftLastAccessTime` and `ftLastWriteTime`, each as the
    /// 100ns tick count its `FILETIME`'s two halves spell.
    pub creation_ticks: u64,
    pub last_access_ticks: u64,
    pub last_write_ticks: u64,
}

/// Native owner for `posix.ScandirIterator` entries and enumeration state.
/// PyPy's `interp_scandir.W_ScandirIterator` keeps the equivalent state on
/// `dirp`; its typedef exposes operations rather than these fields.
#[crate::pyre_class("posix.ScandirIterator", cpython_heaptype)]
#[derive(Default)]
pub struct W_ScandirIterator {
    pub entries: PyObjectRef,
    pub index: i64,
    pub open: bool,
    /// `W_ScandirIterator._in_next` (interp_scandir.py).
    pub in_next: bool,
}

static APPLEVEL_FORK_CALLBACKS: crate::module::thread::ForkMutex<ApplevelForkCallbacks> =
    crate::module::thread::ForkMutex::new(ApplevelForkCallbacks::new_empty());
// PyPy's GIL serializes concurrent fork entry.  Pyre is free-threaded, so the
// corresponding process operation has its own narrow serializer.  This lock is
// held across `fork()` by the surviving thread, the same discipline
// `fork_under_stw` uses for `GcSync::quiesce`: an OS-backed mutex, owned on
// both sides, dropped on both sides.  A `parking_lot` mutex would carry the
// parent's userspace waiter queue into the child.
static FORK_SERIALIZER: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// `rpy_init_mutexes` half for the posix fork tables.  Writes only.
/// `FORK_SERIALIZER` is not in this set: the surviving thread still holds it.
/// Called only from the unix `pthread_atfork` child handler.
#[cfg(unix)]
pub(crate) unsafe fn reinit_fork_tables_after_fork() {
    unsafe {
        APPLEVEL_FORK_CALLBACKS.reinit_after_fork();
    }
}

// `_in_next`'s test-and-set is indivisible under PyPy's GIL. Pyre is
// free-threaded, so every borrow of the native scandir iterator takes this
// narrow serializer. Claiming, taking, and releasing are separate serialized
// accesses, so a second thread arriving during a claimed step observes
// `_in_next` and is refused as interp_scandir.py requires.
static SCANDIR_IN_NEXT_SERIALIZER: Mutex<()> = Mutex::new(());

fn require_env_mapping(
    mapping: PyObjectRef,
    function: &str,
    accepts_none: bool,
) -> Result<(), crate::PyError> {
    if crate::baseobjspace::py_mapping_check(mapping) {
        return Ok(());
    }
    let none_tail = if accepts_none { " or None" } else { "" };
    Err(crate::PyError::type_error(format!(
        "{function}: environment must be a mapping object{none_tail}"
    )))
}

/// The `key=value` byte entries an exec takes from `mapping`, in the order its
/// `keys()` and `values()` hold them.
///
/// Both sequences are snapshotted before any element is encoded, so a
/// `__fspath__` running during the encoding cannot make a later read observe a
/// mutation it performed.  How many variables there are is the mapping's own
/// `len()`, so a snapshot too short to cover it is an error rather than a
/// quietly shorter environment.
///
/// `function` names the caller in the errors, and `accepts_none` spells the
/// message for an entry point that also takes `None` — what `None` means is
/// decided before the call, never here.
fn collect_env_entries(
    mapping: PyObjectRef,
    function: &str,
    accepts_none: bool,
) -> Result<Vec<Vec<u8>>, crate::PyError> {
    let _env_roots = pyre_object::gc_roots::push_roots();
    let mapping_slot = pyre_object::gc_roots::pin_roots(&[mapping]);
    require_env_mapping(
        pyre_object::gc_roots::shadow_stack_get(mapping_slot),
        function,
        accepts_none,
    )?;
    let pair_count =
        crate::baseobjspace::len_w(pyre_object::gc_roots::shadow_stack_get(mapping_slot))? as usize;
    let mut bases = [0usize; 2];
    let mut lengths = [0usize; 2];
    for (i, method) in ["keys", "values"].into_iter().enumerate() {
        let sequence = crate::baseobjspace::call_method(
            pyre_object::gc_roots::shadow_stack_get(mapping_slot),
            method,
            &[],
        );
        if sequence.is_null() {
            return Err(crate::call::take_call_error().unwrap_or_else(|| {
                crate::PyError::type_error(format!("{function}: env must be a mapping"))
            }));
        }
        let items = crate::baseobjspace::unpackiterable(sequence, -1)?;
        bases[i] = pyre_object::gc_roots::pin_roots(&items);
        lengths[i] = items.len();
    }
    // Capacity follows the available snapshots, while iteration still uses the
    // mapping's reported length and rejects a snapshot too short to cover it.
    let mut env = Vec::with_capacity(pair_count.min(lengths[0]).min(lengths[1]));
    for i in 0..pair_count {
        if i >= lengths[0] || i >= lengths[1] {
            return Err(crate::PyError::index_error("list index out of range"));
        }
        let key = crate::gateway::fsencode_bytes_w(pyre_object::gc_roots::shadow_stack_get(
            bases[0] + i,
        ))?;
        let value = crate::gateway::fsencode_bytes_w(pyre_object::gc_roots::shadow_stack_get(
            bases[1] + i,
        ))?;
        // PyPy's `_env2interp` permits the Windows `=C:` form and rejects `=`
        // only after the first byte.
        if key.is_empty() || key.get(1..).is_some_and(|tail| tail.contains(&b'=')) {
            return Err(crate::PyError::value_error(
                "illegal environment variable name",
            ));
        }
        let mut entry = key;
        entry.push(b'=');
        entry.extend_from_slice(&value);
        env.push(entry);
    }
    Ok(env)
}

#[cfg(all(unix, feature = "host_env", not(target_os = "redox")))]
fn sysconf_names() -> &'static [(&'static str, i32)] {
    use rustpython_host_env::posix as host_posix;
    &[
        ("SC_2_CHAR_TERM", host_posix::_SC_2_CHAR_TERM),
        ("SC_2_C_BIND", host_posix::_SC_2_C_BIND),
        ("SC_2_C_DEV", host_posix::_SC_2_C_DEV),
        ("SC_2_FORT_DEV", host_posix::_SC_2_FORT_DEV),
        ("SC_2_FORT_RUN", host_posix::_SC_2_FORT_RUN),
        ("SC_2_LOCALEDEF", host_posix::_SC_2_LOCALEDEF),
        ("SC_2_SW_DEV", host_posix::_SC_2_SW_DEV),
        ("SC_2_UPE", host_posix::_SC_2_UPE),
        ("SC_2_VERSION", host_posix::_SC_2_VERSION),
        ("SC_AIO_LISTIO_MAX", host_posix::_SC_AIO_LISTIO_MAX),
        ("SC_AIO_MAX", host_posix::_SC_AIO_MAX),
        ("SC_AIO_PRIO_DELTA_MAX", host_posix::_SC_AIO_PRIO_DELTA_MAX),
        ("SC_ARG_MAX", host_posix::_SC_ARG_MAX),
        ("SC_ASYNCHRONOUS_IO", host_posix::_SC_ASYNCHRONOUS_IO),
        ("SC_ATEXIT_MAX", host_posix::_SC_ATEXIT_MAX),
        ("SC_BC_BASE_MAX", host_posix::_SC_BC_BASE_MAX),
        ("SC_BC_DIM_MAX", host_posix::_SC_BC_DIM_MAX),
        ("SC_BC_SCALE_MAX", host_posix::_SC_BC_SCALE_MAX),
        ("SC_BC_STRING_MAX", host_posix::_SC_BC_STRING_MAX),
        ("SC_CHILD_MAX", host_posix::_SC_CHILD_MAX),
        ("SC_CLK_TCK", host_posix::_SC_CLK_TCK),
        ("SC_COLL_WEIGHTS_MAX", host_posix::_SC_COLL_WEIGHTS_MAX),
        ("SC_DELAYTIMER_MAX", host_posix::_SC_DELAYTIMER_MAX),
        ("SC_EXPR_NEST_MAX", host_posix::_SC_EXPR_NEST_MAX),
        ("SC_FSYNC", host_posix::_SC_FSYNC),
        ("SC_GETGR_R_SIZE_MAX", host_posix::_SC_GETGR_R_SIZE_MAX),
        ("SC_GETPW_R_SIZE_MAX", host_posix::_SC_GETPW_R_SIZE_MAX),
        ("SC_IOV_MAX", host_posix::_SC_IOV_MAX),
        ("SC_JOB_CONTROL", host_posix::_SC_JOB_CONTROL),
        ("SC_LINE_MAX", host_posix::_SC_LINE_MAX),
        ("SC_LOGIN_NAME_MAX", host_posix::_SC_LOGIN_NAME_MAX),
        ("SC_MAPPED_FILES", host_posix::_SC_MAPPED_FILES),
        ("SC_MEMLOCK", host_posix::_SC_MEMLOCK),
        ("SC_MEMLOCK_RANGE", host_posix::_SC_MEMLOCK_RANGE),
        ("SC_MEMORY_PROTECTION", host_posix::_SC_MEMORY_PROTECTION),
        ("SC_MESSAGE_PASSING", host_posix::_SC_MESSAGE_PASSING),
        ("SC_MQ_OPEN_MAX", host_posix::_SC_MQ_OPEN_MAX),
        ("SC_MQ_PRIO_MAX", host_posix::_SC_MQ_PRIO_MAX),
        ("SC_NGROUPS_MAX", host_posix::_SC_NGROUPS_MAX),
        ("SC_NPROCESSORS_CONF", host_posix::_SC_NPROCESSORS_CONF),
        ("SC_NPROCESSORS_ONLN", host_posix::_SC_NPROCESSORS_ONLN),
        ("SC_OPEN_MAX", host_posix::_SC_OPEN_MAX),
        ("SC_PAGE_SIZE", host_posix::_SC_PAGE_SIZE),
        ("SC_PAGESIZE", host_posix::_SC_PAGE_SIZE),
        #[cfg(any(
            target_os = "linux",
            target_vendor = "apple",
            target_os = "netbsd",
            target_os = "fuchsia"
        ))]
        ("SC_PASS_MAX", host_posix::_SC_PASS_MAX),
        ("SC_PHYS_PAGES", host_posix::_SC_PHYS_PAGES),
        ("SC_PRIORITIZED_IO", host_posix::_SC_PRIORITIZED_IO),
        (
            "SC_PRIORITY_SCHEDULING",
            host_posix::_SC_PRIORITY_SCHEDULING,
        ),
        ("SC_REALTIME_SIGNALS", host_posix::_SC_REALTIME_SIGNALS),
        ("SC_RE_DUP_MAX", host_posix::_SC_RE_DUP_MAX),
        ("SC_RTSIG_MAX", host_posix::_SC_RTSIG_MAX),
        ("SC_SAVED_IDS", host_posix::_SC_SAVED_IDS),
        ("SC_SEMAPHORES", host_posix::_SC_SEMAPHORES),
        ("SC_SEM_NSEMS_MAX", host_posix::_SC_SEM_NSEMS_MAX),
        ("SC_SEM_VALUE_MAX", host_posix::_SC_SEM_VALUE_MAX),
        (
            "SC_SHARED_MEMORY_OBJECTS",
            host_posix::_SC_SHARED_MEMORY_OBJECTS,
        ),
        ("SC_SIGQUEUE_MAX", host_posix::_SC_SIGQUEUE_MAX),
        ("SC_STREAM_MAX", host_posix::_SC_STREAM_MAX),
        ("SC_SYNCHRONIZED_IO", host_posix::_SC_SYNCHRONIZED_IO),
        ("SC_THREADS", host_posix::_SC_THREADS),
        (
            "SC_THREAD_ATTR_STACKADDR",
            host_posix::_SC_THREAD_ATTR_STACKADDR,
        ),
        (
            "SC_THREAD_ATTR_STACKSIZE",
            host_posix::_SC_THREAD_ATTR_STACKSIZE,
        ),
        (
            "SC_THREAD_DESTRUCTOR_ITERATIONS",
            host_posix::_SC_THREAD_DESTRUCTOR_ITERATIONS,
        ),
        ("SC_THREAD_KEYS_MAX", host_posix::_SC_THREAD_KEYS_MAX),
        (
            "SC_THREAD_PRIORITY_SCHEDULING",
            host_posix::_SC_THREAD_PRIORITY_SCHEDULING,
        ),
        (
            "SC_THREAD_PRIO_INHERIT",
            host_posix::_SC_THREAD_PRIO_INHERIT,
        ),
        (
            "SC_THREAD_PRIO_PROTECT",
            host_posix::_SC_THREAD_PRIO_PROTECT,
        ),
        (
            "SC_THREAD_PROCESS_SHARED",
            host_posix::_SC_THREAD_PROCESS_SHARED,
        ),
        (
            "SC_THREAD_SAFE_FUNCTIONS",
            host_posix::_SC_THREAD_SAFE_FUNCTIONS,
        ),
        ("SC_THREAD_STACK_MIN", host_posix::_SC_THREAD_STACK_MIN),
        ("SC_THREAD_THREADS_MAX", host_posix::_SC_THREAD_THREADS_MAX),
        ("SC_TIMERS", host_posix::_SC_TIMERS),
        ("SC_TIMER_MAX", host_posix::_SC_TIMER_MAX),
        ("SC_TTY_NAME_MAX", host_posix::_SC_TTY_NAME_MAX),
        ("SC_TZNAME_MAX", host_posix::_SC_TZNAME_MAX),
        ("SC_VERSION", host_posix::_SC_VERSION),
        ("SC_XOPEN_CRYPT", host_posix::_SC_XOPEN_CRYPT),
        ("SC_XOPEN_ENH_I18N", host_posix::_SC_XOPEN_ENH_I18N),
        ("SC_XOPEN_LEGACY", host_posix::_SC_XOPEN_LEGACY),
        ("SC_XOPEN_REALTIME", host_posix::_SC_XOPEN_REALTIME),
        (
            "SC_XOPEN_REALTIME_THREADS",
            host_posix::_SC_XOPEN_REALTIME_THREADS,
        ),
        ("SC_XOPEN_SHM", host_posix::_SC_XOPEN_SHM),
        ("SC_XOPEN_UNIX", host_posix::_SC_XOPEN_UNIX),
        ("SC_XOPEN_VERSION", host_posix::_SC_XOPEN_VERSION),
        ("SC_XOPEN_XCU_VERSION", host_posix::_SC_XOPEN_XCU_VERSION),
        #[cfg(any(
            target_os = "linux",
            target_vendor = "apple",
            target_os = "netbsd",
            target_os = "fuchsia"
        ))]
        ("SC_XBS5_ILP32_OFF32", host_posix::_SC_XBS5_ILP32_OFF32),
        #[cfg(any(
            target_os = "linux",
            target_vendor = "apple",
            target_os = "netbsd",
            target_os = "fuchsia"
        ))]
        ("SC_XBS5_ILP32_OFFBIG", host_posix::_SC_XBS5_ILP32_OFFBIG),
        #[cfg(any(
            target_os = "linux",
            target_vendor = "apple",
            target_os = "netbsd",
            target_os = "fuchsia"
        ))]
        ("SC_XBS5_LP64_OFF64", host_posix::_SC_XBS5_LP64_OFF64),
        #[cfg(any(
            target_os = "linux",
            target_vendor = "apple",
            target_os = "netbsd",
            target_os = "fuchsia"
        ))]
        ("SC_XBS5_LPBIG_OFFBIG", host_posix::_SC_XBS5_LPBIG_OFFBIG),
    ]
}

pub(crate) fn walk_fork_callback_roots(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    let mut callbacks = APPLEVEL_FORK_CALLBACKS.lock();
    let ApplevelForkCallbacks {
        before_w,
        parent_w,
        child_w,
    } = &mut *callbacks;
    for callbacks in [before_w, parent_w, child_w] {
        for callback in callbacks {
            visitor(unsafe { &mut *(callback as *mut usize as *mut PyObjectRef) });
        }
    }
}

fn run_fork_callbacks(kind: &str) {
    let reverse = kind == "before";
    let initial_len = {
        let callbacks = APPLEVEL_FORK_CALLBACKS.lock();
        match kind {
            "before" => callbacks.before_w.len(),
            "parent" => callbacks.parent_w.len(),
            "child" => callbacks.child_w.len(),
            _ => unreachable!(),
        }
    };
    let indices: Box<dyn Iterator<Item = usize>> = if reverse {
        Box::new((0..initial_len).rev())
    } else {
        Box::new(0..initial_len)
    };
    for index in indices {
        let callback = {
            let callbacks = APPLEVEL_FORK_CALLBACKS.lock();
            match kind {
                "before" => callbacks.before_w.get(index),
                "parent" => callbacks.parent_w.get(index),
                "child" => callbacks.child_w.get(index),
                _ => unreachable!(),
            }
            .copied()
        };
        let Some(callback) = callback else { continue };
        if let Err(mut error) = crate::call::call_function_impl_result(callback as PyObjectRef, &[])
        {
            let _roots = pyre_object::gc_roots::push_roots();
            let mut error = error;
            let slot = error.pin(&_roots);
            let repr = unsafe { crate::display::py_repr_wtf8(callback as PyObjectRef) };
            error.reload(&_roots, slot);
            let repr = repr.unwrap_or_else(|_| {
                rustpython_wtf8::Wtf8Buf::from_string("<callback>".to_string())
            });
            error.write_unraisable(
                pyre_object::w_none(),
                &crate::display::wtf8_format!("Exception ignored in atfork callback ", repr),
                pyre_object::w_none(),
            );
        }
    }
}

fn register_at_fork(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if !pos.is_empty() {
        return Err(crate::PyError::type_error(
            "register_at_fork() takes no positional arguments",
        ));
    }
    crate::builtins::kwarg_reject_unknown(
        kwargs,
        &["before", "after_in_parent", "after_in_child"],
        "register_at_fork",
    )?;
    let before = crate::builtins::kwarg_get(kwargs, "before");
    let parent = crate::builtins::kwarg_get(kwargs, "after_in_parent");
    let child = crate::builtins::kwarg_get(kwargs, "after_in_child");
    if before.is_none() && parent.is_none() && child.is_none() {
        return Err(crate::PyError::type_error(
            "At least one argument is required.",
        ));
    }
    for (name, callback) in [
        ("before", before),
        ("after_in_parent", parent),
        ("after_in_child", child),
    ] {
        if callback.is_some_and(|callback| !crate::baseobjspace::callable_w(callback)) {
            return Err(crate::PyError::type_error(format!(
                "'{name}' must be callable",
            )));
        }
    }
    {
        let mut callbacks = APPLEVEL_FORK_CALLBACKS.lock();
        if let Some(callback) = before {
            callbacks.before_w.push(callback as usize);
        }
        if let Some(callback) = parent {
            callbacks.parent_w.push(callback as usize);
        }
        if let Some(callback) = child {
            callbacks.child_w.push(callback as usize);
        }
    }
    pyre_object::gc_roots::mark_prebuilt_roots_dirty();
    Ok(pyre_object::w_none())
}

/// `posix.waitid_result` structseq — the five `siginfo_t` fields `waitid`
/// fills (`posixmodule.c waitid_result_fields`). The call is one
/// `interp_posix.py:1722` names and does not carry, so the shape here is the
/// one CPython 3.14 publishes.
#[cfg(all(unix, not(feature = "sandbox")))]
fn waitid_result_seq_type() -> PyObjectRef {
    static T: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    T.get_or_init(|| {
        crate::_structseq::make_struct_seq(
            "posix.waitid_result",
            &["si_pid", "si_uid", "si_signo", "si_status", "si_code"],
        )
    })
}

/// `posix.sched_param` structseq — the single field `app_posix.py`
/// declares.  `_structseq.py:102-107` already wraps the scalar a 1-field
/// structseq is handed, so `__new__` only has to name the argument;
/// `__reduce__` has to be replaced outright, because the generic one hands
/// back `(tuple(self), self.__dict__)` and this `__new__` takes one argument.
#[cfg(all(
    unix,
    any(
        target_os = "android",
        target_os = "freebsd",
        target_os = "linux",
        target_os = "netbsd"
    )
))]
fn sched_param_seq_type() -> PyObjectRef {
    static T: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    T.get_or_init(|| {
        let _roots = pyre_object::gc_roots::push_roots();
        let ty = crate::_structseq::make_struct_seq("posix.sched_param", &["sched_priority"]);
        let _ = pyre_object::gc_roots::pin_root(ty);
        let ty_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

        let new_descr = crate::typedef::make_new_descr_with_signature(
            crate::_structseq::structseq_descr_new,
            crate::gateway::Signature::new(vec!["cls", "sched_priority"], None, None, 0, 1),
        );
        let _ = pyre_object::gc_roots::pin_root(new_descr);
        let new_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let reduce = crate::make_builtin_function_with_arity("__reduce__", sched_param_reduce, 1);
        let _ = pyre_object::gc_roots::pin_root(reduce);
        let reduce_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

        unsafe {
            // A store can resize the namespace and collect, which moves the
            // type and the function objects, so every one of them is read back
            // out of its slot and `ns` is re-derived per store.
            let ns = || {
                pyre_object::w_type_get_dict_ptr(pyre_object::gc_roots::shadow_stack_get(ty_slot))
                    as PyObjectRef
            };
            pyre_object::w_dict_setitem_str_no_proxy(
                ns(),
                "__new__",
                pyre_object::gc_roots::shadow_stack_get(new_slot),
            );
            pyre_object::w_dict_setitem_str_no_proxy(
                ns(),
                "__reduce__",
                pyre_object::gc_roots::shadow_stack_get(reduce_slot),
            );
            crate::baseobjspace::mutated_absent(pyre_object::gc_roots::shadow_stack_get(ty_slot));
            pyre_object::gc_roots::shadow_stack_get(ty_slot)
        }
    })
}

/// `os_sched_param_reduce` — `(type(self), (self[0],))`, the one shape this
/// type's own `__new__` can be called back with.
#[cfg(all(
    unix,
    any(
        target_os = "android",
        target_os = "freebsd",
        target_os = "linux",
        target_os = "netbsd"
    )
))]
fn sched_param_reduce(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(&inst) = args.first().filter(|inst| !inst.is_null()) else {
        return Err(crate::PyError::type_error(
            "sched_param.__reduce__ missing self",
        ));
    };
    let cls = unsafe { (*inst).w_class };
    let priority =
        unsafe { pyre_object::w_tuple_getitem(inst, 0) }.unwrap_or_else(pyre_object::w_none);
    // Both tuple allocations can collect, so the class and the element are
    // published first and each one is read back out of its slot at the point
    // it is stored.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[cls, priority]);
    let inner = pyre_object::w_tuple_new(vec![pyre_object::gc_roots::shadow_stack_get(base + 1)]);
    let _ = pyre_object::gc_roots::pin_root(inner);
    Ok(pyre_object::w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(base),
        pyre_object::gc_roots::shadow_stack_get(base + 2),
    ]))
}

/// The `w_param` argument `sched_setparam` and `sched_setscheduler` share.
/// `interp_posix.py` refuses anything that is not a `sched_param`,
/// reads field 0 through the sequence protocol, and refuses a priority the C
/// `int` cannot hold.
#[cfg(all(
    unix,
    not(target_env = "musl"),
    any(
        target_os = "android",
        target_os = "freebsd",
        target_os = "linux",
        target_os = "netbsd"
    )
))]
fn sched_priority_w(w_param: PyObjectRef) -> Result<i32, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let param_slot = pyre_object::gc_roots::pin_roots(&[w_param]);
    if !crate::baseobjspace::isinstance(
        pyre_object::gc_roots::shadow_stack_get(param_slot),
        sched_param_seq_type(),
    )? {
        return Err(crate::PyError::type_error("must have a sched_param object"));
    }
    let idx_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(0));
    let w_priority = crate::baseobjspace::getitem(
        pyre_object::gc_roots::shadow_stack_get(param_slot),
        pyre_object::gc_roots::shadow_stack_get(idx_slot),
    )?;
    let priority = crate::baseobjspace::int_w(w_priority)?;
    i32::try_from(priority)
        .map_err(|_| crate::PyError::overflow_error("sched_priority out of range"))
}

/// Split `path` into the root and everything after it, the way
/// `ntpath.splitroot` splits it three ways with the drive and the root joined
/// back together.
///
/// `_bootstrap_external._path_join` maps this over its parts and classifies
/// each result: a root that starts or ends with a separator is absolute, one
/// ending in `:` is drive-relative, anything else is a plain relative part.
///
/// Both separators count, so the tests are taken on a copy with `/` rewritten
/// to `\`; that rewrite is code-point-for-code-point, so an offset into it
/// addresses the same code point of the original. The offsets are per code
/// point rather than per byte, which is what makes `"ä:\x"` take the drive
/// branch — at byte 1 it would be a continuation byte and take none.
///
/// Compiled everywhere so the tests run on every platform; only the Windows
/// build has a caller.
#[cfg_attr(not(windows), allow(dead_code))]
fn split_root(path: &rustpython_wtf8::Wtf8) -> (&rustpython_wtf8::Wtf8, &rustpython_wtf8::Wtf8) {
    use rustpython_wtf8::Wtf8;

    const SEP: u32 = '\\' as u32;
    const COLON: u32 = ':' as u32;
    const UNC_PREFIX: &str = "\\\\?\\UNC\\";

    // A path is a sequence of code points, and one of them may be a lone
    // surrogate that `str` cannot hold; the separators the split looks for are
    // all ASCII, so the scan runs over the code-point values.
    let norm: Vec<u32> = path
        .code_points()
        .map(|c| {
            if c.to_u32() == '/' as u32 {
                SEP
            } else {
                c.to_u32()
            }
        })
        .collect();
    let byte_at = |index: usize| {
        path.code_point_indices()
            .nth(index)
            .map_or(path.len(), |(offset, _)| offset)
    };
    let split_at = |offset: usize| {
        let bytes = path.as_bytes();
        // The offset comes from `code_point_indices`, so both halves are
        // code-point aligned and stay well-formed WTF-8.
        unsafe {
            (
                Wtf8::from_bytes_unchecked(&bytes[..offset]),
                Wtf8::from_bytes_unchecked(&bytes[offset..]),
            )
        }
    };
    let sep_from = |start: usize| {
        norm.get(start..)
            .and_then(|rest| rest.iter().position(|&c| c == SEP))
            .map(|offset| offset + start)
    };

    if norm.first() != Some(&SEP) {
        if norm.get(1) == Some(&COLON) {
            // `X:\Windows` keeps the separator in the root; `X:Windows` names
            // a location on the drive's own cursor and has no root at all.
            let split = if norm.get(2) == Some(&SEP) { 3 } else { 2 };
            return split_at(byte_at(split));
        }
        return (Wtf8::new(""), path);
    }
    if norm.get(1) != Some(&SEP) {
        // A path rooted on the current drive, e.g. `\Windows`.
        return split_at(byte_at(1));
    }
    // A UNC share (`\\server\share`, `\\?\UNC\server\share`) or a device
    // (`\\.\device`): the root runs to the separator after the share name,
    // and a path that never reaches a second separator is all root.
    let unc = norm.len() >= 8
        && norm[..8]
            .iter()
            .map(|&c| match u8::try_from(c) {
                Ok(b) => u32::from(b.to_ascii_uppercase()),
                Err(_) => c,
            })
            .eq(UNC_PREFIX.chars().map(u32::from));
    let start = if unc { 8 } else { 2 };
    match sep_from(start).and_then(|index| sep_from(index + 1)) {
        Some(index) => split_at(byte_at(index + 1)),
        None => (path, Wtf8::new("")),
    }
}

/// The separator a path is spelled with, and the second one Windows also
/// accepts. Which pair applies is the host's, and it travels as a value rather
/// than a `cfg` so both spellings stay under test wherever the tests run.
#[derive(Clone, Copy)]
struct PathSeps {
    /// `SEP`.
    sep: u16,
    /// `ALTSEP`, absent where the host defines none.
    altsep: Option<u16>,
    /// Whether the drive, UNC share and device prefixes exist at all, which is
    /// the `#ifdef MS_WINDOWS` in `_Py_skiproot` and `_Py_normpath_and_size`.
    drives: bool,
}

const NT_SEPS: PathSeps = PathSeps {
    sep: b'\\' as u16,
    altsep: Some(b'/' as u16),
    drives: true,
};

const POSIX_SEPS: PathSeps = PathSeps {
    sep: b'/' as u16,
    altsep: None,
    drives: false,
};

/// The host's own pair — `ntpath` on Windows, `posixpath` everywhere else.
const HOST_SEPS: PathSeps = if cfg!(windows) { NT_SEPS } else { POSIX_SEPS };

impl PathSeps {
    fn is_sep(self, path: &[u16], index: usize) -> bool {
        match path.get(index) {
            Some(&c) => c == self.sep || self.altsep == Some(c),
            None => false,
        }
    }
}

/// `_Py_skiproot` — how much of `path` is the drive prefix, and how much of
/// what follows is the root separator. Everything past the two is the tail.
///
/// The counts are UTF-16 units, the width the wide spelling of a path is
/// measured in, so the drive test reads the second unit of a surrogate pair
/// rather than the second character: `"\u{1f600}:\\x"` names no drive.
fn skiproot(path: &[u16], seps: PathSeps) -> (usize, usize) {
    let is_sep = |index: usize| seps.is_sep(path, index);
    let is_end = |index: usize| index >= path.len();
    let sep_or_end = |index: usize| is_sep(index) || is_end(index);
    let is_letter = |index: usize, upper: u8| {
        matches!(path.get(index), Some(&c)
            if c == u16::from(upper) || c == u16::from(upper.to_ascii_lowercase()))
    };

    if !seps.drives {
        if !is_sep(0) {
            // Relative path, e.g.: 'foo'
            return (0, 0);
        }
        if !is_sep(1) || is_sep(2) {
            // Absolute path, e.g.: '/foo', '///foo', '////foo', etc.
            return (0, 1);
        }
        // Precisely two leading slashes, e.g.: '//foo'. Implementation defined
        // per POSIX.
        return (0, 2);
    }
    if is_sep(0) {
        if !is_sep(1) {
            // Relative path with root, e.g. \Windows
            return (0, 1);
        }
        // Device drives, e.g. \\.\device or \\?\device
        // UNC drives, e.g. \\server\share or \\?\UNC\server\share
        let unc = path.get(2) == Some(&u16::from(b'?'))
            && is_sep(3)
            && is_letter(4, b'U')
            && is_letter(5, b'N')
            && is_letter(6, b'C')
            && is_sep(7);
        let mut idx = if unc { 8 } else { 2 };
        while !sep_or_end(idx) {
            idx += 1;
        }
        if is_end(idx) {
            return (idx, 0);
        }
        idx += 1;
        while !sep_or_end(idx) {
            idx += 1;
        }
        return (idx, usize::from(!is_end(idx)));
    }
    if !is_end(0) && path.get(1) == Some(&u16::from(b':')) {
        // Absolute drive-letter path, e.g. X:\Windows, or one relative to the
        // drive's own cursor, e.g. X:Windows.
        return (2, usize::from(is_sep(2)));
    }
    // Relative path, e.g. Windows
    (0, 0)
}

/// `_Py_normpath_and_size` — fold `.`, `..` and repeated separators out of
/// `buf` in place, and answer how many units the result occupies.
///
/// `buf` carries the trailing null the wide spelling of a path does, and is
/// one unit longer than the path itself: the scan reads one unit past the
/// segment it is measuring, and the last write is the terminator behind the
/// folded name. The fold never lengthens a path, so it needs no other room.
fn normpath_and_size(buf: &mut [u16], seps: PathSeps) -> usize {
    const DOT: u16 = b'.' as u16;
    let sep = seps.sep;
    let size = buf.len() - 1;
    if size == 0 {
        return 0;
    }
    let is_sep = |buf: &[u16], index: usize| seps.is_sep(buf, index);
    let is_end = |index: usize| index >= size;
    let sep_or_end = |buf: &[u16], index: usize| is_sep(buf, index) || is_end(index);

    let mut p1 = 0; // sequentially scanned index in the path
    let mut p2 = 0; // destination of a scanned unit to be ljusted
    let mut min_p2 = 0; // the beginning of the destination range
    let mut last_c = 0; // the last ljusted unit, buf[p2 - 1] in most cases

    let (drvsize, rootsize) = skiproot(&buf[..size], seps);
    if drvsize != 0 || rootsize != 0 {
        // Skip past root and update min_p2
        p1 = drvsize + rootsize;
        match seps.altsep {
            Some(altsep) => {
                while p2 < p1 {
                    if buf[p2] == altsep {
                        buf[p2] = sep;
                    }
                    p2 += 1;
                }
            }
            None => p2 = p1,
        }
        min_p2 = p2 - 1;
        last_c = buf[min_p2];
        if seps.drives && last_c != sep {
            min_p2 += 1;
        }
    }
    if buf[p1] == DOT && sep_or_end(buf, p1 + 1) {
        // Skip leading '.\'
        p1 += 1;
        last_c = buf[p1];
        if seps.altsep == Some(last_c) {
            last_c = sep;
        }
        while is_sep(buf, p1) {
            p1 += 1;
        }
    }

    while !is_end(p1) {
        let mut c = buf[p1];
        if seps.altsep == Some(c) {
            c = sep;
        }
        if last_c == sep {
            if c == DOT {
                let sep_at_1 = sep_or_end(buf, p1 + 1);
                let sep_at_2 = !sep_at_1 && sep_or_end(buf, p1 + 2);
                if sep_at_2 && buf[p1 + 1] == DOT {
                    let mut p3 = p2;
                    while p3 != min_p2 {
                        p3 -= 1;
                        if buf[p3] != sep {
                            break;
                        }
                    }
                    while p3 != min_p2 && buf[p3 - 1] != sep {
                        p3 -= 1;
                    }
                    if p2 == min_p2 || (buf[p3] == DOT && buf[p3 + 1] == DOT && is_sep(buf, p3 + 2))
                    {
                        // Previous segment is also ../, so append instead.
                        // Relative path does not absorb ../ at min_p2 as well.
                        buf[p2] = DOT;
                        buf[p2 + 1] = DOT;
                        p2 += 2;
                        last_c = DOT;
                    } else if buf[p3] == sep {
                        // Absolute path, so absorb segment
                        p2 = p3 + 1;
                    } else {
                        p2 = p3;
                    }
                    p1 += 1;
                } else if !sep_at_1 {
                    buf[p2] = c;
                    p2 += 1;
                    last_c = c;
                }
            } else if c != sep {
                buf[p2] = c;
                p2 += 1;
                last_c = c;
            }
        } else {
            buf[p2] = c;
            p2 += 1;
            last_c = c;
        }
        p1 += 1;
    }

    buf[p2] = 0;
    if p2 == min_p2 {
        // The destination never advanced, so the fold left nothing behind the
        // root: one unit before the range is where it ends.
        return p2;
    }
    loop {
        p2 -= 1;
        if p2 == min_p2 || buf[p2] != sep {
            return p2 + 1;
        }
        buf[p2] = 0;
    }
}

/// The wide spelling `_Py_skiproot` and `_Py_normpath_and_size` read, with the
/// trailing null they scan up to.
fn wide_with_nul(text: &rustpython_wtf8::Wtf8) -> Vec<u16> {
    let mut wide: Vec<u16> = text.encode_wide().collect();
    wide.push(0);
    wide
}

/// A `path_t(make_wide=True, nonstrict=True)` argument, read as the wide units
/// a path is measured in. The flag reports whether it arrived as `bytes`, so
/// the answer can be spelled back the way the caller spelled the question.
///
/// `nonstrict` is what lifts the embedded-null rejection: the two callers fold
/// and split the name as text and hand it to nobody, so a null is a character
/// like any other there.
fn nonstrict_wide_path(obj: PyObjectRef, func: &str) -> Result<(Vec<u16>, bool), crate::PyError> {
    // `path_converter` reads a `str` argument's own wide units
    // (`PyUnicode_AsWideCharString`) and reaches the filesystem codec only for
    // `bytes`. Taking a `str` through the codec anyway costs a
    // str -> bytes -> str round trip on every call -- and where the legacy
    // filesystem encoding is in force it is not even a round trip, so a name
    // no code page can spell would come back changed.
    if unsafe { pyre_object::is_str(obj) } {
        let units = wide_with_nul(unsafe { pyre_object::w_str_get_wtf8(obj) });
        return Ok((units, false));
    }
    let resolved = crate::gateway::fsencode_path_nonstrict_w(obj, func, "path")?;
    let as_bytes = unsafe { resolved.is_bytes() };
    let text = crate::gateway::fsdecode_filename_wtf8(&resolved.as_bytes);
    Ok((wide_with_nul(&text), as_bytes))
}

/// One of the `_path_*` helpers as a function object, carrying both the
/// docstring `__doc__` reports and the clinic signature line ahead of it that
/// `__text_signature__` does.
fn path_helper_fn(
    name: &'static str,
    func: crate::gateway::BuiltinCodeFn,
    text_signature: &'static str,
    docstring: &'static str,
) -> PyObjectRef {
    let function = crate::make_builtin_function_with_doc(name, func, docstring);
    unsafe {
        crate::function::fset_func_text_signature(function, pyre_object::w_str_new(text_signature));
    }
    function
}

/// A name answered back in the units the question was asked in:
/// `PyUnicode_FromWideChar`, and `PyUnicode_EncodeFSDefault` over that where
/// the argument was `bytes`.
fn wide_result(units: &[u16], as_bytes: bool) -> PyObjectRef {
    let text = rustpython_wtf8::Wtf8Buf::from_wide(units);
    if as_bytes {
        pyre_object::w_bytes_from_bytes(&crate::gateway::fsencode_wtf8_total(&text))
    } else {
        pyre_object::w_str_from_wtf8_managed(text)
    }
}

/// Windows-only `nt` calls — PyPy: pypy/module/posix/interp_nt.py and the
/// `if _WIN32` blocks of interp_posix.py. Registered under the `nt` module
/// name on Windows (moduledef.py `applevel_name = os.name`); `ntpath` reaches
/// for these through `from nt import _getfullpathname` and friends.
#[cfg(all(windows, feature = "host_env"))]
mod win_nt {
    use pyre_object::PyObjectRef;
    use rustpython_host_env::nt as host_nt;
    use rustpython_host_env::winapi as host_winapi;

    /// Wrap a host-layer `io::Error` as an OSError carrying the offending path.
    /// These calls are Win32 APIs, so the code they report is a Win32 error and
    /// lands in `.winerror` (`os_error_win32_syscall2`).
    fn io_err(error: &std::io::Error, path: &str) -> crate::PyError {
        let filename = if path.is_empty() {
            pyre_object::PY_NULL
        } else {
            pyre_object::w_str_new_managed(path)
        };
        io_err_with_filename(error, filename)
    }

    fn io_err_with_filename(error: &std::io::Error, filename: PyObjectRef) -> crate::PyError {
        match error.raw_os_error() {
            Some(winerror) => {
                crate::PyError::os_error_win32_syscall2(winerror, filename, pyre_object::PY_NULL)
            }
            None => crate::PyError::os_error_syscall(
                crate::builtins::io_error_posix_errno(error, 0),
                filename,
            ),
        }
    }

    /// Read argument 0 as a filesystem path; the flag reports whether the
    /// input was bytes so the result can be encoded back to match.
    ///
    /// `path_converter` fills `function_name` and `argument_name` from the
    /// argument clinic, so a rejected argument and an embedded null both name
    /// the call that read them.
    fn arg_path(
        args: &[PyObjectRef],
        func: &str,
    ) -> Result<(widestring::WideCString, bool, crate::gateway::FsEncodedPath), crate::PyError>
    {
        let Some(&arg) = args.first() else {
            return Err(crate::PyError::type_error(format!(
                "{func}() missing required argument 'path'"
            )));
        };
        let resolved = crate::gateway::fsencode_path_named_w(arg, func, "path")?;
        let as_bytes = unsafe { resolved.is_bytes() };
        // Windows names files in UTF-16, so the path reaches the host API as
        // code units rather than bytes. Going through a Rust `String` on the
        // way would replace an undecodable byte with U+FFFD, and the call
        // would then address a different file than the caller named --
        // `interp_posix.py` keeps the syscall spelling intact for the
        // same reason.
        let wide: Vec<u16> = crate::gateway::fsdecode_filename_wtf8(&resolved.as_bytes)
            .encode_wide()
            .collect();
        let path = widestring::WideCString::from_vec(wide)
            .map_err(|_| crate::PyError::value_error("embedded null character"))?;
        Ok((path, as_bytes, resolved))
    }

    /// The wide name these helpers read, answered as the caller spelled the
    /// path it asked about: `PyUnicode_FromWideChar`, and
    /// `PyUnicode_EncodeFSDefault` over that when the argument was `bytes`.
    fn wrap_path(s: &std::ffi::OsStr, as_bytes: bool) -> PyObjectRef {
        // One decode feeds both arms: the bytes form is the filesystem
        // encoding of the same text, not the UTF-8 of a lossy rendering of it.
        let text = crate::gateway::fsdecode_os_str_wtf8(s);
        if as_bytes {
            pyre_object::w_bytes_from_bytes(&crate::gateway::fs_result_bytes(text.as_bytes()))
        } else {
            pyre_object::w_str_from_wtf8_managed(text)
        }
    }

    /// ntpath.abspath helper — resolves `.`/`..` and the drive without
    /// requiring the path to exist.
    pub fn _getfullpathname(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (path, as_bytes, resolved) = arg_path(args, "_getfullpathname")?;
        match host_nt::getfullpathname(&path) {
            Ok(result) => Ok(wrap_path(&result, as_bytes)),
            Err(error) => Err(io_err_with_filename(&error, resolved.w_path())),
        }
    }

    /// ntpath.realpath helper — the canonical `\\?\`-prefixed path, via a
    /// backup-semantics handle so directories open too.
    pub fn _getfinalpathname(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (path, as_bytes, resolved) = arg_path(args, "_getfinalpathname")?;
        // `os__getfinalpathname_impl` leaves the interpreter for the handle it
        // opens and for `GetFinalPathNameByHandleW`, either of which blocks
        // while a network path answers.
        let final_path = {
            let _blocked = crate::module::thread::before_external_block();
            host_nt::getfinalpathname(&path)
        };
        match final_path {
            Ok(result) => Ok(wrap_path(&result, as_bytes)),
            Err(error) => Err(io_err_with_filename(&error, resolved.w_path())),
        }
    }

    /// ntpath.realpath helper — the final component's on-disk name, read out of
    /// the `cFileName` FindFirstFileW fills in. This is the only way to learn
    /// the long name behind an 8.3 alias (`C:\PROGRA~1` → `Program Files`) when
    /// the file cannot be opened, so `_getfinalpathname_nonstrict` falls back
    /// to it on the winerrors that mean "found it, but no handle for you"
    /// (`ntpath.py`). Only the leaf is returned; the caller splits the
    /// parent off itself.
    pub fn _findfirstfile(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (path, _as_bytes, resolved) = arg_path(args, "_findfirstfile")?;
        match host_nt::find_first_file_name(&path) {
            // The one name here that is answered as `str` whatever the
            // argument was: `os__findfirstfile_impl` reports `cFileName`
            // through `PyUnicode_FromWideChar` and asks no codec for a
            // `bytes` spelling of it.
            Ok(name) => Ok(pyre_object::w_str_from_wtf8_managed(
                crate::gateway::fsdecode_os_str_wtf8(&name),
            )),
            Err(error) => Err(io_err_with_filename(&error, resolved.w_path())),
        }
    }

    /// `path_t(suppress_value_error=True)` — the name or the descriptor a file
    /// test reads, or nothing where the conversion reported a `ValueError` and
    /// the test's own answer is `False`. Every other failure is still one:
    /// `path_converter` clears the error only for that one class.
    fn path_or_fd_suppressing(
        obj: PyObjectRef,
        func: &str,
    ) -> Result<Option<crate::gateway::FsEncodedPath>, crate::PyError> {
        match crate::gateway::fsencode_path_or_fd_w(obj, func, true) {
            Ok(path) => Ok(Some(path)),
            Err(mut error) => match crate::builtins::lookup_exc_class("ValueError") {
                Some(value_error)
                    if crate::eval::check_exc_match_against(error.to_exc_object(), value_error) =>
                {
                    Ok(None)
                }
                _ => Err(error),
            },
        }
    }

    /// The wide name a converted path spells. A null would have ended the
    /// conversion already, so the only argument that gets here without one is
    /// a name the host can take.
    fn tested_wide(path: &crate::gateway::FsEncodedPath) -> Option<widestring::WideCString> {
        let wide: Vec<u16> = crate::gateway::fsdecode_filename_wtf8(&path.as_bytes)
            .encode_wide()
            .collect();
        widestring::WideCString::from_vec(wide).ok()
    }

    /// `_testFileType` — the descriptor's handle where the argument named one,
    /// and `_testFileTypeByName` over the name otherwise. A descriptor is
    /// tested `diskOnly`, so a pipe or a console is no kind of file.
    pub fn test_file_type(
        obj: PyObjectRef,
        func: &str,
        tested: host_nt::TestType,
    ) -> Result<PyObjectRef, crate::PyError> {
        let Some(path) = path_or_fd_suppressing(obj, func)? else {
            return Ok(pyre_object::w_bool_from(false));
        };
        // Both arms open handles and query the volume, which blocks while a
        // network name answers.  The guard covers the calls alone, the way
        // `releasegil=True` does: building the answer allocates, and an
        // allocation made outside the RUNNING census can be collected under
        // the thread that made it.
        let result = {
            let _blocked = crate::module::thread::before_external_block();
            if path.is_fd {
                host_nt::test_file_type_by_handle(host_nt::handle_from_fd(path.as_fd), tested, true)
            } else {
                tested_wide(&path)
                    .is_some_and(|wide| host_nt::test_file_type_by_name(&wide, tested))
            }
        };
        Ok(pyre_object::w_bool_from(result))
    }

    /// `_testFileExists` — a descriptor exists where its handle has a type,
    /// and a name where `_testFileExistsByName` reaches it. `follow_links`
    /// separates `_path_exists`, which a broken link is not, from
    /// `_path_lexists`, which it is.
    pub fn test_file_exists(
        obj: PyObjectRef,
        func: &str,
        follow_links: bool,
    ) -> Result<PyObjectRef, crate::PyError> {
        let Some(path) = path_or_fd_suppressing(obj, func)? else {
            return Ok(pyre_object::w_bool_from(false));
        };
        let result = {
            let _blocked = crate::module::thread::before_external_block();
            if path.is_fd {
                unsafe { rustpython_host_env::crt_fd::Borrowed::try_borrow_raw(path.as_fd) }
                    .is_ok_and(host_nt::fd_exists)
            } else {
                tested_wide(&path)
                    .is_some_and(|wide| host_nt::test_file_exists_by_name(&wide, follow_links))
            }
        };
        Ok(pyre_object::w_bool_from(result))
    }

    /// ntpath.isdevdrive helper — whether the volume the name sits on carries
    /// `PERSISTENT_VOLUME_STATE_DEV_VOLUME`. A volume that cannot be reached
    /// is an error rather than a `False`, which is why `ntpath.isdevdrive`
    /// wraps the call in its own `except OSError`.
    pub fn _path_isdevdrive(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (path, _as_bytes, _resolved) = arg_path(args, "_path_isdevdrive")?;
        let state = {
            let _blocked = crate::module::thread::before_external_block();
            host_nt::path_isdevdrive(&path)
        };
        // `PyErr_SetFromWindowsErr` — the volume, not the name, is what
        // failed, so no filename is attached.
        state
            .map(pyre_object::w_bool_from)
            .map_err(|error| io_err(&error, ""))
    }

    /// ntpath.ismount helper — the mount point the name sits under.
    pub fn _getvolumepathname(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (path, as_bytes, resolved) = arg_path(args, "_getvolumepathname")?;
        let volume = {
            let _blocked = crate::module::thread::before_external_block();
            host_nt::getvolumepathname(&path)
        };
        match volume {
            Ok(name) => Ok(wrap_path(&name, as_bytes)),
            Err(error) => Err(io_err_with_filename(&error, resolved.w_path())),
        }
    }

    /// os.stat helper for ntpath.samefile — (volume serial, file index high,
    /// file index low) uniquely identifies a file across handles.
    pub fn _getfileinformation(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let Some(&arg) = args.first() else {
            return Err(crate::PyError::type_error(
                "_getfileinformation() missing required argument 'fd'",
            ));
        };
        let fd = crate::baseobjspace::c_int_w(arg)?;
        let info = host_nt::get_file_information(host_nt::handle_from_fd(fd))
            .map_err(|error| io_err(&error, ""))?;
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(pyre_object::w_int_new(info.volume_serial_number as i64));
        fields.push(pyre_object::w_int_new(info.file_index_high as i64));
        fields.push(pyre_object::w_int_new(info.file_index_low as i64));
        Ok(pyre_object::w_tuple_new(fields.take()))
    }

    /// shutil.disk_usage helper — (total, free) bytes. host_env::getdiskusage
    /// retries against the parent directory when the path names a file
    /// (ERROR_DIRECTORY).
    pub fn _getdiskusage(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (path, _, resolved) = arg_path(args, "_getdiskusage")?;
        // `os__getdiskusage_impl` leaves the interpreter for
        // `GetDiskFreeSpaceExW`, which reaches the volume itself.
        let usage = {
            let _blocked = crate::module::thread::before_external_block();
            host_nt::getdiskusage(&path)
        };
        match usage {
            Ok((total, free)) => {
                let mut fields = pyre_object::gc_roots::RootedItems::new();
                fields.push(pyre_object::w_int_new(total as i64));
                fields.push(pyre_object::w_int_new(free as i64));
                Ok(pyre_object::w_tuple_new(fields.take()))
            }
            Err(error) => Err(io_err_with_filename(&error, resolved.w_path())),
        }
    }

    /// os.get_handle_inheritable — the argument is an OS handle value, not a
    /// CRT fd (rwin32.cast(HANDLE, fd)).
    pub fn get_handle_inheritable(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let Some(&arg) = args.first() else {
            return Err(crate::PyError::type_error(
                "get_handle_inheritable() missing required argument 'handle'",
            ));
        };
        let handle = crate::baseobjspace::c_int_w(arg)? as libc::intptr_t;
        match host_nt::get_handle_inheritable(handle) {
            Ok(value) => Ok(pyre_object::w_bool_from(value)),
            Err(error) => Err(io_err(&error, "")),
        }
    }

    /// os.set_handle_inheritable.
    pub fn set_handle_inheritable(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        if args.len() < 2 {
            return Err(crate::PyError::type_error(
                "set_handle_inheritable() takes 2 arguments",
            ));
        }
        let handle = crate::baseobjspace::c_int_w(args[0])? as libc::intptr_t;
        let inheritable = crate::baseobjspace::is_true(args[1])?;
        match host_nt::set_handle_inheritable(handle, inheritable) {
            Ok(()) => Ok(pyre_object::w_none()),
            Err(error) => Err(io_err(&error, "")),
        }
    }

    /// The reserved instance-dictionary key the cookie object files its
    /// `DLL_DIRECTORY_COOKIE` under, namespaced the way the capsule carrier in
    /// `cpyext` namespaces its own payload.
    const DLL_COOKIE_KEY: &str = "__pyre_dll_directory_cookie__";

    /// The cookies `AddDllDirectory` has issued and `RemoveDllDirectory` has
    /// not taken back.
    ///
    /// The one-shot state that stops a second removal is the cleared payload
    /// below, the way `os__remove_dll_directory_impl` renames the capsule it
    /// was handed.  This list answers the other half: the payload rides an
    /// instance dictionary, which `object.__setattr__` can still write, and the
    /// loader fail-fasts rather than raising on a pointer it never handed out,
    /// so a removal checks the value here before the loader ever sees it.  A
    /// directory added twice has two entries and takes two removals.
    static LIVE_DLL_DIRECTORY_COOKIES: parking_lot::Mutex<Vec<usize>> =
        parking_lot::Mutex::new(Vec::new());

    /// The type of the opaque value `_add_dll_directory` hands back.
    ///
    /// `os__add_dll_directory_impl` returns `PyCapsule_New(cookie, "DLL
    /// directory cookie", NULL)`, and PyPy an `interp_posix.W_DLLCapsule`;
    /// either way the point is that a caller cannot spell a cookie, because
    /// `_remove_dll_directory` reinterprets what it is given as a pointer into
    /// the loader's DLL-directory list.  The type publishes nothing and is
    /// neither instantiable nor subclassable, so `_add_dll_directory` is the
    /// only source of one.
    fn dll_cookie_type() -> PyObjectRef {
        static CELL: pyre_object::gc_roots::RootedOnceRef =
            pyre_object::gc_roots::RootedOnceRef::new();
        CELL.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("nt.DLLDirectoryCookie", |_ns| {});
            unsafe {
                pyre_object::typeobject::w_type_set_hasdict(tp, true);
                pyre_object::typeobject::w_type_set_disallow_instantiation(tp);
                pyre_object::typeobject::w_type_set_acceptable_as_base_class(tp, false);
            }
            tp
        })
    }

    /// Box `cookie`.  The carrier is pinned before the value is built, and both
    /// are read back at the store: building the value allocates, and the store
    /// materialises the dictionary, so either could move the other.
    fn dll_cookie_new(cookie: usize) -> PyObjectRef {
        let _roots = pyre_object::gc_roots::push_roots();
        let carrier_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_instance_new(dll_cookie_type()));
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(cookie as i64));
        crate::baseobjspace::setdictvalue_native(
            pyre_object::gc_roots::shadow_stack_get(carrier_slot),
            DLL_COOKIE_KEY,
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        );
        pyre_object::gc_roots::shadow_stack_get(carrier_slot)
    }

    /// The cookie `w_cookie` carries, or `None` when it is not one of the
    /// objects `_add_dll_directory` returned — the `PyCapsule_IsValid(cookie,
    /// "DLL directory cookie")` test.
    fn dll_cookie_value(w_cookie: PyObjectRef) -> Option<usize> {
        // The first call materialises the type, which allocates, so the
        // argument is pinned across it and re-read: an unrooted local would be
        // stale if that collection moved the object.
        let _roots = pyre_object::gc_roots::push_roots();
        let cookie_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_cookie);
        let cookie_type = dll_cookie_type();
        let w_cookie = pyre_object::gc_roots::shadow_stack_get(cookie_slot);
        let is_cookie = match crate::typedef::r#type(w_cookie) {
            Some(w_type) => w_type.as_ptr() == cookie_type,
            None => false,
        };
        if !is_cookie {
            return None;
        }
        crate::baseobjspace::getdictvalue_native(w_cookie, DLL_COOKIE_KEY)
            .filter(|&value| unsafe { pyre_object::is_int(value) })
            .map(|value| unsafe { pyre_object::w_int_get_value(value) } as usize)
    }

    /// os._add_dll_directory. `os__add_dll_directory_impl` hands the
    /// `DLL_DIRECTORY_COOKIE` back inside a capsule and os.py only round-trips
    /// that object into `_remove_dll_directory`.
    pub fn _add_dll_directory(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (path, _, resolved) = arg_path(args, "_add_dll_directory")?;
        // `arg_path` keeps the code units it decoded the path into; going back
        // through a `str` would have no spelling for a lone surrogate and would
        // address a different directory.
        let cookie = host_winapi::add_dll_directory(&path)
            .map_err(|error| io_err_with_filename(&error, resolved.w_path()))?;
        LIVE_DLL_DIRECTORY_COOKIES.lock().push(cookie as usize);
        Ok(dll_cookie_new(cookie as usize))
    }

    /// os._remove_dll_directory — takes the object `_add_dll_directory`
    /// returned, and returns None.
    ///
    /// `os__remove_dll_directory_impl` opens by rejecting anything else with a
    /// TypeError, then renames the capsule so the same value cannot be removed
    /// twice.  Both guards carry their weight: `_AddedDllDirectory.__exit__`
    /// calls `close()` unconditionally, so a `with` block that closes early
    /// reaches the second removal by ordinary means.
    ///
    /// [3.14-spec] `interp_posix.py _remove_dll_directory` answers the removal
    /// with `space.newbool(...)`, reports a failure as `False` rather than
    /// raising, and has no invalidation at all.
    pub fn _remove_dll_directory(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let Some(&arg) = args.first() else {
            return Err(crate::PyError::type_error(
                "_remove_dll_directory() missing required argument 'cookie'",
            ));
        };
        let not_a_cookie = || {
            crate::PyError::type_error("Provided cookie was not returned from os.add_dll_directory")
        };
        // The argument is read back from the shadow stack at each use: reading
        // its cookie can materialise the type, and clearing it below stores
        // into its dictionary, either of which may move it.
        let _roots = pyre_object::gc_roots::push_roots();
        let cookie_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(arg);
        let w_cookie = || pyre_object::gc_roots::shadow_stack_get(cookie_slot);
        let cookie = dll_cookie_value(w_cookie()).ok_or_else(not_a_cookie)?;
        // Only a cookie the loader actually issued reaches `RemoveDllDirectory`:
        // the payload rides an instance dictionary that `object.__setattr__` can
        // still write, and the loader fail-fasts on a pointer it never handed
        // out.  The Win32 error is read while the lock is still held, since
        // releasing it may enter the OS and overwrite `GetLastError`.
        let failure = {
            let mut live = LIVE_DLL_DIRECTORY_COOKIES.lock();
            let index = live
                .iter()
                .position(|&issued| issued == cookie)
                .ok_or_else(not_a_cookie)?;
            live.swap_remove(index);
            if host_winapi::remove_dll_directory(cookie as host_winapi::DllDirectoryCookie).is_ok()
            {
                None
            } else {
                // The capsule is renamed only after a successful removal, so a
                // failed one leaves the cookie usable.
                let error = std::io::Error::last_os_error();
                live.push(cookie);
                Some(error)
            }
        };
        match failure {
            Some(error) => Err(io_err(&error, "")),
            None => {
                // `PyCapsule_SetName(cookie, NULL)`: the object stops carrying a
                // cookie at all, so a second removal is refused even where the
                // loader has since reissued that pointer for another directory.
                crate::baseobjspace::setdictvalue_native(
                    w_cookie(),
                    DLL_COOKIE_KEY,
                    pyre_object::w_none(),
                );
                Ok(pyre_object::w_none())
            }
        }
    }

    /// os._supports_virtual_terminal — whether stderr's console mode carries
    /// ENABLE_VIRTUAL_TERMINAL_PROCESSING.
    pub fn _supports_virtual_terminal(
        _args: &[PyObjectRef],
    ) -> Result<PyObjectRef, crate::PyError> {
        Ok(pyre_object::w_bool_from(
            host_nt::supports_virtual_terminal(),
        ))
    }
}

/// A fresh dict holding the current process environment.
///
/// PyPy equivalent: posix.State.startup → `_convertenviron`, which copies the
/// environment into `posix.environ` at interpreter startup. os.py seeds
/// `environ` from it at import time and re-reads it from `reload_environ()`.
fn create_environ() -> pyre_object::PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_dict_new());
    // Both halves of an entry stay rooted across the store: either allocation
    // may move what the other one already produced.
    #[cfg_attr(
        not(any(feature = "sandbox", feature = "host_env")),
        expect(dead_code, reason = "no environment source is compiled in")
    )]
    fn store(
        dict_slot: usize,
        key: impl FnOnce() -> pyre_object::PyObjectRef,
        value: impl FnOnce() -> pyre_object::PyObjectRef,
    ) {
        let _entry = pyre_object::gc_roots::push_roots();
        let key_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(key());
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(value());
        unsafe {
            pyre_object::w_dict_store(
                pyre_object::gc_roots::shadow_stack_get(dict_slot),
                pyre_object::gc_roots::shadow_stack_get(key_slot),
                pyre_object::gc_roots::shadow_stack_get(value_slot),
            );
        }
    }
    #[cfg(feature = "sandbox")]
    {
        // The controller delivers the virtual environment as (bytes, bytes).
        if let Ok(items) = crate::host_seam::ops::envitems() {
            for (k_bytes, v_bytes) in items {
                store(
                    dict_slot,
                    || pyre_object::w_bytes_from_bytes(&k_bytes),
                    || pyre_object::w_bytes_from_bytes(&v_bytes),
                );
            }
        }
    }
    #[cfg(all(feature = "host_env", not(feature = "sandbox"), unix))]
    {
        // On POSIX, posix.environ stores bytes → bytes. os.py's
        // _create_environ_mapping wraps this dict in an _Environ object that
        // encodes/decodes via surrogateescape when accessed.
        // `_convertenviron` walks `os.environ.items()`, routed to
        // `rposix_environ.envitems_llimpl`.
        for (key, value) in majit_rlib::rposix_environ::envitems_llimpl() {
            store(
                dict_slot,
                || pyre_object::w_bytes_from_bytes(&key),
                || pyre_object::w_bytes_from_bytes(&value),
            );
        }
    }
    #[cfg(all(feature = "host_env", not(feature = "sandbox"), windows))]
    {
        // On Windows nt.environ stores str → str; os.py's nt branch of
        // _create_environ_mapping demands str keys/values and upper-cases the
        // keys itself. (_convertenviron: `space.newtext(key), newtext(value)`.)
        for (key, value) in host_os::vars_os() {
            // `_wenviron` carries no name beginning with `=`: the C runtime
            // keeps its own per-drive current-directory entries (`=Z:`) out of
            // it, while `vars_os` reads the block those entries live in and
            // splits at the second `=`.  Publishing one would put a name in
            // `os.environ` that `putenv` and `unsetenv` both refuse, so
            // `os.environ.clear()` could not run.
            if key.as_encoded_bytes().first() == Some(&b'=') {
                continue;
            }
            // `_convertenviron`'s Windows arm reads `rwin32._wenviron_items()`,
            // the wide-char environment, and keeps those code units.
            // `fsdecode_os_str` carries them across with `from_wide`; a lossy
            // decode would fold an unpaired one to U+FFFD and stop
            // `os.environ` round-tripping.
            store(
                dict_slot,
                || crate::gateway::fsdecode_os_str(&key),
                || crate::gateway::fsdecode_os_str(&value),
            );
        }
    }
    pyre_object::gc_roots::shadow_stack_get(dict_slot)
}

/// Convert `path` then `attribute` in `interp_posix.py` unwrap order.
///
/// Each [`crate::gateway::FsEncodedPath`] owns a `push_roots` bracket
/// (`gctransform/shadowstack.py` `push_roots`/`pop_roots`). Locals that must
/// survive those conversions are pinned in the returned outer bracket; the
/// caller binds that guard first so the encoded paths drop before it.
#[cfg(any(
    test,
    all(
        not(feature = "sandbox"),
        any(target_os = "linux", target_os = "android")
    )
))]
fn fsencode_path_then_attribute(
    w_path: &mut PyObjectRef,
    w_attribute: &mut PyObjectRef,
    funcname: &str,
) -> Result<
    (
        pyre_object::gc_roots::RootScope,
        crate::gateway::FsEncodedPath,
        crate::gateway::FsEncodedPath,
    ),
    crate::PyError,
> {
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[*w_path, *w_attribute]);
    let path = crate::gateway::fsencode_path_or_fd_w(roots.get(base), funcname, true);
    *w_path = roots.get(base);
    *w_attribute = roots.get(base + 1);
    let path = path?;
    let attribute = crate::gateway::fsencode_path_named_w(*w_attribute, funcname, "attribute");
    *w_path = roots.get(base);
    *w_attribute = roots.get(base + 1);
    let attribute = attribute?;
    Ok((roots, path, attribute))
}

/// posix stub — PyPy: pypy/module/posix/ interp_posix.py
///
/// Provides the minimal surface that os.py module init needs to succeed.
/// Real posix calls are not implemented — they raise or return defaults.
pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), crate::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    crate::__pyre_put_new!(ns_slot, "environ", create_environ());
    crate::__pyre_put_new!(
        ns_slot,
        "_create_environ",
        crate::make_builtin_function_with_arity(
            "_create_environ",
            |_args| Ok(create_environ()),
            0,
        )
    );

    // ── posix.putenv(name, value) / posix.unsetenv(name) ──
    // `os.environ.__setitem__` calls `putenv` before updating its own dict, so
    // without these the mapping and the real process environment drift apart:
    // child processes and any native reader of the environment keep seeing the
    // values captured at startup.
    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
    {
        /// The environment block is a list of NUL-terminated `NAME=VALUE`
        /// strings, so neither half may embed a NUL.
        ///
        /// Bytes, like the path boundary: an entry the process was started with
        /// can hold a byte with no UTF-8 spelling, and `posix.environ` already
        /// hands those back as bytes, so writing one must not fold it first.
        fn env_bytes(arg: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
            let bytes = crate::gateway::fsencode_bytes_w(arg)?;
            if bytes.contains(&0) {
                return Err(crate::PyError::value_error("embedded null byte"));
            }
            Ok(bytes)
        }
        /// The name check that answers with `ValueError`.
        ///
        /// `win32_putenv` refuses an empty name and searches for `=` from
        /// index 1, because a leading `=` names one of the runtime's hidden
        /// per-drive entries and is left to `_wputenv` to turn away.
        /// Elsewhere the check is `os_putenv_impl`'s `strchr(name, '=')` over
        /// the whole name, and an empty name is left to `setenv`.
        fn illegal_name(name: &[u8]) -> bool {
            if cfg!(windows) {
                name.is_empty() || name[1.min(name.len())..].contains(&b'=')
            } else {
                name.contains(&b'=')
            }
        }
        /// The refusal the host call makes for a name [`illegal_name`] lets
        /// through -- a hidden `=NAME` entry on Windows, an empty name
        /// elsewhere.  `_wputenv` and `setenv` report `EINVAL` for it, and
        /// `set_var` / `remove_var` panic on such a key rather than returning,
        /// so it is spelled here instead of reaching them.
        fn refused_by_host(name: &[u8]) -> Option<crate::PyError> {
            (name.is_empty() || name.contains(&b'='))
                .then(|| crate::PyError::os_error_syscall(libc::EINVAL, pyre_object::PY_NULL))
        }
        /// `win32_putenv` measures the whole `NAME=VALUE` entry it is about to
        /// hand the block and turns away one longer than `_MAX_ENV`, which is
        /// as much as the block can hold. The count is in UTF-16 units, the
        /// width the entry is written at.
        #[cfg(windows)]
        fn entry_fits(name: &[u8], value: &[u8]) -> Result<(), crate::PyError> {
            const MAX_ENV: usize = 32767;
            // Through the same decoder the name reaches the API by, so a
            // lone surrogate is counted as the one unit it is written as
            // rather than as whatever a lossy decode substitutes for it.
            let units = |bytes: &[u8]| {
                crate::typedef::fsdecode_wtf8_total(bytes)
                    .encode_wide()
                    .count()
            };
            if units(name) + 1 + units(value) > MAX_ENV {
                return Err(crate::PyError::value_error(format!(
                    "the environment variable is longer than {MAX_ENV} characters"
                )));
            }
            Ok(())
        }
        crate::__pyre_put_new!(
            ns_slot,
            "putenv",
            crate::make_builtin_function_with_arity(
                "putenv",
                |args| {
                    // interp_posix.py putenv_impl rejects the name itself...
                    let w_name = args[0];
                    let mut w_value = args[1];
                    let name = pyre_object::with_roots!(w_value => env_bytes(w_name))?;
                    if illegal_name(&name) {
                        return Err(crate::PyError::value_error(
                            "illegal environment variable name",
                        ));
                    }
                    let value = env_bytes(w_value)?;
                    #[cfg(windows)]
                    entry_fits(&name, &value)?;
                    if let Some(err) = refused_by_host(&name) {
                        return Err(err);
                    }
                    #[cfg(unix)]
                    {
                        majit_rlib::rposix_environ::putenv_llimpl(&name, &value)
                            .map_err(|errno| errno_err(errno, ""))?;
                    }
                    #[cfg(not(unix))]
                    {
                        unsafe {
                            host_os::set_var(os_str_from_bytes(&name), os_str_from_bytes(&value))
                        };
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            )
        );
        crate::__pyre_put_new!(
            ns_slot,
            "unsetenv",
            crate::make_builtin_function_with_arity(
                "unsetenv",
                |args| {
                    let name = env_bytes(args[0])?;
                    // ...while on POSIX unsetenv leaves the same rejection to
                    // the syscall, which reports EINVAL.  On Windows
                    // `os_unsetenv_impl` reaches the very `win32_putenv`
                    // `os.putenv` does, so the name is judged there and the
                    // entry it would write is measured too.
                    if illegal_name(&name) {
                        return Err(if cfg!(windows) {
                            crate::PyError::value_error("illegal environment variable name")
                        } else {
                            crate::PyError::os_error_syscall(libc::EINVAL, pyre_object::PY_NULL)
                        });
                    }
                    #[cfg(windows)]
                    entry_fits(&name, b"")?;
                    if let Some(err) = refused_by_host(&name) {
                        return Err(err);
                    }
                    #[cfg(unix)]
                    {
                        // `interp_posix.unsetenv` swallows KeyError from
                        // `rposix.unsetenv` (absent key) and wraps OSError
                        // with `eintr_retry=False`. `unsetenv_llimpl` does
                        // not raise KeyError.
                        if let Err(errno) = majit_rlib::rposix_environ::unsetenv_llimpl(&name) {
                            return Err(errno_err(errno, ""));
                        }
                    }
                    #[cfg(not(unix))]
                    {
                        unsafe { host_os::remove_var(os_str_from_bytes(&name)) };
                    }
                    Ok(pyre_object::w_none())
                },
                1,
            )
        );
    }

    // _have_functions — list of HAVE_* macro names that were defined at
    // build time. os.py uses this to populate the supports_* capability sets
    // (supports_dir_fd / supports_effective_ids / supports_fd /
    // supports_follow_symlinks), which callers like shutil.rmtree consult to
    // choose between fd-relative and path-based implementations. Only the
    // macros whose functionality is actually implemented may be listed — the
    // entry beside each one below names the claim os.py reads out of it, so a
    // bit whose call is still missing has no entry at all rather than a
    // qualified one.
    //
    // Each bit is the same constant the entry point itself branches on, so the
    // advertisement cannot drift from the behaviour: a build where `chdir`
    // rejects a descriptor is a build that does not claim HAVE_FCHDIR. That is
    // what drops the whole fd-relative family under sandbox, where the host
    // probes/mutators are raising stubs, and on the hosts that carry no
    // `host_env::posix` at all.
    let have_functions: &[(&str, bool)] = &[
        // os.py:117,137,158 reads this as `access` honouring all three of its
        // modifiers, which is the one `faccessat` its body makes.
        ("HAVE_FACCESSAT", HAVE_FACCESSAT),
        ("HAVE_FCHDIR", HAVE_FCHDIR),
        ("HAVE_FCHMOD", HAVE_FCHMOD),
        ("HAVE_FCHOWN", HAVE_FCHOWN),
        // os.py:119,180 reads this as `chown` honouring both dir_fd and
        // follow_symlinks. HAVE_LCHOWN is not listed beside it: os.py:186
        // reads either one as the same follow_symlinks capability.
        // `follow_symlinks=False` with no directory descriptor calls
        // `rposix.c_lchown`. A directory descriptor still uses `fchownat`.
        // os.py:118 reads this as `chmod` honouring dir_fd.
        ("HAVE_FCHMODAT", HAVE_FCHMODAT),
        ("HAVE_FCHOWNAT", HAVE_FCHOWNAT),
        // Do not advertise HAVE_FEXECVE until execve() accepts an open file
        // descriptor.  os.py uses this bit to add execve to supports_fd, and
        // test_posix then runs a fork+fexecve path whose child must never
        // return to the libregrtest worker.
        //
        // os.py reads this as `listdir` and `scandir` taking a
        // descriptor, which `fdlistdir` serves through `fdopendir`.
        ("HAVE_FDOPENDIR", HAVE_FDOPENDIR),
        ("HAVE_FPATHCONF", HAVE_FPATHCONF),
        // os.py:120-121 reads this as `stat` and `lstat` honouring dir_fd.
        ("HAVE_FSTATAT", HAVE_FSTATAT),
        // os.py:122 reads this as `link` honouring `src_dir_fd`/`dst_dir_fd`,
        // which its `linkat` call is.
        ("HAVE_LINKAT", HAVE_LINKAT),
        ("HAVE_FSTATVFS", HAVE_FSTATVFS),
        ("HAVE_FTRUNCATE", HAVE_FTRUNCATE),
        // os.py reads this as `chflags` honouring follow_symlinks, which
        // is its `lchflags` arm.
        ("HAVE_LCHFLAGS", HAVE_LCHFLAGS),
        // os.py:183 reads this as `chmod` honouring follow_symlinks; os.py:179
        // shows why HAVE_FCHMODAT is not read for that claim.
        ("HAVE_LCHMOD", HAVE_LCHMOD),
        // HAVE_FUTIMES is not listed beside it: nothing here calls `futimes`,
        // and os.py:150-151 reads either one as the same `utime` capability.
        ("HAVE_FUTIMENS", HAVE_FUTIMENS),
        ("HAVE_LSTAT", HAVE_LSTAT),
        // os.py reads these as `mkdir`, `mkfifo`, `mknod` and `open`
        // honouring dir_fd.
        ("HAVE_MKDIRAT", HAVE_MKDIRAT),
        ("HAVE_MKFIFOAT", HAVE_MKFIFOAT),
        ("HAVE_MKNODAT", HAVE_MKNODAT),
        ("HAVE_OPENAT", HAVE_OPENAT),
        // os.py reads these as `readlink`, `rename`/`replace`, and `symlink`
        // honouring dir_fd, which are `readlinkat`, `renameat`, and
        // `symlinkat`.
        ("HAVE_READLINKAT", HAVE_READLINKAT),
        ("HAVE_RENAMEAT", HAVE_RENAMEAT),
        ("HAVE_SYMLINKAT", HAVE_SYMLINKAT),
        // os.py:131-132 reads this as `unlink` and `rmdir` honouring dir_fd,
        // which is the one `unlinkat` both of them make. os.remove is not in
        // that set — os.py never names it — even though the call takes the
        // modifier all the same.
        ("HAVE_UNLINKAT", HAVE_UNLINKAT),
        // os.py:133,191 reads this as `utime` honouring both dir_fd and
        // follow_symlinks, which is the one `utimensat` the name form makes.
        // HAVE_LUTIMES is not listed beside it for the same reason as
        // HAVE_FUTIMES above: os.py:188 reads it as the same follow_symlinks
        // capability and nothing here calls `lutimes`.
        ("HAVE_UTIMENSAT", HAVE_UTIMENSAT),
        // `interp_posix.py:2854-2855` appends this after the HAVE_* loop, so
        // it keeps that position here too.
        ("MS_WINDOWS", MS_WINDOWS),
    ];
    let w_have_functions = {
        // Stay on the function guard. `with_roots!(ns => ...)` assigns `ns`
        // and drops that guard's pin for every later store.
        let mut items = pyre_object::gc_roots::RootedItems::new();
        for &(n, have) in have_functions {
            if have {
                items.push(pyre_object::w_str_new(n));
            }
        }
        pyre_object::w_list_new(items.take())
    };
    crate::__pyre_put_new!(ns_slot, "_have_functions", w_have_functions);
    // POSIX constants — real libc values (cross-platform subset).
    for (name, val) in [
        #[cfg(feature = "host_env")]
        ("F_OK", rustpython_host_env::os::F_OK as i64),
        #[cfg(all(not(feature = "host_env"), unix))]
        ("F_OK", libc::F_OK as i64),
        #[cfg(all(not(feature = "host_env"), not(unix)))]
        ("F_OK", 0i64),
        #[cfg(feature = "host_env")]
        ("R_OK", rustpython_host_env::os::R_OK as i64),
        #[cfg(all(not(feature = "host_env"), unix))]
        ("R_OK", libc::R_OK as i64),
        #[cfg(all(not(feature = "host_env"), not(unix)))]
        ("R_OK", 4i64),
        #[cfg(feature = "host_env")]
        ("W_OK", rustpython_host_env::os::W_OK as i64),
        #[cfg(all(not(feature = "host_env"), unix))]
        ("W_OK", libc::W_OK as i64),
        #[cfg(all(not(feature = "host_env"), not(unix)))]
        ("W_OK", 2i64),
        #[cfg(feature = "host_env")]
        ("X_OK", rustpython_host_env::os::X_OK as i64),
        #[cfg(all(not(feature = "host_env"), unix))]
        ("X_OK", libc::X_OK as i64),
        #[cfg(all(not(feature = "host_env"), not(unix)))]
        ("X_OK", 1i64),
        ("O_RDONLY", libc::O_RDONLY as i64),
        ("O_WRONLY", libc::O_WRONLY as i64),
        ("O_RDWR", libc::O_RDWR as i64),
        ("O_APPEND", libc::O_APPEND as i64),
        ("O_CREAT", libc::O_CREAT as i64),
        ("O_EXCL", libc::O_EXCL as i64),
        ("O_TRUNC", libc::O_TRUNC as i64),
        // O_NONBLOCK, O_DSYNC, O_SYNC are Unix-only, and `nt` does not carry
        // them -- a zero there is a flag that silently does nothing. The
        // targets that are neither keep the zero they were given.
        #[cfg(unix)]
        ("O_NONBLOCK", libc::O_NONBLOCK as i64),
        #[cfg(not(any(unix, windows)))]
        ("O_NONBLOCK", 0i64),
        #[cfg(unix)]
        ("O_NDELAY", libc::O_NONBLOCK as i64),
        #[cfg(not(any(unix, windows)))]
        ("O_NDELAY", 0i64),
        // `moduledef.py:264-266` publishes O_CLOEXEC by name wherever the host
        // has it, which is every Unix. `nt` has no such flag -- it spells the
        // same intent O_NOINHERIT -- so a zero there would be a flag that
        // silently leaves the descriptor inheritable.
        #[cfg(unix)]
        ("O_CLOEXEC", libc::O_CLOEXEC as i64),
        // The rest of the `<fcntl.h>` set. Each value is the host header's own,
        // and the split below is the hosts' own too: these six are on every
        // Unix, the next two groups are one platform's each. `nt` has none of
        // them and is left with the flags it does have.
        #[cfg(unix)]
        ("O_ACCMODE", libc::O_ACCMODE as i64),
        #[cfg(unix)]
        ("O_ASYNC", libc::O_ASYNC as i64),
        #[cfg(unix)]
        ("O_DIRECTORY", libc::O_DIRECTORY as i64),
        #[cfg(unix)]
        ("O_FSYNC", libc::O_FSYNC as i64),
        #[cfg(unix)]
        ("O_NOCTTY", libc::O_NOCTTY as i64),
        #[cfg(unix)]
        ("O_NOFOLLOW", libc::O_NOFOLLOW as i64),
        // Linux's own. O_LARGEFILE is 0 on the targets that are already 64-bit,
        // which is the header answering that there is nothing to widen.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        ("O_DIRECT", libc::O_DIRECT as i64),
        #[cfg(any(target_os = "linux", target_os = "android"))]
        ("O_LARGEFILE", libc::O_LARGEFILE as i64),
        #[cfg(any(target_os = "linux", target_os = "android"))]
        ("O_NOATIME", libc::O_NOATIME as i64),
        #[cfg(any(target_os = "linux", target_os = "android"))]
        ("O_PATH", libc::O_PATH as i64),
        #[cfg(any(target_os = "linux", target_os = "android"))]
        ("O_RSYNC", libc::O_RSYNC as i64),
        #[cfg(any(target_os = "linux", target_os = "android"))]
        ("O_TMPFILE", libc::O_TMPFILE as i64),
        // The Apple targets' own.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ("O_EVTONLY", libc::O_EVTONLY as i64),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ("O_EXEC", libc::O_EXEC as i64),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ("O_EXLOCK", libc::O_EXLOCK as i64),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ("O_NOFOLLOW_ANY", libc::O_NOFOLLOW_ANY as i64),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ("O_SEARCH", libc::O_SEARCH as i64),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ("O_SHLOCK", libc::O_SHLOCK as i64),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ("O_SYMLINK", libc::O_SYMLINK as i64),
        #[cfg(unix)]
        ("O_DSYNC", libc::O_DSYNC as i64),
        #[cfg(not(any(unix, windows)))]
        ("O_DSYNC", 0i64),
        #[cfg(unix)]
        ("O_SYNC", libc::O_SYNC as i64),
        #[cfg(not(any(unix, windows)))]
        ("O_SYNC", 0i64),
        // SEEK_SET/SEEK_CUR/SEEK_END are os.py's own (`SEEK_SET = 0`,
        // os.py) on every platform, named in its own `__all__`, so a
        // binding here is counted a second time through the star-import.
        // Neither `posix` nor `nt` carries them; the other SEEK_* values are
        // the module's to publish.
    ] {
        crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
    }
    // Windows-only open() mode flags (fcntl.h). os.py exposes these off `nt`,
    // and stdlib callers reach for them behind `hasattr(os, 'O_BINARY')`
    // probes (tempfile) that only succeed once the names are bound.
    #[cfg(windows)]
    for (name, val) in [
        ("O_BINARY", libc::O_BINARY as i64),
        ("O_TEXT", libc::O_TEXT as i64),
        ("O_NOINHERIT", libc::O_NOINHERIT as i64),
        ("O_TEMPORARY", libc::O_TEMPORARY as i64),
        ("O_SHORT_LIVED", {
            #[cfg(feature = "host_env")]
            {
                rustpython_host_env::msvcrt::O_SHORT_LIVED as i64
            }
            #[cfg(not(feature = "host_env"))]
            {
                libc::_O_SHORT_LIVED as i64
            }
        }),
        ("O_RANDOM", libc::O_RANDOM as i64),
        ("O_SEQUENTIAL", libc::O_SEQUENTIAL as i64),
    ] {
        crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
    }
    // Placeholders the POSIX blocks further down overwrite with the real libc
    // values — the wait options beside the `W*` predicates, the `PRIO_*` trio
    // beside `getpriority`. A build that reaches neither keeps the zero.
    fn install_zero_constants(ns: PyObjectRef, names: &[&str]) {
        let _root_scope = pyre_object::gc_roots::push_roots();
        let ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ns = pyre_object::gc_roots::pin_root(ns);
        for &name in names {
            crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(0));
        }
    }

    // The wait flags and the priority classes are `nt`'s absentees for the same
    // reason its calls are: they exist only where the platform defines them,
    // and code reads their presence to decide whether the facility is there at
    // all.
    #[cfg(unix)]
    install_zero_constants(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        &[
            "WNOHANG",
            "WCONTINUED",
            "WUNTRACED",
            "PRIO_PROCESS",
            "PRIO_PGRP",
            "PRIO_USER",
        ],
    );

    // `nt`'s own constants. The spawn modes are the C runtime's `_P_*`
    // (process.h) and carry its values: registering the set at zero made
    // `P_NOWAIT` mean `P_WAIT`. On POSIX these are os.py's, not the module's —
    // it defines `P_WAIT = 0` and `P_NOWAIT = P_NOWAITO = 1` for itself in the
    // branch that has `fork`, which is why `P_NOWAITO` is 3 here and 1 there.
    //
    // `EX_OK` is the one member of the `<sysexits.h>` family Windows answers to
    // as well, and it carries the same 0 there.
    #[cfg(all(windows, feature = "host_env"))]
    {
        use rustpython_host_env::msvcrt as host_msvcrt;
        use rustpython_host_env::nt as host_nt;
        for (name, val) in [
            ("EX_OK", host_msvcrt::EX_OK as i64),
            ("P_WAIT", host_msvcrt::P_WAIT as i64),
            ("P_NOWAIT", host_msvcrt::P_NOWAIT as i64),
            ("P_OVERLAY", host_msvcrt::P_OVERLAY as i64),
            ("P_NOWAITO", host_msvcrt::P_NOWAITO as i64),
            ("P_DETACH", host_msvcrt::P_DETACH as i64),
            ("TMP_MAX", host_msvcrt::TMP_MAX as i64),
            (
                "_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR",
                host_nt::LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR as i64,
            ),
            (
                "_LOAD_LIBRARY_SEARCH_APPLICATION_DIR",
                host_nt::LOAD_LIBRARY_SEARCH_APPLICATION_DIR as i64,
            ),
            (
                "_LOAD_LIBRARY_SEARCH_USER_DIRS",
                host_nt::LOAD_LIBRARY_SEARCH_USER_DIRS as i64,
            ),
            (
                "_LOAD_LIBRARY_SEARCH_SYSTEM32",
                host_nt::LOAD_LIBRARY_SEARCH_SYSTEM32 as i64,
            ),
            (
                "_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS",
                host_nt::LOAD_LIBRARY_SEARCH_DEFAULT_DIRS as i64,
            ),
        ] {
            crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
        }
    }
    #[cfg(all(windows, not(feature = "host_env")))]
    for (name, val) in [
        ("EX_OK", 0i64),
        ("P_WAIT", 0i64),
        ("P_NOWAIT", 1),
        ("P_OVERLAY", 2),
        ("P_NOWAITO", 3),
        ("P_DETACH", 4),
        ("TMP_MAX", 2_147_483_647),
        ("_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR", 0x100),
        ("_LOAD_LIBRARY_SEARCH_APPLICATION_DIR", 0x200),
        ("_LOAD_LIBRARY_SEARCH_USER_DIRS", 0x400),
        ("_LOAD_LIBRARY_SEARCH_SYSTEM32", 0x800),
        ("_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS", 0x1000),
    ] {
        crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
    }
    #[cfg(unix)]
    {
        // `<sysexits.h>`. The header is a verbatim descendant of the 4.3BSD one
        // wherever it is carried, so the values are the same on every host that
        // has it and the `libc` crate binds none of them.
        #[cfg(feature = "host_env")]
        for (name, val) in [
            ("EX_OK", rustpython_host_env::posix::EX_OK as i64),
            ("EX_USAGE", rustpython_host_env::posix::EX_USAGE as i64),
            ("EX_DATAERR", rustpython_host_env::posix::EX_DATAERR as i64),
            ("EX_NOINPUT", rustpython_host_env::posix::EX_NOINPUT as i64),
            ("EX_NOUSER", rustpython_host_env::posix::EX_NOUSER as i64),
            ("EX_NOHOST", rustpython_host_env::posix::EX_NOHOST as i64),
            (
                "EX_UNAVAILABLE",
                rustpython_host_env::posix::EX_UNAVAILABLE as i64,
            ),
            (
                "EX_SOFTWARE",
                rustpython_host_env::posix::EX_SOFTWARE as i64,
            ),
            ("EX_OSERR", rustpython_host_env::posix::EX_OSERR as i64),
            ("EX_OSFILE", rustpython_host_env::posix::EX_OSFILE as i64),
            (
                "EX_CANTCREAT",
                rustpython_host_env::posix::EX_CANTCREAT as i64,
            ),
            ("EX_IOERR", rustpython_host_env::posix::EX_IOERR as i64),
            (
                "EX_TEMPFAIL",
                rustpython_host_env::posix::EX_TEMPFAIL as i64,
            ),
            (
                "EX_PROTOCOL",
                rustpython_host_env::posix::EX_PROTOCOL as i64,
            ),
            ("EX_NOPERM", rustpython_host_env::posix::EX_NOPERM as i64),
            ("EX_CONFIG", rustpython_host_env::posix::EX_CONFIG as i64),
        ] {
            crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
        }
        #[cfg(not(feature = "host_env"))]
        for (name, val) in [
            ("EX_OK", 0i64),
            ("EX_USAGE", 64),
            ("EX_DATAERR", 65),
            ("EX_NOINPUT", 66),
            ("EX_NOUSER", 67),
            ("EX_NOHOST", 68),
            ("EX_UNAVAILABLE", 69),
            ("EX_SOFTWARE", 70),
            ("EX_OSERR", 71),
            ("EX_OSFILE", 72),
            ("EX_CANTCREAT", 73),
            ("EX_IOERR", 74),
            ("EX_TEMPFAIL", 75),
            ("EX_PROTOCOL", 76),
            ("EX_NOPERM", 77),
            ("EX_CONFIG", 78),
        ] {
            crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
        }
        // The `f_flag` bits `statvfs` answers with, which is the only reader
        // there is for them.
        for (name, val) in [
            ("ST_RDONLY", libc::ST_RDONLY as i64),
            ("ST_NOSUID", libc::ST_NOSUID as i64),
        ] {
            crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
        }
        // `<dlfcn.h>` — `rdynload.py:50-82` reads the same set, and
        // `sys.setdlopenflags` and `ctypes` hand them straight back to
        // `dlopen`, where a zero would ask for `RTLD_LOCAL | RTLD_LAZY`
        // whatever was named.
        for (name, val) in [
            ("RTLD_LAZY", libc::RTLD_LAZY as i64),
            ("RTLD_NOW", libc::RTLD_NOW as i64),
            ("RTLD_GLOBAL", libc::RTLD_GLOBAL as i64),
            ("RTLD_LOCAL", libc::RTLD_LOCAL as i64),
            ("RTLD_NODELETE", libc::RTLD_NODELETE as i64),
            ("RTLD_NOLOAD", libc::RTLD_NOLOAD as i64),
            // A glibc extension, absent from the header anywhere else.
            #[cfg(all(target_os = "linux", target_env = "gnu"))]
            ("RTLD_DEEPBIND", libc::RTLD_DEEPBIND as i64),
        ] {
            crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
        }
        // `<sched.h>`, read as `rposix.py` reads it: present where the
        // header defines it, and with the host's own numbering rather than a
        // shared one — the three disagree between Linux, the BSDs and Darwin.
        // The Apple targets declare them in `<pthread/pthread_impl.h>`, which
        // the `libc` crate does not mirror.
        for (name, val) in [
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            ("SCHED_OTHER", libc::SCHED_OTHER as i64),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            ("SCHED_OTHER", 1i64),
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            ("SCHED_FIFO", libc::SCHED_FIFO as i64),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            ("SCHED_FIFO", 4i64),
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            ("SCHED_RR", libc::SCHED_RR as i64),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            ("SCHED_RR", 2i64),
            #[cfg(any(target_os = "linux", target_os = "android"))]
            ("SCHED_BATCH", libc::SCHED_BATCH as i64),
            #[cfg(any(target_os = "linux", target_os = "android"))]
            ("SCHED_IDLE", libc::SCHED_IDLE as i64),
            // `<linux/sched.h>` names these two; `SCHED_RESET_ON_FORK` is a
            // flag OR-ed into a policy rather than a policy of its own.
            #[cfg(any(target_os = "linux", target_os = "android"))]
            ("SCHED_NORMAL", libc::SCHED_NORMAL as i64),
            #[cfg(any(target_os = "linux", target_os = "android"))]
            ("SCHED_DEADLINE", libc::SCHED_DEADLINE as i64),
            #[cfg(any(target_os = "linux", target_os = "android"))]
            ("SCHED_RESET_ON_FORK", libc::SCHED_RESET_ON_FORK as i64),
        ] {
            crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
        }
        // `moduledef.py` publishes these when `rposix.posix_fadvise` exists.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        for (name, val) in [
            ("POSIX_FADV_WILLNEED", libc::POSIX_FADV_WILLNEED as i64),
            ("POSIX_FADV_NORMAL", libc::POSIX_FADV_NORMAL as i64),
            ("POSIX_FADV_SEQUENTIAL", libc::POSIX_FADV_SEQUENTIAL as i64),
            ("POSIX_FADV_RANDOM", libc::POSIX_FADV_RANDOM as i64),
            ("POSIX_FADV_NOREUSE", libc::POSIX_FADV_NOREUSE as i64),
            ("POSIX_FADV_DONTNEED", libc::POSIX_FADV_DONTNEED as i64),
        ] {
            crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
        }
        // `moduledef.py` publishes `MFD_*` when `rposix.memfd_create` exists.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        for (name, val) in [
            ("MFD_CLOEXEC", libc::MFD_CLOEXEC as i64),
            ("MFD_ALLOW_SEALING", libc::MFD_ALLOW_SEALING as i64),
            ("MFD_HUGETLB", libc::MFD_HUGETLB as i64),
            ("MFD_HUGE_SHIFT", libc::MFD_HUGE_SHIFT as i64),
            ("MFD_HUGE_MASK", libc::MFD_HUGE_MASK as i64),
            ("MFD_HUGE_64KB", libc::MFD_HUGE_64KB as i64),
            ("MFD_HUGE_512KB", libc::MFD_HUGE_512KB as i64),
            ("MFD_HUGE_1MB", libc::MFD_HUGE_1MB as i64),
            ("MFD_HUGE_2MB", libc::MFD_HUGE_2MB as i64),
            ("MFD_HUGE_8MB", libc::MFD_HUGE_8MB as i64),
            ("MFD_HUGE_16MB", libc::MFD_HUGE_16MB as i64),
            ("MFD_HUGE_32MB", libc::MFD_HUGE_32MB as i64),
            ("MFD_HUGE_256MB", libc::MFD_HUGE_256MB as i64),
            ("MFD_HUGE_512MB", libc::MFD_HUGE_512MB as i64),
            ("MFD_HUGE_1GB", libc::MFD_HUGE_1GB as i64),
            ("MFD_HUGE_2GB", libc::MFD_HUGE_2GB as i64),
            ("MFD_HUGE_16GB", libc::MFD_HUGE_16GB as i64),
        ] {
            crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
        }
        // `moduledef.py` publishes these when `hasattr(rposix, 'getxattr')`.
        // `rposix.XATTR_SIZE_MAX` is `linux/limits.h` (65536); libc has no
        // `XATTR_SIZE_MAX`.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        for (name, val) in [
            ("XATTR_SIZE_MAX", 65536i64),
            ("XATTR_CREATE", libc::XATTR_CREATE as i64),
            ("XATTR_REPLACE", libc::XATTR_REPLACE as i64),
        ] {
            crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
        }
        // The `cmd` `lockf` takes, which is the whole of its vocabulary and
        // which os.py neither writes nor names.
        #[cfg(all(
            feature = "host_env",
            any(
                target_os = "android",
                target_os = "dragonfly",
                target_os = "freebsd",
                target_os = "linux",
                target_os = "macos",
                target_os = "netbsd",
                target_os = "redox"
            )
        ))]
        for (name, val) in [
            ("F_ULOCK", rustpython_host_env::fcntl::F_ULOCK as i64),
            ("F_LOCK", rustpython_host_env::fcntl::F_LOCK as i64),
            ("F_TLOCK", rustpython_host_env::fcntl::F_TLOCK as i64),
            ("F_TEST", rustpython_host_env::fcntl::F_TEST as i64),
        ] {
            crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
        }
        #[cfg(not(all(
            feature = "host_env",
            any(
                target_os = "android",
                target_os = "dragonfly",
                target_os = "freebsd",
                target_os = "linux",
                target_os = "macos",
                target_os = "netbsd",
                target_os = "redox"
            )
        )))]
        for (name, val) in [
            ("F_ULOCK", libc::F_ULOCK as i64),
            ("F_LOCK", libc::F_LOCK as i64),
            ("F_TLOCK", libc::F_TLOCK as i64),
            ("F_TEST", libc::F_TEST as i64),
        ] {
            crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
        }
        // The two `whence` values beyond the three os.py fixes itself: they
        // seek to the next hole or the next data in a sparse file. A host that
        // cannot answer that question defines neither, and the OpenBSD/NetBSD
        // and AIX headers are among those — so the set is named rather than
        // excluded, and a host left out of it is one short of a name rather
        // than one carrying a wrong value.
        #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "linux",
            target_os = "android",
            target_os = "freebsd",
            target_os = "dragonfly",
            target_os = "solaris",
            target_os = "illumos",
            target_os = "hurd",
        ))]
        for (name, val) in [
            ("SEEK_HOLE", libc::SEEK_HOLE as i64),
            ("SEEK_DATA", libc::SEEK_DATA as i64),
        ] {
            crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
        }
        // Darwin-only names. `NGROUPS_MAX` (`<sys/syslimits.h>`) and `TMP_MAX`
        // (`<stdio.h>`) exist on linux too but with the glibc numbering, so
        // they are answered per platform rather than shared. The `PRIO_DARWIN_*`
        // family is what `setpriority` takes there instead of a nice value, and
        // the `_COPYFILE_*` bits are the `flags` word `shutil` hands
        // `copyfile()` through `posix._fcopyfile`.
        #[cfg(target_vendor = "apple")]
        for (name, val) in [
            ("NGROUPS_MAX", 16i64),
            ("TMP_MAX", libc::TMP_MAX as i64),
            ("PRIO_DARWIN_BG", libc::PRIO_DARWIN_BG as i64),
            ("PRIO_DARWIN_NONUI", libc::PRIO_DARWIN_NONUI as i64),
            ("PRIO_DARWIN_PROCESS", libc::PRIO_DARWIN_PROCESS as i64),
            ("PRIO_DARWIN_THREAD", libc::PRIO_DARWIN_THREAD as i64),
            ("_COPYFILE_ACL", libc::COPYFILE_ACL as i64),
            ("_COPYFILE_DATA", libc::COPYFILE_DATA as i64),
            ("_COPYFILE_STAT", libc::COPYFILE_STAT as i64),
            ("_COPYFILE_XATTR", libc::COPYFILE_XATTR as i64),
        ] {
            crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
        }
    }
    // Remaining noop stubs — functions os.py references at module level.
    // Functions with real implementations are registered individually below.
    fn install_noop_stubs(ns: PyObjectRef, names: &[&'static str]) {
        let _root_scope = pyre_object::gc_roots::push_roots();
        let ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ns = pyre_object::gc_roots::pin_root(ns);
        for &name in names {
            crate::__pyre_put_new!(
                ns_slot,
                name,
                crate::make_builtin_function(name, |_| Ok(pyre_object::w_none()))
            );
        }
    }

    // Names both `posix` and `nt` answer to.
    install_noop_stubs(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        &[
            "dup",
            "dup2",
            "chdir",
            "link",
            "symlink",
            "chmod",
            "fchmod",
            "access",
            "execve",
            "execv",
            "waitpid",
            "truncate",
            "ftruncate",
            "getppid",
            "umask",
            "getlogin",
            "pipe",
            "fsync",
            "get_inheritable",
            "set_inheritable",
            // "get_terminal_size" — implemented below
            "cpu_count",
            "kill",
            "device_encoding",
            "waitstatus_to_exitcode",
            "_exit",
            "abort",
            // "spawnv"/"spawnve" — os.py builds the spawn family out of
            // fork+exec+waitpid, but only `if not _exists("spawnv")`, so a name
            // bound here is not a placeholder waiting to be overwritten: it is
            // what stops the real implementation from ever being defined. That
            // reading is the POSIX one: `nt` carries `_spawnv` itself and the
            // os.py block is behind `_exists("fork")`, so on Windows the name
            // is the module's or it is nowhere, and it is registered below.
            "system",
        ],
    );

    // `spawnv` / `spawnve` are the entry points `nt` has of its own, and are
    // registered further down; os.py builds `spawnl` and `spawnle` out of them
    // rather than out of fork+exec, because its fork branch is behind
    // `_exists("fork")`.

    // The calls `nt` has not got. Each is probed for presence rather than
    // called blind — `os.py` gates `supports_fd` on `_exists`, `shutil` picks
    // its `disk_usage` implementation on `hasattr(os, 'statvfs')`, and
    // `multiprocessing` picks a start method on `hasattr(os, 'fork')` — so a
    // stub answering `None` here does not add a call, it wins the POSIX branch
    // on a host that cannot serve it. Registered where the platform is one that
    // can, which is the only place the name exists at all.
    #[cfg(unix)]
    install_noop_stubs(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        &[
            // "fstatat"/"faccessat"/"futimens"/"futimes"/"fdopendir" — the `*at`
            // and `f*` C entry points the module calls to serve `dir_fd` and a
            // descriptor path. They are how the calls above are made, not calls
            // of their own, and `moduledef.py` publishes none of them.
            "statvfs",
            "fstatvfs",
            "fchdir",
            "fchown",
            "fork",
            "forkpty",
            "wait",
            "pathconf",
            "fpathconf",
            "setsid",
            "setpgid",
            "getgroups",
            "setgroups",
            "getgrouplist",
            "setpgrp",
            "nice",
            // "pipe2" — the flag-taking form of `pipe`, published below on the
            // hosts whose libc declares it. "dup3" is not a name `moduledef.py`
            // defines, nor one `os` publishes on any host, so there is nothing
            // here for it to stand in for.
            "fdatasync",
            "mkfifo",
            "getloadavg",
            "killpg",
            "getpriority",
            "setpriority",
            "sched_get_priority_max",
            "sched_get_priority_min",
            // "sched_getparam"/"sched_setparam"/"sched_getscheduler"/
            // "sched_setscheduler" — the policy calls, published below together
            // with the `sched_param` type they hand back and forth.
            // `moduledef.py:168-174` gates the five as one group.
            "sched_yield",
            // "confstr"/"confstr_names" — the host's string-valued configuration
            // table, published below where the host defines one. A build with no
            // `<unistd.h>` behind it has no `confstr` at all, which is what the
            // name being absent says.
            "sysconf",
            "sysconf_names",
            // "setenv" — the entry point is spelled `putenv`, and there is no
            // second name for it.
            "ttyname",
            "openpty",
            "login_tty",
            "tcgetpgrp",
            "tcsetpgrp",
            // "get_exec_path" — `os.py` writes it in Python and lists it in
            // its own `__all__`, so a name bound here is not overwritten by that
            // definition; it is counted a second time, through the star-import.
            "WIFEXITED",
            "WEXITSTATUS",
            "WIFSIGNALED",
            "WTERMSIG",
            "WIFSTOPPED",
            "WSTOPSIG",
            // "WEXITED"/"WNOWAIT"/"WSTOPPED" — `waitid`'s option flags, which
            // are numbers rather than calls; bound with the other wait options
            // below.
            "_cpu_count",
            // "spawnvp"/"spawnvpe" — the same os.py branch defines these,
            // for the same reason the two above are not bound.
            // "popen" — `os.py` writes it over `subprocess` and
            // appends it to its own `__all__`, with no guard on this name being
            // free.
        ],
    );

    // putenv/unsetenv are implemented above unless the host environment is out
    // of reach.
    #[cfg(any(not(feature = "host_env"), feature = "sandbox"))]
    install_noop_stubs(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        &["unsetenv", "putenv"],
    );
    // There is no fork to register against on Windows, and `os.py` reaches for
    // the name to decide whether it has one.
    #[cfg(not(windows))]
    crate::__pyre_put_new!(
        ns_slot,
        "register_at_fork",
        crate::make_builtin_function("register_at_fork", register_at_fork)
    );

    // os.major(device) / os.minor(device) / os.makedev(major, minor)
    // (`interp_posix.py`) — how a device number is taken apart and
    // put back together, which is the host's own encoding and not arithmetic
    // that can be spelled portably. `tarfile` reads a node's pair out of
    // `st_rdev` to write a header (`tarfile.py`) and puts one back
    // together to recreate the node (`:2735`), so a `None` here writes a
    // header field that is not a number.
    //
    // No syscall, but the encoding is still the host's, and the sandbox build
    // reaches libc through a shim that carries no `dev_t` — so the names are
    // absent there rather than answering with another host's arithmetic.
    // `moduledef.py:152-157` registers each only where the host has it.
    #[cfg(all(unix, not(feature = "sandbox")))]
    {
        fn device_u64_w(value: PyObjectRef) -> Result<u64, crate::PyError> {
            let value = if unsafe {
                pyre_object::is_bool(value)
                    || pyre_object::is_int(value)
                    || pyre_object::is_long(value)
            } {
                value
            } else {
                crate::baseobjspace::space_index(value)?
            };
            crate::baseobjspace::uint_w(value)
        }
        fn device_value_w(value: PyObjectRef) -> Result<libc::dev_t, crate::PyError> {
            #[cfg_attr(
                not(all(target_os = "linux", not(target_env = "musl"))),
                allow(unused_mut)
            )]
            let mut indexed = crate::baseobjspace::space_index(value)?;
            // Reading the sentinel must not be fallible: a device number above
            // `i64::MAX` has no machine-word form, so propagating `int_w`'s
            // overflow here would refuse a value `uint_w` below accepts.
            #[cfg(all(target_os = "linux", not(target_env = "musl")))]
            if matches!(
                pyre_object::with_roots!(indexed => crate::baseobjspace::int_w(indexed)),
                Ok(-1)
            ) {
                return Ok(-1i64 as libc::dev_t);
            }
            let value = crate::baseobjspace::uint_w(indexed)?;
            // `dev_t` is signed on some targets and unsigned on others, so the
            // ceiling comes from the type rather than from its signedness.
            let max = u64::try_from(libc::dev_t::MAX).unwrap_or(u64::MAX);
            if value > max {
                return Err(crate::PyError::overflow_error(
                    "Python int too large to convert to C dev_t",
                ));
            }
            Ok(value as libc::dev_t)
        }
        fn device_w(args: &[PyObjectRef]) -> Result<libc::dev_t, crate::PyError> {
            let Some(&value) = args.first() else {
                return Err(crate::PyError::type_error("device is required"));
            };
            device_value_w(value)
        }
        fn major_minor_result(value: i64) -> PyObjectRef {
            #[cfg(all(target_os = "linux", not(target_env = "musl")))]
            if value == -1 || value == libc::c_uint::MAX as i64 {
                return pyre_object::w_int_new(-1);
            }
            pyre_object::w_int_new(value)
        }
        fn major_minor_arg(value: PyObjectRef) -> Result<libc::dev_t, crate::PyError> {
            // Where `NODEV` is spelled -1 that one value passes through, rather
            // than being rejected as out of range for an unsigned field.
            #[cfg(all(target_os = "linux", not(target_env = "musl")))]
            let value = {
                let mut indexed = crate::baseobjspace::space_index(value)?;
                if pyre_object::with_roots!(indexed => crate::baseobjspace::int_w(indexed))? == -1 {
                    return Ok(-1i64 as libc::dev_t);
                }
                crate::baseobjspace::uint_w(indexed)?
            };
            #[cfg(not(all(target_os = "linux", not(target_env = "musl"))))]
            let value = device_u64_w(value).map_err(|err| {
                if err.kind == crate::PyErrorKind::OverflowError {
                    crate::PyError::overflow_error(
                        "Python int too large to convert to C unsigned int",
                    )
                } else {
                    err
                }
            })?;
            if value > libc::c_uint::MAX as u64 {
                return Err(crate::PyError::overflow_error(
                    "Python int too large to convert to C unsigned int",
                ));
            }
            Ok(value as libc::dev_t)
        }
        crate::__pyre_put_new!(
            ns_slot,
            "major",
            crate::make_builtin_function_with_arity(
                "major",
                |args| {
                    Ok(major_minor_result(
                        unsafe { majit_rlib::rposix::c_major(device_w(args)?) } as i64,
                    ))
                },
                1,
            )
        );
        crate::__pyre_put_new!(
            ns_slot,
            "minor",
            crate::make_builtin_function_with_arity(
                "minor",
                |args| {
                    Ok(major_minor_result(
                        unsafe { majit_rlib::rposix::c_minor(device_w(args)?) } as i64,
                    ))
                },
                1,
            )
        );
        crate::__pyre_put_new!(
            ns_slot,
            "makedev",
            crate::make_builtin_function_with_arity(
                "makedev",
                |args| {
                    let (major, mut minor) = match args {
                        [major, minor, ..] => (*major, *minor),
                        _ => return Err(crate::PyError::type_error("makedev takes 2 arguments")),
                    };
                    let major = pyre_object::with_roots!(minor => major_minor_arg(major))?;
                    let minor = major_minor_arg(minor)?;
                    Ok(pyre_object::w_int_new(unsafe {
                        majit_rlib::rposix::c_makedev(major as _, minor as _)
                    } as i64))
                },
                2,
            )
        );
    }

    // PyPy `interp_posix.get_blocking/set_blocking` → rposix
    // `get_blocking/set_blocking`: inspect or update O_NONBLOCK with fcntl.
    fn get_blocking(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        if args.len() != 1 {
            return Err(crate::PyError::type_error(format!(
                "get_blocking() takes exactly one argument ({} given)",
                args.len()
            )));
        }
        let mut w_fd = args[0];
        let fd_value =
            pyre_object::with_roots!(w_fd => crate::builtins::space_index_w(w_fd))?;
        let fd = libc::c_int::try_from(fd_value)
            .map_err(|_| crate::PyError::overflow_error("fd is greater than maximum"))?;
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            // `interp_posix.get_blocking`: `eintr_retry=False`.
            let flags = pyre_object::with_roots!(w_fd => unsafe {
                majit_rlib::rposix::c_get_status_flags(fd)
            });
            if flags < 0 {
                return Err(errno_err(majit_rlib::rposix::get_saved_errno(), ""));
            }
            Ok(pyre_object::w_bool_from(flags & libc::O_NONBLOCK == 0))
        }
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        {
            use std::os::windows::io::AsRawHandle;

            // `PIPE_NOWAIT` is the named-pipe mode bit `get_blocking` reads.
            const PIPE_NOWAIT: u32 = 0x0000_0001;
            let borrowed = unsafe { rustpython_host_env::crt_fd::Borrowed::try_borrow_raw(fd) }
                .map_err(|error| {
                    crate::PyError::os_error_syscall(
                        crate::builtins::io_error_posix_errno(&error, libc::EBADF),
                        pyre_object::PY_NULL,
                    )
                })?;
            let handle = rustpython_host_env::crt_fd::as_handle(borrowed).map_err(|error| {
                crate::PyError::os_error_syscall(
                    crate::builtins::io_error_posix_errno(&error, libc::EBADF),
                    pyre_object::PY_NULL,
                )
            })?;
            let mode =
                rustpython_host_env::winapi::get_named_pipe_handle_state(handle.as_raw_handle())
                    .map_err(|error| {
                        crate::PyError::os_error_win32_syscall2(
                            error
                                .raw_os_error()
                                .unwrap_or(rustpython_host_env::winapi::get_last_error() as i32),
                            pyre_object::PY_NULL,
                            pyre_object::PY_NULL,
                        )
                    })?;
            Ok(pyre_object::w_bool_from(mode & PIPE_NOWAIT == 0))
        }
        #[cfg(any(
            feature = "sandbox",
            all(not(unix), not(all(windows, feature = "host_env")))
        ))]
        {
            let _ = fd;
            Err(crate::PyError::not_implemented(
                "get_blocking is unavailable on this target",
            ))
        }
    }

    fn set_blocking(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        if args.len() != 2 {
            return Err(crate::PyError::type_error(format!(
                "set_blocking() takes exactly two arguments ({} given)",
                args.len()
            )));
        }
        let w_fd = args[0];
        let mut w_blocking = args[1];
        let fd_value =
            pyre_object::with_roots!(w_blocking => crate::builtins::space_index_w(w_fd))?;
        let fd = libc::c_int::try_from(fd_value)
            .map_err(|_| crate::PyError::overflow_error("fd is greater than maximum"))?;
        // CPython 3.14's Argument Clinic declares this parameter `bool`, so
        // it truth-tests arbitrary objects.  This intentionally differs from
        // PyPy's older `@unwrap_spec(blocking=int)` gateway.
        let blocking =
            pyre_object::with_roots!(w_blocking => crate::baseobjspace::is_true(w_blocking))?;
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            // `interp_posix.set_blocking`: `eintr_retry=False`.
            let flags = pyre_object::with_roots!(w_blocking => unsafe {
                majit_rlib::rposix::c_get_status_flags(fd)
            });
            if flags < 0 {
                return Err(errno_err(majit_rlib::rposix::get_saved_errno(), ""));
            }
            let flags = if blocking {
                flags & !libc::O_NONBLOCK
            } else {
                flags | libc::O_NONBLOCK
            };
            let result = pyre_object::with_roots!(w_blocking => unsafe {
                majit_rlib::rposix::c_set_status_flags(fd, flags)
            });
            if result < 0 {
                return Err(errno_err(majit_rlib::rposix::get_saved_errno(), ""));
            }
            Ok(pyre_object::w_none())
        }
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        {
            use std::os::windows::io::AsRawHandle;

            // `PIPE_NOWAIT` is the named-pipe mode bit `set_blocking` flips.
            const PIPE_NOWAIT: u32 = 0x0000_0001;
            let borrowed = unsafe { rustpython_host_env::crt_fd::Borrowed::try_borrow_raw(fd) }
                .map_err(|error| {
                    crate::PyError::os_error_syscall(
                        crate::builtins::io_error_posix_errno(&error, libc::EBADF),
                        pyre_object::PY_NULL,
                    )
                })?;
            let handle = rustpython_host_env::crt_fd::as_handle(borrowed).map_err(|error| {
                crate::PyError::os_error_syscall(
                    crate::builtins::io_error_posix_errno(&error, libc::EBADF),
                    pyre_object::PY_NULL,
                )
            })?;
            let mut mode =
                rustpython_host_env::winapi::get_named_pipe_handle_state(handle.as_raw_handle())
                    .map_err(|error| {
                        crate::PyError::os_error_win32_syscall2(
                            error
                                .raw_os_error()
                                .unwrap_or(rustpython_host_env::winapi::get_last_error() as i32),
                            pyre_object::PY_NULL,
                            pyre_object::PY_NULL,
                        )
                    })?;
            if blocking {
                mode &= !PIPE_NOWAIT;
            } else {
                mode |= PIPE_NOWAIT;
            }
            rustpython_host_env::winapi::set_named_pipe_handle_state(
                handle.as_raw_handle(),
                Some(mode),
                None,
                None,
            )
            .map_err(|error| match error.raw_os_error() {
                Some(winerror) => crate::PyError::os_error_win32_syscall2(
                    winerror,
                    pyre_object::PY_NULL,
                    pyre_object::PY_NULL,
                ),
                None => crate::PyError::os_error_syscall(
                    crate::builtins::io_error_posix_errno(&error, libc::EINVAL),
                    pyre_object::PY_NULL,
                ),
            })?;
            Ok(pyre_object::w_none())
        }
        #[cfg(any(
            feature = "sandbox",
            all(not(unix), not(all(windows, feature = "host_env")))
        ))]
        {
            let _ = (fd, blocking);
            Err(crate::PyError::not_implemented(
                "set_blocking is unavailable on this target",
            ))
        }
    }

    crate::__pyre_put_new!(
        ns_slot,
        "get_blocking",
        crate::make_builtin_function("get_blocking", get_blocking)
    );
    crate::__pyre_put_new!(
        ns_slot,
        "set_blocking",
        crate::make_builtin_function("set_blocking", set_blocking)
    );

    // `baseobjspace.py fsencode_w` returns filesystem bytes; syscall
    // boundaries must not pass through a Rust `String`.
    use crate::gateway::fsencode_bytes_w as extract_path;

    /// The descriptor an `fd` argument names, as the borrowed handle the host
    /// API takes one as. `-1` is the single value `BorrowedFd::borrow_raw`
    /// refuses — the standard library reserves it as the niche that makes
    /// `Option<BorrowedFd>` free — so a caller who names it gets the `EBADF`
    /// the call would have answered with rather than a handle built out of the
    /// one integer that may not become one.
    #[cfg(unix)]
    fn fd_borrow(fd: libc::c_int) -> Result<std::os::fd::BorrowedFd<'static>, crate::PyError> {
        if fd == -1 {
            return Err(errno_err(libc::EBADF, ""));
        }
        Ok(unsafe { std::os::fd::BorrowedFd::borrow_raw(fd) })
    }

    /// The host-API view of OS bytes — a filename, or a half of an environment
    /// entry. Unix spells both in bytes and takes them back unchanged.
    #[cfg(not(all(unix, feature = "host_env")))]
    fn os_str_from_bytes(bytes: &[u8]) -> std::borrow::Cow<'_, std::ffi::OsStr> {
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            std::borrow::Cow::Borrowed(std::ffi::OsStr::from_bytes(bytes))
        }
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStringExt;
            // PEP 529 spells a name as UTF-8 over the host's UTF-16, so the
            // bytes decode back to the code units the API takes — including an
            // unpaired surrogate, which is a unit a name may carry.  A lossy
            // decode would substitute U+FFFD, addressing a different name and
            // making distinct entries alias onto one another.
            let units: Vec<u16> = crate::typedef::fsdecode_wtf8_total(bytes)
                .encode_wide()
                .collect();
            std::borrow::Cow::Owned(std::ffi::OsString::from_wide(&units))
        }
        #[cfg(not(any(unix, windows)))]
        {
            // This platform has no byte spelling, so the host API necessarily
            // receives the best text representation of these bytes.
            std::borrow::Cow::Owned(std::ffi::OsString::from(
                String::from_utf8_lossy(bytes).into_owned(),
            ))
        }
    }

    #[cfg(not(all(unix, feature = "host_env")))]
    fn path_from_bytes(path: &[u8]) -> std::borrow::Cow<'_, std::path::Path> {
        match os_str_from_bytes(path) {
            std::borrow::Cow::Borrowed(s) => std::borrow::Cow::Borrowed(std::path::Path::new(s)),
            std::borrow::Cow::Owned(s) => std::borrow::Cow::Owned(std::path::PathBuf::from(s)),
        }
    }

    // ── Helper: convert std::io::Error → PyError (OSError) ──
    fn errno_err(errno: i32, path: &str) -> crate::PyError {
        let w_filename = if path.is_empty() {
            pyre_object::PY_NULL
        } else {
            pyre_object::w_str_new_managed(path)
        };
        crate::PyError::os_error_syscall(errno, w_filename)
    }

    // `interp_posix.py` keeps the resolved `Path.w_path` as
    // `OSError.filename`.
    fn errno_err_with_filename(errno: i32, w_path: PyObjectRef) -> crate::PyError {
        crate::PyError::os_error_syscall(errno, w_path)
    }

    fn io_err(e: std::io::Error, path: &str) -> crate::PyError {
        errno_err(crate::builtins::io_error_posix_errno(&e, 0), path)
    }

    fn io_err_with_filename(e: std::io::Error, w_path: PyObjectRef) -> crate::PyError {
        errno_err_with_filename(crate::builtins::io_error_posix_errno(&e, 0), w_path)
    }

    /// The OSError for a failed *filesystem* call, whose error code Windows
    /// reports through `GetLastError` rather than `errno`: it becomes the
    /// `.winerror` attribute and picks up the system's message, the way
    /// `os.stat` and `os.mkdir` report `[WinError 3]`.  The descriptor calls
    /// (`open`, `read`, `write`, `close`, `lseek`) go through the C runtime and
    /// keep `io_err`'s errno form, which is what they report as well.
    ///
    /// `default_errno` stands in when the error carries no OS code at all.
    fn fs_err_with_filename2(
        e: std::io::Error,
        default_errno: i32,
        w_path: PyObjectRef,
        w_path2: PyObjectRef,
    ) -> crate::PyError {
        #[cfg(windows)]
        if let Some(winerror) = e.raw_os_error() {
            return crate::PyError::os_error_win32_syscall2(winerror, w_path, w_path2);
        }
        crate::PyError::os_error_syscall2(
            crate::builtins::io_error_posix_errno(&e, default_errno),
            w_path,
            w_path2,
        )
    }

    /// The width the platform's truncating call takes its length in.
    #[cfg(windows)]
    type TruncateLen = i64;
    #[cfg(not(windows))]
    type TruncateLen = libc::off_t;

    /// `space.int_w` over the `r_longlong` half of `interp_posix.py:404`.
    ///
    /// `Py_off_t_converter` names the C type it could not fit the value
    /// into, and that type is the platform's: a `long` where the converter
    /// is `PyLong_AsLong`, and nothing at all where it is
    /// `PyLong_AsLongLong`.
    fn truncate_length_w(obj: PyObjectRef) -> Result<TruncateLen, crate::PyError> {
        const TOO_BIG: &str = if cfg!(windows) {
            "int too big to convert"
        } else {
            "Python int too large to convert to C long"
        };
        let w_length = crate::baseobjspace::space_index(obj)?;
        let length = crate::baseobjspace::int_w(w_length).map_err(|err| {
            if err.kind == crate::PyErrorKind::OverflowError {
                crate::PyError::overflow_error(TOO_BIG)
            } else {
                err
            }
        })?;
        // `off_t` is the width the call takes the length in, and a value
        // above it is not a size the file can be given. An `as` cast would
        // wrap it into one the caller never asked for and truncate the file
        // to that instead.
        TruncateLen::try_from(length).map_err(|_| crate::PyError::overflow_error(TOO_BIG))
    }

    fn fs_err_with_filename(e: std::io::Error, w_path: PyObjectRef) -> crate::PyError {
        fs_err_with_filename2(e, 0, w_path, pyre_object::PY_NULL)
    }

    /// The wide (UTF-16) spelling of a path, for the Windows entry points that
    /// take one.  The narrow entry points re-encode through the ANSI code page,
    /// which has no spelling for most of what a filesystem name may hold, so
    /// every path call takes the `W` form.  A name holding an interior NUL is
    /// no more nameable there than it is to `CString`.
    #[cfg(all(windows, feature = "host_env"))]
    fn wide_path(bytes: &[u8]) -> Result<widestring::WideCString, crate::PyError> {
        let name = os_str_from_bytes(bytes);
        widestring::WideCString::from_os_str(&*name)
            .map_err(|_| crate::PyError::value_error("embedded null in path"))
    }

    // Both arms name their directory entries the same way, so the one
    // implementation lives beside the `stat_result` they also share.
    use super::fs_name_obj;

    /// interp_posix.py `unwrap_fd`.
    ///
    /// ```python
    /// def unwrap_fd(space, w_value, allowed_types='integer'):
    ///     try:
    ///         result = space.c_int_w(w_value)
    ///     except OperationError as e:
    ///         if not e.match(space, space.w_OverflowError):
    ///             raise oefmt(space.w_TypeError,
    ///                 "argument should be %s, not %T", allowed_types, w_value)
    ///         else:
    ///             raise
    ///     if result == -1:
    ///         # -1 is used as sentinel value for not a fd
    ///         raise oefmt(space.w_OSError, "invalid file descriptor: -1")
    ///     return result
    /// ```
    ///
    /// `c_int_w` is the load-bearing part: it converts through `__index__`,
    /// so an `int` subclass — an `IntEnum` member, say — reaches the syscall,
    /// where an exact-type test would reject it and a raw payload read would
    /// interpret the instance's first word as the descriptor.
    fn unwrap_fd(mut value: PyObjectRef, allowed_types: &str) -> Result<i32, crate::PyError> {
        if unsafe { pyre_object::is_bool(value) } {
            pyre_object::with_roots!(value => crate::warn::warn_category("bool is used as a file descriptor", "RuntimeWarning", 1))?;
        }
        let result = pyre_object::with_roots!(value => crate::baseobjspace::c_int_w(value))
            .map_err(|err| {
                if err.kind == crate::PyErrorKind::OverflowError {
                    err
                } else {
                    crate::PyError::type_error(format!(
                        "argument should be {allowed_types}, not {}",
                        crate::baseobjspace::object_functionstr_type_name(value)
                    ))
                }
            })?;
        if result == -1 {
            return Err(crate::PyError::os_error("invalid file descriptor: -1"));
        }
        Ok(result)
    }

    /// The `*, dir_fd=None` tail, read the way `DirFD(available)` reads it
    /// (`interp_posix.py`): `None` and an absent argument are the same
    /// `DEFAULT_DIR_FD`, and the value is converted before the platform is
    /// reported, so a wrongly typed one is a TypeError even on a build that
    /// carries no `*at` call to honour it.
    fn dir_fd_kwarg(
        kwargs: Option<pyre_object::PyObjectRef>,
        have: bool,
    ) -> Result<Option<i32>, crate::PyError> {
        match crate::builtins::kwarg_get(kwargs, "dir_fd") {
            Some(v) if !unsafe { pyre_object::is_none(v) } => {
                let fd = unwrap_fd(v, "integer or None")?;
                if !have {
                    return Err(dir_fd_unavailable());
                }
                Ok(Some(fd))
            }
            _ => Ok(None),
        }
    }

    /// Bind an entry point whose positional parameters all sit before the
    /// clinic `/`. None of them binds by name, so the count is over
    /// positionals alone.
    ///
    /// `kwonly` decides which parser reports a bad count, and the two word it
    /// differently. With no keyword-capable parameter at all the call is
    /// parsed by `_PyArg_CheckPositional`, whose wording carries neither the
    /// trailing `()` nor a parenthesised count, and every keyword is refused
    /// against the module-qualified name. A keyword-only tail puts the call
    /// back on `_PyArg_UnpackKeywords`, which reports the positional count in
    /// the parenthesised form and accepts the named modifiers.
    fn bind_posonly_args(
        args: &[pyre_object::PyObjectRef],
        name: &str,
        qualname: &str,
        total: usize,
        required: usize,
        kwonly: &[&'static str],
    ) -> Result<
        (
            Vec<Option<pyre_object::PyObjectRef>>,
            Option<pyre_object::PyObjectRef>,
        ),
        crate::PyError,
    > {
        let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
        if kwonly.is_empty() && crate::builtins::real_kwarg_count(kwargs) > 0 {
            return Err(crate::PyError::type_error(format!(
                "{qualname}() takes no keyword arguments"
            )));
        }
        // The count is checked before the keyword names: a call that supplies
        // neither the positionals nor a recognised keyword is reported against
        // the positionals.
        if pos.len() < required || pos.len() > total {
            let plural = if total == 1 { "" } else { "s" };
            let text = if !kwonly.is_empty() {
                let limit = if required == total {
                    "exactly"
                } else {
                    "at most"
                };
                format!(
                    "{name}() takes {limit} {total} positional argument{plural} ({} given)",
                    pos.len()
                )
            } else if required == total {
                format!(
                    "{name} expected {total} argument{plural}, got {}",
                    pos.len()
                )
            } else {
                let bound = if pos.len() > total { total } else { required };
                let plural = if bound == 1 { "" } else { "s" };
                let at = if pos.len() > total {
                    "at most"
                } else {
                    "at least"
                };
                format!(
                    "{name} expected {at} {bound} argument{plural}, got {}",
                    pos.len()
                )
            };
            return Err(crate::PyError::type_error(text));
        }
        crate::builtins::kwarg_reject_unknown(kwargs, kwonly, name)?;
        Ok((
            (0..total).map(|index| pos.get(index).copied()).collect(),
            kwargs,
        ))
    }

    /// Bind the positional-or-keyword prefix of a path-taking entry point.
    /// `params` names that prefix in order, `path` first, and the leading
    /// `required` of them carry no default; the rest are reported absent as
    /// `None`. `kwonly` names the keyword-only tail, which is left in the
    /// returned kwargs dict — which `HAVE_*` bit each modifier answers to is
    /// the caller's business.
    ///
    /// A surplus argument is reported the way the entry point's own generated
    /// parser reports it, and the two forms differ. Where there is a
    /// keyword-only tail the count is over positionals alone, and a signature
    /// with no defaults says "exactly" where one with defaults says "at most";
    /// where there is none, every argument counts toward the one limit and it
    /// is always "at most" — which is why `os.lchflags(p, 0, follow_symlinks=1)`
    /// is a count error and not an unknown keyword.
    fn bind_path_args(
        args: &[pyre_object::PyObjectRef],
        name: &str,
        params: &[&'static str],
        required: usize,
        kwonly: &[&'static str],
    ) -> Result<
        (
            Vec<Option<pyre_object::PyObjectRef>>,
            Option<pyre_object::PyObjectRef>,
        ),
        crate::PyError,
    > {
        let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
        let count = params.len();
        let plural = if count == 1 { "" } else { "s" };
        if kwonly.is_empty() {
            let given = pos.len() + crate::builtins::real_kwarg_count(kwargs);
            if given > count {
                return Err(crate::PyError::type_error(format!(
                    "{name}() takes at most {count} argument{plural} ({given} given)"
                )));
            }
        } else if pos.len() > count {
            let limit = if required == count {
                "exactly"
            } else {
                "at most"
            };
            return Err(crate::PyError::type_error(format!(
                "{name}() takes {limit} {count} positional argument{plural} ({} given)",
                pos.len()
            )));
        }
        // The names are checked after the count is satisfied:
        // `_PyArg_UnpackKeywords` fills the slots first and reports a required
        // one still empty, so `os.stat(other=1)` names `path` as missing
        // rather than `other` as unexpected.
        let mut bound = Vec::with_capacity(params.len());
        for (index, key) in params.iter().enumerate() {
            let value = crate::builtins::bind_pos_or_kw(pos, kwargs, index, key, name, index + 1)?;
            if value.is_none() && index < required {
                return Err(crate::PyError::type_error(format!(
                    "{name}() missing required argument '{key}' (pos {})",
                    index + 1
                )));
            }
            bound.push(value);
        }
        let mut allowed: Vec<&str> = params.to_vec();
        allowed.extend_from_slice(kwonly);
        crate::builtins::kwarg_reject_unknown(kwargs, &allowed, name)?;
        Ok((bound, kwargs))
    }

    // ── posix.open(path, flags, mode=0o777, *, dir_fd=None) → fd ──
    crate::__pyre_put_new!(
        ns_slot,
        "open",
        crate::make_builtin_function("open", |args| {
            let (bound, mut kwargs) =
                bind_path_args(args, "open", &["path", "flags", "mode"], 2, &["dir_fd"])?;
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
            let path = crate::gateway::fsencode_path_or_fd_w(
                bound[0].expect("path is required"),
                "open",
                false,
            );
            let w = roots.get(base);
            kwargs = if w.is_null() { None } else { Some(w) };
            let path = path?;
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
            let flags = crate::baseobjspace::c_int_w(bound[1].expect("flags is required"));
            let w = roots.get(base);
            kwargs = if w.is_null() { None } else { Some(w) };
            drop(roots);
            let flags = flags? as libc::c_int;
            let mode: u32 = match bound[2] {
                Some(value) => {
                    let roots = pyre_object::gc_roots::push_roots();
                    let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
                    let r = crate::baseobjspace::c_int_w(value);
                    let w = roots.get(base);
                    kwargs = if w.is_null() { None } else { Some(w) };
                    drop(roots);
                    r? as u32
                }
                None => 0o777,
            };
            // `open` types `dir_fd` as `DirFD(rposix.HAVE_OPENAT)`
            // (`interp_posix.py`). Only the `openat` arm below reads it;
            // every other build has already turned a descriptor away, because
            // `HAVE_OPENAT` is what those builds do not claim.
            let _dir_fd = dir_fd_kwarg(kwargs, HAVE_OPENAT)?;
            #[cfg(not(feature = "sandbox"))]
            let fd = {
                // Open the fd non-inheritable (PEP 446) so the descriptor does
                // not leak across exec into child processes: O_CLOEXEC on unix,
                // O_NOINHERIT on Windows (O_CLOEXEC is unix-only in libc). Moot
                // under sandbox, where the controller hands out virtual fds, so
                // it is applied only here.
                #[cfg(unix)]
                let flags = flags | libc::O_CLOEXEC;
                #[cfg(windows)]
                let flags = flags | libc::O_NOINHERIT;
                // `_wopen`, so the name reaches the filesystem intact.
                #[cfg(all(windows, feature = "host_env"))]
                let (fd, errno) = {
                    let wide = wide_path(&path.as_bytes)?;
                    crate::module::thread::call_external_function(|| {
                        rustpython_host_env::crt_fd::wopen(&wide, flags, mode as i32)
                            .map_or(-1, |owned| owned.into_raw())
                    })
                };
                #[cfg(not(all(windows, feature = "host_env")))]
                let (fd, errno) = {
                    let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    // interp_posix.py `open`: the syscall sits inside the
                    // `eintr_retry=True` loop.  A FIFO opened without O_NONBLOCK
                    // is the case that makes this reachable — it waits for a peer,
                    // and an alarm arriving meanwhile must run its handler rather
                    // than surface as `InterruptedError`.
                    loop {
                        // `c_open` / `c_openat` release the GIL and save errno.
                        // `openat` resolves the name against the descriptor;
                        // the plain `open` is what a name without one means
                        // (`interp_posix.py`).
                        #[cfg(unix)]
                        let (fd, errno) = {
                            let fd = if let Some(dir_fd) = _dir_fd {
                                unsafe {
                                    majit_rlib::rposix::c_openat(
                                        dir_fd,
                                        c_path.as_ptr(),
                                        flags,
                                        mode as _,
                                    )
                                }
                            } else {
                                unsafe {
                                    majit_rlib::rposix::c_open(c_path.as_ptr(), flags, mode as _)
                                }
                            };
                            let errno = if fd < 0 {
                                majit_rlib::rposix::get_saved_errno()
                            } else {
                                0
                            };
                            (fd, errno)
                        };
                        #[cfg(not(unix))]
                        let (fd, errno) =
                            crate::module::thread::call_external_function(|| unsafe {
                                libc::open(c_path.as_ptr(), flags, mode as libc::c_uint)
                            });
                        if fd >= 0 {
                            break (fd, errno);
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(errno),
                            |e| {
                                errno_err_with_filename(
                                    e.raw_os_error().unwrap_or(0),
                                    path.w_path(),
                                )
                            },
                        )?;
                    }
                };
                if fd < 0 {
                    return Err(errno_err_with_filename(errno, path.w_path()));
                }
                fd
            };
            #[cfg(feature = "sandbox")]
            let fd = crate::host_seam::ops::open(&path.as_bytes, flags, mode)
                .map_err(|e| crate::host_seam::seam_os_err_with_filename(e, path.w_path()))?;
            Ok(pyre_object::w_int_new(fd as i64))
        })
    );

    // ── posix.close(fd) ──
    crate::__pyre_put_new!(
        ns_slot,
        "close",
        crate::make_builtin_function_with_arity(
            "close",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("close() requires 1 argument"));
                }
                let fd = crate::baseobjspace::c_int_w(args[0])? as libc::c_int;
                #[cfg(not(feature = "sandbox"))]
                {
                    // `rposix.c_close` is `releasegil=False` and saves errno.
                    #[cfg(unix)]
                    let (ret, errno) = {
                        let ret = unsafe { majit_rlib::rposix::c_close(fd) };
                        let errno = if ret < 0 {
                            majit_rlib::rposix::get_saved_errno()
                        } else {
                            0
                        };
                        (ret, errno)
                    };
                    #[cfg(not(unix))]
                    let (ret, errno) = {
                        let ret = crate::builtins::crt_call!(libc::close(fd));
                        let errno = if ret < 0 {
                            crate::builtins::crt_errno()
                        } else {
                            0
                        };
                        (ret, errno)
                    };
                    if ret < 0 {
                        return Err(errno_err(errno, ""));
                    }
                }
                #[cfg(feature = "sandbox")]
                crate::host_seam::ops::close(fd)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                Ok(pyre_object::w_none())
            },
            1,
        )
    );

    // ── posix.closerange(fd_low, fd_high) ──
    // Half-open, and every failure is dropped: the point of the call is to shut
    // whatever is open in the range without first asking what that is, which is
    // how `subprocess` closes the parent's descriptors in the child. Closing one
    // the process does not have is the ordinary case, not an error. Unix uses
    // `rposix.c_close`; Windows goes through `crt_call!` so the C runtime's
    // invalid-parameter handler does not abort the process.
    crate::__pyre_put_new!(
        ns_slot,
        "closerange",
        crate::make_builtin_function_with_arity(
            "closerange",
            |args| {
                if args.len() != 2 {
                    return Err(crate::PyError::type_error(format!(
                        "closerange() takes exactly 2 arguments ({} given)",
                        args.len()
                    )));
                }
                let mut w_low = args[0];
                let mut w_high = args[1];
                let low = pyre_object::with_roots!(w_low, w_high => crate::baseobjspace::c_int_w(w_low))?
                    as libc::c_int;
                let high = crate::baseobjspace::c_int_w(w_high)? as libc::c_int;
                for fd in low..high {
                    #[cfg(all(not(feature = "sandbox"), unix))]
                    let _ = unsafe { majit_rlib::rposix::c_close(fd) };
                    #[cfg(all(not(feature = "sandbox"), not(unix)))]
                    let _ = crate::builtins::crt_call!(libc::close(fd));
                    #[cfg(feature = "sandbox")]
                    let _ = crate::host_seam::ops::close(fd);
                }
                Ok(pyre_object::w_none())
            },
            2,
        )
    );

    // ── posix.strerror(code) ──
    // The C runtime's message table, which is the one `OSError.strerror`
    // already reports from — the two answer alike for the same errno.
    crate::__pyre_put_new!(
        ns_slot,
        "strerror",
        crate::make_builtin_function_with_arity(
            "strerror",
            |args| {
                if args.len() != 1 {
                    return Err(crate::PyError::type_error(format!(
                        "strerror() takes exactly 1 argument ({} given)",
                        args.len()
                    )));
                }
                let code = crate::baseobjspace::c_int_w(args[0])?;
                Ok(pyre_object::w_str_new_managed(
                    &crate::PyError::clean_strerror(code),
                ))
            },
            1,
        )
    );

    // ── posix.read(fd, n) → bytes ──
    crate::__pyre_put_new!(
        ns_slot,
        "read",
        crate::make_builtin_function_with_arity(
            "read",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("read() requires 2 arguments"));
                }
                let mut w_fd = args[0];
                let mut w_n = args[1];
                let fd = pyre_object::with_roots!(w_fd, w_n => crate::baseobjspace::c_int_w(w_fd))?
                    as libc::c_int;
                let n_signed = crate::baseobjspace::int_w(w_n)?;
                // A negative size would wrap to a huge `usize` (and allocation);
                // os.read rejects it with EINVAL, matching the host read(2).
                if n_signed < 0 {
                    return Err(crate::PyError::os_error_with_errno(
                        libc::EINVAL,
                        "read: negative size",
                    ));
                }
                // `os_read_impl` then clamps the request to `_PY_READ_MAX`
                // before it allocates, and `_Py_read` clamps it again.
                // Unclamped, the whole request reaches the allocator, and a
                // Windows `n > INT_MAX` reaches the C runtime as EINVAL
                // instead of data.
                let n = n_signed.min(crate::builtins::PY_READ_MAX) as usize;
                #[cfg(not(feature = "sandbox"))]
                let buf = {
                    // `PyBytes_FromStringAndSize(NULL, length)` reserves the
                    // whole request up front and reports a failure it cannot
                    // satisfy as MemoryError; an infallible `vec![0u8; n]`
                    // aborts the process there instead.  The block stays
                    // uninitialised until the read reports how much of it was
                    // filled, so nothing writes the tail either.
                    let mut buf: Vec<u8> = Vec::new();
                    buf.try_reserve_exact(n)
                        .map_err(|_| crate::PyError::memory_error(""))?;
                    // interp_posix.py `read`: the syscall sits inside
                    // the `eintr_retry=True` loop, so an interrupted read runs
                    // the pending signal handlers and is re-issued rather than
                    // surfacing as `InterruptedError`.  The blocking guard is
                    // scoped to the syscall alone: `checksignals` runs Python.
                    loop {
                        // `rposix.c_read` releases the GIL and saves errno.
                        #[cfg(unix)]
                        let (ret, errno) = {
                            let ret = unsafe {
                                majit_rlib::rposix::c_read(
                                    fd,
                                    buf.as_mut_ptr() as majit_rlib::rffi::VOIDP,
                                    n,
                                )
                            };
                            let errno = if ret < 0 {
                                majit_rlib::rposix::get_saved_errno()
                            } else {
                                0
                            };
                            (ret, errno)
                        };
                        #[cfg(not(unix))]
                        let (ret, errno) =
                            crate::module::thread::call_external_function(|| unsafe {
                                libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, n as _)
                            });
                        if ret >= 0 {
                            // Only the prefix the call reports was written.
                            unsafe { buf.set_len(ret as usize) };
                            break buf;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(errno),
                            |e| errno_err(e.raw_os_error().unwrap_or(0), ""),
                        )?;
                    }
                };
                #[cfg(feature = "sandbox")]
                let buf = crate::host_seam::ops::read(fd, n as i64)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                Ok(pyre_object::w_bytes_from_bytes(&buf))
            },
            2,
        )
    );

    // Python 3.14 `os.readinto`: acquire one writable buffer export for the
    // complete `_Py_read(fd, buffer->buf, buffer->len)` call and return the
    // number of bytes transferred without allocating an intermediate bytes
    // object on the real-host path.
    crate::__pyre_put_new!(
        ns_slot,
        "readinto",
        crate::make_builtin_function_with_arity(
            "readinto",
            |args| {
                let w_fd = args[0];
                let mut w_buffer = args[1];
                let fd_value =
                    pyre_object::with_roots!(w_buffer => crate::baseobjspace::int_w(w_fd))?;
                let fd = libc::c_int::try_from(fd_value)
                    .map_err(|_| crate::PyError::overflow_error("fd is greater than maximum"))?;
                let mut buffer = unsafe { crate::builtins::WritableBuffer::acquire(w_buffer) }?;
                let target = unsafe { buffer.as_mut_slice() };
                #[cfg(not(feature = "sandbox"))]
                let result = loop {
                    #[cfg(all(windows, feature = "host_env"))]
                    let read_result = crate::builtins::fd_read_into(fd, &mut *target);
                    #[cfg(unix)]
                    let (result, errno) = {
                        let result = unsafe {
                            majit_rlib::rposix::c_read(
                                fd,
                                target.as_mut_ptr() as majit_rlib::rffi::VOIDP,
                                target.len(),
                            )
                        };
                        let errno = if result < 0 {
                            majit_rlib::rposix::get_saved_errno()
                        } else {
                            0
                        };
                        (result, errno)
                    };
                    #[cfg(not(any(unix, all(windows, feature = "host_env"))))]
                    let (result, errno) =
                        crate::module::thread::call_external_function(|| unsafe {
                            libc::read(
                                fd,
                                target.as_mut_ptr() as *mut libc::c_void,
                                target.len() as _,
                            )
                        });
                    #[cfg(all(windows, feature = "host_env"))]
                    let (result, errno) = match read_result {
                        Ok(result) => (result as i64, 0),
                        Err(error) => (-1, error.raw_os_error().unwrap_or(libc::EIO)),
                    };
                    if result >= 0 {
                        break result as i64;
                    }
                    // The retry used to skip the handlers, so an interrupted
                    // read never let the handler that supplies the remaining
                    // bytes run.  Guard dropped first: `checksignals` runs
                    // Python.
                    crate::builtins::eintr_retry_with(
                        std::io::Error::from_raw_os_error(errno),
                        |e| errno_err(e.raw_os_error().unwrap_or(0), ""),
                    )?;
                };
                #[cfg(feature = "sandbox")]
                let result = {
                    let data = crate::host_seam::ops::read(fd, target.len() as i64)
                        .map_err(|error| crate::host_seam::seam_os_err(error, ""))?;
                    let length = data.len().min(target.len());
                    target[..length].copy_from_slice(&data[..length]);
                    length as i64
                };
                Ok(pyre_object::w_int_new(result))
            },
            2,
        )
    );

    // ── posix.write(fd, data) → nbytes ──
    crate::__pyre_put_new!(
        ns_slot,
        "write",
        crate::make_builtin_function_with_arity(
            "write",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("write() requires 2 arguments"));
                }
                let mut w_fd = args[0];
                let mut w_data = args[1];
                let fd = pyre_object::with_roots!(w_fd, w_data => crate::baseobjspace::c_int_w(w_fd))?
                    as libc::c_int;
                // CPython `os_write_impl` receives a `Py_buffer`: text is not
                // accepted, while every contiguous readable exporter is.
                let data = unsafe { crate::builtins::file_write_buffer_bytes(w_data) }
                    .map_err(|_| crate::PyError::type_error("write() arg 2 must be bytes-like"))?;
                #[cfg(not(feature = "sandbox"))]
                let ret = {
                    // interp_posix.py `write`: the syscall sits inside
                    // the `eintr_retry=True` loop, so an interrupted write runs
                    // the pending signal handlers and is re-issued rather than
                    // surfacing as `InterruptedError`.  The blocking guard is
                    // scoped to the syscall alone: `checksignals` runs Python.
                    loop {
                        // `rposix.c_write` releases the GIL and saves errno.
                        #[cfg(unix)]
                        let (ret, errno) = {
                            let count = data.len().min(i32::MAX as usize);
                            let ret = unsafe {
                                majit_rlib::rposix::c_write(
                                    fd,
                                    data.as_ptr() as majit_rlib::rffi::VOIDP,
                                    count,
                                )
                            };
                            let errno = if ret < 0 {
                                majit_rlib::rposix::get_saved_errno()
                            } else {
                                0
                            };
                            (ret as i64, errno)
                        };
                        #[cfg(not(unix))]
                        let (ret, errno) = crate::builtins::crt_write_once(fd, &data);
                        if ret >= 0 {
                            break ret as i64;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(errno),
                            |e| errno_err(e.raw_os_error().unwrap_or(0), ""),
                        )?;
                    }
                };
                #[cfg(feature = "sandbox")]
                let ret = crate::host_seam::ops::write(fd, &data)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                Ok(pyre_object::w_int_new(ret))
            },
            2,
        )
    );

    // ── posix.lseek(fd, offset, whence) → position ──
    crate::__pyre_put_new!(
        ns_slot,
        "lseek",
        crate::make_builtin_function_with_arity(
            "lseek",
            |args| {
                if args.len() < 3 {
                    return Err(crate::PyError::type_error("lseek() requires 3 arguments"));
                }
                // interp_posix.py `@unwrap_spec(fd=c_int, position=r_longlong,
                // how=c_int)` — the position is a 64-bit offset, not a C int.
                let w_fd = args[0];
                let mut w_offset = args[1];
                let mut w_whence = args[2];
                let fd = pyre_object::with_roots!(w_offset, w_whence =>
                    crate::baseobjspace::c_int_w(w_fd))? as libc::c_int;
                let offset =
                    pyre_object::with_roots!(w_whence => crate::baseobjspace::int_w(w_offset))?;
                let whence = crate::baseobjspace::c_int_w(w_whence)? as libc::c_int;
                #[cfg(not(feature = "sandbox"))]
                let ret = {
                    // `rposix.c_lseek` is `macro=_MACRO_ON_POSIX` and saves
                    // errno.
                    #[cfg(unix)]
                    {
                        let ret = unsafe {
                            majit_rlib::rposix::c_lseek(
                                fd,
                                offset as majit_rlib::rffi::LONGLONG,
                                whence,
                            )
                        };
                        if ret < 0 {
                            return Err(errno_err(majit_rlib::rposix::get_saved_errno(), ""));
                        }
                        ret as i64
                    }
                    #[cfg(not(unix))]
                    {
                        let ret = crate::builtins::crt_lseek(fd, offset, whence);
                        if ret < 0 {
                            return Err(errno_err(crate::builtins::crt_errno(), ""));
                        }
                        ret
                    }
                };
                #[cfg(feature = "sandbox")]
                let ret = crate::host_seam::ops::lseek(fd, offset, whence)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                Ok(pyre_object::w_int_new(ret))
            },
            3,
        )
    );

    // ── posix.unlink(path, *, dir_fd=None) / posix.remove(path, *, dir_fd=None) ──
    // `remove` is `unlink` written out a second time under its own name
    // (`interp_posix.py`), so it reports itself by that name.
    fn posix_unlink(
        args: &[pyre_object::PyObjectRef],
        name: &str,
    ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        let (bound, mut kwargs) = bind_path_args(args, name, &["path"], 1, &["dir_fd"])?;
        let roots = pyre_object::gc_roots::push_roots();
        let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
        let path =
            crate::gateway::fsencode_path_or_fd_w(bound[0].expect("path is required"), name, false);
        let w = roots.get(base);
        kwargs = if w.is_null() { None } else { Some(w) };
        let path = path?;
        // Both take `DirFD(rposix.HAVE_UNLINKAT)` (`interp_posix.py`).
        let _dir_fd = dir_fd_kwarg(kwargs, HAVE_UNLINKAT)?;
        // `DeleteFileW`, except on a directory symlink, which `RemoveDirectoryW`
        // unlinks without following (`os_unlink_impl`).
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        rustpython_host_env::nt::remove(&wide_path(&path.as_bytes)?)
            .map_err(|e| fs_err_with_filename(e, path.w_path()))?;
        #[cfg(all(not(all(windows, feature = "host_env")), not(feature = "sandbox")))]
        {
            let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
            // `unlinkat` without `AT_REMOVEDIR` is the name form resolved
            // against a descriptor (`rposix.py`). The no-descriptor call is
            // `rposix.c_unlink`, which releases the GIL and saves errno.
            #[cfg(unix)]
            let (ret, err) = match _dir_fd {
                Some(dir_fd) => {
                    let ret = unsafe { majit_rlib::rposix::c_unlinkat(dir_fd, c_path.as_ptr(), 0) };
                    (
                        ret,
                        std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                    )
                }
                None => {
                    let ret = unsafe { majit_rlib::rposix::c_unlink(c_path.as_ptr()) };
                    (
                        ret,
                        std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                    )
                }
            };
            #[cfg(not(unix))]
            let (ret, err) = {
                let ret = unsafe { libc::unlink(c_path.as_ptr()) };
                (ret, std::io::Error::last_os_error())
            };
            if ret < 0 {
                return Err(fs_err_with_filename(err, path.w_path()));
            }
        }
        #[cfg(feature = "sandbox")]
        crate::host_seam::ops::unlink(&path.as_bytes)
            .map_err(|e| crate::host_seam::seam_os_err_with_filename(e, path.w_path()))?;
        Ok(pyre_object::w_none())
    }
    crate::__pyre_put_new!(
        ns_slot,
        "unlink",
        crate::make_builtin_function("unlink", |args| posix_unlink(args, "unlink"))
    );
    crate::__pyre_put_new!(
        ns_slot,
        "remove",
        crate::make_builtin_function("remove", |args| posix_unlink(args, "remove"))
    );

    // ── posix.readlink(path, *, dir_fd=None) ──
    // Returns the symlink target; a non-symlink raises OSError(EINVAL), which
    // `posixpath.realpath` relies on to stop following links.
    // Under sandbox readlink is unavailable (the controller has no ll_os
    // readlink handler); the stub override loop registers a raising stub, so
    // keep the real body out of the sandbox build.
    #[cfg(not(feature = "sandbox"))]
    crate::__pyre_put_new!(
        ns_slot,
        "readlink",
        crate::make_builtin_function("readlink", |args| {
            let (bound, kwargs) = bind_path_args(args, "readlink", &["path"], 1, &["dir_fd"])?;
            // `readlink` types `dir_fd` as `DirFD(rposix.HAVE_READLINKAT)`.
            let _dir_fd = dir_fd_kwarg(kwargs, HAVE_READLINKAT)?;
            let path = crate::gateway::fsencode_path_named_w(
                bound[0].expect("path is required"),
                "readlink",
                "path",
            )?;
            let bytes_mode = unsafe { path.is_bytes() };
            // `FSCTL_GET_REPARSE_POINT` and the substitute name out of the
            // buffer it fills, which is what spells a junction's target
            // `\\?\C:\...`; `std::fs::read_link` hands back a name with
            // that prefix already taken off.
            #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
            {
                use rustpython_host_env::nt::ReadlinkError;
                return match rustpython_host_env::nt::readlink(&wide_path(&path.as_bytes)?) {
                    Ok(target) => Ok(fs_name_obj(
                        bytes_mode,
                        target.as_os_str().as_encoded_bytes(),
                    )),
                    Err(ReadlinkError::Io(error)) => {
                        Err(fs_err_with_filename(error, path.w_path()))
                    }
                    // A reparse point of any other kind names nothing this
                    // call can answer with.
                    Err(ReadlinkError::NotSymbolicLink)
                    | Err(ReadlinkError::InvalidReparseData) => {
                        Err(crate::PyError::value_error("not a symbolic link"))
                    }
                };
            }
            #[cfg(unix)]
            {
                // `rposix.readlink` starts at 1023 bytes and multiplies by 4
                // while `c_readlink` fills the buffer. `c_readlink` releases
                // the GIL and saves errno.
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                let mut bufsize = 1023usize;
                let target = loop {
                    let mut buf = Vec::new();
                    buf.try_reserve_exact(bufsize)
                        .map_err(|_| crate::PyError::memory_error(""))?;
                    buf.resize(bufsize, 0);
                    let res = match _dir_fd {
                        Some(dir_fd) => unsafe {
                            majit_rlib::rposix::c_readlinkat(
                                dir_fd,
                                c_path.as_ptr(),
                                buf.as_mut_ptr().cast(),
                                bufsize,
                            )
                        },
                        None => unsafe {
                            majit_rlib::rposix::c_readlink(
                                c_path.as_ptr(),
                                buf.as_mut_ptr().cast(),
                                bufsize,
                            )
                        },
                    };
                    if res < 0 {
                        return Err(fs_err_with_filename(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            path.w_path(),
                        ));
                    }
                    let n = res as usize;
                    if n < bufsize {
                        buf.truncate(n);
                        break buf;
                    }
                    bufsize *= 4;
                };
                Ok(fs_name_obj(bytes_mode, &target))
            }
            #[cfg(not(unix))]
            match std::fs::read_link(path_from_bytes(&path.as_bytes).as_ref()) {
                Ok(target) => {
                    let target = target.as_os_str().as_encoded_bytes();
                    Ok(fs_name_obj(bytes_mode, target))
                }
                Err(e) => Err(fs_err_with_filename(e, path.w_path())),
            }
        })
    );

    // ── posix.mkdir(path, mode=0o777, *, dir_fd=None) ──
    crate::__pyre_put_new!(
        ns_slot,
        "mkdir",
        crate::make_builtin_function("mkdir", |args| {
            let (bound, mut kwargs) =
                bind_path_args(args, "mkdir", &["path", "mode"], 1, &["dir_fd"])?;
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
            let path = crate::gateway::fsencode_path_or_fd_w(
                bound[0].expect("path is required"),
                "mkdir",
                false,
            );
            let w = roots.get(base);
            kwargs = if w.is_null() { None } else { Some(w) };
            let path = path?;
            let _mode: u32 = match bound[1] {
                Some(value) => {
                    let roots = pyre_object::gc_roots::push_roots();
                    let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
                    let r = crate::baseobjspace::c_int_w(value);
                    let w = roots.get(base);
                    kwargs = if w.is_null() { None } else { Some(w) };
                    drop(roots);
                    r? as u32
                }
                None => 0o777,
            };
            // `mkdir` types `dir_fd` as `DirFD(rposix.HAVE_MKDIRAT)`
            // (`interp_posix.py`).
            let _dir_fd = dir_fd_kwarg(kwargs, HAVE_MKDIRAT)?;
            // `CreateDirectoryW`; a mode of 0o700 is served by the security
            // descriptor that denies everyone but the owner (`os_mkdir_impl`).
            #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
            rustpython_host_env::nt::mkdir(&wide_path(&path.as_bytes)?, _mode as i32)
                .map_err(|e| fs_err_with_filename(e, path.w_path()))?;
            #[cfg(all(not(all(windows, feature = "host_env")), not(feature = "sandbox")))]
            {
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                // `mkdirat` resolves the name against the descriptor
                // (`rposix.py`). The no-descriptor call is `rposix.c_mkdir`,
                // which releases the GIL and saves errno.
                #[cfg(unix)]
                let (ret, err) = match _dir_fd {
                    Some(dir_fd) => {
                        let ret = unsafe {
                            majit_rlib::rposix::c_mkdirat(
                                dir_fd,
                                c_path.as_ptr(),
                                _mode as libc::mode_t,
                            )
                        };
                        (
                            ret,
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                        )
                    }
                    None => {
                        let ret = unsafe {
                            majit_rlib::rposix::c_mkdir(c_path.as_ptr(), _mode as libc::mode_t)
                        };
                        (
                            ret,
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                        )
                    }
                };
                #[cfg(windows)]
                let (ret, err) = {
                    let ret = unsafe { libc::mkdir(c_path.as_ptr()) };
                    (ret, std::io::Error::last_os_error())
                };
                if ret < 0 {
                    return Err(fs_err_with_filename(err, path.w_path()));
                }
            }
            #[cfg(feature = "sandbox")]
            crate::host_seam::ops::mkdir(&path.as_bytes, _mode)
                .map_err(|e| crate::host_seam::seam_os_err_with_filename(e, path.w_path()))?;
            Ok(pyre_object::w_none())
        })
    );

    // ── posix.rmdir(path, *, dir_fd=None) ──
    // Mutates the host filesystem; stubbed under sandbox, so the real body
    // (and its libc call) is compiled out.
    #[cfg(not(feature = "sandbox"))]
    crate::__pyre_put_new!(
        ns_slot,
        "rmdir",
        crate::make_builtin_function("rmdir", |args| {
            let (bound, mut kwargs) = bind_path_args(args, "rmdir", &["path"], 1, &["dir_fd"])?;
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
            let path = crate::gateway::fsencode_path_or_fd_w(
                bound[0].expect("path is required"),
                "rmdir",
                false,
            );
            let w = roots.get(base);
            kwargs = if w.is_null() { None } else { Some(w) };
            let path = path?;
            // Removing a directory is the same call as removing a file, so
            // `rmdir` reads the same bit: `DirFD(rposix.HAVE_UNLINKAT)`
            // (`interp_posix.py`).
            let _dir_fd = dir_fd_kwarg(kwargs, HAVE_UNLINKAT)?;
            // `RemoveDirectoryW`, which is what `std::fs::remove_dir` is on
            // Windows (`os_rmdir_impl`).
            #[cfg(windows)]
            std::fs::remove_dir(path_from_bytes(&path.as_bytes).as_ref())
                .map_err(|e| fs_err_with_filename(e, path.w_path()))?;
            #[cfg(not(windows))]
            {
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                // `AT_REMOVEDIR` is what makes the one `unlinkat` a `rmdir`
                // (`rposix.unlinkat` `removedir=True`). The no-descriptor
                // call is `rposix.c_rmdir`, which releases the GIL and saves
                // errno.
                let (ret, err) = match _dir_fd {
                    Some(dir_fd) => {
                        let ret = unsafe {
                            majit_rlib::rposix::c_unlinkat(
                                dir_fd,
                                c_path.as_ptr(),
                                libc::AT_REMOVEDIR,
                            )
                        };
                        (
                            ret,
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                        )
                    }
                    None => {
                        let ret = unsafe { majit_rlib::rposix::c_rmdir(c_path.as_ptr()) };
                        (
                            ret,
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                        )
                    }
                };
                if ret < 0 {
                    return Err(fs_err_with_filename(err, path.w_path()));
                }
            }
            Ok(pyre_object::w_none())
        })
    );

    // ── posix.rename / posix.replace(src, dst, *, src_dir_fd=None,
    //    dst_dir_fd=None) ──
    // A non-None `src_dir_fd` / `dst_dir_fd` resolves the path relative to the
    // open directory descriptor (`renameat`); the descriptors are only usable
    // where `renameat` exists (unix).
    //
    // The two entry points take the same arguments and differ only in what a
    // pre-existing `dst` does: `replace` overwrites it on every platform,
    // `rename` leaves that to the platform call. On Windows the host layer's
    // `replace` uses MoveFileExW(MOVEFILE_REPLACE_EXISTING); plain `rename`
    // deliberately omits that flag.
    fn rename_impl(
        args: &[PyObjectRef],
        name: &'static str,
    ) -> Result<PyObjectRef, crate::PyError> {
        let (pos, mut kwargs) = crate::builtins::split_builtin_kwargs(args);
        if pos.len() < 2 {
            return Err(crate::PyError::type_error(format!(
                "{name}() requires 2 arguments"
            )));
        }
        if pos.len() > 2 {
            return Err(crate::PyError::type_error(format!(
                "{name}() takes exactly 2 positional arguments ({} given)",
                pos.len()
            )));
        }
        crate::builtins::kwarg_reject_unknown(kwargs, &["src_dir_fd", "dst_dir_fd"], name)?;
        // `rename` and `replace` are one body here and two argument-clinic
        // declarations there, so the rejected argument is named after whichever
        // of the two the caller reached.
        let npos = pos.len();
        let roots = pyre_object::gc_roots::push_roots();
        let base = roots.publish(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
        let pos_base = roots.publish(pos);
        roots.normalize(base, 1 + npos);
        let src = crate::gateway::fsencode_path_named_w(roots.get(pos_base), name, "src")?;
        let w = roots.get(base);
        kwargs = if w.is_null() { None } else { Some(w) };
        let dst = crate::gateway::fsencode_path_named_w(roots.get(pos_base + 1), name, "dst")?;
        let w = roots.get(base);
        kwargs = if w.is_null() { None } else { Some(w) };
        let dir_fd = |name: &str| -> Result<Option<i32>, crate::PyError> {
            match crate::builtins::kwarg_get(kwargs, name) {
                // interp_posix.py `_unwrap_dirfd` — a non-`None` value
                // goes through `unwrap_fd` with `allowed_types="integer or
                // None"`.
                Some(v) if !unsafe { pyre_object::is_none(v) } => {
                    Ok(Some(unwrap_fd(v, "integer or None")?))
                }
                _ => Ok(None),
            }
        };
        let src_fd = dir_fd("src_dir_fd")?;
        let dst_fd = dir_fd("dst_dir_fd")?;
        // `rposix.c_rename` is the call with no directory descriptor.
        // `rposix.c_renameat` is the call when either side has one. On unix
        // `replace` is the same syscall. Both release the GIL and save errno.
        #[cfg(unix)]
        {
            let src_c = std::ffi::CString::new(src.as_bytes.as_slice())
                .map_err(|_| crate::PyError::value_error("embedded null in src"))?;
            let dst_c = std::ffi::CString::new(dst.as_bytes.as_slice())
                .map_err(|_| crate::PyError::value_error("embedded null in dst"))?;
            let ret = if src_fd.is_none() && dst_fd.is_none() {
                unsafe { majit_rlib::rposix::c_rename(src_c.as_ptr(), dst_c.as_ptr()) }
            } else {
                unsafe {
                    majit_rlib::rposix::c_renameat(
                        src_fd.unwrap_or(libc::AT_FDCWD),
                        src_c.as_ptr(),
                        dst_fd.unwrap_or(libc::AT_FDCWD),
                        dst_c.as_ptr(),
                    )
                }
            };
            if ret < 0 {
                return Err(fs_err_with_filename2(
                    std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                    0,
                    src.w_path(),
                    dst.w_path(),
                ));
            }
            return Ok(pyre_object::w_none());
        }
        #[cfg(not(unix))]
        {
            if src_fd.is_some() || dst_fd.is_some() {
                return Err(crate::PyError::not_implemented(
                    "dir_fd unavailable on this platform",
                ));
            }
            let src_path = path_from_bytes(&src.as_bytes);
            let dst_path = path_from_bytes(&dst.as_bytes);
            let result = if name == "replace" {
                host_os::replace(src_path.as_ref(), None, dst_path.as_ref(), None)
            } else {
                host_os::rename(src_path.as_ref(), None, dst_path.as_ref(), None)
            };
            // interp_posix.py hands both resolved `Path.w_path` objects to
            // `wrap_oserror2`.
            result.map_err(|e| fs_err_with_filename2(e, 0, src.w_path(), dst.w_path()))?;
            Ok(pyre_object::w_none())
        }
    }
    crate::__pyre_put_new!(
        ns_slot,
        "rename",
        crate::make_builtin_function("rename", |args| rename_impl(args, "rename"))
    );
    crate::__pyre_put_new!(
        ns_slot,
        "replace",
        crate::make_builtin_function("replace", |args| rename_impl(args, "replace"))
    );

    // os.utime(path, times=None, *, ns=None, dir_fd=None, follow_symlinks=True)
    /// One of the two times `utime` writes, kept the way `rposix.futimens` and
    /// `rposix.utimensat` keep it (`rposix.py`): the seconds and the
    /// nanoseconds apart, both signed, so a time before the epoch is the
    /// negative second it names rather than a value with no representation.
    /// The nanoseconds are always the ones after that second — `1969-12-31
    /// 23:59:59.999999999` is `(-1, 999999999)`, which is the `ns=-1` a caller
    /// asked for.
    #[derive(Clone, Copy)]
    struct UTime {
        sec: i64,
        nsec: i64,
    }

    /// `interp_posix.py:1901-1904` answers a descriptor with `futimens`, which
    /// is the call HAVE_FUTIMENS names.
    fn utime_fd(
        fd: i32,
        now: bool,
        access: UTime,
        modified: UTime,
    ) -> Result<PyObjectRef, crate::PyError> {
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            let times = [timespec_of(access, now), timespec_of(modified, now)];
            // `rposix.c_futimens` releases the GIL and saves errno.
            if unsafe { majit_rlib::rposix::c_futimens(fd, times.as_ptr()) } < 0 {
                return Err(io_err(
                    std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                    "",
                ));
            }
            return Ok(pyre_object::w_none());
        }
        #[allow(unreachable_code)]
        {
            let _ = (fd, now, access, modified);
            Err(crate::PyError::not_implemented(
                "utime: fd is unavailable on this platform",
            ))
        }
    }

    /// `do_utimens` (`interp_posix.py`) writes `UTIME_NOW` over both
    /// nanosecond fields when the caller named no time, rather than reading a
    /// time off its own clock and asking for that one. The two are different
    /// requests: `UTIME_NOW` on both stamps is granted to anyone the file is
    /// writable to, while naming a timestamp asks for ownership, so a writable
    /// descriptor onto someone else's file answers `utime(fd)` and refuses
    /// `utime(fd, ns=(now, now))` with EPERM.
    #[cfg(all(unix, not(feature = "sandbox")))]
    fn timespec_of(t: UTime, now: bool) -> libc::timespec {
        libc::timespec {
            tv_sec: t.sec as libc::time_t,
            tv_nsec: if now {
                libc::UTIME_NOW as _
            } else {
                t.nsec as _
            },
        }
    }

    // PyPy `interp_posix.utime` → rposix `utimensat`/`SetFileTime`.  `times` is a
    // `(atime, mtime)` pair in seconds; `ns` the same pair in integer
    // nanoseconds; the two are mutually exclusive.  Both `None` means "now".
    fn utime_impl(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (pos, mut kwargs) = crate::builtins::split_builtin_kwargs(args);
        if pos.is_empty() {
            return Err(crate::PyError::type_error(
                "utime() missing required argument 'path' (pos 1)",
            ));
        }
        if pos.len() > 2 {
            return Err(crate::PyError::type_error(format!(
                "utime() takes from 1 to 2 positional arguments but {} were given",
                pos.len()
            )));
        }
        // `interp_posix.py` puts `__kwonly__` after `w_times`, so `times`
        // is the one argument here a caller may spell either way.
        crate::builtins::kwarg_reject_unknown(
            kwargs,
            &["times", "ns", "dir_fd", "follow_symlinks"],
            "utime",
        )?;
        let mut w_times = crate::builtins::kwarg_get(kwargs, "times");
        if w_times.is_some() && pos.len() > 1 {
            return Err(crate::PyError::type_error(
                "utime() got multiple values for argument 'times'",
            ));
        }
        // interp_posix.py `path_or_fd(allow_fd=rposix.HAVE_FUTIMENS or
        // rposix.HAVE_FUTIMES)`.
        let npos = pos.len();
        let roots = pyre_object::gc_roots::push_roots();
        let base = roots.publish(&[
            kwargs.unwrap_or(pyre_object::PY_NULL),
            w_times.unwrap_or(pyre_object::PY_NULL),
        ]);
        let pos_base = roots.publish(pos);
        roots.normalize(base, 2 + npos);
        let path =
            crate::gateway::fsencode_path_or_fd_w(roots.get(pos_base), "utime", HAVE_FUTIMENS);
        let w = roots.get(base);
        kwargs = if w.is_null() { None } else { Some(w) };
        let w = roots.get(base + 1);
        w_times = if w.is_null() { None } else { Some(w) };
        let mut pos_buf = vec![pyre_object::PY_NULL; npos];
        pyre_object::gc_roots::shadow_stack_copy_range(pos_base, &mut pos_buf);
        let path = path?;

        let present = |v: PyObjectRef| (!unsafe { pyre_object::is_none(v) }).then_some(v);
        let mut times = pos_buf.get(1).copied().or(w_times).and_then(present);
        let mut ns = crate::builtins::kwarg_get(kwargs, "ns").and_then(present);
        let follow_symlinks = match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
            Some(v) => {
                let roots = pyre_object::gc_roots::push_roots();
                let base = roots.pin_roots(&[
                    kwargs.unwrap_or(pyre_object::PY_NULL),
                    ns.unwrap_or(pyre_object::PY_NULL),
                    times.unwrap_or(pyre_object::PY_NULL),
                ]);
                let r = crate::baseobjspace::is_true(v);
                let w = roots.get(base);
                kwargs = if w.is_null() { None } else { Some(w) };
                let w = roots.get(base + 1);
                ns = if w.is_null() { None } else { Some(w) };
                let w = roots.get(base + 2);
                times = if w.is_null() { None } else { Some(w) };
                drop(roots);
                r?
            }
            None => true,
        };
        let dir_fd = match crate::builtins::kwarg_get(kwargs, "dir_fd").and_then(present) {
            // interp_posix.py types `dir_fd` as `DirFD(...)`, whose
            // `unwrap` is `_unwrap_dirfd` (:274-278).
            Some(v) => {
                let roots = pyre_object::gc_roots::push_roots();
                let base = roots.pin_roots(&[
                    ns.unwrap_or(pyre_object::PY_NULL),
                    times.unwrap_or(pyre_object::PY_NULL),
                ]);
                let r = unwrap_fd(v, "integer or None");
                let w = roots.get(base);
                ns = if w.is_null() { None } else { Some(w) };
                let w = roots.get(base + 1);
                times = if w.is_null() { None } else { Some(w) };
                drop(roots);
                Some(r?)
            }
            None => None,
        };

        let unpack_two =
            |obj: PyObjectRef, what: &str| -> Result<(PyObjectRef, PyObjectRef), crate::PyError> {
                if !unsafe { pyre_object::is_tuple(obj) }
                    || unsafe { pyre_object::w_tuple_len(obj) } != 2
                {
                    // `times` is the argument that also has a `None` spelling
                    // — it is the one whose default means "now" — and its
                    // message names that spelling. `ns` has no such form.
                    let shape = if what == "times" {
                        "either a tuple of two ints or None"
                    } else {
                        "a tuple of two ints"
                    };
                    return Err(crate::PyError::type_error(format!(
                        "utime: '{what}' must be {shape}"
                    )));
                }
                Ok((
                    unsafe { pyre_object::w_tuple_getitem(obj, 0) }.unwrap(),
                    unsafe { pyre_object::w_tuple_getitem(obj, 1) }.unwrap(),
                ))
            };
        /// The out-of-range answer for both spellings of a second.
        ///
        /// `_PyTime_ObjectToDenominator` refuses a value no `time_t` can hold
        /// by overflow rather than by value, and `_PyLong_AsTime_t` gives the
        /// same words for an integer too wide to be one — `(2**200, 0)` and
        /// `(1e30, 0)` answer alike.
        fn time_t_overflow() -> crate::PyError {
            crate::PyError::overflow_error("timestamp out of range for platform time_t")
        }
        // `_PyTime_ObjectToTimespec(..., _PyTime_ROUND_FLOOR)`: the seconds are
        // the floor of the value and the nanoseconds are what is left above
        // that floor, so they stay in `0..1_000_000_000` however negative the
        // time is. `utime(p, (-1.5, -2.5))` is `(-2, 500000000)` and
        // `(-3, 500000000)`, which reads back as `-1_500_000_000` and
        // `-2_500_000_000` nanoseconds.
        let time_from_secs = |mut v: PyObjectRef| -> Result<UTime, crate::PyError> {
            // An integer names its second exactly, so it is read as one
            // rather than through a float that would round the seconds it is
            // too wide to hold.
            if unsafe { pyre_object::is_int_or_long(v) } {
                let sec = crate::builtins::space_index_w(v).map_err(|_| time_t_overflow())?;
                return Ok(UTime { sec, nsec: 0 });
            }
            // Everything else has to name a float. What cannot is not a time
            // at all, and is refused by type: `utime(p, ('a', 'b'))` names the
            // type it was given rather than reporting a failed float parse.
            let f = pyre_object::with_roots!(v => crate::builtins::builtin_float(&[v])).map_err(
                |err| {
                    if err.kind == crate::PyErrorKind::OverflowError {
                        time_t_overflow()
                    } else {
                        crate::PyError::type_error(format!(
                            "argument must be int or float, not {}",
                            crate::type_methods::arg_type_name(v)
                        ))
                    }
                },
            )?;
            let secs = unsafe { pyre_object::w_float_get_value(f) };
            // A NaN has no floor to take, and it is the one non-finite value
            // answered by value rather than by range.
            if secs.is_nan() {
                return Err(crate::PyError::value_error(
                    "Invalid value NaN (not a number)",
                ));
            }
            let floor = secs.floor();
            // The floor of an infinite or too-large value is not a second any
            // clock names; `i64::MIN`/`MAX` are what an `as` cast would answer
            // for both, so the range is checked before the cast rather than
            // read back out of it.
            if !(floor >= -(2f64.powi(63)) && floor < 2f64.powi(63)) {
                return Err(time_t_overflow());
            }
            let mut sec = floor as i64;
            let mut nsec = ((secs - floor) * 1e9).floor() as i64;
            if nsec >= 1_000_000_000 {
                nsec -= 1_000_000_000;
                sec = sec.checked_add(1).ok_or_else(time_t_overflow)?;
            }
            Ok(UTime { sec, nsec })
        };
        let time_from_ns = |v: PyObjectRef| -> Result<UTime, crate::PyError> {
            // `split_py_long_to_s_and_ns` splits with `divmod` before it
            // narrows anything, so a count of nanoseconds too wide for a
            // `time_t` is only refused when the SECOND it names is — `ns=2**80`
            // is a second that fits. Dividing after the narrowing turned away
            // the whole range instead. `divmod` is also what answers for a
            // value that is not a number at all.
            let _roots = pyre_object::gc_roots::push_roots();
            let v_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(v);
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(1_000_000_000));
            let split = crate::baseobjspace::divmod(
                pyre_object::gc_roots::shadow_stack_get(v_slot),
                pyre_object::gc_roots::shadow_stack_get(ns_slot),
            )?;
            let split_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(split);
            let (w_sec, w_nsec) = unsafe {
                (
                    pyre_object::w_tuple_getitem(
                        pyre_object::gc_roots::shadow_stack_get(split_slot),
                        0,
                    ),
                    pyre_object::w_tuple_getitem(
                        pyre_object::gc_roots::shadow_stack_get(split_slot),
                        1,
                    ),
                )
            };
            let (Some(w_sec), Some(w_nsec)) = (w_sec, w_nsec) else {
                return Err(crate::PyError::type_error(
                    "utime: divmod() returned a non-pair",
                ));
            };
            // Both words are already live; publish them together so the
            // first normalize cannot move `w_nsec` before it is a root.
            let sec_slot = pyre_object::gc_roots::pin_roots(&[w_sec, w_nsec]);
            let nsec_slot = sec_slot + 1;
            // Python's own `//` and `%`, so a negative count of nanoseconds
            // lands on the second below it with a positive remainder.
            //
            // Only an integer second can be out of a `time_t`'s range. A
            // quotient that is not one at all — `divmod` answers a float pair
            // for `ns=(1.5, 2.5)` — keeps the conversion's own refusal.
            let sec =
                crate::builtins::space_index_w(pyre_object::gc_roots::shadow_stack_get(sec_slot))
                    .map_err(|err| {
                    if err.kind == crate::PyErrorKind::OverflowError {
                        time_t_overflow()
                    } else {
                        err
                    }
                })?;
            Ok(UTime {
                sec,
                nsec: crate::builtins::space_index_w(pyre_object::gc_roots::shadow_stack_get(
                    nsec_slot,
                ))?,
            })
        };

        // `parse_utime_args` (`interp_posix.py`) answers a "now" flag
        // beside the pair and leaves the pair itself at zero when it is set;
        // each of the calls below is what turns that flag into its own spelling
        // of "now".
        let (now, access, modified) = match (times, ns) {
            (Some(_), Some(_)) => {
                return Err(crate::PyError::value_error(
                    "utime: you may specify either 'times' or 'ns' but not both",
                ));
            }
            (Some(t), None) => {
                let (a, mut m) = unpack_two(t, "times")?;
                let access = pyre_object::with_roots!(m => time_from_secs(a))?;
                (false, access, time_from_secs(m)?)
            }
            (None, Some(n)) => {
                let (a, mut m) = unpack_two(n, "ns")?;
                let access = pyre_object::with_roots!(m => time_from_ns(a))?;
                (false, access, time_from_ns(m)?)
            }
            (None, None) => (true, UTime { sec: 0, nsec: 0 }, UTime { sec: 0, nsec: 0 }),
        };

        if path.is_fd {
            // interp_posix.py:1893-1900 — both modifiers reinterpret a *name*,
            // and a descriptor is not one. 3.14, which the parity suite reads
            // as the oracle, words the first "can't specify dir_fd without
            // matching path" where `interp_posix.py:1895` says "can't specify
            // both dir_fd and fd".
            if dir_fd.is_some() {
                return Err(crate::PyError::value_error(
                    "utime: can't specify dir_fd without matching path",
                ));
            }
            if !follow_symlinks {
                return Err(crate::PyError::value_error(
                    "utime: cannot use fd and follow_symlinks together",
                ));
            }
            return utime_fd(path.as_fd, now, access, modified);
        }

        #[cfg(all(windows, feature = "host_env"))]
        {
            if dir_fd.is_some() || !follow_symlinks {
                return Err(crate::PyError::not_implemented(
                    "utime: dir_fd and follow_symlinks=False are unavailable on this platform",
                ));
            }
            // `rposix.py` reads a clock here when the caller named no time —
            // `GetSystemTime` into both stamps. `SetFileTime` has no word for
            // "now", which is what the `utimensat` arm below spells
            // `UTIME_NOW`; the pair arrives at zero while the flag carries the
            // meaning, so reading it here is what keeps `os.utime(path)` off
            // 1970.
            let (access, modified) = if now {
                let d = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or(std::time::Duration::ZERO);
                let t = UTime {
                    sec: d.as_secs() as i64,
                    nsec: d.subsec_nanos() as i64,
                };
                (t, t)
            } else {
                (access, modified)
            };
            let wide = wide_path(&path.as_bytes)?;
            rustpython_host_env::nt::set_file_times(
                &wide,
                access.sec,
                access.nsec,
                modified.sec,
                modified.nsec,
            )
            .map_err(|error| fs_err_with_filename(error, path.w_path()))?;
            return Ok(pyre_object::w_none());
        }
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                .map_err(|_| crate::PyError::value_error("embedded null character"))?;
            // `rposix.c_utimensat` is the whole name form. It releases the
            // GIL and saves errno. The descriptor the name resolves against
            // is `AT_FDCWD` when the caller named none, and
            // `follow_symlinks=False` is `AT_SYMLINK_NOFOLLOW`.
            let flag = if follow_symlinks {
                0
            } else {
                libc::AT_SYMLINK_NOFOLLOW
            };
            let times = [timespec_of(access, now), timespec_of(modified, now)];
            let error = unsafe {
                majit_rlib::rposix::c_utimensat(
                    dir_fd.unwrap_or(libc::AT_FDCWD),
                    c_path.as_ptr(),
                    times.as_ptr(),
                    flag,
                )
            };
            if error < 0 {
                return Err(io_err_with_filename(
                    std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                    path.w_path(),
                ));
            }
            return Ok(pyre_object::w_none());
        }
        #[allow(unreachable_code)]
        {
            let _ = (now, access, modified, dir_fd, follow_symlinks, &path, pos);
            Err(crate::PyError::not_implemented(
                "utime is unavailable on this platform",
            ))
        }
    }
    crate::__pyre_put_new!(ns_slot, "utime", crate::make_builtin_function("utime", utime_impl));

    // ── posix._path_splitroot(path) → (root, tail) ──
    // Registered only where `sys.platform` is `win32`, the same condition
    // `_bootstrap_external` gates its use on.
    #[cfg(windows)]
    crate::__pyre_put_new!(
        ns_slot,
        "_path_splitroot",
        path_helper_fn(
            "_path_splitroot",
            |args| {
                let (bound, _) = bind_path_args(args, "_path_splitroot", &["path"], 1, &[])?;
                let path = crate::gateway::fsencode_path_named_w(
                    bound[0].expect("path is required"),
                    "_path_splitroot",
                    "path",
                )?
                .as_bytes;
                // Splitting a drive or UNC prefix is a text operation on a
                // Windows path, and both halves are handed back as `str`, so
                // this one stays in the text domain rather than the byte one.
                let path = crate::gateway::fsdecode_filename_wtf8(&path);
                let (root, tail) = split_root(&path);
                let mut fields = pyre_object::gc_roots::RootedItems::new();
                fields.push(pyre_object::w_str_from_wtf8_managed(root.to_wtf8_buf()));
                fields.push(pyre_object::w_str_from_wtf8_managed(tail.to_wtf8_buf()));
                Ok(pyre_object::w_tuple_new(fields.take()))
            },
            "($module, /, path)",
            "Removes everything after the root on Win32.",
        )
    );

    // ── posix._path_splitroot_ex(p) → (drive, root, tail) ──
    // `_Py_skiproot` measures the two prefixes and the three pieces are slices
    // of the one name. This is `ntpath.splitroot` and `posixpath.splitroot`
    // themselves: both import it and fall back to their own Python split only
    // where a build does not carry it.
    crate::__pyre_put_new!(
        ns_slot,
        "_path_splitroot_ex",
        path_helper_fn(
            "_path_splitroot_ex",
            |args| {
                let (bound, _) = bind_path_args(args, "_path_splitroot_ex", &["p"], 1, &[])?;
                let (wide, as_bytes) =
                    nonstrict_wide_path(bound[0].expect("p is required"), "_path_splitroot_ex")?;
                let units = &wide[..wide.len() - 1];
                let (drvsize, rootsize) = skiproot(units, HOST_SEPS);
                let mut fields = pyre_object::gc_roots::RootedItems::new();
                fields.push(wide_result(&units[..drvsize], as_bytes));
                fields.push(wide_result(&units[drvsize..drvsize + rootsize], as_bytes));
                fields.push(wide_result(&units[drvsize + rootsize..], as_bytes));
                Ok(pyre_object::w_tuple_new(fields.take()))
            },
            "($module, /, p)",
            "Split a pathname into drive, root and tail.\n\nThe tail contains \
             anything after the root.",
        )
    );

    // ── posix._path_normpath(path) → path ──
    // `ntpath.normpath` and `posixpath.normpath` themselves, on the same
    // import-or-fall-back terms as `_path_splitroot_ex` above.
    crate::__pyre_put_new!(
        ns_slot,
        "_path_normpath",
        path_helper_fn(
            "_path_normpath",
            |args| {
                let (bound, _) = bind_path_args(args, "_path_normpath", &["path"], 1, &[])?;
                let (mut wide, as_bytes) =
                    nonstrict_wide_path(bound[0].expect("path is required"), "_path_normpath")?;
                let norm_len = normpath_and_size(&mut wide, HOST_SEPS);
                // A name that folds away to nothing names the current
                // directory, which `PyUnicode_FromOrdinal('.')` spells.
                if norm_len == 0 {
                    return Ok(wide_result(&[u16::from(b'.')], as_bytes));
                }
                Ok(wide_result(&wide[..norm_len], as_bytes))
            },
            "($module, /, path)",
            "Normalize path, eliminating double slashes, etc.",
        )
    );

    // ── nt._path_isdir / _path_isfile / _path_islink / _path_isjunction /
    //    _path_exists / _path_lexists ──
    // `path_t(allow_fd=True, suppress_value_error=True)`: a descriptor is
    // tested through its handle, and a name the conversion cannot spell is not
    // an error but a `False` — `_testFileType` and `_testFileExists` both open
    // with `if (path->value_error) return FALSE`. `ntpath` imports the six in
    // one `try`, in place of the `genericpath` predicates that would stat.
    #[cfg(all(windows, feature = "host_env"))]
    {
        crate::__pyre_put_new!(
            ns_slot,
            "_path_isdir",
            path_helper_fn(
                "_path_isdir",
                |args| {
                    let (bound, _) = bind_path_args(args, "_path_isdir", &["s"], 1, &[])?;
                    win_nt::test_file_type(
                        bound[0].expect("s is required"),
                        "_path_isdir",
                        rustpython_host_env::nt::TestType::Directory,
                    )
                },
                "($module, /, s)",
                "Return true if the pathname refers to an existing directory.",
            )
        );
        crate::__pyre_put_new!(
            ns_slot,
            "_path_isfile",
            path_helper_fn(
                "_path_isfile",
                |args| {
                    let (bound, _) = bind_path_args(args, "_path_isfile", &["path"], 1, &[])?;
                    win_nt::test_file_type(
                        bound[0].expect("path is required"),
                        "_path_isfile",
                        rustpython_host_env::nt::TestType::RegularFile,
                    )
                },
                "($module, /, path)",
                "Test whether a path is a regular file",
            )
        );
        crate::__pyre_put_new!(
            ns_slot,
            "_path_islink",
            path_helper_fn(
                "_path_islink",
                |args| {
                    let (bound, _) = bind_path_args(args, "_path_islink", &["path"], 1, &[])?;
                    win_nt::test_file_type(
                        bound[0].expect("path is required"),
                        "_path_islink",
                        rustpython_host_env::nt::TestType::Symlink,
                    )
                },
                "($module, /, path)",
                "Test whether a path is a symbolic link",
            )
        );
        crate::__pyre_put_new!(
            ns_slot,
            "_path_isjunction",
            path_helper_fn(
                "_path_isjunction",
                |args| {
                    let (bound, _) = bind_path_args(args, "_path_isjunction", &["path"], 1, &[])?;
                    win_nt::test_file_type(
                        bound[0].expect("path is required"),
                        "_path_isjunction",
                        rustpython_host_env::nt::TestType::Junction,
                    )
                },
                "($module, /, path)",
                "Test whether a path is a junction",
            )
        );
        crate::__pyre_put_new!(
            ns_slot,
            "_path_exists",
            path_helper_fn(
                "_path_exists",
                |args| {
                    let (bound, _) = bind_path_args(args, "_path_exists", &["path"], 1, &[])?;
                    win_nt::test_file_exists(
                        bound[0].expect("path is required"),
                        "_path_exists",
                        true,
                    )
                },
                "($module, /, path)",
                "Test whether a path exists.  Returns False for broken symbolic links.",
            )
        );
        crate::__pyre_put_new!(
            ns_slot,
            "_getvolumepathname",
            path_helper_fn(
                "_getvolumepathname",
                |args| {
                    let (bound, _) = bind_path_args(args, "_getvolumepathname", &["path"], 1, &[])?;
                    win_nt::_getvolumepathname(&[bound[0].expect("path is required")])
                },
                "($module, /, path)",
                "A helper function for ismount on Win32.",
            )
        );
        crate::__pyre_put_new!(
            ns_slot,
            "_path_isdevdrive",
            path_helper_fn(
                "_path_isdevdrive",
                |args| {
                    let (bound, _) = bind_path_args(args, "_path_isdevdrive", &["path"], 1, &[])?;
                    win_nt::_path_isdevdrive(&[bound[0].expect("path is required")])
                },
                "($module, /, path)",
                "Determines whether the specified path is on a Windows Dev Drive.",
            )
        );
        crate::__pyre_put_new!(
            ns_slot,
            "_path_lexists",
            path_helper_fn(
                "_path_lexists",
                |args| {
                    let (bound, _) = bind_path_args(args, "_path_lexists", &["path"], 1, &[])?;
                    win_nt::test_file_exists(
                        bound[0].expect("path is required"),
                        "_path_lexists",
                        false,
                    )
                },
                "($module, /, path)",
                "Test whether a path exists.  Returns True for broken symbolic links.",
            )
        );
    }

    // ── Windows-only nt calls — moduledef.py `if os.name == 'nt'` block ──
    // ntpath imports these from `nt` behind try/except ImportError, so a build
    // that omits one silently falls back to its pure-Python path implementation.
    #[cfg(all(windows, feature = "host_env"))]
    {
        for (name, func, arity) in [
            (
                "_getfullpathname",
                win_nt::_getfullpathname as crate::gateway::BuiltinCodeFn,
                1u16,
            ),
            ("_getfinalpathname", win_nt::_getfinalpathname, 1),
            ("_findfirstfile", win_nt::_findfirstfile, 1),
            ("_getfileinformation", win_nt::_getfileinformation, 1),
            ("_getdiskusage", win_nt::_getdiskusage, 1),
            ("get_handle_inheritable", win_nt::get_handle_inheritable, 1),
            ("set_handle_inheritable", win_nt::set_handle_inheritable, 2),
            ("_add_dll_directory", win_nt::_add_dll_directory, 1),
            ("_remove_dll_directory", win_nt::_remove_dll_directory, 1),
        ] {
            crate::__pyre_put_new!(
                ns_slot,
                name,
                crate::make_builtin_function_with_arity(name, func, arity)
            );
        }
        crate::__pyre_put_new!(
            ns_slot,
            "_supports_virtual_terminal",
            crate::make_builtin_function_with_arity(
                "_supports_virtual_terminal",
                win_nt::_supports_virtual_terminal,
                0,
            )
        );
    }

    /// Drive an open `DIR*` to its end, handing each real entry (`.` and `..`
    /// left out) to `f` as `(name, d_ino, d_type)` — the `get_name_bytes`,
    /// `get_inode`, and `get_known_type` a `nextentry` yields
    /// (`interp_scandir.py:148-153`).  Returns the errno at the end: `0` for a
    /// clean end, or the failure `readdir` reported.  Does not close `dirp`.
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    fn readdir_collect(dirp: *mut libc::DIR, mut f: impl FnMut(&[u8], i64, u8)) -> i32 {
        loop {
            // `readdir` reports the end of the directory and a failure the
            // same way — a null return. `rposix.c_readdir` is
            // `RFFI_FULL_ERRNO_ZERO`, so errno is cleared before the call and
            // the saved errno is what a null return reports.
            let entry = unsafe { majit_rlib::rposix::c_readdir(dirp) };
            if entry.is_null() {
                return majit_rlib::rposix::get_saved_errno();
            }
            let name = unsafe { std::ffi::CStr::from_ptr((*entry).d_name.as_ptr()) };
            let name = name.to_bytes();
            if name != b"." && name != b".." {
                let ino = unsafe { (*entry).d_ino } as i64;
                let d_type = unsafe { (*entry).d_type };
                f(name, ino, d_type);
            }
        }
    }

    /// Read a directory descriptor's entries through `f`
    /// (`rposix.py` `_listdir`/`fdlistdir`).
    ///
    /// `fdopendir` takes the descriptor over and `closedir` closes it, so the
    /// caller's own is duplicated first — `interp_posix.listdir` spells that
    /// `rposix.dup(fd, inheritable=False)`, which is `c_dup_noninheritable`.
    /// The duplicate shares its file description — and so its directory offset
    /// — with the caller's descriptor, which would be left at the end of the
    /// directory and read as empty next time; `_listdir`'s `rewind=True`
    /// (`rposix.py`) puts it back before the close.
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    fn fd_readdir(fd: i32, f: impl FnMut(&[u8], i64, u8)) -> Result<(), i32> {
        let dup = unsafe { majit_rlib::rposix::c_dup_noninheritable(fd) };
        if dup < 0 {
            return Err(majit_rlib::rposix::get_saved_errno());
        }
        // `rposix.c_fdopendir` releases the GIL and saves errno. Capture
        // that errno before `c_close`, which saves errno of its own.
        let dirp = unsafe { majit_rlib::rposix::c_fdopendir(dup) };
        if dirp.is_null() {
            let errno = majit_rlib::rposix::get_saved_errno();
            unsafe {
                let _ = majit_rlib::rposix::c_close(dup);
            }
            return Err(errno);
        }
        let errno = readdir_collect(dirp, f);
        // `rposix.c_rewinddir` returns void and does not save errno.
        unsafe { majit_rlib::rposix::c_rewinddir(dirp) };
        // `closedir` closes the duplicate, so nothing here outlives the call.
        // `rposix.c_closedir` is `releasegil=False` and its result is dropped.
        unsafe {
            let _ = majit_rlib::rposix::c_closedir(dirp);
        }
        if errno != 0 {
            return Err(errno);
        }
        Ok(())
    }

    /// The names a directory descriptor holds, `.` and `..` left out.
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    fn fdlistdir(fd: i32) -> Result<Vec<Vec<u8>>, i32> {
        let mut names = Vec::new();
        fd_readdir(fd, |name, _ino, _d_type| names.push(name.to_vec()))?;
        Ok(names)
    }

    /// `rposix.listdir`: `c_opendir` then `_listdir` with `rewind=False`.
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    fn path_listdir(path: &[u8]) -> Result<Vec<Vec<u8>>, i32> {
        let c_path = std::ffi::CString::new(path).map_err(|_| libc::EINVAL)?;
        let dirp = unsafe { majit_rlib::rposix::c_opendir(c_path.as_ptr()) };
        if dirp.is_null() {
            return Err(majit_rlib::rposix::get_saved_errno());
        }
        let mut names = Vec::new();
        let errno = readdir_collect(dirp, |name, _ino, _d_type| names.push(name.to_vec()));
        unsafe {
            let _ = majit_rlib::rposix::c_closedir(dirp);
        }
        if errno != 0 {
            Err(errno)
        } else {
            Ok(names)
        }
    }

    // ── posix.listdir(path=".") → list of str ──
    crate::__pyre_put_new!(
        ns_slot,
        "listdir",
        crate::make_builtin_function("listdir", |args| {
            let (bound, _kwargs) = bind_path_args(args, "listdir", &["path"], 0, &[])?;
            // One resolution yields both the path and its bytes-ness, so
            // `__fspath__` runs exactly once. The omitted argument is the same
            // `None` the signature names, which resolves to `"."` there.
            let mut w_arg = bound[0].unwrap_or(pyre_object::w_none());
            let path_roots = pyre_object::gc_roots::push_roots();
            let path_base = path_roots.pin_roots(&[w_arg]);
            let resolved = crate::gateway::fsencode_path_or_fd_nullable_w(
                path_roots.get(path_base),
                "listdir",
                HAVE_FDOPENDIR,
            );
            w_arg = path_roots.get(path_base);
            let resolved = resolved?;
            // A descriptor names no directory to prefix and is not `bytes`, so
            // its names come back as `str` whatever the caller held
            // (`interp_posix.py:1112-1121`).
            #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
            if resolved.is_fd {
                // The descriptor is what named the directory, so it is what
                // names the failure. `c_dup_noninheritable` / `c_fdopendir`
                // release the GIL (`rposix.dup` / `c_fdopendir`).
                let mut w_path = resolved.w_path();
                let names = pyre_object::with_roots!(w_path, w_arg => fdlistdir(resolved.as_fd))
                    .map_err(|errno| errno_err_with_filename(errno, w_path))?;
                // Each name is freshly allocated and the next one allocates
                // again, so they are pinned as they arrive.
                let mut items = pyre_object::gc_roots::RootedItems::new();
                for n in &names {
                    items.push(fs_name_obj(false, n));
                }
                return Ok(pyre_object::w_list_new(items.take()));
            }
            let bytes_mode = unsafe { resolved.is_bytes() };
            let path = resolved.as_bytes.as_slice();
            let w_path = || resolved.w_path();
            #[cfg(feature = "sandbox")]
            {
                let names = crate::host_seam::ops::listdir(path)
                    .map_err(|e| crate::host_seam::seam_os_err_with_filename(e, w_path()))?;
                let mut items = pyre_object::gc_roots::RootedItems::new();
                for n in names {
                    items.push(fs_name_obj(bytes_mode, &n));
                }
                return Ok(pyre_object::w_list_new(items.take()));
            }
            #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
            {
                let mut w_path = w_path();
                let names = pyre_object::with_roots!(w_path, w_arg => path_listdir(path))
                    .map_err(|errno| errno_err_with_filename(errno, w_path))?;
                let mut items = pyre_object::gc_roots::RootedItems::new();
                for n in &names {
                    items.push(fs_name_obj(bytes_mode, n));
                }
                Ok(pyre_object::w_list_new(items.take()))
            }
            #[cfg(all(not(feature = "sandbox"), not(all(unix, feature = "host_env"))))]
            {
                let entries = host_fs::read_dir(path_from_bytes(path).as_ref())
                    .map_err(|e| fs_err_with_filename(e, w_path()))?;
                let mut items = pyre_object::gc_roots::RootedItems::new();
                for entry in entries {
                    let entry = entry.map_err(|e| fs_err_with_filename(e, w_path()))?;
                    let name = entry.file_name();
                    items.push(fs_name_obj(bytes_mode, name.as_encoded_bytes()));
                }
                Ok(pyre_object::w_list_new(items.take()))
            }
        })
    );

    // ── posix.isatty(fd) → bool ──
    crate::__pyre_put_new!(
        ns_slot,
        "isatty",
        crate::make_builtin_function_with_arity(
            "isatty",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let mut w_fd = args[0];
                let fd = pyre_object::with_roots!(w_fd => crate::baseobjspace::c_int_w(w_fd))?;
                #[cfg(feature = "sandbox")]
                {
                    return Ok(pyre_object::w_bool_from(
                        crate::host_seam::ops::isatty(fd).unwrap_or(false),
                    ));
                }
                #[cfg(all(unix, not(feature = "sandbox")))]
                {
                    // `rposix.isatty`: `c_isatty(fd) != 0` (`save_err` is `RFFI_ERR_NONE`).
                    let res = pyre_object::with_roots!(w_fd => unsafe {
                        majit_rlib::rposix::c_isatty(fd)
                    });
                    Ok(pyre_object::w_bool_from(res != 0))
                }
                #[cfg(all(not(unix), not(feature = "sandbox")))]
                Ok(pyre_object::w_bool_from(host_os::isatty(fd)))
            },
            1,
        )
    );

    // ── posix._inputhook() / posix._is_inputhook_installed() ──
    // `PyOS_InputHook` is the seam a C extension (readline, Tk) installs a
    // callback in so the REPL can pump its event loop while it waits for a
    // line.  `_pyrepl`'s console reads both at start-up — `unix_console.py`
    // and `windows_console.py` each answer `input_hook` with
    // `posix._inputhook` when `posix._is_inputhook_installed()` says one is
    // there — so a missing name is an `AttributeError` out of the interactive
    // prompt, not a feature nobody asks for.
    //
    // Nothing in pyre publishes that seam, so the answers are the ones the
    // calls give with no hook installed: `False`, and the 0 the absent hook
    // would have returned.
    crate::__pyre_put_new!(
        ns_slot,
        "_inputhook",
        crate::make_builtin_function_with_arity(
            "_inputhook",
            |_| Ok(pyre_object::w_int_new(0)),
            0,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "_is_inputhook_installed",
        crate::make_builtin_function_with_arity(
            "_is_inputhook_installed",
            |_| Ok(pyre_object::w_bool_from(false)),
            0,
        )
    );

    // ── posix.urandom(n) → bytes ──
    crate::__pyre_put_new!(
        ns_slot,
        "urandom",
        crate::make_builtin_function_with_arity(
            "urandom",
            |args| {
                crate::gateway::check_declared_arity("urandom", 1, args.len())?;
                // `__index__` conversion, so a non-integer is a TypeError
                // instead of a raw field read that asks for an arbitrary
                // number of bytes.
                let mut w_n = args[0];
                let n = pyre_object::with_roots!(w_n => crate::builtins::space_index_w(w_n))?;
                if n < 0 {
                    return Err(crate::PyError::value_error("negative argument not allowed"));
                }
                // A 32-bit target's `usize` is narrower than the `__index__`
                // result, so a size it cannot hold is reported rather than
                // wrapped to a smaller request.
                let n = usize::try_from(n)
                    .map_err(|_| crate::PyError::overflow_error("argument out of range"))?;
                // `interp_posix.urandom` calls `rurandom.urandom`. A size the
                // allocator cannot meet is refused rather than reaching the
                // entropy source.
                #[cfg(all(unix, not(feature = "sandbox")))]
                let buf = {
                    let _ = crate::builtins::try_vec_zeroed(n)?;
                    // `interp_posix.urandom` / `_signal_checker`. Invoked
                    // from `rurandom._getrandom` on Linux `EINTR`.
                    let mut signal_error: Option<crate::PyError> = None;
                    let outcome = {
                        let mut checker = || {
                            match crate::module::signal::interp_signal::checksignals_now() {
                                Ok(()) => true,
                                Err(err) => {
                                    signal_error = Some(err);
                                    false
                                }
                            }
                        };
                        pyre_object::with_roots!(w_n => {
                            majit_rlib::rurandom::urandom(n, Some(&mut checker))
                        })
                    };
                    match outcome {
                        Ok(buf) => buf,
                        Err(errno) => {
                            if let Some(err) = signal_error.take() {
                                return Err(err);
                            }
                            // `wrap_oserror(..., w_exception_class=NotImplementedError)`.
                            return Err(crate::PyError::not_implemented(
                                std::io::Error::from_raw_os_error(errno).to_string(),
                            ));
                        }
                    }
                };
                #[cfg(all(not(unix), not(feature = "sandbox")))]
                let buf = {
                    let mut buf = crate::builtins::try_vec_zeroed(n)?;
                    getrandom::fill(&mut buf).map_err(|e| io_err(std::io::Error::from(e), ""))?;
                    buf
                };
                // Route host entropy through the trusted controller instead of
                // reaching host getrandom directly.
                #[cfg(feature = "sandbox")]
                let buf = crate::host_seam::ops::urandom(n as i64)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                let block = pyre_object::bytesobject::try_alloc_bytes_block(&buf)
                    .ok_or_else(crate::builtins::reservation_failed)?;
                Ok(pyre_object::bytesobject::w_bytes_from_block(block))
            },
            1,
        )
    );
    // os.terminal_size — structseq (columns, lines).
    fn make_terminal_size(cols: i64, lines: i64) -> pyre_object::PyObjectRef {
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(pyre_object::w_int_new(cols));
        fields.push(pyre_object::w_int_new(lines));
        crate::_structseq::new_instance(super::terminal_size_seq_type(), fields.take())
    }
    crate::__pyre_put_new!(ns_slot, "terminal_size", super::terminal_size_seq_type());
    crate::__pyre_put_new!(ns_slot, "statvfs_result", super::statvfs_result_seq_type());
    crate::__pyre_put_new!(ns_slot, "times_result", super::times_result_seq_type());
    crate::__pyre_put_new!(ns_slot, "uname_result", super::uname_result_seq_type());

    // `interp_posix.device_encoding`: only a terminal has one. UTF-8 mode
    // answers "utf-8"; otherwise the active locale's `nl_langinfo(CODESET)`,
    // or None when that is empty. The Windows arm (console code pages) is
    // registered in the `host_env` block below.
    #[cfg(all(not(windows), not(feature = "sandbox")))]
    crate::__pyre_put_new!(
        ns_slot,
        "device_encoding",
        crate::make_builtin_function_with_arity(
            "device_encoding",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "device_encoding() requires 1 argument",
                    ));
                }
                let fd = crate::baseobjspace::c_int_w(args[0])?;
                if !crate::importing::host::os::isatty(fd) {
                    return Ok(pyre_object::w_none());
                }
                if crate::importing::utf8_mode_flag() != 0 {
                    return Ok(pyre_object::w_str_new_managed("utf-8"));
                }
                #[cfg(feature = "host_env")]
                let codeset = rustpython_host_env::locale::nl_langinfo_codeset();
                // host_env owns nl_langinfo; without it there is no locale to ask.
                #[cfg(not(feature = "host_env"))]
                let codeset: Option<Vec<u8>> = None;
                match codeset {
                    Some(codeset) if !codeset.is_empty() => Ok(pyre_object::w_str_new_managed(
                        &String::from_utf8_lossy(&codeset),
                    )),
                    _ => Ok(pyre_object::w_none()),
                }
            },
            1,
        )
    );

    // ── posix.get_terminal_size(fd=1) → os.terminal_size(columns, lines) ──
    // The size belongs to the terminal the descriptor names, read through
    // `ioctl(TIOCGWINSZ)` or `GetConsoleScreenBufferInfo`.  A descriptor that
    // names no terminal has no size, which is reported rather than answered
    // with a guess — `shutil.get_terminal_size` is where the guess lives, and
    // it is reached by catching this.  Stubbed under sandbox, so the real body
    // is compiled out.
    #[cfg(not(feature = "sandbox"))]
    crate::__pyre_put_new!(
        ns_slot,
        "get_terminal_size",
        crate::make_builtin_function("get_terminal_size", |args| {
            // `($module, fd=<unrepresentable>, /)` — the descriptor is
            // positional-only, so `fd=1` is a keyword this entry point does
            // not take rather than a binding.
            // A keyword is refused against the module-qualified name, and this
            // module answers to `nt` on Windows.
            let qualname = if cfg!(windows) {
                "nt.get_terminal_size"
            } else {
                "posix.get_terminal_size"
            };
            let (bound, _kwargs) =
                bind_posonly_args(args, "get_terminal_size", qualname, 1, 0, &[])?;
            let fd = match bound[0] {
                Some(w) => crate::baseobjspace::c_int_w(w)?,
                None => 1,
            };
            #[cfg(unix)]
            {
                // `interp_posix._get_terminal_size`: Unix arm is
                // `rposix.c_ioctl_voidp(fd, rposix.TIOCGWINSZ, winsize)`;
                // nonzero is `exception_from_saved_errno`. Columns/lines are
                // `ws_col`/`ws_row`.
                let mut winsize: libc::winsize = unsafe { core::mem::zeroed() };
                let failed = if let Some(mut w_fd) = bound[0] {
                    pyre_object::with_roots!(w_fd => unsafe {
                        majit_rlib::rposix::c_ioctl_voidp(
                            fd,
                            majit_rlib::rposix::TIOCGWINSZ as majit_rlib::rffi::UINT,
                            &mut winsize as *mut libc::winsize as majit_rlib::rffi::VOIDP,
                        )
                    })
                } else {
                    unsafe {
                        majit_rlib::rposix::c_ioctl_voidp(
                            fd,
                            majit_rlib::rposix::TIOCGWINSZ as majit_rlib::rffi::UINT,
                            &mut winsize as *mut libc::winsize as majit_rlib::rffi::VOIDP,
                        )
                    }
                };
                if failed != 0 {
                    return Err(errno_err(majit_rlib::rposix::get_saved_errno(), ""));
                }
                Ok(make_terminal_size(
                    winsize.ws_col as i64,
                    winsize.ws_row as i64,
                ))
            }
            #[cfg(all(windows, feature = "host_env"))]
            {
                let handle = rustpython_host_env::nt::handle_from_fd(fd);
                let (columns, lines) = rustpython_host_env::nt::get_terminal_size_handle(handle)
                    .map_err(|e| fs_err_with_filename(e, pyre_object::PY_NULL))?;
                Ok(make_terminal_size(columns as i64, lines as i64))
            }
            // A target with neither call has no terminal to measure.
            #[cfg(not(any(unix, all(windows, feature = "host_env"))))]
            {
                let _ = fd;
                Ok(make_terminal_size(80, 24))
            }
        })
    );
    // os.fspath() — posixmodule.c posix_fspath / PyOS_FSPath.
    crate::__pyre_put_new!(
        ns_slot,
        "fspath",
        crate::make_builtin_function_with_arity(
            "fspath",
            |args| super::fspath(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        )
    );
    // os.stat / os.lstat / os.fstat — return stat_result structseq.
    // PyPy: posixmodule.c posix_do_stat → build_stat_result.
    //
    // The returned object is a tuple subclass with named attributes
    // (st_mode, st_ino, ...). We expose it as a plain instance with
    // attributes so that both `os.stat(p).st_mode` and
    // `os.stat(p)[0]` work.
    // Unix path/descriptor forms fill a `STAT_STRUCT` through
    // `rposix_stat.c_stat` / `c_lstat` / `c_fstat` and read `st_flags`
    // off that struct. Under sandbox the fields arrive over the wire.

    /// A whole timestamp in nanoseconds.
    ///
    /// `i64` nanoseconds run out in 2262, and a file can carry a later time
    /// than that — every Windows FILETIME up to the year 30828 is one. The
    /// product is taken wide so `st_mtime_ns` is the number it is rather than
    /// the wrap `sec * 1_000_000_000` would answer with, and
    /// [`w_time_ns`] hands back an int of whatever width it needs.
    fn whole_ns(sec: i64, nsec: i64) -> i128 {
        sec as i128 * 1_000_000_000 + nsec as i128
    }

    /// The `st_*_ns` field as a Python int, which has no width to run out of.
    fn w_time_ns(ns: i128) -> pyre_object::PyObjectRef {
        match i64::try_from(ns) {
            Ok(n) => pyre_object::w_int_new(n),
            Err(_) => pyre_object::longobject::w_long_new(majit_rlib::rbigint::RBigInt::from(ns)),
        }
    }

    /// `_pystat_l128_from_l64_l64` -- Windows reports a 128-bit file id, whose
    /// upper half is the only part some volumes fill in, so `st_ino` is an int
    /// of whatever width the id needs rather than the low word alone.
    fn w_ino(ino: u128) -> pyre_object::PyObjectRef {
        match i64::try_from(ino) {
            Ok(n) => pyre_object::w_int_new(n),
            Err(_) => pyre_object::longobject::w_long_new(majit_rlib::rbigint::RBigInt::from(ino)),
        }
    }

    struct StatFields {
        mode: i64,
        ino: u128,
        dev: i64,
        nlink: i64,
        uid: i64,
        gid: i64,
        size: i64,
        atime: i64,
        mtime: i64,
        ctime: i64,
        atime_ns: i128,
        mtime_ns: i128,
        ctime_ns: i128,
        #[cfg(unix)]
        blksize: i64,
        #[cfg(unix)]
        blocks: i64,
        #[cfg(unix)]
        rdev: i64,
        /// `st_file_attributes` / `st_reparse_tag` -- the raw Win32 attribute
        /// word and, where it says the name is a reparse point, that point's
        /// tag.
        #[cfg(windows)]
        file_attributes: u32,
        #[cfg(windows)]
        reparse_tag: u32,
        /// `st_birthtime` -- the creation time, which `st_ctime` reports here
        /// as well.
        #[cfg(windows)]
        birthtime: i64,
        #[cfg(windows)]
        birthtime_ns: i128,
    }

    fn stat_result_from_fields(f: &StatFields, st_flags: u32) -> pyre_object::PyObjectRef {
        let (
            st_mode,
            st_ino,
            st_dev,
            st_nlink,
            st_uid,
            st_gid,
            st_size,
            st_atime,
            st_mtime,
            st_ctime,
            st_atime_ns,
            st_mtime_ns,
            st_ctime_ns,
        ) = (
            f.mode, f.ino, f.dev, f.nlink, f.uid, f.gid, f.size, f.atime, f.mtime, f.ctime,
            f.atime_ns, f.mtime_ns, f.ctime_ns,
        );
        #[cfg(unix)]
        let (st_blksize, st_blocks, st_rdev) = (f.blksize, f.blocks, f.rdev);
        // The 10 sequence slots are the integer fields (integer-seconds
        // times at 7..10, named `_integer_*`); the float times, `st_*_ns`,
        // and the platform block/device extras are named-only fields.
        // `_ll_get_st_atime` — float times keep sub-second precision:
        // `float(seconds) + 1e-9 * nanosecond_fraction`, where the
        // fraction is recovered from the full-nanosecond field.
        let st_atime_f = st_atime as f64 + 1e-9 * (st_atime_ns - whole_ns(st_atime, 0)) as f64;
        let st_mtime_f = st_mtime as f64 + 1e-9 * (st_mtime_ns - whole_ns(st_mtime, 0)) as f64;
        let st_ctime_f = st_ctime as f64 + 1e-9 * (st_ctime_ns - whole_ns(st_ctime, 0)) as f64;
        let _roots = pyre_object::gc_roots::push_roots();
        let mut extra_slots: Vec<(&str, usize)> = Vec::new();
        let mut put_extra = |name: &'static str, value: pyre_object::PyObjectRef| {
            extra_slots.push((name, pyre_object::gc_roots::shadow_stack_len()));
            let _ = pyre_object::gc_roots::pin_root(value);
        };
        put_extra("st_atime", pyre_object::w_float_new(st_atime_f));
        put_extra("st_mtime", pyre_object::w_float_new(st_mtime_f));
        put_extra("st_ctime", pyre_object::w_float_new(st_ctime_f));
        put_extra("st_atime_ns", w_time_ns(st_atime_ns));
        put_extra("st_mtime_ns", w_time_ns(st_mtime_ns));
        put_extra("st_ctime_ns", w_time_ns(st_ctime_ns));
        // `build_stat_result` (interp_posix.py): the
        // sub-second remainder of each full-nanosecond timestamp,
        // `value % 1_000_000_000` (non-negative for pre-1970 times).
        put_extra(
            "nsec_atime",
            pyre_object::w_int_new(st_atime_ns.rem_euclid(1_000_000_000) as i64),
        );
        put_extra(
            "nsec_mtime",
            pyre_object::w_int_new(st_mtime_ns.rem_euclid(1_000_000_000) as i64),
        );
        put_extra(
            "nsec_ctime",
            pyre_object::w_int_new(st_ctime_ns.rem_euclid(1_000_000_000) as i64),
        );
        #[cfg(unix)]
        {
            put_extra("st_blksize", pyre_object::w_int_new(st_blksize));
            put_extra("st_blocks", pyre_object::w_int_new(st_blocks));
            put_extra("st_rdev", pyre_object::w_int_new(st_rdev));
        }
        #[cfg(target_os = "macos")]
        put_extra("st_flags", pyre_object::w_int_new(st_flags as i64));
        #[cfg(not(target_os = "macos"))]
        let _ = st_flags;
        #[cfg(windows)]
        {
            let birthtime_f =
                f.birthtime as f64 + 1e-9 * (f.birthtime_ns - whole_ns(f.birthtime, 0)) as f64;
            put_extra(
                "st_file_attributes",
                pyre_object::w_int_new(f.file_attributes as i64),
            );
            put_extra(
                "st_reparse_tag",
                pyre_object::w_int_new(f.reparse_tag as i64),
            );
            put_extra("st_birthtime", pyre_object::w_float_new(birthtime_f));
            put_extra("st_birthtime_ns", w_time_ns(f.birthtime_ns));
        }
        drop(put_extra);
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(pyre_object::w_int_new(st_mode));
        fields.push(w_ino(st_ino));
        fields.push(pyre_object::w_int_new(st_dev));
        fields.push(pyre_object::w_int_new(st_nlink));
        fields.push(pyre_object::w_int_new(st_uid));
        fields.push(pyre_object::w_int_new(st_gid));
        fields.push(pyre_object::w_int_new(st_size));
        fields.push(pyre_object::w_int_new(st_atime));
        fields.push(pyre_object::w_int_new(st_mtime));
        fields.push(pyre_object::w_int_new(st_ctime));
        let extras: Vec<_> = extra_slots
            .into_iter()
            .map(|(name, slot)| (name, pyre_object::gc_roots::shadow_stack_get(slot)))
            .collect();
        crate::_structseq::new_instance_with_extra(
            super::stat_result_seq_type(),
            fields.take(),
            extras,
        )
    }
    /// Build a `stat_result` from the sandbox wire `StatBuf` (sandbox build
    /// only, hence unix-only): the controller delivers the 10 protocol fields
    /// plus integer atime/mtime/ctime; the sub-second and block/device extras
    /// are whatever `StatBuf` carries (zero over the wire). Mirrors the unix
    /// slot/extra layout of `make_stat_result`.
    #[cfg(feature = "sandbox")]
    fn make_stat_result_from_statbuf(st: &crate::host_seam::StatBuf) -> pyre_object::PyObjectRef {
        let st_atime = st.atime;
        let st_mtime = st.mtime;
        let st_ctime = st.ctime;
        let st_atime_ns = whole_ns(st.atime, st.atime_nsec);
        let st_mtime_ns = whole_ns(st.mtime, st.mtime_nsec);
        let st_ctime_ns = whole_ns(st.ctime, st.ctime_nsec);
        let st_atime_f = st_atime as f64 + 1e-9 * (st_atime_ns - whole_ns(st_atime, 0)) as f64;
        let st_mtime_f = st_mtime as f64 + 1e-9 * (st_mtime_ns - whole_ns(st_mtime, 0)) as f64;
        let st_ctime_f = st_ctime as f64 + 1e-9 * (st_ctime_ns - whole_ns(st_ctime, 0)) as f64;
        let _roots = pyre_object::gc_roots::push_roots();
        let mut extra_slots: Vec<(&str, usize)> = Vec::new();
        let mut put_extra = |name: &'static str, value: pyre_object::PyObjectRef| {
            extra_slots.push((name, pyre_object::gc_roots::shadow_stack_len()));
            let _ = pyre_object::gc_roots::pin_root(value);
        };
        put_extra("st_atime", pyre_object::w_float_new(st_atime_f));
        put_extra("st_mtime", pyre_object::w_float_new(st_mtime_f));
        put_extra("st_ctime", pyre_object::w_float_new(st_ctime_f));
        put_extra("st_atime_ns", w_time_ns(st_atime_ns));
        put_extra("st_mtime_ns", w_time_ns(st_mtime_ns));
        put_extra("st_ctime_ns", w_time_ns(st_ctime_ns));
        put_extra(
            "nsec_atime",
            pyre_object::w_int_new(st_atime_ns.rem_euclid(1_000_000_000) as i64),
        );
        put_extra(
            "nsec_mtime",
            pyre_object::w_int_new(st_mtime_ns.rem_euclid(1_000_000_000) as i64),
        );
        put_extra(
            "nsec_ctime",
            pyre_object::w_int_new(st_ctime_ns.rem_euclid(1_000_000_000) as i64),
        );
        put_extra("st_blksize", pyre_object::w_int_new(st.blksize as i64));
        put_extra("st_blocks", pyre_object::w_int_new(st.blocks as i64));
        put_extra("st_rdev", pyre_object::w_int_new(st.rdev as i64));
        #[cfg(target_os = "macos")]
        put_extra("st_flags", pyre_object::w_int_new(st.st_flags as i64));
        drop(put_extra);
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(pyre_object::w_int_new(st.mode as i64));
        fields.push(pyre_object::w_int_new(st.ino as i64));
        fields.push(pyre_object::w_int_new(st.dev as i64));
        fields.push(pyre_object::w_int_new(st.nlink as i64));
        fields.push(pyre_object::w_int_new(st.uid as i64));
        fields.push(pyre_object::w_int_new(st.gid as i64));
        fields.push(pyre_object::w_int_new(st.size as i64));
        fields.push(pyre_object::w_int_new(st_atime));
        fields.push(pyre_object::w_int_new(st_mtime));
        fields.push(pyre_object::w_int_new(st_ctime));
        let extras: Vec<_> = extra_slots
            .into_iter()
            .map(|(name, slot)| (name, pyre_object::gc_roots::shadow_stack_get(slot)))
            .collect();
        crate::_structseq::new_instance_with_extra(
            super::stat_result_seq_type(),
            fields.take(),
            extras,
        )
    }
    /// `os.stat(path, *, dir_fd=None, follow_symlinks=True)` /
    /// `os.lstat(path, *, dir_fd=None)` — `follow_symlinks` is keyword-only,
    /// so `stat` cannot take the fixed-arity carrier that rejects keywords.
    /// The three argument forms are the three arms of `do_stat`
    /// (`interp_posix.py`): an open descriptor as `path` goes to
    /// `fstat`, a `dir_fd`-relative name to `fstatat`, and a bare name to
    /// `stat`/`lstat`.
    fn stat_entry(
        args: &[pyre_object::PyObjectRef],
        default_follow: bool,
    ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        let name = if default_follow { "stat" } else { "lstat" };
        let (pos, mut kwargs) = crate::builtins::split_builtin_kwargs(args);
        let allowed: &[&str] = if default_follow {
            &["path", "dir_fd", "follow_symlinks"]
        } else {
            &["path", "dir_fd"]
        };
        crate::builtins::kwarg_reject_unknown(kwargs, allowed, name)?;
        if pos.len() > 1 {
            return Err(crate::PyError::type_error(format!(
                "{name}() takes at most 1 positional argument ({} given)",
                pos.len()
            )));
        }
        let path = match crate::builtins::bind_pos_or_kw(pos, kwargs, 0, "path", name, 1)? {
            Some(path) => path,
            None => {
                return Err(crate::PyError::type_error(format!(
                    "{name}() missing required argument 'path' (pos 1)"
                )));
            }
        };
        // The three arguments are unwrapped in signature order — `path`,
        // `dir_fd`, `follow_symlinks` (`interp_posix.py:610-614`) — because
        // `gateway.py` applies the unwrap specs in that order and each can
        // both raise and run user code: `__fspath__` for `path`, `__index__`
        // for `dir_fd`, `__bool__` for `follow_symlinks`.
        //
        // interp_posix.py — `stat` takes `path_or_fd(allow_fd=True)`
        // and `lstat` takes `allow_fd=False`, which is also what makes their
        // type errors name different allowed types.
        let roots = pyre_object::gc_roots::push_roots();
        let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
        let path = crate::gateway::fsencode_path_or_fd_w(path, name, default_follow);
        let w = roots.get(base);
        kwargs = if w.is_null() { None } else { Some(w) };
        let path = path?;
        // `stat`/`lstat` type `dir_fd` as `DirFD(rposix.HAVE_FSTATAT)`
        // (`interp_posix.py`), whose `unwrap` is `_unwrap_dirfd`
        // (:274-278).
        let dir_fd = match crate::builtins::kwarg_get(kwargs, "dir_fd")
            .filter(|&v| !unsafe { pyre_object::is_none(v) })
        {
            Some(v) => {
                let roots = pyre_object::gc_roots::push_roots();
                let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
                let fd = unwrap_fd(v, "integer or None");
                let w = roots.get(base);
                kwargs = if w.is_null() { None } else { Some(w) };
                drop(roots);
                let fd = fd?;
                // `DirFD(available=False)` is `_DirFD_Unavailable`
                // (:285-292), which turns a non-default `dir_fd` away while
                // unwrapping — so where the platform has no `fstatat` the
                // answer is this, not the descriptor conflict `do_stat`
                // would reach first.
                if !HAVE_FSTATAT {
                    return Err(dir_fd_unavailable());
                }
                Some(fd)
            }
            None => None,
        };
        let follow_symlinks = match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
            Some(v) => crate::baseobjspace::is_true(v)?,
            None => default_follow,
        };
        // interp_posix.py `do_stat` tests the descriptor first: with one
        // in hand neither other argument has anything to apply to. Only the
        // `follow_symlinks` rejection precedes the platform's dir_fd
        // availability, though — `_DirFD_Unavailable` (`interp_posix.py`,
        // the `!HAVE_FSTATAT` arm above) turns the argument away while
        // unwrapping it, a step earlier than this, so where `fstatat` does not
        // exist a descriptor passed with `dir_fd` reports the platform rather
        // than the conflict.
        if path.is_fd {
            if dir_fd.is_some() {
                // 3.14 words this "can't specify dir_fd without matching
                // path"; `interp_posix.py:639` says "can't specify both
                // dir_fd and fd". The parity suite's oracle is CPython.
                return Err(crate::PyError::value_error(format!(
                    "{name}: can't specify dir_fd without matching path"
                )));
            }
            if !follow_symlinks {
                return Err(crate::PyError::value_error(format!(
                    "{name}: cannot use fd and follow_symlinks together"
                )));
            }
            return fstat_fd(path.as_fd);
        }
        match dir_fd {
            Some(dir_fd) => stat_at(name, &path, dir_fd, follow_symlinks),
            None => stat_path(&path, follow_symlinks),
        }
    }

    /// `rposix_stat.build_stat_result` reads the same fields off the raw
    /// `struct stat` the `*at` calls fill in.
    #[cfg(all(unix, not(feature = "sandbox")))]
    fn stat_fields_from_libc(st: &libc::stat) -> StatFields {
        StatFields {
            mode: st.st_mode as i64,
            ino: st.st_ino as u128,
            dev: st.st_dev as i64,
            nlink: st.st_nlink as i64,
            uid: st.st_uid as i64,
            gid: st.st_gid as i64,
            size: st.st_size,
            atime: st.st_atime,
            mtime: st.st_mtime,
            ctime: st.st_ctime,
            atime_ns: whole_ns(st.st_atime, st.st_atime_nsec),
            mtime_ns: whole_ns(st.st_mtime, st.st_mtime_nsec),
            ctime_ns: whole_ns(st.st_ctime, st.st_ctime_nsec),
            blksize: st.st_blksize as i64,
            blocks: st.st_blocks,
            rdev: st.st_rdev as i64,
        }
    }

    /// `rposix_stat.build_stat_result` over a filled `STAT_STRUCT`.
    #[cfg(all(unix, not(feature = "sandbox")))]
    fn stat_result_from_libc_stat(st: &libc::stat) -> pyre_object::PyObjectRef {
        #[cfg(target_os = "macos")]
        let st_flags = st.st_flags;
        #[cfg(not(target_os = "macos"))]
        let st_flags = 0u32;
        stat_result_from_fields(&stat_fields_from_libc(st), st_flags)
    }

    /// `rposix_stat.stat` / `lstat`: `c_stat` / `c_lstat` into a `STAT_STRUCT`.
    #[cfg(all(unix, not(feature = "sandbox")))]
    fn libc_stat_path(path: &[u8], follow: bool) -> Result<libc::stat, i32> {
        let c_path = std::ffi::CString::new(path).map_err(|_| libc::EINVAL)?;
        let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
        let ret = if follow {
            unsafe { majit_rlib::rposix::c_stat(c_path.as_ptr(), st.as_mut_ptr()) }
        } else {
            unsafe { majit_rlib::rposix::c_lstat(c_path.as_ptr(), st.as_mut_ptr()) }
        };
        if ret != 0 {
            return Err(majit_rlib::rposix::get_saved_errno());
        }
        Ok(unsafe { st.assume_init() })
    }

    /// The Windows counterpart of `stat_fields_from_libc`.  `win32_xstat` and
    /// `fstat` fill the same `StatStruct`, so routing both entry points
    /// through it is what lets a name and a descriptor for one file report one
    /// identity — `st_ino`/`st_dev` were previously reported as 0 from a path,
    /// which made every file look like every other one.
    ///
    /// `st_ctime` carries the creation time, which is the value the field has
    /// always held here; `StatStruct` keeps that under `st_birthtime` and
    /// leaves its own `st_ctime` at zero.
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    fn stat_fields_from_statstruct(st: &rustpython_host_env::fileutils::StatStruct) -> StatFields {
        StatFields {
            mode: st.st_mode as i64,
            // `_pystat_l128_from_l64_l64` -- a volume that numbers its files
            // above 2**64 leaves the low word at zero, which is how every
            // directory on such a volume used to answer with the same id.
            ino: (st.st_ino_high as u128) << 64 | st.st_ino as u128,
            dev: st.st_dev as i64,
            nlink: st.st_nlink as i64,
            uid: st.st_uid as i64,
            gid: st.st_gid as i64,
            size: st.st_size as i64,
            atime: st.st_atime as i64,
            mtime: st.st_mtime as i64,
            ctime: st.st_birthtime as i64,
            atime_ns: whole_ns(st.st_atime as i64, st.st_atime_nsec as i64),
            mtime_ns: whole_ns(st.st_mtime as i64, st.st_mtime_nsec as i64),
            ctime_ns: whole_ns(st.st_birthtime as i64, st.st_birthtime_nsec as i64),
            file_attributes: st.st_file_attributes as u32,
            reparse_tag: st.st_reparse_tag,
            birthtime: st.st_birthtime as i64,
            birthtime_ns: whole_ns(st.st_birthtime as i64, st.st_birthtime_nsec as i64),
        }
    }

    /// `win32_xstat` for a byte path.  `os.stat`, `DirEntry.stat` and
    /// `DirEntry.inode` all go through here so that one file has one identity
    /// however it was reached; reaching only some of them would leave
    /// `entry.inode()` and `os.stat(entry.path).st_ino` disagreeing.
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    fn win_stat_fields(path: &[u8], follow_symlinks: bool) -> std::io::Result<StatFields> {
        let wide =
            widestring::WideCString::from_os_str(&*os_str_from_bytes(path)).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "embedded null character in path",
                )
            })?;
        rustpython_host_env::nt::win32_xstat(&wide, follow_symlinks)
            .map(|st| stat_fields_from_statstruct(&st))
    }

    /// `find_data_to_file_info` followed by `_Py_attribute_data_to_stat` --
    /// the `lstat` a directory walk already knows, read out of the
    /// `WIN32_FIND_DATAW` it reported rather than from a call of its own.
    ///
    /// A find record carries no link count, no file index and no volume, and
    /// `find_data_to_file_info` zeroes the `BY_HANDLE_FILE_INFORMATION` it
    /// fills, so `st_nlink`, `st_ino` and `st_dev` all read `0` on an entry
    /// that came out of `scandir`. `DirEntry.inode` is what goes back to the
    /// name for the identity.
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    fn stat_fields_from_find_data(data: &WinFindData) -> StatFields {
        const IO_REPARSE_TAG_SYMLINK: u32 = rustpython_host_env::nt::IO_REPARSE_TAG_SYMLINK;
        // A `FILETIME` counts 100ns ticks from 1601-01-01.
        const SECS_BETWEEN_EPOCHS: i64 = 11_644_473_600;

        fn filetime_to_time(ticks: u64) -> (i64, i64) {
            let ticks = ticks as i64;
            (
                ticks / 10_000_000 - SECS_BETWEEN_EPOCHS,
                (ticks % 10_000_000) * 100,
            )
        }

        let attrs = data.file_attributes;
        let reparse_tag = if attrs & FileAttributes::REPARSE_POINT.bits() != 0 {
            data.reserved0
        } else {
            0
        };
        let mut mode = if attrs & FileAttributes::DIRECTORY.bits() != 0 {
            S_IFDIR | 0o111
        } else {
            S_IFREG
        };
        mode |= if attrs & FileAttributes::READONLY.bits() != 0 {
            0o444
        } else {
            0o666
        };
        if reparse_tag == IO_REPARSE_TAG_SYMLINK {
            mode = (mode & !S_IFMT) | S_IFLNK;
        }

        let (birthtime, birthtime_nsec) = filetime_to_time(data.creation_ticks);
        let (mtime, mtime_nsec) = filetime_to_time(data.last_write_ticks);
        let (atime, atime_nsec) = filetime_to_time(data.last_access_ticks);
        StatFields {
            mode: mode as i64,
            ino: 0,
            dev: 0,
            nlink: 0,
            uid: 0,
            gid: 0,
            size: data.file_size as i64,
            atime,
            mtime,
            // `st_ctime` is the creation time here, which is the copy
            // `DirEntry_from_find_data` makes across from `st_birthtime`.
            ctime: birthtime,
            atime_ns: whole_ns(atime, atime_nsec),
            mtime_ns: whole_ns(mtime, mtime_nsec),
            ctime_ns: whole_ns(birthtime, birthtime_nsec),
            file_attributes: attrs,
            reparse_tag,
            birthtime,
            birthtime_ns: whole_ns(birthtime, birthtime_nsec),
        }
    }

    /// The `DT_*` an entry's find data decides, so `is_dir`, `is_file` and
    /// `is_symlink` answer from the walk rather than from the name.  A reparse
    /// point that is not a symlink -- a junction, or any other tag -- is the
    /// directory or file its attribute word says it is, which is how
    /// `DirEntry_test_mode` reads `FILE_ATTRIBUTE_DIRECTORY` for everything
    /// its `win32_lstat` did not spell `S_IFLNK`.
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    fn find_data_known_type(fields: &StatFields) -> u8 {
        match fields.mode as u32 & S_IFMT {
            S_IFLNK => DT_LNK,
            S_IFDIR => DT_DIR,
            _ => DT_REG,
        }
    }

    /// `os_scandir_impl`'s Windows arm -- `FindFirstFileW` over the directory
    /// joined to `*.*`.  It is the one directory walk that hands each entry's
    /// `WIN32_FIND_DATAW` back, which is what lets a `DirEntry` answer
    /// `is_dir`, `is_file` and `stat` without returning to the name, and so
    /// keep answering after the name is removed.
    ///
    /// `join_path_filenameW` puts a separator between the directory and the
    /// name unless the directory already ends in one or in a drive's colon,
    /// and it leaves an empty directory empty -- `FindFirstFileW("")` then
    /// fails the way `os.scandir("")` reports.
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    fn win_scandir_each(
        path: &[u8],
        mut each: impl FnMut(&[u8], &[u8], WinFindData),
    ) -> std::io::Result<()> {
        let wide =
            widestring::WideCString::from_os_str(&*os_str_from_bytes(path)).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "embedded null character in path",
                )
            })?;
        let mut scan = rustpython_host_env::nt::scandir(&wide)?;
        while let Some(entry) = scan.next_entry()? {
            if entry.is_dot_or_dotdot() {
                continue;
            }
            let full = scan.entry_path(&entry);
            each(
                entry.name.as_encoded_bytes(),
                full.as_encoded_bytes(),
                WinFindData {
                    file_attributes: entry.file_attributes,
                    reserved0: entry.reserved0,
                    file_size: entry.file_size,
                    creation_ticks: entry.creation_ticks,
                    last_access_ticks: entry.last_access_ticks,
                    last_write_ticks: entry.last_write_ticks,
                },
            );
        }
        Ok(())
    }

    /// Where the `host_env::posix`-backed implementations below are compiled.
    /// Elsewhere those names are the noop placeholders registered near the top
    /// of this function, which cannot serve a descriptor.
    const HOST_POSIX: bool = cfg!(all(unix, feature = "host_env", not(feature = "sandbox")));
    /// The same, for the calls Windows serves through the C runtime.
    const HOST_WINDOWS_CRT: bool =
        cfg!(all(windows, feature = "host_env", not(feature = "sandbox")));

    /// The `HAVE_*` macros `_have_functions` advertises, each spelled as the
    /// condition under which the entry point it names really does take an open
    /// descriptor. `os.py:140-155` reads them into `supports_fd`, so a bit set
    /// where the call still rejects an integer hands the caller a capability it
    /// cannot use.
    /// `rposix.HAVE_FACCESSAT` — what `access` types its `dir_fd` as
    /// (`interp_posix.py`) and what its two flag modifiers are tested
    /// against (`:771-775`). All three of `access`'s modifiers are the one
    /// `faccessat` call, so the same bit carries them: `os.py:117,137,158` read
    /// it into `supports_dir_fd`, `supports_effective_ids` and
    /// `supports_follow_symlinks` alike, and it is the only bit any of those
    /// three reads for `access`.
    const HAVE_FACCESSAT: bool = HOST_POSIX;
    const HAVE_FCHDIR: bool = HOST_POSIX;
    const HAVE_FCHMOD: bool = HOST_POSIX;
    const HAVE_FCHOWN: bool = HOST_POSIX;
    /// `chown` resolves a name against `dir_fd` and honours
    /// `follow_symlinks=False` through the one `fchownat` call
    /// (`interp_posix.py`), which is also how `lchown` is spelled.
    const HAVE_FCHOWNAT: bool = HOST_POSIX;
    const HAVE_FPATHCONF: bool = HOST_POSIX;
    const HAVE_FSTATVFS: bool = HOST_POSIX;
    /// Windows serves this one too — `_chsize_s` behind `os.ftruncate` — so
    /// `os.truncate` belongs in `supports_fd` on both.
    const HAVE_FTRUNCATE: bool = HOST_POSIX || HOST_WINDOWS_CRT;
    /// `utime` reaches `futimens` and `utimensat` through `libc` rather than
    /// `host_env`, so it needs one less condition than the rest.
    const HAVE_FUTIMENS: bool = cfg!(all(unix, not(feature = "sandbox")));
    /// The name form of `utime` is one `utimensat`, so the same bit carries
    /// both of its modifiers: `dir_fd` is the descriptor the name resolves
    /// against and `follow_symlinks=False` is `AT_SYMLINK_NOFOLLOW`.
    const HAVE_UTIMENSAT: bool = cfg!(all(unix, not(feature = "sandbox")));
    /// `rposix.HAVE_FSTATAT` — what `DirFD` is parameterised on
    /// (`interp_posix.py`). `stat_at` calls `fstatat` through `libc`.
    const HAVE_FSTATAT: bool = cfg!(all(unix, not(feature = "sandbox")));
    /// `rposix.HAVE_FCHMODAT` — what `chmod` types its `dir_fd` as
    /// (`interp_posix.py`), and what `_chmod_path` calls to honour either
    /// modifier. `os.py:118` reads it as `chmod` honouring `dir_fd`; `os.py:179`
    /// deliberately does *not* read it for `follow_symlinks`, because a host can
    /// have `fchmodat` and still not honour `AT_SYMLINK_NOFOLLOW`.
    const HAVE_FCHMODAT: bool = cfg!(all(unix, not(feature = "sandbox")));
    /// `os.py:183` reads this as `chmod` honouring `follow_symlinks`, which is
    /// the `AT_SYMLINK_NOFOLLOW` arm of that same `fchmodat`. It is a narrower
    /// claim than `HAVE_FCHMODAT`: `os.py:159-177` records that where a host's
    /// `lchmod` is a stub returning ENOTSUP, the flag does not work either, so
    /// only the hosts that carry a working `lchmod` may say it — and those are
    /// exactly the ones `os.lchmod` is registered on below.
    const HAVE_LCHMOD: bool = HOST_POSIX && BSD_FLAVOURED;
    /// `os.py` reads this as `chflags` honouring `follow_symlinks`, which
    /// is the `lchflags` arm of the pair. `chflags` is a BSD interface, so the
    /// two names exist on exactly the hosts this is true for — and where they
    /// do not, `shutil.copystat`'s `lookup("chflags")` finds nothing and skips
    /// the flags rather than believing a stub that copied none.
    const HAVE_LCHFLAGS: bool = HOST_POSIX && BSD_FLAVOURED;
    /// Where `lchmod` and `chflags` are the host's own calls rather than stubs
    /// that report ENOTSUP — the platform half of the two bits above.
    const BSD_FLAVOURED: bool = cfg!(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "dragonfly",
    ));
    /// `rposix.HAVE_MKNODAT` — what `mknod` types its `dir_fd` as
    /// (`interp_posix.py`). Registered beside `mkfifo`, so it carries the
    /// same condition.
    const HAVE_MKNODAT: bool = HOST_POSIX;
    /// `rposix.HAVE_OPENAT` — what `open` types its `dir_fd` as
    /// (`interp_posix.py`). `openat` is reached through `libc`, so it needs
    /// no `host_env`; the Windows arm serves the name through `_wopen` and
    /// resolves nothing against a descriptor.
    const HAVE_OPENAT: bool = cfg!(all(unix, not(feature = "sandbox")));
    /// `rposix.HAVE_MKDIRAT` — `mkdir`'s (`interp_posix.py`).
    const HAVE_MKDIRAT: bool = cfg!(all(unix, not(feature = "sandbox")));
    /// `rposix.HAVE_UNLINKAT`. `os.py:131-132` reads it twice, for `unlink` and
    /// for `rmdir`, which are the same `unlinkat` told apart by `AT_REMOVEDIR`
    /// (`rposix.py`) — so the two cannot be advertised apart.
    const HAVE_UNLINKAT: bool = cfg!(all(unix, not(feature = "sandbox")));
    /// `rposix.HAVE_MKFIFOAT` — `mkfifo`'s (`interp_posix.py`). Narrower
    /// than the three above only because `os.mkfifo` itself is registered on
    /// the `host_env` POSIX builds; elsewhere the name is a noop placeholder
    /// that resolves nothing.
    const HAVE_MKFIFOAT: bool = HOST_POSIX;
    /// `rposix.HAVE_LINKAT` — what `link` takes its `src_dir_fd`/`dst_dir_fd`
    /// from. Its `linkat` call sits with the rest of the `host_env`-backed
    /// entry points, so it carries their condition and not `HAVE_FSTATAT`'s.
    const HAVE_LINKAT: bool = HOST_POSIX;
    /// `rposix.HAVE_READLINKAT` — what `readlink` types its `dir_fd` as.
    const HAVE_READLINKAT: bool = cfg!(all(unix, not(feature = "sandbox")));
    /// `rposix.HAVE_RENAMEAT` — what `rename`/`replace` take `src_dir_fd` and
    /// `dst_dir_fd` as.
    const HAVE_RENAMEAT: bool = cfg!(all(unix, not(feature = "sandbox")));
    /// `rposix.HAVE_SYMLINKAT` — what `symlink` types its `dir_fd` as.
    const HAVE_SYMLINKAT: bool = cfg!(all(unix, not(feature = "sandbox")));
    /// `os.py` reads this as `listdir` and `scandir` taking a
    /// descriptor, which `fdlistdir` serves through `fdopendir`. `readdir`
    /// reports the end of a directory and a failure alike, so reading it apart
    /// needs the errno seam the host layer wraps — hence `HOST_POSIX` and not
    /// `HAVE_FSTATAT`'s weaker condition, which `HOST_POSIX` implies anyway for
    /// the `fstatat` a descriptor's entries stat themselves with.
    const HAVE_FDOPENDIR: bool = HOST_POSIX;
    /// `os.py:118` reads this as `lstat` honouring `dir_fd`, which is the same
    /// `fstatat` `stat` resolves one with — so the two cannot be advertised
    /// apart. `os.py:189` reads it a second time as `stat` honouring
    /// `follow_symlinks`; where this bit is false, `MS_WINDOWS` carries that
    /// second claim instead (`os.py:192`).
    const HAVE_LSTAT: bool = HAVE_FSTATAT;
    /// `_WIN32` (`interp_posix.py`). `os.py` reads it as `chmod`
    /// taking a descriptor (`:143`), and as `chmod` and `stat` honouring
    /// `follow_symlinks` (`:184`, `:192`) — all three of which the C runtime
    /// block below serves, and the noop placeholders do not.
    const MS_WINDOWS: bool = HOST_WINDOWS_CRT;

    /// `_DirFD_Unavailable` (`interp_posix.py`) names the argument and
    /// not the call it was passed to, which is also how
    /// `dir_fd_unavailable` reports it.
    fn dir_fd_unavailable() -> crate::PyError {
        crate::PyError::not_implemented("dir_fd unavailable on this platform")
    }

    /// `argument_unavailable` (`interp_posix.py`) — a modifier this
    /// platform has no call to apply, named together with the entry point that
    /// was asked to apply it.
    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
    fn argument_unavailable(funcname: &str, arg: &str) -> crate::PyError {
        crate::PyError::not_implemented(format!("{funcname}: {arg} unavailable on this platform"))
    }

    /// `os.link` takes its two names positionally and everything else by
    /// keyword, so a third positional argument is a `src_dir_fd` that would
    /// otherwise be dropped on the floor.
    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
    fn link_positional(args: &[pyre_object::PyObjectRef]) -> Result<(), crate::PyError> {
        if args.len() == 2 {
            return Ok(());
        }
        Err(crate::PyError::type_error(format!(
            "link() takes exactly 2 positional arguments ({} given)",
            args.len()
        )))
    }

    /// `do_stat` (`interp_posix.py`) resolves a name against an open
    /// directory descriptor with `fstatat`, where `AT_SYMLINK_NOFOLLOW`
    /// carries `follow_symlinks=False`. An absolute name ignores `dir_fd`,
    /// which is why the caller does not have to test for one.
    fn stat_at(
        name: &str,
        path: &crate::gateway::FsEncodedPath,
        dir_fd: i32,
        follow_symlinks: bool,
    ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                .map_err(|_| crate::PyError::value_error("embedded null character"))?;
            let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
            let flags = if follow_symlinks {
                0
            } else {
                libc::AT_SYMLINK_NOFOLLOW
            };
            // `rposix_stat.c_fstatat` releases the GIL and saves errno.
            let ret = unsafe {
                majit_rlib::rposix::c_fstatat(dir_fd, c_path.as_ptr(), st.as_mut_ptr(), flags)
            };
            if ret != 0 {
                let errno = majit_rlib::rposix::get_saved_errno();
                let errno = if errno == 0 { libc::EBADF } else { errno };
                return Err(errno_err_with_filename(errno, path.w_path()));
            }
            let st = unsafe { st.assume_init() };
            return Ok(stat_result_from_libc_stat(&st));
        }
        // Unreachable in practice — `stat_entry` turns a `dir_fd` away at
        // unwrap time wherever `HAVE_FSTATAT` is false — but the arm has to
        // exist for those targets to compile.
        #[allow(unreachable_code)]
        {
            let _ = (path, dir_fd, follow_symlinks);
            Err(dir_fd_unavailable())
        }
    }

    fn stat_path(
        path: &crate::gateway::FsEncodedPath,
        follow_symlinks: bool,
    ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        #[cfg(feature = "sandbox")]
        {
            let buf = if follow_symlinks {
                crate::host_seam::ops::stat(&path.as_bytes)
            } else {
                crate::host_seam::ops::lstat(&path.as_bytes)
            }
            .map_err(|e| crate::host_seam::seam_os_err_with_filename(e, path.w_path()))?;
            return Ok(make_stat_result_from_statbuf(&buf));
        }
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        {
            return match win_stat_fields(&path.as_bytes, follow_symlinks) {
                Ok(fields) => Ok(stat_result_from_fields(&fields, 0)),
                Err(e) => Err(fs_err_with_filename2(
                    e,
                    2,
                    path.w_path(),
                    pyre_object::PY_NULL,
                )),
            };
        }
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            let mut w_path = path.w_path();
            let st = pyre_object::with_roots!(w_path => {
                libc_stat_path(&path.as_bytes, follow_symlinks)
            })
            .map_err(|errno| errno_err_with_filename(errno, w_path))?;
            return Ok(stat_result_from_libc_stat(&st));
        }
        #[allow(unreachable_code)]
        {
            let _ = (path, follow_symlinks);
            Err(crate::PyError::os_error_with_errno(
                libc::ENOSYS,
                "stat".to_string(),
            ))
        }
    }

    // ── posix.scandir(path=".") → ScandirIterator of DirEntry ──
    // `posix_scandir` / `posixmodule.c` DirEntry + ScandirIterator. The
    // entries are read eagerly into a list backing a context-manager
    // iterator so `with os.scandir(p) as it:` and `for e in it:` both work.
    //
    // DirEntry holds `name`/`path` as instance attributes; the type carries
    // is_dir/is_file/is_symlink/is_junction/stat/inode/__fspath__ which stat
    // the stored path on demand.
    type PyObjectRef = pyre_object::PyObjectRef;

    fn dir_entry_path(self_obj: PyObjectRef) -> Result<(PyObjectRef, Vec<u8>), crate::PyError> {
        let de = W_DirEntry::from_obj(self_obj)
            .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?;
        let path = extract_path(de.w_path)?;
        Ok((de.w_path, path))
    }
    fn dir_entry_get_name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // GetSetProperty get: `(descriptor, w_obj)` — the entry is `args[1]`.
        let de = W_DirEntry::from_obj(args[1])
            .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?;
        Ok(de.w_name)
    }
    fn dir_entry_get_path(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let de = W_DirEntry::from_obj(args[1])
            .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?;
        Ok(de.w_path)
    }
    /// The descriptor the `scandir` that produced this entry was handed, or
    /// `-1` where it was given a name instead. An entry from a descriptor
    /// carries no directory in its `path` — `interp_scandir.py` leaves the
    /// prefix empty — so its own stat calls have to resolve the name against
    /// that descriptor rather than against the process's working directory.
    fn dir_entry_dir_fd(self_obj: PyObjectRef) -> Result<i32, crate::PyError> {
        let de = W_DirEntry::from_obj(self_obj)
            .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?;
        Ok(de.dir_fd)
    }
    /// `rposix_stat.fstatat(name, dirfd, follow)` — the call
    /// `interp_scandir.py:252-254,297-299` makes for exactly that reason. The
    /// errno comes back raw because the entry's own `path` is what names the
    /// failure (`:328` `wrap_oserror2(…, self.fget_path(space))`), and only the
    /// callers that report one hold it.
    #[cfg(all(unix, not(feature = "sandbox")))]
    fn dir_entry_stat_at(
        name: &[u8],
        dir_fd: i32,
        follow_symlinks: bool,
    ) -> Result<libc::stat, i32> {
        let c_name = std::ffi::CString::new(name).map_err(|_| libc::EINVAL)?;
        let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
        let flags = if follow_symlinks {
            0
        } else {
            libc::AT_SYMLINK_NOFOLLOW
        };
        // `rposix_stat.c_fstatat` releases the GIL and saves errno.
        let ret = unsafe {
            majit_rlib::rposix::c_fstatat(dir_fd, c_name.as_ptr(), st.as_mut_ptr(), flags)
        };
        if ret != 0 {
            let errno = majit_rlib::rposix::get_saved_errno();
            return Err(if errno == 0 { libc::EBADF } else { errno });
        }
        Ok(unsafe { st.assume_init() })
    }
    fn dir_entry_follow(args: &[PyObjectRef]) -> Result<bool, crate::PyError> {
        let (_pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
        match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
            Some(v) => crate::baseobjspace::is_true(v),
            None => Ok(true),
        }
    }
    /// The `S_IFMT` half of a mode, which is all `is_dir`/`is_file`/
    /// `is_symlink` read. The values are the same on every POSIX host, and
    /// spelling them here keeps the two arms below comparable.
    const S_IFMT: u32 = 0o170_000;
    const S_IFDIR: u32 = 0o040_000;
    const S_IFREG: u32 = 0o100_000;
    const S_IFLNK: u32 = 0o120_000;

    /// The `d_type` byte `readdir` reports — the `known_type` half of
    /// `interp_scandir.py`'s `flags`.  `DT_UNKNOWN` (`0`) is the `W_DirEntry`
    /// default, so an entry whose type the host did not report (a non-unix
    /// host, or a filesystem that answers `DT_UNKNOWN`) falls through to the
    /// stat `dir_entry_kind` runs.  Only the three types the tests read need a
    /// name.
    const DT_UNKNOWN: u8 = 0;
    const DT_DIR: u8 = 4;
    const DT_REG: u8 = 8;
    const DT_LNK: u8 = 10;

    fn dir_entry_known_type(self_obj: PyObjectRef) -> u8 {
        W_DirEntry::from_obj(self_obj).map_or(DT_UNKNOWN, |de| de.enum_type as u8)
    }

    /// Answer `is_dir`/`is_file`/`is_symlink` from the enumeration `d_type`
    /// when it decides the question, else `None` to fall through to a stat
    /// (`W_DirEntry.is_dir` / `.is_file` / `.is_symlink`).  A Windows walk
    /// reports the type for every entry, so only a followed query on a symlink
    /// reaches the stat there -- which is the
    /// `need_stat = follow_symlinks && is_symlink` of `DirEntry_test_mode`.
    /// `target` is the `DT_*` the query wants.
    /// An unknown type never decides.  A symlink decides every query but a
    /// followed `is_dir`/`is_file`, which need the target's type instead.
    fn dir_entry_kind_from_type(known: u8, target: u8, follow: bool) -> Option<bool> {
        if known == DT_UNKNOWN {
            None
        } else if known == target {
            Some(true)
        } else if follow && known == DT_LNK {
            None
        } else {
            Some(false)
        }
    }

    /// The file type an entry's name resolves to, or `None` for a name that has
    /// gone away — `check_mode` (`interp_scandir.py`) answers "no, not
    /// this type" for `ENOENT` alone, on the reasoning that a vanished entry is
    /// better reported as not being of the asked-for kind than as an error.
    /// Every other failure is the caller's to see, named by the entry.
    fn dir_entry_kind(
        mut self_obj: PyObjectRef,
        follow: bool,
    ) -> Result<Option<u32>, crate::PyError> {
        let (mut w_path, path) = pyre_object::with_roots!(self_obj => dir_entry_path(self_obj))?;
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            let dir_fd = pyre_object::with_roots!(w_path => dir_entry_dir_fd(self_obj))?;
            let st = if dir_fd != -1 {
                pyre_object::with_roots!(w_path => dir_entry_stat_at(&path, dir_fd, follow))
            } else {
                pyre_object::with_roots!(w_path => libc_stat_path(&path, follow))
            };
            return match st {
                Ok(st) => Ok(Some(st.st_mode as u32 & S_IFMT)),
                Err(errno) if errno == libc::ENOENT => Ok(None),
                Err(errno) => Err(errno_err_with_filename(errno, w_path)),
            };
        }
        #[cfg(feature = "sandbox")]
        {
            let _ = (w_path, path, follow);
            return Err(crate::host_seam::stub("posix.DirEntry"));
        }
        #[cfg(all(not(feature = "sandbox"), not(unix)))]
        let meta = if follow {
            host_fs::metadata(path_from_bytes(&path).as_ref())
        } else {
            host_fs::symlink_metadata(path_from_bytes(&path).as_ref())
        };
        #[cfg(all(not(feature = "sandbox"), not(unix)))]
        match meta {
            Ok(m) => {
                let ft = m.file_type();
                Ok(Some(if ft.is_dir() {
                    S_IFDIR
                } else if ft.is_symlink() {
                    S_IFLNK
                } else if ft.is_file() {
                    S_IFREG
                } else {
                    0
                }))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(fs_err_with_filename(e, w_path)),
        }
    }
    fn dir_entry_is_dir(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let mut w_self = args[0];
        let follow = pyre_object::with_roots!(w_self => dir_entry_follow(args))?;
        let ans = match dir_entry_kind_from_type(dir_entry_known_type(w_self), DT_DIR, follow) {
            Some(b) => b,
            None => dir_entry_kind(w_self, follow)? == Some(S_IFDIR),
        };
        Ok(pyre_object::w_bool_from(ans))
    }
    fn dir_entry_is_file(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let mut w_self = args[0];
        let follow = pyre_object::with_roots!(w_self => dir_entry_follow(args))?;
        let ans = match dir_entry_kind_from_type(dir_entry_known_type(w_self), DT_REG, follow) {
            Some(b) => b,
            None => dir_entry_kind(w_self, follow)? == Some(S_IFREG),
        };
        Ok(pyre_object::w_bool_from(ans))
    }
    /// `os_DirEntry_is_symlink_impl`.  `is_symlink` never follows, so a known
    /// non-`DT_LNK` type answers `false` and `DT_LNK` answers `true`; only
    /// `DT_UNKNOWN` needs the lstat.
    fn dir_entry_is_symlink_value(args: &[PyObjectRef]) -> Result<bool, crate::PyError> {
        match dir_entry_kind_from_type(dir_entry_known_type(args[0]), DT_LNK, false) {
            Some(b) => Ok(b),
            None => Ok(dir_entry_kind(args[0], false)? == Some(S_IFLNK)),
        }
    }
    fn dir_entry_is_symlink(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        Ok(pyre_object::w_bool_from(dir_entry_is_symlink_value(args)?))
    }
    fn dir_entry_is_junction(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // `DirEntry_is_junction` -- the lstat's reparse tag naming a mount
        // point. A name that is no reparse point at all carries tag 0.
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        {
            // The walk reported that tag, so the answer costs no call and
            // survives the name being removed.
            const IO_REPARSE_TAG_MOUNT_POINT: i64 =
                rustpython_host_env::nt::IO_REPARSE_TAG_MOUNT_POINT as i64;
            let de = W_DirEntry::from_obj(_args[0])
                .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?;
            return Ok(pyre_object::w_bool_from(
                de.enum_tag == IO_REPARSE_TAG_MOUNT_POINT,
            ));
        }
        // POSIX has no junction points.
        #[allow(unreachable_code)]
        Ok(pyre_object::w_bool_from(false))
    }
    fn dir_entry_inode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // `descr_inode` returns the inode `readdir` reported at enumeration when
        // the entry carries one (every unix `scandir` path, name or descriptor),
        // with no stat; `-1` means it has none and the stat paths below answer
        // instead.
        if let Some(de) = W_DirEntry::from_obj(args[0])
            && de.enum_ino != -1
        {
            return Ok(pyre_object::w_int_new(de.enum_ino));
        }
        let mut w_self = args[0];
        let (mut w_path, path) = pyre_object::with_roots!(w_self => dir_entry_path(w_self))?;
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            let dir_fd = pyre_object::with_roots!(w_path => dir_entry_dir_fd(w_self))?;
            let st = if dir_fd != -1 {
                pyre_object::with_roots!(w_path => dir_entry_stat_at(&path, dir_fd, false))
            } else {
                pyre_object::with_roots!(w_path => libc_stat_path(&path, false))
            };
            return match st {
                Ok(st) => Ok(pyre_object::w_int_new(st.st_ino as i64)),
                Err(errno) => Err(errno_err_with_filename(errno, w_path)),
            };
        }
        // `posixmodule.c DirEntry_inode` reads the same file index `os.stat`
        // reports, so the 0 this used to answer with on Windows disagreed with
        // `os.stat(entry.path).st_ino`.
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        {
            let fields =
                win_stat_fields(&path, false).map_err(|e| fs_err_with_filename(e, w_path))?;
            return Ok(w_ino(fields.ino));
        }
        #[cfg(feature = "sandbox")]
        {
            let _ = (w_path, path);
            return Err(crate::host_seam::stub("posix.DirEntry"));
        }
        #[cfg(not(any(feature = "sandbox", unix, all(windows, feature = "host_env"))))]
        {
            let _ = (w_path, path);
            Ok(pyre_object::w_int_new(0))
        }
    }
    /// Fetch a fresh `stat` (`follow=true`) / `lstat` (`follow=false`) result —
    /// the uncached path shared by `dir_entry_stat`'s cache miss.  `dir_fd` is
    /// the descriptor the entry's `scandir` was handed (or `-1`); a real one
    /// resolves the entry's bare `name` through `fstatat`.
    fn dir_entry_fetch_stat(
        mut w_path: PyObjectRef,
        path: &[u8],
        follow: bool,
        dir_fd: i32,
    ) -> Result<PyObjectRef, crate::PyError> {
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            if dir_fd != -1 {
                let st = pyre_object::with_roots!(w_path => {
                    dir_entry_stat_at(path, dir_fd, follow)
                })
                .map_err(|errno| errno_err_with_filename(errno, w_path))?;
                return Ok(stat_result_from_libc_stat(&st));
            }
            let st = pyre_object::with_roots!(w_path => libc_stat_path(path, follow))
                .map_err(|errno| errno_err_with_filename(errno, w_path))?;
            return Ok(stat_result_from_libc_stat(&st));
        }
        #[cfg(not(all(unix, not(feature = "sandbox"))))]
        let _ = dir_fd;
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        {
            return match win_stat_fields(path, follow) {
                Ok(fields) => Ok(stat_result_from_fields(&fields, 0)),
                Err(e) => Err(fs_err_with_filename(e, w_path)),
            };
        }
        #[cfg(feature = "sandbox")]
        {
            let _ = (w_path, path, follow);
            return Err(crate::host_seam::stub("posix.DirEntry"));
        }
        #[cfg(not(any(feature = "sandbox", unix, all(windows, feature = "host_env"))))]
        {
            let _ = (w_path, path, follow);
            Err(crate::PyError::os_error_with_errno(
                libc::ENOSYS,
                "stat".to_string(),
            ))
        }
    }
    /// `posixmodule.c DirEntry_get_stat` caches the built result and hands back
    /// the same object — `follow_symlinks=True` into `w_stat`, `False` into
    /// `w_lstat` — so `entry.stat() is entry.stat()`.  (`interp_scandir.py
    /// descr_stat` caches only the raw stat data and rebuilds a fresh
    /// `build_stat_result` on every call, so under it the result identity
    /// differs; the 3.14 behavior is to cache the object, which is what this
    /// does.)  Only a successful fetch is cached; an error re-raises on each
    /// call.  The entry never moves (`allocate_stable`), so the raw receiver
    /// stays valid across the fetch's allocation.
    fn dir_entry_get_lstat(mut self_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        {
            let de = W_DirEntry::from_obj(self_obj)
                .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?;
            if !de.w_lstat.is_null() {
                return Ok(de.w_lstat);
            }
        }
        // A walk that read the find record answers from it rather than
        // returning to the name, which is what keeps a removed entry
        // answering. The build waits until here so a listing nobody stats
        // never pays for one.
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        {
            let found = W_DirEntry::from_obj(self_obj)
                .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?
                .win32_lstat;
            if let Some(found) = found {
                let result = stat_result_from_fields(&stat_fields_from_find_data(&found), 0);
                let de = W_DirEntry::from_obj(self_obj).ok_or_else(|| {
                    crate::PyError::type_error("expected a 'posix.DirEntry' object")
                })?;
                de.w_lstat = result;
                unsafe { pyre_object::gc_hook::try_gc_write_barrier(self_obj as pyre_object::gc_hook::GCREF) };
                return Ok(result);
            }
        }
        let dir_fd = dir_entry_dir_fd(self_obj)?;
        let (w_path, path) = pyre_object::with_roots!(self_obj => dir_entry_path(self_obj))?;
        let result = pyre_object::with_roots!(self_obj => dir_entry_fetch_stat(w_path, &path, false, dir_fd))?;
        let de = W_DirEntry::from_obj(self_obj)
            .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?;
        de.w_lstat = result;
        unsafe { pyre_object::gc_hook::try_gc_write_barrier(self_obj as pyre_object::gc_hook::GCREF) };
        Ok(result)
    }
    fn dir_entry_stat(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let mut self_obj = args[0];
        let follow = pyre_object::with_roots!(self_obj => dir_entry_follow(args))?;
        if !follow {
            return dir_entry_get_lstat(self_obj);
        }
        {
            let de = W_DirEntry::from_obj(self_obj)
                .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?;
            if !de.w_stat.is_null() {
                return Ok(de.w_stat);
            }
        }
        // `os_DirEntry_stat_impl` returns to the name for a followed stat only
        // where the entry is a symlink; anything else is its own `lstat`, and
        // `get_stat` (interp_scandir.py) decides it the same way. An entry
        // whose walk reported the `lstat` has therefore answered both.
        let receiver = [self_obj];
        let is_symlink =
            pyre_object::with_roots!(self_obj => dir_entry_is_symlink_value(&receiver))?;
        let result = if is_symlink {
            let dir_fd = dir_entry_dir_fd(self_obj)?;
            let (w_path, path) = pyre_object::with_roots!(self_obj => dir_entry_path(self_obj))?;
            pyre_object::with_roots!(self_obj => dir_entry_fetch_stat(w_path, &path, true, dir_fd))?
        } else {
            pyre_object::with_roots!(self_obj => dir_entry_get_lstat(self_obj))?
        };
        let de = W_DirEntry::from_obj(self_obj)
            .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?;
        de.w_stat = result;
        // `result` may be a nursery object stored into the stable entry; join
        // the remembered set so the next minor collection forwards it.
        unsafe { pyre_object::gc_hook::try_gc_write_barrier(self_obj as pyre_object::gc_hook::GCREF) };
        Ok(result)
    }
    fn dir_entry_fspath(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let de = W_DirEntry::from_obj(args[0])
            .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?;
        Ok(de.w_path)
    }
    /// `posixmodule.c DirEntry_repr` — `"<DirEntry %R>"`, so the name is
    /// rendered by its own `repr`.  That keeps `os.scandir(b'.')`'s bytes
    /// name spelled `b'…'` and lets a name whose bytes have no UTF-8 form
    /// keep the lone surrogate `fs_name_obj` decoded it to.
    fn dir_entry_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let name = W_DirEntry::from_obj(args[0])
            .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?
            .w_name;
        let mut out = rustpython_wtf8::Wtf8Buf::new();
        out.push_str("<DirEntry ");
        out.push_wtf8(&unsafe { crate::py_repr_wtf8(name)? });
        out.push_str(">");
        // interp_scandir.py:230 returns `space.newtext(...)`: this rendered
        // value is an ordinary collectable runtime string.
        Ok(pyre_object::w_str_from_wtf8_managed(out))
    }
    /// `interp_scandir.py descr_reduce_ex` — an entry names a live
    /// position in a directory listing, so it refuses to be pickled.  `%T` is
    /// `error.py space.type(value).name`, the qualified typedef name
    /// (`interp_scandir.py 'posix.DirEntry'`), which is what
    /// `w_type_get_name` returns; the flag-driven `reduce_newobj` refusal in
    /// `reduce_protocol_app.py` `reduce_1` spells it with `__name__`.
    fn dir_entry_reduce_ex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let type_name = match crate::typedef::r#type(args[0]) {
            Some(tp) => unsafe { pyre_object::typeobject::w_type_get_name(tp.as_ptr()) },
            None => "posix.DirEntry",
        };
        Err(crate::PyError::type_error(format!(
            "cannot pickle '{type_name}' object"
        )))
    }
    fn dir_entry_type() -> PyObjectRef {
        static CELL: pyre_object::gc_roots::RootedOnceRef =
            pyre_object::gc_roots::RootedOnceRef::new();
        CELL.get_or_init(|| {
            // `interp_scandir.py` names the typedef `'posix.DirEntry'`.
            // `typedef.rs`'s `new_typeobject_with_base_and_layout` turns the
            // leading component of a qualified
            // builtin name into a `__module__` entry, so `type(e).__module__`
            // reports `posix` and every type-name-bearing error message is
            // spelled the way the typedef spells it.
            // The native `W_DirEntry` layout backs the type; `name`/`path` are
            // read-only getset descriptors over the inline fields, so instances
            // carry no `__dict__` (matching a `W_DirEntry` typedef with no
            // `makedict`).
            let tp = crate::typedef::make_builtin_type_with_layout(
                "posix.DirEntry",
                |ns| {
                    let _root_scope = pyre_object::gc_roots::push_roots();
                    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = pyre_object::gc_roots::pin_root(ns);
                    for (name, getter) in [
                        ("name", dir_entry_get_name as crate::gateway::BuiltinCodeFn),
                        ("path", dir_entry_get_path),
                    ] {
                        let getter =
                            crate::gateway::make_builtin_function_with_arity(name, getter, 2);
                        let getter_slot = pyre_object::gc_roots::shadow_stack_len();
                        let _ = pyre_object::gc_roots::pin_root(getter);
                        crate::__pyre_put_new!(
                            ns_slot,
                            name,
                            crate::typedef::make_getset_descriptor_named(
                                pyre_object::gc_roots::shadow_stack_get(getter_slot),
                                name,
                            )
                        );
                    }
                    for (name, f) in [
                        ("is_dir", dir_entry_is_dir as crate::gateway::BuiltinCodeFn),
                        ("is_file", dir_entry_is_file),
                        ("is_symlink", dir_entry_is_symlink),
                        ("is_junction", dir_entry_is_junction),
                        ("inode", dir_entry_inode),
                        ("stat", dir_entry_stat),
                        ("__fspath__", dir_entry_fspath),
                        ("__repr__", dir_entry_repr),
                        ("__reduce_ex__", dir_entry_reduce_ex),
                    ] {
                        crate::__pyre_put_new!(
                            ns_slot,
                            name,
                            crate::make_builtin_function(name, f)
                        );
                    }
                    // CPython 3.14 Modules/posixmodule.c DirEntry_methods —
                    // Py_GenericAlias with METH_CLASS.
                    {
                        let func = crate::make_builtin_function(
                            "__class_getitem__",
                            crate::_pypy_generic_alias::generic_alias_class_getitem,
                        );
                        let func_slot = pyre_object::gc_roots::shadow_stack_len();
                        let _ = pyre_object::gc_roots::pin_root(func);
                        crate::__pyre_put_new!(
                            ns_slot,
                            "__class_getitem__",
                            pyre_object::function::w_classmethod_new(
                                pyre_object::gc_roots::shadow_stack_get(func_slot)
                            )
                        );
                    }
                },
                crate::typedef::w_object(),
                <W_DirEntry as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
            );
            // CPython 3.14 Modules/posixmodule.c creates DirEntryType from an
            // immutable module type spec.
            crate::typedef::mark_cpython_heap_type(tp, true);
            pyre_object::pyobject::set_instantiate(
                unsafe { &*<W_DirEntry as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
                tp,
            );
            // `interp_scandir.py:468-487` declares no `__new__` on the typedef
            // and `:487` sets `acceptable_as_base_class = False`; `typedef.py:55
            // acceptable_as_base_class = '__new__' in rawdict` is the rule, and
            // `typedef.py:754 assert not PyFrame.typedef.acceptable_as_base_class
            // # no __new__` is the same shape `typedef.rs`'s `init_typeobjects`
            // already ports for the `frame` and `traceback` types.
            // `scandir_fn` below allocates entries with `W_DirEntry::
            // allocate_stable`, which never enters `type.__call__`, so the
            // producer is untouched by the instantiation gate.
            unsafe {
                pyre_object::typeobject::w_type_set_disallow_instantiation(tp);
                pyre_object::typeobject::w_type_set_acceptable_as_base_class(tp, false);
            }
            tp
        })
    }

    fn scandir_iter_self(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        Ok(args[0])
    }

    /// Every mutable borrow of the native iterator is derived, used and dropped
    /// inside the serializer, so no two callers ever hold overlapping
    /// references to the same `W_ScandirIterator`.
    fn with_scandir_iter<R>(
        self_obj: PyObjectRef,
        body: impl FnOnce(&mut W_ScandirIterator) -> R,
    ) -> Option<R> {
        let _serialized = SCANDIR_IN_NEXT_SERIALIZER.lock();
        W_ScandirIterator::from_obj(self_obj).map(body)
    }

    fn scandir_iter_mark_closed(self_obj: PyObjectRef) {
        // `W_ScandirIterator._close` clears the state inspected by
        // `_finalize_`, whether closure is explicit or due to exhaustion.
        // `dirp` is already gone on this eager iterator, so the queued
        // finalizer is a no-op; `may_ignore_finalizer` is the same answer
        // the prompt-finalization census needs so `gen.close()` does not
        // collect the whole heap for an already-closed scandir.
        let _ = with_scandir_iter(self_obj, |iterator| {
            iterator.open = false;
        });
        crate::executioncontext::may_ignore_finalizer(self_obj);
    }
    fn scandir_iter_close(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        scandir_iter_mark_closed(args[0]);
        Ok(pyre_object::w_none())
    }
    /// What a `next()` may do, decided in one serialized read of the
    /// iterator's flags.
    enum ScandirStep {
        /// Enumeration is over, either by `close()` or by exhaustion.
        Ended,
        /// Another step holds `_in_next` (interp_scandir.py:133-135).
        InProgress,
        /// This call owns the step and must release it.
        Claimed,
    }

    /// `_in_next` around one enumeration step: `true` on the way in, `false` on
    /// every way out (interp_scandir.py:136,158).  The open flag is read in the
    /// same serialized region, so the answer names a state no concurrent
    /// `close()` can be halfway through.
    fn scandir_iter_claim_next(self_obj: PyObjectRef) -> Option<ScandirStep> {
        with_scandir_iter(self_obj, |iterator| {
            // `W_ScandirIterator.next_w` ends enumeration after `close()`, without
            // yielding entries already buffered in the native owner.
            if !iterator.open {
                return ScandirStep::Ended;
            }
            if iterator.in_next {
                return ScandirStep::InProgress;
            }
            iterator.in_next = true;
            ScandirStep::Claimed
        })
    }

    fn scandir_iter_release_next(self_obj: PyObjectRef) {
        let _ = with_scandir_iter(self_obj, |iterator| {
            iterator.in_next = false;
        });
    }

    /// One enumeration step, with the step already claimed.
    fn scandir_iter_next_entry(self_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let result = with_scandir_iter(self_obj, |iterator| {
            let idx = iterator.index;
            let entries = iterator.entries;
            let len = unsafe { pyre_object::w_list_len(entries) } as i64;
            if idx >= len {
                iterator.open = false;
                return Err(crate::PyError::stop_iteration());
            }
            let Some(item) = (unsafe { pyre_object::w_list_getitem(entries, idx) }) else {
                iterator.open = false;
                return Err(crate::PyError::stop_iteration());
            };
            iterator.index = idx + 1;
            Ok(item)
        })
        .unwrap_or_else(|| {
            Err(crate::PyError::type_error(
                "expected a 'posix.ScandirIterator' object",
            ))
        });
        result
    }

    fn scandir_iter_next(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = args[0];
        let step = scandir_iter_claim_next(self_obj).ok_or_else(|| {
            crate::PyError::type_error("expected a 'posix.ScandirIterator' object")
        })?;
        match step {
            ScandirStep::Ended => return Err(crate::PyError::stop_iteration()),
            // interp_scandir.py:133-135 refuses a step taken while another is
            // in progress, and refuses it through `fail`, which closes the
            // iterator before raising.  Without this two steps read one `index`
            // and hand out the same entry twice.
            ScandirStep::InProgress => {
                scandir_iter_mark_closed(self_obj);
                return Err(crate::PyError::runtime_error(
                    "cannot use ScandirIterator from multiple threads concurrently",
                ));
            }
            ScandirStep::Claimed => {}
        }
        let result = scandir_iter_next_entry(self_obj);
        scandir_iter_release_next(self_obj);
        // interp_scandir.py `fail()` → `_close()` when `nextentry` is exhausted.
        // `_close` is what makes `_finalize_` a no-op; `may_ignore_finalizer`
        // is the same answer the prompt-finalization census needs so a
        // consumer still holding the iterator (`os.fwalk` after its
        // `for entry` loop) does not pay a whole-heap collect on `gen.close()`.
        let result = match result {
            Err(error) => {
                let _stop_roots = pyre_object::gc_roots::push_roots();
                let self_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(self_obj);
                let (stop, error) = error.matches_stop_iteration_keep();
                if stop {
                    scandir_iter_mark_closed(pyre_object::gc_roots::shadow_stack_get(self_slot));
                }
                Err(error)
            }
            other => other,
        };
        result
    }
    fn scandir_iter_is_open(self_obj: PyObjectRef) -> bool {
        with_scandir_iter(self_obj, |iterator| iterator.open).unwrap_or(false)
    }
    fn scandir_iter_del(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let mut self_obj = args[0];
        if !scandir_iter_is_open(self_obj) {
            return Ok(pyre_object::w_none());
        }

        let message = match pyre_object::with_roots!(self_obj => unsafe { crate::display::py_repr_wtf8(self_obj) })
        {
            Ok(repr) => format!("unclosed scandir iterator {}", repr.to_string_lossy()),
            Err(_) => "unclosed scandir iterator".to_string(),
        };
        if let Err(mut error) = pyre_object::with_roots!(self_obj => crate::warn::warn_category_source(&message, "ResourceWarning", 1, self_obj))
        {
            // `W_ScandirIterator._finalize_` reports a warning promoted to an
            // error as unraisable because finalization cannot propagate it.
            pyre_object::with_roots!(self_obj => error.write_unraisable(
                pyre_object::w_none(),
                rustpython_wtf8::Wtf8::new(""),
                self_obj,
            ));
        }
        scandir_iter_mark_closed(self_obj);
        Ok(pyre_object::w_none())
    }
    fn scandir_iter_type() -> PyObjectRef {
        static CELL: pyre_object::gc_roots::RootedOnceRef =
            pyre_object::gc_roots::RootedOnceRef::new();
        CELL.get_or_init(|| {
            // `interp_scandir.py` names the typedef `'posix.ScandirIterator'`.
            let tp = crate::typedef::make_builtin_type_with_layout(
                "posix.ScandirIterator",
                |ns| {
                    let _roots = pyre_object::gc_roots::push_roots();
                    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = pyre_object::gc_roots::pin_root(ns);
                    for (name, f) in [
                        (
                            "__iter__",
                            scandir_iter_self as crate::gateway::BuiltinCodeFn,
                        ),
                        ("__next__", scandir_iter_next),
                        ("__enter__", scandir_iter_self),
                        ("__exit__", scandir_iter_close),
                        ("close", scandir_iter_close),
                        // `interp_scandir.py:172-180` keeps finalization on the
                        // RPython-internal `_finalize_` and publishes no
                        // `__del__`.  3.14 makes `__del__` a real entry in
                        // `posix.ScandirIterator`'s type dict, so it is
                        // published here, with the arity-1 binding below that
                        // makes it callable as an ordinary method.
                        ("__del__", scandir_iter_del),
                    ] {
                        let function = if name == "__del__" {
                            crate::make_builtin_function_with_arity(name, f, 1)
                        } else {
                            crate::make_builtin_function(name, f)
                        };
                        unsafe {
                            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                                pyre_object::gc_roots::shadow_stack_get(ns_slot),
                                name,
                                function,
                            )
                        };
                    }
                },
                crate::typedef::w_object(),
                <W_ScandirIterator as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
            );
            // CPython 3.14 Modules/posixmodule.c creates ScandirIteratorType
            // from an immutable module type spec.
            crate::typedef::mark_cpython_heap_type(tp, true);
            pyre_object::pyobject::set_instantiate(
                unsafe { &*<W_ScandirIterator as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
                tp,
            );
            unsafe { pyre_object::w_type_set_hasuserdel(tp, true) };
            // PyPy's `W_ScandirIterator.typedef` has no `__new__` and disallows
            // subclassing; `scandir_fn` creates instances with
            // `W_ScandirIterator::allocate_stable`.
            unsafe {
                pyre_object::typeobject::w_type_set_disallow_instantiation(tp);
                pyre_object::typeobject::w_type_set_acceptable_as_base_class(tp, false);
            }
            tp
        })
    }

    /// Allocate one `W_DirEntry` and append it to `list` (pinned at `list_slot`
    /// on the shadow stack).  Each string is pinned before the next allocation
    /// so a moving collection during `allocate_stable` forwards it; the entry
    /// is stable but its young strings join the remembered set.  `dir_fd` is the
    /// descriptor the entry resolves its own `name` against, or `-1`.  `enum_ino`
    /// is the `readdir` inode (or `-1` when the enumeration did not carry one).
    /// Join a `scandir` path prefix to an entry name the way
    /// `interp_scandir.py` builds `w_path_prefix`: a separator goes
    /// between them unless the prefix is empty or already ends in one.
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    fn join_dir_name(prefix: &[u8], name: &[u8]) -> Vec<u8> {
        let mut full = Vec::with_capacity(prefix.len() + 1 + name.len());
        full.extend_from_slice(prefix);
        if !prefix.is_empty() && full.last() != Some(&b'/') {
            full.push(b'/');
        }
        full.extend_from_slice(name);
        full
    }
    /// Returns the appended entry so a caller with more to record -- the
    /// Windows walk, which also carries the find record -- writes it without a
    /// second lookup.  The entry is `allocate_stable`, so that address stays
    /// valid for as long as the list holds it.
    fn scandir_push_entry(
        list_slot: usize,
        bytes_mode: bool,
        name: &[u8],
        full: &[u8],
        dir_fd: i32,
        enum_ino: i64,
        enum_type: u8,
    ) -> PyObjectRef {
        let _entry_scope = pyre_object::gc_roots::push_roots();
        let base = pyre_object::gc_roots::shadow_stack_len();
        // The names are pinned before the entry is allocated, so a moving
        // collection during `allocate_stable` forwards them.
        let _ = pyre_object::gc_roots::pin_root(fs_name_obj(bytes_mode, name));
        let _ = pyre_object::gc_roots::pin_root(fs_name_obj(bytes_mode, full));
        let _ = pyre_object::gc_roots::pin_root(W_DirEntry::allocate_stable(W_DirEntry::default()));
        let obj =
            pyre_object::gc_roots::shadow_stack_get(pyre_object::gc_roots::shadow_stack_len() - 1);
        let de = W_DirEntry::from_obj(obj).expect("freshly allocated posix.DirEntry");
        de.w_name = pyre_object::gc_roots::shadow_stack_get(base);
        de.w_path = pyre_object::gc_roots::shadow_stack_get(base + 1);
        de.dir_fd = dir_fd;
        de.enum_ino = enum_ino;
        de.enum_type = enum_type as i32;
        unsafe { pyre_object::gc_hook::try_gc_write_barrier(obj as pyre_object::gc_hook::GCREF) };
        let list = pyre_object::gc_roots::shadow_stack_get(list_slot);
        unsafe { pyre_object::w_list_append(list, obj) };
        obj
    }
    fn scandir_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (bound, _kwargs) = bind_path_args(args, "scandir", &["path"], 0, &[])?;
        // One resolution yields both the path and its bytes-ness, so
        // `__fspath__` runs exactly once. The omitted argument is the same
        // `None` the signature names, which resolves to `"."` there.
        let mut w_arg = bound[0].unwrap_or(pyre_object::w_none());
        let path_roots = pyre_object::gc_roots::push_roots();
        let path_base = path_roots.pin_roots(&[w_arg]);
        let resolved = crate::gateway::fsencode_path_or_fd_nullable_w(
            path_roots.get(path_base),
            "scandir",
            HAVE_FDOPENDIR,
        );
        w_arg = path_roots.get(path_base);
        let resolved = resolved?;
        let bytes_mode = unsafe { resolved.is_bytes() };
        let path = resolved.as_bytes.as_slice();
        let w_path = || resolved.w_path();
        // Initialise the DirEntry type (`set_instantiate`) before allocating
        // any entry of it.
        let _ = dir_entry_type();
        w_arg = path_roots.get(path_base);
        let list = pyre_object::with_roots!(w_arg => pyre_object::w_list_new(Vec::new()));
        // The entries are `allocate_stable` (non-nursery) objects, but a stable
        // allocation can still drive a moving collection over a large listing,
        // so `list` and each per-entry temporary live on the shadow stack.
        let _list_scope = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(list);
        let list_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        // A `scandir(fd)` lists through the descriptor and every entry records
        // it (`-1` for the entries a name produced), so its own stat resolves
        // the bare name against that descriptor.
        #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
        let from_fd = resolved.is_fd.then_some(resolved.as_fd);
        #[cfg(not(all(unix, feature = "host_env", not(feature = "sandbox"))))]
        let from_fd: Option<i32> = None;
        match from_fd {
            #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
            Some(fd) => {
                // `interp_scandir.py:50` leaves the path prefix empty for a
                // descriptor — there is no directory to join — so an entry's
                // `path` is its bare `name`, and a descriptor is not `bytes`,
                // so both come back as `str`. Every entry records the
                // descriptor so its own stat resolves the bare name against it.
                let mut w_path = w_path();
                pyre_object::with_roots!(w_path, w_arg => {
                    fd_readdir(fd, |name, ino, d_type| {
                        scandir_push_entry(list_slot, false, name, name, fd, ino, d_type);
                    })
                })
                .map_err(|errno| errno_err_with_filename(errno, w_path))?;
            }
            // A name is enumerated through `opendir`/`readdir` so each entry
            // carries the `d_ino` and `d_type` the dirent reports
            // (`interp_scandir.py:148-153`): `inode()` answers from `d_ino`
            // and `is_dir`/`is_file`/`is_symlink` from `d_type`, both without
            // a stat.
            #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
            _ => {
                let c_path = std::ffi::CString::new(path)
                    .map_err(|_| crate::PyError::value_error("embedded null byte"))?;
                // `rposix.c_opendir` releases the GIL and saves errno.
                let mut w_path = w_path();
                let dirp = pyre_object::with_roots!(w_path, w_arg => unsafe {
                    majit_rlib::rposix::c_opendir(c_path.as_ptr())
                });
                if dirp.is_null() {
                    return Err(errno_err_with_filename(
                        majit_rlib::rposix::get_saved_errno(),
                        w_path,
                    ));
                }
                let errno = pyre_object::with_roots!(w_path, w_arg => {
                    readdir_collect(dirp, |name, ino, d_type| {
                        let full = join_dir_name(path, name);
                        scandir_push_entry(list_slot, bytes_mode, name, &full, -1, ino, d_type);
                    })
                });
                unsafe {
                    let _ = majit_rlib::rposix::c_closedir(dirp);
                }
                if errno != 0 {
                    return Err(errno_err_with_filename(errno, w_path));
                }
            }
            // `FindFirstFileW` reports each entry's whole `WIN32_FIND_DATAW`,
            // so the type and the `lstat` are both known without a second
            // call and the entry keeps answering after the name is removed.
            // A find record carries no file index, so `inode()` alone goes
            // back to the name.
            #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
            _ => {
                win_scandir_each(path, |name, full, found| {
                    // The find record decides the type and the reparse tag
                    // without allocating; only the `stat_result` waits for a
                    // caller to ask for it.
                    let fields = stat_fields_from_find_data(&found);
                    let obj = scandir_push_entry(
                        list_slot,
                        bytes_mode,
                        name,
                        full,
                        -1,
                        -1,
                        find_data_known_type(&fields),
                    );
                    let de = W_DirEntry::from_obj(obj).expect("freshly appended posix.DirEntry");
                    de.enum_tag = fields.reparse_tag as i64;
                    de.win32_lstat = Some(found);
                })
                .map_err(|e| fs_err_with_filename(e, w_path()))?;
            }
            // No raw `readdir` to read the dirent from (wasm, the sandbox seam,
            // or a build without `host_env`), so `d_type` is unknown and
            // `is_dir` stats; the inode is still free from the dirent on unix.
            #[cfg(feature = "sandbox")]
            _ => {
                return Err(crate::host_seam::stub("posix.scandir"));
            }
            #[cfg(all(
                not(feature = "sandbox"),
                not(all(any(unix, windows), feature = "host_env"))
            ))]
            _ => {
                let entries = host_fs::read_dir(path_from_bytes(path).as_ref())
                    .map_err(|e| fs_err_with_filename(e, w_path()))?;
                for entry in entries {
                    let entry = entry.map_err(|e| fs_err_with_filename(e, w_path()))?;
                    let name = entry.file_name();
                    let full = entry.path().into_os_string();
                    #[cfg(unix)]
                    let enum_ino = {
                        use std::os::unix::fs::DirEntryExt;
                        entry.ino() as i64
                    };
                    #[cfg(not(unix))]
                    let enum_ino = -1i64;
                    scandir_push_entry(
                        list_slot,
                        bytes_mode,
                        name.as_encoded_bytes(),
                        full.as_encoded_bytes(),
                        -1,
                        enum_ino,
                        DT_UNKNOWN,
                    );
                }
            }
        }
        // Initialise the iterator type before allocating its native owner, then
        // pin that stable owner while connecting it to the entries list.
        let _ = scandir_iter_type();
        let _ = pyre_object::gc_roots::pin_root(W_ScandirIterator::allocate_stable(
            W_ScandirIterator::default(),
        ));
        let it_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let it = pyre_object::gc_roots::shadow_stack_get(it_slot);
        let list = pyre_object::gc_roots::shadow_stack_get(list_slot);
        let iterator =
            W_ScandirIterator::from_obj(it).expect("freshly allocated posix.ScandirIterator");
        iterator.entries = list;
        iterator.open = true;
        unsafe { pyre_object::gc_hook::try_gc_write_barrier(it as pyre_object::gc_hook::GCREF) };
        // `StdObjSpace.allocate_instance` immediately queues instances whose
        // type has `hasuserdel`. This native allocation bypasses that helper,
        // so it must register the new iterator explicitly.
        pyre_object::gc_hook::maybe_register_finalizer(it);
        let it = pyre_object::gc_roots::shadow_stack_get(it_slot);
        drop(_list_scope);
        Ok(it)
    }
    crate::__pyre_put_new!(ns_slot, "scandir", crate::make_builtin_function("scandir", scandir_fn));
    crate::__pyre_put_new!(ns_slot, "DirEntry", dir_entry_type());

    // os.uname() — returns structseq (sysname, nodename, release, version, machine).
    // `rposix.c_uname` fills `struct utsname`. POSIX only, the way
    // `HAVE_UNAME` gates it. Its callers read presence as "the POSIX
    // identification is available": `platform.uname` falls back to
    // `sys.platform` on AttributeError, and `sysconfig.get_platform` tests
    // `hasattr(os, 'uname')` directly.
    #[cfg(unix)]
    crate::__pyre_put_new!(
        ns_slot,
        "uname",
        crate::make_builtin_function_with_arity(
            "uname",
            |_| {
                // `rposix.c_uname` releases the GIL and saves errno.
                let mut uts = std::mem::MaybeUninit::<libc::utsname>::zeroed();
                let ret = unsafe { majit_rlib::rposix::c_uname(uts.as_mut_ptr()) };
                if ret < 0 {
                    return Err(io_err(
                        std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                        "",
                    ));
                }
                let uts = unsafe { uts.assume_init() };
                let field = |bytes: &[libc::c_char]| {
                    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                    let raw =
                        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u8>(), end) };
                    String::from_utf8_lossy(raw).into_owned()
                };
                let sysname = field(&uts.sysname);
                let nodename = field(&uts.nodename);
                let release = field(&uts.release);
                let version = field(&uts.version);
                let machine = field(&uts.machine);
                let mut fields = pyre_object::gc_roots::RootedItems::new();
                fields.push(pyre_object::w_str_new_managed(&sysname));
                fields.push(pyre_object::w_str_new_managed(&nodename));
                fields.push(pyre_object::w_str_new_managed(&release));
                fields.push(pyre_object::w_str_new_managed(&version));
                fields.push(pyre_object::w_str_new_managed(&machine));
                Ok(crate::_structseq::new_instance(
                    super::uname_result_seq_type(),
                    fields.take(),
                ))
            },
            0,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "stat",
        crate::gateway::make_builtin_function_with_text_signature(
            "stat",
            |args| stat_entry(args, true),
            "(path, *, dir_fd=None, follow_symlinks=True)",
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "lstat",
        crate::make_builtin_function("lstat", |args| stat_entry(args, false))
    );
    /// `rposix_stat.py fstat`: the descriptor form both `os.fstat` and
    /// `os.stat` with a descriptor answer through, so the two cannot drift.
    fn fstat_fd(fd: i32) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        // `rposix_stat.py fstat` passes the descriptor to `c_fstat`, where
        // `-1` reports EBADF.  The sandbox seam does not, so keep the
        // refusal here.  Windows answers a descriptor it cannot name with
        // the Win32 error instead (`_Py_fstat_noraise` sets
        // ERROR_INVALID_HANDLE itself).
        #[cfg(feature = "sandbox")]
        if fd == -1 {
            return Err(crate::PyError::os_error_with_errno(
                libc::EBADF,
                std::io::Error::from_raw_os_error(libc::EBADF).to_string(),
            ));
        }
        #[cfg(feature = "sandbox")]
        {
            let buf = crate::host_seam::ops::fstat(fd)
                .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
            Ok(make_stat_result_from_statbuf(&buf))
        }
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            // interp_posix.py `fstat`: `rposix_stat.fstat` then
            // `wrap_oserror(..., eintr_retry=True)`.
            loop {
                let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
                let ret = unsafe { majit_rlib::rposix::c_fstat(fd, st.as_mut_ptr()) };
                if ret == 0 {
                    return Ok(stat_result_from_libc_stat(&unsafe { st.assume_init() }));
                }
                let errno = majit_rlib::rposix::get_saved_errno();
                crate::builtins::eintr_retry_with(
                    std::io::Error::from_raw_os_error(errno),
                    |e| {
                        crate::PyError::os_error_with_errno(
                            crate::builtins::io_error_posix_errno(&e, libc::EBADF),
                            format!("{e}"),
                        )
                    },
                )?;
            }
        }
        // `_Py_fstat_noraise`: the descriptor's underlying handle, then
        // `GetFileInformationByHandle` — which is what `File::metadata`
        // is here.  A descriptor with no handle is reported as the
        // Win32 `ERROR_INVALID_HANDLE`, the errno spelling of which is
        // `EBADF`.
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        {
            let invalid_handle = || {
                crate::PyError::os_error_win32_syscall2(
                    rustpython_host_env::nt::ERROR_INVALID_HANDLE_I32,
                    pyre_object::PY_NULL,
                    pyre_object::PY_NULL,
                )
            };
            let borrowed = unsafe { rustpython_host_env::crt_fd::Borrowed::borrow_raw(fd) };
            // Same `StatStruct` the path forms take, for the reason
            // [`win_stat_fields`] states: `std::fs::Metadata` carries no file
            // index or volume serial, so answering from it would report a
            // different identity for the very file `os.stat(path)` just named.
            // It also answers a character device or pipe with the format bits
            // `GetFileType` reports rather than a disk file's.
            match rustpython_host_env::fileutils::fstat(borrowed) {
                Ok(st) => Ok(stat_result_from_fields(
                    &stat_fields_from_statstruct(&st),
                    0,
                )),
                Err(e) => Err(match e.raw_os_error() {
                    Some(winerror) => crate::PyError::os_error_win32_syscall2(
                        winerror,
                        pyre_object::PY_NULL,
                        pyre_object::PY_NULL,
                    ),
                    None => invalid_handle(),
                }),
            }
        }
        #[cfg(not(any(unix, feature = "sandbox", all(windows, feature = "host_env"))))]
        Err(crate::PyError::os_error_with_errno(
            9,
            "fstat unsupported".to_string(),
        ))
    }

    crate::__pyre_put_new!(
        ns_slot,
        "fstat",
        crate::make_builtin_function_with_arity(
            "fstat",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("fstat() missing argument"));
                }
                fstat_fd(crate::baseobjspace::c_int_w(args[0])?)
            },
            1,
        )
    );
    // stat_result type — structseq (tuple subclass). Exported so that
    // `posix.stat_result` and `isinstance(os.stat(p), os.stat_result)` work.
    crate::__pyre_put_new!(ns_slot, "stat_result", super::stat_result_seq_type());
    // `interp_posix.getcwdb` is `os.getcwd()` bytes; Unix `getcwd` is
    // `space.fsdecode(getcwdb(space))`. `rposix.getcwd` returns those bytes.
    #[cfg(all(unix, not(feature = "sandbox")))]
    fn getcwdb_bytes() -> Result<Vec<u8>, crate::PyError> {
        match majit_rlib::rposix::getcwd() {
            Ok(cwd) => Ok(cwd),
            Err(_) => Err(errno_err(majit_rlib::rposix::get_saved_errno(), "")),
        }
    }
    // os.getcwd() — PyPy: posixmodule.c posix_getcwd.
    crate::__pyre_put_new!(
        ns_slot,
        "getcwd",
        crate::make_builtin_function_with_arity(
            "getcwd",
            |_| {
                // `interp_posix.py` is `space.fsdecode(getcwdb(space))`,
                // so the directory's bytes reach Python through the filesystem
                // decoder and a byte with no UTF-8 spelling survives as its
                // surrogate escape. A lossy decode would fold it to U+FFFD,
                // which breaks the `os.fsencode(os.getcwd())` round trip and
                // makes two different directories compare equal.
                #[cfg(feature = "sandbox")]
                {
                    let cwd = crate::host_seam::ops::getcwd()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    Ok(crate::gateway::fsdecode_filename_bytes(&cwd))
                }
                #[cfg(all(unix, not(feature = "sandbox")))]
                {
                    let cwd = getcwdb_bytes()?;
                    Ok(crate::gateway::fsdecode_filename_bytes(&cwd))
                }
                #[cfg(all(not(unix), not(feature = "sandbox")))]
                {
                    #[cfg(feature = "host_env")]
                    {
                        if let Ok(cwd) = host_os::current_dir() {
                            return Ok(crate::gateway::fsdecode_os_str(cwd.as_os_str()));
                        }
                    }
                    Ok(pyre_object::w_str_new_managed(""))
                }
            },
            0,
        )
    );
    // os.getcwdb() — bytes form of getcwd.
    crate::__pyre_put_new!(
        ns_slot,
        "getcwdb",
        crate::make_builtin_function_with_arity(
            "getcwdb",
            |_| {
                #[cfg(feature = "sandbox")]
                {
                    let cwd = crate::host_seam::ops::getcwd()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    Ok(pyre_object::w_bytes_from_bytes(&cwd))
                }
                #[cfg(all(unix, not(feature = "sandbox")))]
                {
                    let cwd = getcwdb_bytes()?;
                    Ok(pyre_object::w_bytes_from_bytes(&cwd))
                }
                #[cfg(all(not(unix), not(feature = "sandbox")))]
                {
                    #[cfg(feature = "host_env")]
                    {
                        if let Ok(cwd) = host_os::current_dir() {
                            return Ok(pyre_object::w_bytes_from_bytes(
                                &crate::gateway::fs_result_bytes(
                                    cwd.as_os_str().as_encoded_bytes(),
                                ),
                            ));
                        }
                    }
                    Ok(pyre_object::w_bytes_from_bytes(b""))
                }
            },
            0,
        )
    );
    // os.getuid / geteuid / getgid / getegid — `rposix.c_getuid` and the three
    // siblings. Each builtin's `#[cfg(feature = "sandbox")]` arm routes
    // through `host_seam::ops` instead.
    // The user and group ids are POSIX's; `nt` has none of the four, and code
    // reads `hasattr(os, 'geteuid')` to decide whether an ownership check is
    // meaningful at all.
    #[cfg(not(windows))]
    crate::__pyre_put_new!(
        ns_slot,
        "getuid",
        crate::make_builtin_function_with_arity(
            "getuid",
            |_| {
                #[cfg(feature = "sandbox")]
                {
                    let v = crate::host_seam::ops::getuid()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    return Ok(pyre_object::w_int_new(v));
                }
                #[cfg(all(unix, not(feature = "sandbox")))]
                unsafe {
                    // `rposix.c_getuid` releases the GIL. It does not save errno.
                    Ok(pyre_object::w_int_new(majit_rlib::rposix::c_getuid() as i64))
                }
                #[cfg(not(any(unix, feature = "sandbox")))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        )
    );
    #[cfg(not(windows))]
    crate::__pyre_put_new!(
        ns_slot,
        "geteuid",
        crate::make_builtin_function_with_arity(
            "geteuid",
            |_| {
                #[cfg(feature = "sandbox")]
                {
                    let v = crate::host_seam::ops::geteuid()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    return Ok(pyre_object::w_int_new(v));
                }
                #[cfg(all(unix, not(feature = "sandbox")))]
                unsafe {
                    Ok(pyre_object::w_int_new(
                        majit_rlib::rposix::c_geteuid() as i64
                    ))
                }
                #[cfg(not(any(unix, feature = "sandbox")))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        )
    );
    #[cfg(not(windows))]
    crate::__pyre_put_new!(
        ns_slot,
        "getgid",
        crate::make_builtin_function_with_arity(
            "getgid",
            |_| {
                #[cfg(feature = "sandbox")]
                {
                    let v = crate::host_seam::ops::getgid()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    return Ok(pyre_object::w_int_new(v));
                }
                #[cfg(all(unix, not(feature = "sandbox")))]
                unsafe {
                    Ok(pyre_object::w_int_new(majit_rlib::rposix::c_getgid() as i64))
                }
                #[cfg(not(any(unix, feature = "sandbox")))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        )
    );
    #[cfg(not(windows))]
    crate::__pyre_put_new!(
        ns_slot,
        "getegid",
        crate::make_builtin_function_with_arity(
            "getegid",
            |_| {
                #[cfg(feature = "sandbox")]
                {
                    let v = crate::host_seam::ops::getegid()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    return Ok(pyre_object::w_int_new(v));
                }
                #[cfg(all(unix, not(feature = "sandbox")))]
                unsafe {
                    Ok(pyre_object::w_int_new(
                        majit_rlib::rposix::c_getegid() as i64
                    ))
                }
                #[cfg(not(any(unix, feature = "sandbox")))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        )
    );
    // os.getpid — `rposix.c_getpid` (`releasegil=False`, saves errno).
    // `rposix.getpid` reports a negative result through `handle_posix_error`.
    // Windows keeps `host_os::process_id`.
    crate::__pyre_put_new!(
        ns_slot,
        "getpid",
        crate::make_builtin_function_with_arity(
            "getpid",
            |_| {
                #[cfg(unix)]
                {
                    let pid = unsafe { majit_rlib::rposix::c_getpid() };
                    if pid < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_int_new(pid as i64))
                }
                #[cfg(not(unix))]
                Ok(pyre_object::w_int_new(host_os::process_id() as i64))
            },
            0,
        )
    );
    // `getenv` is not bound here. `os.py` writes it against `environ`
    // — the dict this module publishes and that os.py's `_Environ` wrapper
    // writes back into — and names it in its own `__all__`, so a binding is
    // both shadowed and counted twice through the star-import.
    // ── host_env::posix-backed real implementations (override the noop
    //    placeholders registered above) ───────────────────────────────
    // `sandbox` implies `host_env`, but these bodies are real host syscalls.
    // The sandbox overwrite below can only stub names it lists; keep the
    // real implementations out of that build so a missed name cannot openat.
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    {
        use rustpython_host_env::posix as host_posix;

        /// `interp_posix._pipe_inhcache`.
        static PIPE_INHCACHE: majit_rlib::rposix::SetNonInheritableCache =
            majit_rlib::rposix::SetNonInheritableCache::new();

        fn exec_argv(
            w_argv: PyObjectRef,
            function: &str,
        ) -> Result<Vec<std::ffi::CString>, crate::PyError> {
            // PyPy interp_posix.execv: `space.unpackiterable(w_argv)` followed
            // by `space.fsencode_w` for every argument.
            let items = crate::baseobjspace::unpackiterable(w_argv, -1).map_err(|error| {
                if error.kind == crate::PyErrorKind::TypeError {
                    crate::PyError::type_error(format!(
                        "{function}() arg 2 must be an iterable of strings"
                    ))
                } else {
                    error
                }
            })?;
            if items.is_empty() {
                return Err(crate::PyError::value_error(format!(
                    "{function}() arg 2 must not be empty"
                )));
            }
            let mut argv = Vec::with_capacity(items.len());
            for item in items {
                // An element is converted on the sequence's behalf, not as an
                // argument of the call, so the caller-less message is the one
                // it reports — measured, and the same for the environment
                // below and for `posix_spawn`'s file actions.
                let value = extract_path(item)?;
                argv.push(std::ffi::CString::new(value).map_err(|_| {
                    crate::PyError::value_error(format!(
                        "{function}() arg 2 contains an embedded null byte"
                    ))
                })?);
            }
            if argv[0].as_bytes().is_empty() {
                return Err(crate::PyError::value_error(format!(
                    "{function}() arg 2 first element cannot be empty"
                )));
            }
            Ok(argv)
        }

        fn exec_pointer_array(values: &[std::ffi::CString]) -> Vec<*const libc::c_char> {
            let mut pointers: Vec<_> = values.iter().map(|value| value.as_ptr()).collect();
            pointers.push(std::ptr::null());
            pointers
        }

        // PyPy interp_posix.execv: this call replaces the current process and
        // returns only to translate the host errno into OSError.
        crate::__pyre_put_new!(
            ns_slot,
            "execv",
            crate::make_builtin_function_with_arity(
                "execv",
                |args| {
                    // The path names itself; the argv entries below do not,
                    // because each of those is converted on the sequence's
                    // behalf rather than as an argument of its own.
                    let mut w_path = args[0];
                    let mut w_argv = args[1];
                    let command = pyre_object::with_roots!(w_path, w_argv =>
                        crate::gateway::fsencode_path_named_w(w_path, "execv", "path")
                            .map(|path| path.as_bytes))?;
                    let command_c = std::ffi::CString::new(command).map_err(|_| {
                        crate::PyError::value_error("execv() path contains an embedded null byte")
                    })?;
                    let argv =
                        pyre_object::with_roots!(w_path, w_argv => exec_argv(w_argv, "execv"))?;
                    let argv_ptrs = exec_pointer_array(&argv);
                    // `rposix.c_execv` releases the GIL and saves errno.
                    // The call returns only on failure.
                    pyre_object::with_roots!(w_path, w_argv => unsafe {
                        majit_rlib::rposix::c_execv(command_c.as_ptr(), argv_ptrs.as_ptr());
                    });
                    let errno = majit_rlib::rposix::get_saved_errno();
                    // interp_posix.py:1814-1817 uses `wrap_oserror`, which does
                    // not attach the command path.
                    Err(errno_err(errno, ""))
                },
                2,
            )
        );

        // PyPy interp_posix.execve/_env2interp: accept a mapping, fsencode
        // names and values, reject illegal names, then replace the process.
        crate::__pyre_put_new!(
            ns_slot,
            "execve",
            crate::make_builtin_function_with_arity(
                "execve",
                |args| {
                    let mut w_path = args[0];
                    let mut w_argv = args[1];
                    let mut w_env = args[2];
                    let command = pyre_object::with_roots!(w_path, w_argv, w_env =>
                        crate::gateway::fsencode_path_named_w(w_path, "execve", "path")
                            .map(|path| path.as_bytes))?;
                    let command_c = std::ffi::CString::new(command).map_err(|_| {
                        crate::PyError::value_error("execve() path contains an embedded null byte")
                    })?;
                    let argv = pyre_object::with_roots!(w_path, w_argv, w_env =>
                        exec_argv(w_argv, "execve"))?;
                    let argv_ptrs = exec_pointer_array(&argv);

                    let env = pyre_object::with_roots!(w_path, w_argv, w_env => {
                        collect_env_entries(w_env, "execve", false)
                    })?
                        .into_iter()
                        .map(|entry| {
                            std::ffi::CString::new(entry).map_err(|_| {
                                crate::PyError::value_error(
                                    "execve() environment contains an embedded null byte",
                                )
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let env_ptrs = exec_pointer_array(&env);
                    // `rposix.c_execve` releases the GIL and saves errno.
                    // The call returns only on failure.
                    pyre_object::with_roots!(w_path, w_argv, w_env => unsafe {
                        majit_rlib::rposix::c_execve(
                            command_c.as_ptr(),
                            argv_ptrs.as_ptr(),
                            env_ptrs.as_ptr(),
                        );
                    });
                    let errno = majit_rlib::rposix::get_saved_errno();
                    // `interp_posix.py:1812,1817` wraps the failure with
                    // `wrap_oserror`, which names no file, so the path stays
                    // out of the error the same way `execv` leaves it out.
                    Err(errno_err(errno, ""))
                },
                3,
            )
        );

        // os.strerror(code) -> str
        crate::__pyre_put_new!(
            ns_slot,
            "strerror",
            crate::make_builtin_function_with_arity(
                "strerror",
                |args| {
                    let code = match args.first() {
                        Some(&o) => crate::baseobjspace::c_int_w(o)?,
                        None => {
                            return Err(crate::PyError::type_error(
                                "strerror() requires 1 argument",
                            ));
                        }
                    };
                    #[cfg(feature = "sandbox")]
                    {
                        let msg = crate::host_seam::ops::strerror(code)
                            .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                        // The text comes from the C library in the current
                        // locale, so a byte with no UTF-8 spelling takes the
                        // surrogateescape rather than U+FFFD.
                        return Ok(crate::typedef::charp2uni(&msg));
                    }
                    #[cfg(not(feature = "sandbox"))]
                    Ok(pyre_object::w_str_new_managed(
                        &rustpython_host_env::time::strerror(code),
                    ))
                },
                1,
            )
        );

        // os.pipe() -> (r_fd, w_fd)
        crate::__pyre_put_new!(
            ns_slot,
            "pipe",
            crate::make_builtin_function_with_arity(
                "pipe",
                |_| {
                    // `rposix.c_pipe` releases the GIL and saves errno.
                    // `interp_posix.pipe` then `_pipe_inhcache.set_non_inheritable`.
                    let mut fds = [0; 2];
                    if unsafe { majit_rlib::rposix::c_pipe(fds.as_mut_ptr()) } < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    if PIPE_INHCACHE.set_non_inheritable(fds[0]) < 0
                        || PIPE_INHCACHE.set_non_inheritable(fds[1]) < 0
                    {
                        let err = majit_rlib::rposix::get_saved_errno();
                        unsafe {
                            let _ = majit_rlib::rposix::c_close(fds[0]);
                            let _ = majit_rlib::rposix::c_close(fds[1]);
                        }
                        return Err(io_err(std::io::Error::from_raw_os_error(err), ""));
                    }
                    let mut fields = pyre_object::gc_roots::RootedItems::new();
                    fields.push(pyre_object::w_int_new(fds[0] as i64));
                    fields.push(pyre_object::w_int_new(fds[1] as i64));
                    Ok(pyre_object::w_tuple_new(fields.take()))
                },
                0,
            )
        );

        // os.pipe2(flags) -> (r_fd, w_fd)
        //
        // `interp_posix.py`, which — unlike `pipe` two blocks up —
        // forces no inheritance on the pair afterwards: the flags argument is
        // the whole of the caller's control over it.
        #[cfg(any(
            target_os = "android",
            target_os = "dragonfly",
            target_os = "freebsd",
            target_os = "linux",
            target_os = "netbsd",
            target_os = "openbsd"
        ))]
        crate::__pyre_put_new!(
            ns_slot,
            "pipe2",
            crate::make_builtin_function_with_arity(
                "pipe2",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("pipe2() requires 1 argument"));
                    }
                    // interp_posix.py `@unwrap_spec(flags=c_int)`.
                    let flags = crate::baseobjspace::c_int_w(args[0])?;
                    // `rposix.c_pipe2` releases the GIL and saves errno. The
                    // flags are the caller's whole control over inheritance.
                    let mut fds = [0; 2];
                    if unsafe { majit_rlib::rposix::c_pipe2(fds.as_mut_ptr(), flags) } < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    let mut fields = pyre_object::gc_roots::RootedItems::new();
                    fields.push(pyre_object::w_int_new(fds[0] as i64));
                    fields.push(pyre_object::w_int_new(fds[1] as i64));
                    Ok(pyre_object::w_tuple_new(fields.take()))
                },
                1,
            )
        );

        // interp_posix.py `pread`: `rposix.pread` plus `eintr_retry=True`.
        crate::__pyre_put_new!(
            ns_slot,
            "pread",
            crate::make_builtin_function_with_arity(
                "pread",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error("pread() requires 3 arguments"));
                    }
                    // `@unwrap_spec(fd=c_int, length=int, offset=r_longlong)`.
                    let w_fd = args[0];
                    let mut w_length = args[1];
                    let mut w_offset = args[2];
                    let fd = pyre_object::with_roots!(w_length, w_offset =>
                        crate::baseobjspace::c_int_w(w_fd))?;
                    let length =
                        pyre_object::with_roots!(w_offset => crate::baseobjspace::int_w(w_length))?;
                    let offset = crate::baseobjspace::int_w(w_offset)? as libc::off_t;
                    if length < 0 {
                        return Err(crate::PyError::os_error_with_errno(
                            libc::EINVAL,
                            "pread: negative length",
                        ));
                    }
                    // unwrap_spec(length=int): space.int_w is a machine
                    // word. A value that does not fit usize is OverflowError,
                    // not a wrapped allocation size.
                    let n = usize::try_from(length).map_err(|_| {
                        crate::PyError::overflow_error(
                            "Python int too large to convert to C ssize_t",
                        )
                    })?;
                    let mut buf = Vec::new();
                    buf.try_reserve_exact(n)
                        .map_err(|_| crate::PyError::memory_error(""))?;
                    buf.resize(n, 0);
                    // `rposix.c_pread` releases the GIL and saves errno.
                    loop {
                        let got = unsafe {
                            majit_rlib::rposix::c_pread(fd, buf.as_mut_ptr().cast(), n, offset)
                        };
                        if got >= 0 {
                            buf.truncate(got as usize);
                            break;
                        }
                        let err = std::io::Error::from_raw_os_error(
                            majit_rlib::rposix::get_saved_errno(),
                        );
                        crate::builtins::eintr_retry_with(err, |e| io_err(e, ""))?;
                    }
                    Ok(pyre_object::w_bytes_from_bytes(&buf))
                },
                3,
            )
        );

        // interp_posix.py `pwrite`: `space.bufferstr_w` plus `eintr_retry=True`.
        crate::__pyre_put_new!(
            ns_slot,
            "pwrite",
            crate::make_builtin_function_with_arity(
                "pwrite",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error("pwrite() requires 3 arguments"));
                    }
                    // `@unwrap_spec(fd=c_int, offset=r_longlong)`.
                    let w_fd = args[0];
                    let mut w_data = args[1];
                    let mut w_offset = args[2];
                    let fd = pyre_object::with_roots!(w_data, w_offset =>
                        crate::baseobjspace::c_int_w(w_fd))?;
                    let data = pyre_object::with_roots!(w_offset => unsafe {
                        crate::builtins::file_write_buffer_bytes(w_data)
                    })
                    .map_err(|_| crate::PyError::type_error("pwrite() arg 2 must be bytes-like"))?;
                    let offset = crate::baseobjspace::int_w(w_offset)? as libc::off_t;
                    // `rposix.c_pwrite` releases the GIL and saves errno.
                    let written = loop {
                        let n = unsafe {
                            majit_rlib::rposix::c_pwrite(
                                fd,
                                data.as_ptr() as *mut libc::c_void,
                                data.len(),
                                offset,
                            )
                        };
                        if n >= 0 {
                            break n as usize;
                        }
                        let err = std::io::Error::from_raw_os_error(
                            majit_rlib::rposix::get_saved_errno(),
                        );
                        crate::builtins::eintr_retry_with(err, |e| io_err(e, ""))?;
                    };
                    Ok(pyre_object::w_int_new(written as i64))
                },
                3,
            )
        );

        // interp_posix.py `posix_fallocate`: `eintr_retry=True`.
        // `rposix.c_posix_fallocate` releases the GIL and saves errno.
        #[cfg(all(
            not(feature = "sandbox"),
            any(target_os = "linux", target_os = "android")
        ))]
        crate::__pyre_put_new!(
            ns_slot,
            "posix_fallocate",
            crate::make_builtin_function_with_arity(
                "posix_fallocate",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error(
                            "posix_fallocate() requires 3 arguments",
                        ));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int, length=r_longlong,
                    // offset=r_longlong)`.
                    let mut w_fd = args[0];
                    let mut w_offset = args[1];
                    let mut w_length = args[2];
                    let fd = pyre_object::with_roots!(w_fd, w_offset, w_length =>
                        crate::baseobjspace::c_int_w(w_fd))?;
                    let offset = pyre_object::with_roots!(w_fd, w_offset, w_length =>
                        crate::baseobjspace::int_w(w_offset))?
                        as libc::off_t;
                    let length = pyre_object::with_roots!(w_fd, w_offset, w_length =>
                        crate::baseobjspace::int_w(w_length))?
                        as libc::off_t;
                    let ret = loop {
                        let ret = pyre_object::with_roots!(w_fd, w_offset, w_length => unsafe {
                            majit_rlib::rposix::c_posix_fallocate(fd, offset, length)
                        });
                        if ret >= 0 {
                            break ret;
                        }
                        pyre_object::with_roots!(w_fd, w_offset, w_length => {
                            crate::builtins::eintr_retry_with(
                                std::io::Error::from_raw_os_error(
                                    majit_rlib::rposix::get_saved_errno(),
                                ),
                                |e| io_err(e, ""),
                            )
                        })?;
                    };
                    Ok(pyre_object::with_roots!(w_fd, w_offset, w_length => {
                        pyre_object::w_int_new(ret as i64)
                    }))
                },
                3,
            )
        );

        // interp_posix.py `posix_fadvise`: `eintr_retry=True`.
        // `rposix.posix_fadvise` uses the C return value as the errno.
        #[cfg(all(
            not(feature = "sandbox"),
            any(target_os = "linux", target_os = "android")
        ))]
        crate::__pyre_put_new!(
            ns_slot,
            "posix_fadvise",
            crate::make_builtin_function_with_arity(
                "posix_fadvise",
                |args| {
                    if args.len() < 4 {
                        return Err(crate::PyError::type_error(
                            "posix_fadvise() requires 4 arguments",
                        ));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int, offset=r_longlong,
                    // length=r_longlong, advice=int)`.
                    let mut w_fd = args[0];
                    let mut w_offset = args[1];
                    let mut w_length = args[2];
                    let mut w_advice = args[3];
                    let fd = pyre_object::with_roots!(w_fd, w_offset, w_length, w_advice =>
                        crate::baseobjspace::c_int_w(w_fd))?;
                    let offset = pyre_object::with_roots!(w_fd, w_offset, w_length, w_advice =>
                        crate::baseobjspace::int_w(w_offset))?
                        as libc::off_t;
                    let length = pyre_object::with_roots!(w_fd, w_offset, w_length, w_advice =>
                        crate::baseobjspace::int_w(w_length))?
                        as libc::off_t;
                    let advice = pyre_object::with_roots!(w_fd, w_offset, w_length, w_advice =>
                        crate::baseobjspace::int_w(w_advice))?
                        as majit_rlib::rffi::INT;
                    loop {
                        let error = pyre_object::with_roots!(
                            w_fd, w_offset, w_length, w_advice => unsafe {
                            majit_rlib::rposix::c_posix_fadvise(fd, offset, length, advice)
                        });
                        if error == 0 {
                            break;
                        }
                        pyre_object::with_roots!(w_fd, w_offset, w_length, w_advice => {
                            crate::builtins::eintr_retry_with(
                                std::io::Error::from_raw_os_error(error),
                                |e| io_err(e, ""),
                            )
                        })?;
                    }
                    Ok(pyre_object::w_none())
                },
                4,
            )
        );

        // os.sched_yield()
        crate::__pyre_put_new!(
            ns_slot,
            "sched_yield",
            crate::make_builtin_function_with_arity(
                "sched_yield",
                |_| {
                    // interp_posix.py `sched_yield`: retry on EINTR.
                    // `rposix.c_sched_yield` releases the GIL and saves errno.
                    loop {
                        let ret = unsafe { majit_rlib::rposix::c_sched_yield() };
                        if ret >= 0 {
                            break;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            |e| io_err(e, ""),
                        )?;
                    }
                    Ok(pyre_object::w_none())
                },
                0,
            )
        );

        // os.nice(increment) -> new niceness
        crate::__pyre_put_new!(
            ns_slot,
            "nice",
            crate::make_builtin_function_with_arity(
                "nice",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("nice() requires 1 argument"));
                    }
                    // interp_posix.py `@unwrap_spec(increment=c_int)`.
                    let inc = crate::baseobjspace::c_int_w(args[0])?;
                    // `rposix.c_nice` clears errno before the call and saves
                    // it after. -1 is a successful niceness when the saved
                    // errno stays 0. `rposix.nice` does not retry EINTR.
                    let n = unsafe { majit_rlib::rposix::c_nice(inc) };
                    if n == -1 {
                        let err = majit_rlib::rposix::get_saved_errno();
                        if err != 0 {
                            return Err(io_err(std::io::Error::from_raw_os_error(err), ""));
                        }
                    }
                    Ok(pyre_object::w_int_new(n as i64))
                },
                1,
            )
        );

        // os.umask(mask) -> previous mask
        crate::__pyre_put_new!(
            ns_slot,
            "umask",
            crate::make_builtin_function_with_arity(
                "umask",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("umask() requires 1 argument"));
                    }
                    // interp_posix.py `@unwrap_spec(mask=c_int)`.
                    let mask = crate::baseobjspace::c_int_w(args[0])? as libc::mode_t;
                    // `rposix.c_umask` returns the previous mask. It does not
                    // fail and does not save errno.
                    let prev = unsafe { majit_rlib::rposix::c_umask(mask) };
                    Ok(pyre_object::w_int_new(prev as i64))
                },
                1,
            )
        );

        // os.getlogin() -> str
        crate::__pyre_put_new!(
            ns_slot,
            "getlogin",
            crate::make_builtin_function_with_arity(
                "getlogin",
                |_| {
                    // `rposix.c_getlogin` is `releasegil=False` and saves
                    // errno. A null return is `OSError` of that errno.
                    let name = unsafe { majit_rlib::rposix::c_getlogin() };
                    if name.is_null() {
                        return Err(crate::PyError::os_error_with_errno(
                            majit_rlib::rposix::get_saved_errno(),
                            "getlogin",
                        ));
                    }
                    // A login name is an OS string; decode it the way every
                    // other one is so an undecodable byte keeps its escape.
                    let bytes = unsafe { std::ffi::CStr::from_ptr(name) }.to_bytes();
                    Ok(crate::gateway::fsdecode_filename_bytes(bytes))
                },
                0,
            )
        );

        // `getgroups(2)` reports at most `NGROUPS_MAX` entries, so a process in
        // more groups than that gets a truncated list. `<unistd.h>` aliases the
        // name to an unlimited variant under `_DARWIN_C_SOURCE`.
        // `rposix.c_getgroups` is that alias on Apple (`getgroups$DARWIN_EXTSN`)
        // and saves errno.
        /// The group list, sized by the count the kernel reports first.
        /// `rposix.c_getgroups` is `getgroups$DARWIN_EXTSN` on Apple and the
        /// plain symbol elsewhere. It releases the GIL and saves errno.
        fn host_getgroups() -> std::io::Result<Vec<libc::gid_t>> {
            let count = unsafe { majit_rlib::rposix::c_getgroups(0, std::ptr::null_mut()) };
            if count < 0 {
                return Err(std::io::Error::from_raw_os_error(
                    majit_rlib::rposix::get_saved_errno(),
                ));
            }
            let mut groups = Vec::<libc::gid_t>::with_capacity(count as usize);
            let filled = unsafe { majit_rlib::rposix::c_getgroups(count, groups.as_mut_ptr()) };
            if filled < 0 {
                return Err(std::io::Error::from_raw_os_error(
                    majit_rlib::rposix::get_saved_errno(),
                ));
            }
            // A `gidsetsize` of 0 asks for the count instead of the list, so
            // a process that was in no groups at the first call and is in
            // some by the second gets back a count with nothing written.
            let filled = (filled as usize).min(groups.capacity());
            unsafe { groups.set_len(filled) };
            Ok(groups)
        }

        /// Replace the supplementary group list. `rposix.c_setgroups` releases
        /// the GIL and saves errno.
        fn host_setgroups(groups: &[libc::gid_t]) -> std::io::Result<()> {
            let ret = unsafe { majit_rlib::rposix::c_setgroups(groups.len(), groups.as_ptr()) };
            if ret < 0 {
                return Err(std::io::Error::from_raw_os_error(
                    majit_rlib::rposix::get_saved_errno(),
                ));
            }
            Ok(())
        }

        // os.getgroups() -> list[int]
        crate::__pyre_put_new!(
            ns_slot,
            "getgroups",
            crate::make_builtin_function_with_arity(
                "getgroups",
                |_| {
                    let gs = host_getgroups().map_err(|e| io_err(e, ""))?;
                    let mut items = pyre_object::gc_roots::RootedItems::new();
                    for g in gs {
                        items.push(pyre_object::w_int_new(g as i64));
                    }
                    Ok(pyre_object::w_list_new(items.take()))
                },
                0,
            )
        );

        // os.setgroups(list) -> None
        crate::__pyre_put_new!(
            ns_slot,
            "setgroups",
            crate::make_builtin_function_with_arity(
                "setgroups",
                |args| {
                    let Some(&w_list) = args.first() else {
                        return Err(crate::PyError::type_error(
                            "setgroups() requires 1 argument",
                        ));
                    };
                    let mut w_list = w_list;
                    // interp_posix.py:1053-1064 — the list is unpacked as any
                    // iterable and each element read with `c_uid_t_w`, which is
                    // what lets -1 name `(gid_t)-1` instead of being refused.
                    let items = pyre_object::with_roots!(w_list => {
                        crate::builtins::collect_iterable(w_list)
                    })?;
                    // `c_uid_t_w` reaches `__index__` for a non-int entry, so
                    // converting one entry can collect and move the entries not
                    // yet converted -- `collect_iterable` hands back a plain
                    // vector, its own roots already dropped.  Publish `w_list`
                    // with that sequence as one live set: a later entry's
                    // `__index__` must not move the argument list still read
                    // after the loop (`interp_posix.py setgroups`).
                    let seq_roots = pyre_object::gc_roots::push_roots();
                    let mut live = Vec::with_capacity(items.len() + 1);
                    live.push(w_list);
                    live.extend_from_slice(&items);
                    let base = seq_roots.pin_roots(&live);
                    let mut groups: Vec<libc::gid_t> = Vec::with_capacity(items.len());
                    for offset in 0..items.len() {
                        let w_gid = seq_roots.get(base + 1 + offset);
                        groups.push(crate::baseobjspace::c_uid_t_w(w_gid)?);
                    }
                    w_list = seq_roots.get(base);
                    host_setgroups(&groups).map_err(|e| io_err(e, ""))?;
                    let _ = seq_roots.get(base);
                    Ok(pyre_object::w_none())
                },
                1,
            )
        );

        // os.getgrouplist(user, group) -> list of groups
        crate::__pyre_put_new!(
            ns_slot,
            "getgrouplist",
            crate::make_builtin_function_with_arity(
                "getgrouplist",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "getgrouplist() requires username, gid",
                        ));
                    }
                    let w_user = args[0];
                    let mut w_gid = args[1];
                    let user = unsafe {
                        if pyre_object::is_str(w_user) {
                            pyre_object::with_roots!(w_gid => crate::baseobjspace::str_utf8_w(w_user))?
                                .to_string()
                        } else {
                            return Err(crate::PyError::type_error(
                                "getgrouplist(): username must be str",
                            ));
                        }
                    };
                    let cuser = std::ffi::CString::new(user.as_bytes()).map_err(|_| {
                        crate::PyError::value_error("getgrouplist: embedded null in username")
                    })?;
                    // interp_posix.py `@unwrap_spec(username='text', gid=c_gid_t)`.
                    // Darwin's `getgrouplist` takes `int`; other unix hosts take `gid_t`.
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    type GroupId = libc::c_int;
                    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                    type GroupId = libc::gid_t;
                    let gid = crate::baseobjspace::c_uid_t_w(w_gid)? as GroupId;
                    let mut ngroups: libc::c_int = 64;
                    let mut groups = vec![0 as GroupId; 64];
                    // `rposix.c_getgroupslist` releases the GIL and saves errno.
                    let mut ret = unsafe {
                        majit_rlib::rposix::c_getgroupslist(
                            cuser.as_ptr(),
                            gid,
                            groups.as_mut_ptr(),
                            &mut ngroups,
                        )
                    };
                    if ret < 0 && ngroups > 64 {
                        groups.resize(ngroups as usize, 0);
                        ret = unsafe {
                            majit_rlib::rposix::c_getgroupslist(
                                cuser.as_ptr(),
                                gid,
                                groups.as_mut_ptr(),
                                &mut ngroups,
                            )
                        };
                    }
                    if ret < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    let groups = groups
                        .into_iter()
                        .take(ngroups as usize)
                        .map(|g| g as i64)
                        .collect::<Vec<_>>();
                    let mut items = pyre_object::gc_roots::RootedItems::new();
                    for g in groups {
                        items.push(pyre_object::w_int_new(g));
                    }
                    Ok(pyre_object::w_list_new(items.take()))
                },
                2,
            )
        );

        // os.sched_get_priority_max(policy) -> int
        crate::__pyre_put_new!(
            ns_slot,
            "sched_get_priority_max",
            crate::make_builtin_function_with_arity(
                "sched_get_priority_max",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "sched_get_priority_max() requires 1 argument",
                        ));
                    }
                    // interp_posix.py:2977 `@unwrap_spec(policy=int)`.
                    let policy = crate::baseobjspace::int_w(args[0])? as i32;
                    // interp_posix.py `sched_get_priority_max`: retry on EINTR.
                    // `rposix.c_sched_get_priority_max` uses
                    // `RFFI_FULL_ERRNO_ZERO` and releases the GIL.
                    let m = loop {
                        let m = unsafe { majit_rlib::rposix::c_sched_get_priority_max(policy) };
                        if m >= 0 {
                            break m;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            |e| io_err(e, ""),
                        )?;
                    };
                    Ok(pyre_object::w_int_new(m as i64))
                },
                1,
            )
        );

        // os.sched_get_priority_min(policy) -> int
        crate::__pyre_put_new!(
            ns_slot,
            "sched_get_priority_min",
            crate::make_builtin_function_with_arity(
                "sched_get_priority_min",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "sched_get_priority_min() requires 1 argument",
                        ));
                    }
                    // interp_posix.py:2991 `@unwrap_spec(policy=int)`.
                    let policy = crate::baseobjspace::int_w(args[0])? as i32;
                    // interp_posix.py `sched_get_priority_min`: retry on EINTR.
                    // `rposix.c_sched_get_priority_min` releases the GIL and
                    // saves errno.
                    let m = loop {
                        let m = unsafe { majit_rlib::rposix::c_sched_get_priority_min(policy) };
                        if m >= 0 {
                            break m;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            |e| io_err(e, ""),
                        )?;
                    };
                    Ok(pyre_object::w_int_new(m as i64))
                },
                1,
            )
        );

        // The scheduling-policy group `moduledef.py:168-174` publishes as one —
        // the two getters, the two setters and the `sched_param` type they
        // exchange — plus `sched_rr_get_interval`, which `moduledef.py:166-167`
        // gates on its own but which the same libcs carry. The setters are left
        // out where the libc is musl, which declares neither.
        #[cfg(any(
            target_os = "android",
            target_os = "freebsd",
            target_os = "linux",
            target_os = "netbsd"
        ))]
        {
            crate::__pyre_put_new!(ns_slot, "sched_param", sched_param_seq_type());

            // os.sched_rr_get_interval(pid) -> seconds
            //
            // host_env wraps none of this one, so the call is made here — which
            // is why it is absent from a sandbox build: `host_seam::sys`
            // re-exports no syscall function, and the name is served there by
            // the raising stub registered at the end of this module instead.
            #[cfg(not(feature = "sandbox"))]
            crate::__pyre_put_new!(
                ns_slot,
                "sched_rr_get_interval",
                crate::make_builtin_function_with_arity(
                    "sched_rr_get_interval",
                    |args| {
                        if args.is_empty() {
                            return Err(crate::PyError::type_error(
                                "sched_rr_get_interval() requires 1 argument",
                            ));
                        }
                        // interp_posix.py:3061 `@unwrap_spec(pid=int)`; the
                        // timespec the call fills is answered as one float
                        // (`rposix.py:2525`).
                        let pid = crate::baseobjspace::c_int_w(args[0])? as libc::pid_t;
                        let mut interval: libc::timespec =
                            unsafe { core::mem::zeroed::<libc::timespec>() };
                        // interp_posix.py `sched_rr_get_interval`: retry on EINTR.
                        // `rposix.c_sched_rr_get_interval` uses
                        // `RFFI_FULL_ERRNO_ZERO` and releases the GIL.
                        loop {
                            let ret = unsafe {
                                majit_rlib::rposix::c_sched_rr_get_interval(pid, &mut interval)
                            };
                            if ret >= 0 {
                                break;
                            }
                            crate::builtins::eintr_retry_with(
                                std::io::Error::from_raw_os_error(
                                    majit_rlib::rposix::get_saved_errno(),
                                ),
                                |e| io_err(e, ""),
                            )?;
                        }
                        Ok(pyre_object::w_float_new(
                            interval.tv_sec as f64 + 1e-9 * interval.tv_nsec as f64,
                        ))
                    },
                    1,
                )
            );

            // os.sched_getscheduler(pid) -> policy
            crate::__pyre_put_new!(
                ns_slot,
                "sched_getscheduler",
                crate::make_builtin_function_with_arity(
                    "sched_getscheduler",
                    |args| {
                        if args.is_empty() {
                            return Err(crate::PyError::type_error(
                                "sched_getscheduler() requires 1 argument",
                            ));
                        }
                        // interp_posix.py:3073 `@unwrap_spec(pid=int)`.
                        let pid = crate::baseobjspace::c_int_w(args[0])? as libc::pid_t;
                        // interp_posix.py `sched_getscheduler`: retry on EINTR.
                        // `rposix.c_sched_getscheduler` uses
                        // `RFFI_FULL_ERRNO_ZERO` and releases the GIL.
                        let policy = loop {
                            let policy = unsafe { majit_rlib::rposix::c_sched_getscheduler(pid) };
                            if policy >= 0 {
                                break policy;
                            }
                            crate::builtins::eintr_retry_with(
                                std::io::Error::from_raw_os_error(
                                    majit_rlib::rposix::get_saved_errno(),
                                ),
                                |e| io_err(e, ""),
                            )?;
                        };
                        Ok(pyre_object::w_int_new(policy as i64))
                    },
                    1,
                )
            );

            // os.sched_getparam(pid) -> sched_param
            crate::__pyre_put_new!(
                ns_slot,
                "sched_getparam",
                crate::make_builtin_function_with_arity(
                    "sched_getparam",
                    |args| {
                        if args.is_empty() {
                            return Err(crate::PyError::type_error(
                                "sched_getparam() requires 1 argument",
                            ));
                        }
                        // interp_posix.py:3103 `@unwrap_spec(pid=int)`; the
                        // priority the call fills in is handed back wrapped in
                        // the type, not bare (`interp_posix.py:3113`).
                        let pid = crate::baseobjspace::c_int_w(args[0])? as libc::pid_t;
                        // interp_posix.py `sched_getparam`: retry on EINTR.
                        // `rposix.c_sched_getparam` uses `RFFI_FULL_ERRNO_ZERO`
                        // and releases the GIL.
                        let param = loop {
                            let mut param: libc::sched_param = unsafe { core::mem::zeroed() };
                            let ret =
                                unsafe { majit_rlib::rposix::c_sched_getparam(pid, &mut param) };
                            if ret >= 0 {
                                break param;
                            }
                            crate::builtins::eintr_retry_with(
                                std::io::Error::from_raw_os_error(
                                    majit_rlib::rposix::get_saved_errno(),
                                ),
                                |e| io_err(e, ""),
                            )?;
                        };
                        Ok(crate::_structseq::new_instance(
                            sched_param_seq_type(),
                            vec![pyre_object::w_int_new(param.sched_priority as i64)],
                        ))
                    },
                    1,
                )
            );

            // Both setters answer None. `interp_posix.py:3097`/`:3131` hand
            // back the raw `handle_posix_error` result instead, which is 0 on
            // every success and which `os.sched_setparam` does not publish.
            #[cfg(not(target_env = "musl"))]
            {
                // os.sched_setscheduler(pid, policy, param)
                crate::__pyre_put_new!(
                    ns_slot,
                    "sched_setscheduler",
                    crate::make_builtin_function_with_arity(
                        "sched_setscheduler",
                        |args| {
                            if args.len() < 3 {
                                return Err(crate::PyError::type_error(
                                    "sched_setscheduler() requires 3 arguments",
                                ));
                            }
                            // interp_posix.py `@unwrap_spec(pid=int, policy=int)`.
                            let w_pid = args[0];
                            let mut w_policy = args[1];
                            let mut w_param = args[2];
                            let pid = pyre_object::with_roots!(w_policy, w_param => {
                                crate::baseobjspace::c_int_w(w_pid)
                            })? as libc::pid_t;
                            let policy = pyre_object::with_roots!(w_param => {
                                crate::baseobjspace::int_w(w_policy)
                            })? as libc::c_int;
                            let priority = sched_priority_w(w_param)?;
                            let mut param: libc::sched_param =
                                unsafe { core::mem::zeroed::<libc::sched_param>() };
                            param.sched_priority = priority;
                            // interp_posix.py `sched_setscheduler`: retry on EINTR.
                            // `rposix.c_sched_setscheduler` uses
                            // `RFFI_FULL_ERRNO_ZERO` and releases the GIL.
                            loop {
                                let ret = unsafe {
                                    majit_rlib::rposix::c_sched_setscheduler(pid, policy, &param)
                                };
                                if ret >= 0 {
                                    break;
                                }
                                crate::builtins::eintr_retry_with(
                                    std::io::Error::from_raw_os_error(
                                        majit_rlib::rposix::get_saved_errno(),
                                    ),
                                    |e| io_err(e, ""),
                                )?;
                            }
                            Ok(pyre_object::w_none())
                        },
                        3,
                    )
                );

                // os.sched_setparam(pid, param)
                crate::__pyre_put_new!(
                    ns_slot,
                    "sched_setparam",
                    crate::make_builtin_function_with_arity(
                        "sched_setparam",
                        |args| {
                            if args.len() < 2 {
                                return Err(crate::PyError::type_error(
                                    "sched_setparam() requires 2 arguments",
                                ));
                            }
                            // interp_posix.py:3117 `@unwrap_spec(pid=int)`.
                            let w_pid = args[0];
                            let mut w_param = args[1];
                            let pid = pyre_object::with_roots!(w_param => {
                                crate::baseobjspace::c_int_w(w_pid)
                            })? as libc::pid_t;
                            let priority = sched_priority_w(w_param)?;
                            let mut param: libc::sched_param =
                                unsafe { core::mem::zeroed::<libc::sched_param>() };
                            param.sched_priority = priority;
                            // interp_posix.py `sched_setparam`: retry on EINTR.
                            // `rposix.c_sched_setparam` uses
                            // `RFFI_FULL_ERRNO_ZERO` and releases the GIL.
                            loop {
                                let ret =
                                    unsafe { majit_rlib::rposix::c_sched_setparam(pid, &param) };
                                if ret >= 0 {
                                    break;
                                }
                                crate::builtins::eintr_retry_with(
                                    std::io::Error::from_raw_os_error(
                                        majit_rlib::rposix::get_saved_errno(),
                                    ),
                                    |e| io_err(e, ""),
                                )?;
                            }
                            Ok(pyre_object::w_none())
                        },
                        2,
                    )
                );
            }
        }

        // interp_posix.py `sched_getaffinity` / `sched_setaffinity`. The mask
        // is `rposix.CPU_MASK_P` (`CArrayPtr(rffi.ULONG)`). Neither retries
        // EINTR. `rposix.c_sched_getaffinity` / `c_sched_setaffinity` release
        // the GIL and save errno.
        #[cfg(all(
            not(feature = "sandbox"),
            any(target_os = "linux", target_os = "android")
        ))]
        {
            crate::__pyre_put_new!(
                ns_slot,
                "sched_getaffinity",
                crate::make_builtin_function_with_arity(
                    "sched_getaffinity",
                    |args| {
                        const CPU_MASK_BITS: usize = core::mem::size_of::<libc::c_ulong>() * 8;
                        if args.is_empty() {
                            return Err(crate::PyError::type_error(
                                "sched_getaffinity() requires 1 argument",
                            ));
                        }
                        // interp_posix.py `@unwrap_spec(pid=int)`.
                        let mut w_pid = args[0];
                        let pid = pyre_object::with_roots!(w_pid => {
                            crate::baseobjspace::c_int_w(w_pid)
                        })? as libc::pid_t;
                        let mut ncpus = CPU_MASK_BITS;
                        loop {
                            let nwords = ncpus / CPU_MASK_BITS;
                            let mut mask = vec![0 as libc::c_ulong; nwords];
                            let size = nwords * core::mem::size_of::<libc::c_ulong>();
                            let res = pyre_object::with_roots!(w_pid => unsafe {
                                majit_rlib::rposix::c_sched_getaffinity(
                                    pid,
                                    size,
                                    mask.as_mut_ptr() as majit_rlib::rffi::VOIDP,
                                )
                            });
                            if res >= 0 {
                                return Ok(pyre_object::with_roots!(w_pid => {
                                    let mut items = pyre_object::gc_roots::RootedItems::new();
                                    for (i, word) in mask.iter().copied().enumerate() {
                                        if word == 0 {
                                            continue;
                                        }
                                        for bit in 0..CPU_MASK_BITS {
                                            if word & ((1 as libc::c_ulong) << bit) != 0 {
                                                items.push(pyre_object::w_int_new(
                                                    (i * CPU_MASK_BITS + bit) as i64,
                                                ));
                                            }
                                        }
                                    }
                                    pyre_object::w_set_from_items(&items.take())
                                }));
                            }
                            let err = majit_rlib::rposix::get_saved_errno();
                            if err != libc::EINVAL {
                                return Err(io_err(std::io::Error::from_raw_os_error(err), ""));
                            }
                            if ncpus > (libc::c_int::MAX as usize) / 2 {
                                return Err(crate::PyError::overflow_error(
                                    "could not allocate a large enough CPU set",
                                ));
                            }
                            ncpus *= 2;
                        }
                    },
                    1,
                )
            );

            crate::__pyre_put_new!(
                ns_slot,
                "sched_setaffinity",
                crate::make_builtin_function_with_arity(
                    "sched_setaffinity",
                    |args| {
                        const CPU_MASK_BITS: usize = core::mem::size_of::<libc::c_ulong>() * 8;
                        if args.len() < 2 {
                            return Err(crate::PyError::type_error(
                                "sched_setaffinity() requires 2 arguments",
                            ));
                        }
                        // interp_posix.py `@unwrap_spec(pid=int)`.
                        let mut w_pid = args[0];
                        let mut w_mask = args[1];
                        let pid = pyre_object::with_roots!(w_pid, w_mask => {
                            crate::baseobjspace::c_int_w(w_pid)
                        })? as libc::pid_t;
                        let items = pyre_object::with_roots!(w_pid, w_mask => {
                            crate::builtins::collect_iterable(w_mask)
                        })?;
                        let mut cpus = Vec::new();
                        let _seq_roots = pyre_object::gc_roots::push_roots();
                        let items_base = pyre_object::gc_roots::pin_roots(&items);
                        for offset in 0..items.len() {
                            let item = pyre_object::gc_roots::shadow_stack_get(items_base + offset);
                            if !crate::baseobjspace::isinstance(
                                item,
                                crate::typedef::gettypeobject(&pyre_object::pyobject::INT_TYPE),
                            )? {
                                return Err(crate::PyError::type_error(format!(
                                    "expected an iterator of ints, but iterator yielded <class '{}'>",
                                    crate::type_methods::arg_type_name(item)
                                )));
                            }
                            let cpu = crate::baseobjspace::int_w(item)?;
                            if cpu < 0 {
                                return Err(crate::PyError::value_error("negative CPU number"));
                            }
                            if cpu > (libc::c_int::MAX as i64) - 1 {
                                return Err(crate::PyError::overflow_error("CPU number too large"));
                            }
                            cpus.push(cpu as usize);
                        }
                        drop(_seq_roots);
                        drop(items);
                        let mut maxcpu = 0usize;
                        for &cpu in &cpus {
                            if cpu > maxcpu {
                                maxcpu = cpu;
                            }
                        }
                        let nwords = maxcpu / CPU_MASK_BITS + 1;
                        let mut mask = vec![0 as libc::c_ulong; nwords];
                        for cpu in cpus {
                            let i = cpu / CPU_MASK_BITS;
                            mask[i] |= (1 as libc::c_ulong) << (cpu % CPU_MASK_BITS);
                        }
                        let size = nwords * core::mem::size_of::<libc::c_ulong>();
                        let res = pyre_object::with_roots!(w_pid, w_mask => unsafe {
                            majit_rlib::rposix::c_sched_setaffinity(
                                pid,
                                size,
                                mask.as_mut_ptr() as majit_rlib::rffi::VOIDP,
                            )
                        });
                        if res < 0 {
                            return Err(io_err(
                                std::io::Error::from_raw_os_error(
                                    majit_rlib::rposix::get_saved_errno(),
                                ),
                                "",
                            ));
                        }
                        Ok(pyre_object::w_none())
                    },
                    2,
                )
            );
        }

        // interp_posix.py `memfd_create`. `rposix.c_memfd_create` releases
        // the GIL and saves errno. Default flags are `rposix.MFD_CLOEXEC`.
        #[cfg(all(
            not(feature = "sandbox"),
            any(target_os = "linux", target_os = "android")
        ))]
        crate::__pyre_put_new!(
            ns_slot,
            "memfd_create",
            crate::make_builtin_function("memfd_create", |args| {
                // interp_posix.py `@unwrap_spec(name='text', flags=int)`.
                let (mut w_name, mut w_flags) = {
                    let (bound, _kwargs) =
                        bind_path_args(args, "memfd_create", &["name", "flags"], 1, &[])?;
                    (
                        bound[0].expect("name is required"),
                        bound[1].unwrap_or(pyre_object::PY_NULL),
                    )
                };
                // `interp_posix.memfd_create` unwraps `name='text'` then
                // `flags=int`; default flags stay `MFD_CLOEXEC`.
                let name = pyre_object::with_roots!(w_name, w_flags => {
                    crate::baseobjspace::text_w(w_name)
                })?;
                let flags = if w_flags.is_null() {
                    libc::MFD_CLOEXEC as majit_rlib::rffi::UINT
                } else {
                    pyre_object::with_roots!(w_name, w_flags => {
                        crate::baseobjspace::int_w(w_flags)
                    })? as majit_rlib::rffi::UINT
                };
                let c_name = std::ffi::CString::new(name)
                    .map_err(|_| crate::PyError::value_error("embedded null character"))?;
                let fd = pyre_object::with_roots!(w_name, w_flags => unsafe {
                    majit_rlib::rposix::c_memfd_create(c_name.as_ptr(), flags)
                });
                if fd < 0 {
                    return Err(io_err(
                        std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                        "",
                    ));
                }
                Ok(pyre_object::w_int_new(fd as i64))
            })
        );

        // interp_posix.py `getxattr` / `setxattr` / `removexattr` / `listxattr`.
        // `rposix.c_*xattr` release the GIL and save errno. ERANGE retries
        // `rposix.buf_sizes` = [256, XATTR_SIZE_MAX] in the product builtin.
        #[cfg(all(
            not(feature = "sandbox"),
            any(target_os = "linux", target_os = "android")
        ))]
        {
            const XATTR_BUF_SIZES: [usize; 2] = [256, 65536];
            crate::__pyre_put_new!(
                ns_slot,
                "getxattr",
                crate::make_builtin_function("getxattr", |args| {
                    let (mut w_path, mut w_attribute, mut w_follow) = {
                        let (bound, kwargs) = bind_path_args(
                            args,
                            "getxattr",
                            &["path", "attribute"],
                            2,
                            &["follow_symlinks"],
                        )?;
                        let w_path = bound[0].expect("path is required");
                        let w_attribute = bound[1].expect("attribute is required");
                        drop(bound);
                        let w_follow = crate::builtins::kwarg_get(kwargs, "follow_symlinks")
                            .unwrap_or(pyre_object::PY_NULL);
                        (w_path, w_attribute, w_follow)
                    };
                    // `interp_posix.getxattr` unwraps path, attribute, then
                    // follow_symlinks. Path then attribute share one root
                    // scope (`fsencode_path_then_attribute`). The original
                    // path argument is pinned here for `wrap_oserror2`
                    // (`path.w_path`): `FsEncodedPath::w_path` is a shadow-
                    // stack slot a later `with_roots!` can reuse.
                    let hold = pyre_object::gc_roots::push_roots();
                    let hold_base = hold.pin_roots(&[w_path, w_attribute, w_follow]);
                    let (_path_roots, path, attribute) =
                        fsencode_path_then_attribute(&mut w_path, &mut w_attribute, "getxattr")?;
                    w_path = hold.get(hold_base);
                    w_attribute = hold.get(hold_base + 1);
                    w_follow = hold.get(hold_base + 2);
                    let follow_symlinks = if w_follow.is_null() {
                        true
                    } else {
                        pyre_object::with_roots!(w_path, w_attribute, w_follow => {
                            crate::baseobjspace::is_true(w_follow)
                        })?
                    };
                    if path.is_fd && !follow_symlinks {
                        return Err(crate::PyError::value_error(
                            "getxattr: cannot use fd and follow_symlinks together",
                        ));
                    }
                    let c_attr = std::ffi::CString::new(attribute.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null character"))?;
                    let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    let mut got = None;
                    for size in XATTR_BUF_SIZES {
                        let mut buf = vec![0u8; size];
                        let res = if path.is_fd {
                            pyre_object::with_roots!(w_path, w_attribute, w_follow => unsafe {
                                majit_rlib::rposix::c_fgetxattr(
                                    path.as_fd,
                                    c_attr.as_ptr(),
                                    buf.as_mut_ptr() as majit_rlib::rffi::VOIDP,
                                    size,
                                )
                            })
                        } else if follow_symlinks {
                            pyre_object::with_roots!(w_path, w_attribute, w_follow => unsafe {
                                majit_rlib::rposix::c_getxattr(
                                    c_path.as_ptr(),
                                    c_attr.as_ptr(),
                                    buf.as_mut_ptr() as majit_rlib::rffi::VOIDP,
                                    size,
                                )
                            })
                        } else {
                            pyre_object::with_roots!(w_path, w_attribute, w_follow => unsafe {
                                majit_rlib::rposix::c_lgetxattr(
                                    c_path.as_ptr(),
                                    c_attr.as_ptr(),
                                    buf.as_mut_ptr() as majit_rlib::rffi::VOIDP,
                                    size,
                                )
                            })
                        };
                        if res >= 0 {
                            buf.truncate(res as usize);
                            got = Some(buf);
                            break;
                        }
                        let err = majit_rlib::rposix::get_saved_errno();
                        if err != libc::ERANGE {
                            return Err(io_err_with_filename(
                                std::io::Error::from_raw_os_error(err),
                                hold.get(hold_base),
                            ));
                        }
                    }
                    match got {
                        Some(buf) => Ok(pyre_object::with_roots!(w_path, w_attribute, w_follow => {
                            pyre_object::bytesobject::w_bytes_from_bytes(&buf)
                        })),
                        None => Err(io_err_with_filename(
                            std::io::Error::from_raw_os_error(libc::ERANGE),
                            hold.get(hold_base),
                        )),
                    }
                })
            );

            crate::__pyre_put_new!(
                ns_slot,
                "setxattr",
                crate::make_builtin_function("setxattr", |args| {
                    let (mut w_path, mut w_attribute, mut w_value, mut w_flags, mut w_follow) = {
                        let (bound, kwargs) = bind_path_args(
                            args,
                            "setxattr",
                            &["path", "attribute", "value", "flags"],
                            3,
                            &["follow_symlinks"],
                        )?;
                        let w_path = bound[0].expect("path is required");
                        let w_attribute = bound[1].expect("attribute is required");
                        let w_value = bound[2].expect("value is required");
                        let w_flags = bound[3].unwrap_or(pyre_object::PY_NULL);
                        drop(bound);
                        let w_follow = crate::builtins::kwarg_get(kwargs, "follow_symlinks")
                            .unwrap_or(pyre_object::PY_NULL);
                        (w_path, w_attribute, w_value, w_flags, w_follow)
                    };
                    // `interp_posix.setxattr` unwraps path, attribute, flags,
                    // follow_symlinks, then `space.bufferstr_w(w_value)`.
                    // Path then attribute share one root scope, matching
                    // `getxattr`. The original path argument is pinned for
                    // `wrap_oserror2`.
                    let hold = pyre_object::gc_roots::push_roots();
                    let hold_base =
                        hold.pin_roots(&[w_path, w_attribute, w_value, w_flags, w_follow]);
                    let (_path_roots, path, attribute) =
                        fsencode_path_then_attribute(&mut w_path, &mut w_attribute, "setxattr")?;
                    w_path = hold.get(hold_base);
                    w_attribute = hold.get(hold_base + 1);
                    w_value = hold.get(hold_base + 2);
                    w_flags = hold.get(hold_base + 3);
                    w_follow = hold.get(hold_base + 4);
                    let flags = if w_flags.is_null() {
                        0
                    } else {
                        pyre_object::with_roots!(
                            w_path,
                            w_attribute,
                            w_value,
                            w_flags,
                            w_follow => crate::baseobjspace::c_int_w(w_flags)
                        )?
                    };
                    let follow_symlinks = if w_follow.is_null() {
                        true
                    } else {
                        pyre_object::with_roots!(
                            w_path,
                            w_attribute,
                            w_value,
                            w_flags,
                            w_follow => crate::baseobjspace::is_true(w_follow)
                        )?
                    };
                    let value = pyre_object::with_roots!(
                        w_path,
                        w_attribute,
                        w_value,
                        w_flags,
                        w_follow => crate::baseobjspace::charbuf_w(w_value)
                    )?;
                    if path.is_fd && !follow_symlinks {
                        return Err(crate::PyError::value_error(
                            "setxattr: cannot use fd and follow_symlinks together",
                        ));
                    }
                    let c_attr = std::ffi::CString::new(attribute.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null character"))?;
                    let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    let ret = if path.is_fd {
                        pyre_object::with_roots!(
                            w_path,
                            w_attribute,
                            w_value,
                            w_flags,
                            w_follow => unsafe {
                            majit_rlib::rposix::c_fsetxattr(
                                path.as_fd,
                                c_attr.as_ptr(),
                                value.as_ptr() as *const libc::c_char,
                                value.len(),
                                flags,
                            )
                        })
                    } else if follow_symlinks {
                        pyre_object::with_roots!(
                            w_path,
                            w_attribute,
                            w_value,
                            w_flags,
                            w_follow => unsafe {
                            majit_rlib::rposix::c_setxattr(
                                c_path.as_ptr(),
                                c_attr.as_ptr(),
                                value.as_ptr() as *const libc::c_char,
                                value.len(),
                                flags,
                            )
                        })
                    } else {
                        pyre_object::with_roots!(
                            w_path,
                            w_attribute,
                            w_value,
                            w_flags,
                            w_follow => unsafe {
                            majit_rlib::rposix::c_lsetxattr(
                                c_path.as_ptr(),
                                c_attr.as_ptr(),
                                value.as_ptr() as *const libc::c_char,
                                value.len(),
                                flags,
                            )
                        })
                    };
                    if ret < 0 {
                        return Err(io_err_with_filename(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            hold.get(hold_base),
                        ));
                    }
                    Ok(pyre_object::w_none())
                })
            );

            crate::__pyre_put_new!(
                ns_slot,
                "removexattr",
                crate::make_builtin_function("removexattr", |args| {
                    let (mut w_path, mut w_attribute, mut w_follow) = {
                        let (bound, kwargs) = bind_path_args(
                            args,
                            "removexattr",
                            &["path", "attribute"],
                            2,
                            &["follow_symlinks"],
                        )?;
                        let w_path = bound[0].expect("path is required");
                        let w_attribute = bound[1].expect("attribute is required");
                        drop(bound);
                        let w_follow = crate::builtins::kwarg_get(kwargs, "follow_symlinks")
                            .unwrap_or(pyre_object::PY_NULL);
                        (w_path, w_attribute, w_follow)
                    };
                    // `interp_posix.removexattr` unwraps path, attribute, then
                    // follow_symlinks. The original path argument is pinned
                    // for `wrap_oserror2`.
                    let hold = pyre_object::gc_roots::push_roots();
                    let hold_base = hold.pin_roots(&[w_path, w_attribute, w_follow]);
                    let (_path_roots, path, attribute) =
                        fsencode_path_then_attribute(&mut w_path, &mut w_attribute, "removexattr")?;
                    w_path = hold.get(hold_base);
                    w_attribute = hold.get(hold_base + 1);
                    w_follow = hold.get(hold_base + 2);
                    let follow_symlinks = if w_follow.is_null() {
                        true
                    } else {
                        pyre_object::with_roots!(w_path, w_attribute, w_follow => {
                            crate::baseobjspace::is_true(w_follow)
                        })?
                    };
                    if path.is_fd && !follow_symlinks {
                        return Err(crate::PyError::value_error(
                            "removexattr: cannot use fd and follow_symlinks together",
                        ));
                    }
                    let c_attr = std::ffi::CString::new(attribute.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null character"))?;
                    let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    let ret = if path.is_fd {
                        pyre_object::with_roots!(w_path, w_attribute, w_follow => unsafe {
                            majit_rlib::rposix::c_fremovexattr(path.as_fd, c_attr.as_ptr())
                        })
                    } else if follow_symlinks {
                        pyre_object::with_roots!(w_path, w_attribute, w_follow => unsafe {
                            majit_rlib::rposix::c_removexattr(c_path.as_ptr(), c_attr.as_ptr())
                        })
                    } else {
                        pyre_object::with_roots!(w_path, w_attribute, w_follow => unsafe {
                            majit_rlib::rposix::c_lremovexattr(c_path.as_ptr(), c_attr.as_ptr())
                        })
                    };
                    if ret < 0 {
                        return Err(io_err_with_filename(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            hold.get(hold_base),
                        ));
                    }
                    Ok(pyre_object::w_none())
                })
            );

            crate::__pyre_put_new!(
                ns_slot,
                "listxattr",
                crate::make_builtin_function("listxattr", |args| {
                    let (mut w_path, mut w_follow) = {
                        let (bound, kwargs) =
                            bind_path_args(args, "listxattr", &["path"], 1, &["follow_symlinks"])?;
                        let w_path = bound[0].expect("path is required");
                        drop(bound);
                        let w_follow = crate::builtins::kwarg_get(kwargs, "follow_symlinks")
                            .unwrap_or(pyre_object::PY_NULL);
                        (w_path, w_follow)
                    };
                    // `interp_posix.listxattr` unwraps path, then follow_symlinks.
                    // The path owns a bracket of its own, above this one; this
                    // one stays open until the path is gone.
                    let path_roots = pyre_object::gc_roots::push_roots();
                    let path_base = path_roots.pin_roots(&[w_path, w_follow]);
                    let path = crate::gateway::fsencode_path_or_fd_w(
                        path_roots.get(path_base),
                        "listxattr",
                        true,
                    );
                    w_path = path_roots.get(path_base);
                    w_follow = path_roots.get(path_base + 1);
                    let path = path?;
                    let follow_symlinks = if w_follow.is_null() {
                        true
                    } else {
                        pyre_object::with_roots!(w_path, w_follow => {
                            crate::baseobjspace::is_true(w_follow)
                        })?
                    };
                    if path.is_fd && !follow_symlinks {
                        return Err(crate::PyError::value_error(
                            "listxattr: cannot use fd and follow_symlinks together",
                        ));
                    }
                    let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    let mut got = None;
                    for size in XATTR_BUF_SIZES {
                        let mut buf = vec![0u8; size];
                        let res = if path.is_fd {
                            pyre_object::with_roots!(w_path, w_follow => unsafe {
                                majit_rlib::rposix::c_flistxattr(
                                    path.as_fd,
                                    buf.as_mut_ptr() as *mut libc::c_char,
                                    size,
                                )
                            })
                        } else if follow_symlinks {
                            pyre_object::with_roots!(w_path, w_follow => unsafe {
                                majit_rlib::rposix::c_listxattr(
                                    c_path.as_ptr(),
                                    buf.as_mut_ptr() as *mut libc::c_char,
                                    size,
                                )
                            })
                        } else {
                            pyre_object::with_roots!(w_path, w_follow => unsafe {
                                majit_rlib::rposix::c_llistxattr(
                                    c_path.as_ptr(),
                                    buf.as_mut_ptr() as *mut libc::c_char,
                                    size,
                                )
                            })
                        };
                        if res >= 0 {
                            buf.truncate(res as usize);
                            got = Some(buf);
                            break;
                        }
                        let err = majit_rlib::rposix::get_saved_errno();
                        if err != libc::ERANGE {
                            return Err(io_err_with_filename(
                                std::io::Error::from_raw_os_error(err),
                                path_roots.get(path_base),
                            ));
                        }
                    }
                    let buf = match got {
                        Some(buf) => buf,
                        None => {
                            return Err(io_err_with_filename(
                                std::io::Error::from_raw_os_error(libc::ERANGE),
                                path_roots.get(path_base),
                            ));
                        }
                    };
                    // `rposix._unpack_attrs`: `split('\0'); del result[-1]`.
                    let mut names: Vec<&[u8]> = buf.split(|b| *b == 0).collect();
                    if !names.is_empty() {
                        names.pop();
                    }
                    Ok(pyre_object::with_roots!(w_path, w_follow => {
                        let mut items = pyre_object::gc_roots::RootedItems::new();
                        for name in names {
                            items.push(crate::gateway::fsdecode_filename_bytes(name));
                        }
                        pyre_object::w_list_new(items.take())
                    }))
                })
            );
        }

        // os.sync()
        #[cfg(not(any(target_os = "redox", target_os = "android")))]
        crate::__pyre_put_new!(
            ns_slot,
            "sync",
            crate::make_builtin_function_with_arity(
                "sync",
                |_| {
                    // `rposix.c_sync` releases the GIL and does not save errno.
                    unsafe { majit_rlib::rposix::c_sync() };
                    Ok(pyre_object::w_none())
                },
                0,
            )
        );

        // os.chdir(path)
        crate::__pyre_put_new!(
            ns_slot,
            "chdir",
            crate::make_builtin_function_with_arity(
                "chdir",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("chdir() requires 1 argument"));
                    }
                    // interp_posix.py dispatches `rposix.chdir` with
                    // `allow_fd_fn=os.fchdir` when `rposix.HAVE_FCHDIR`.
                    // The path owns a bracket of its own, above this one; this
                    // one stays open until the path is gone.
                    let mut w_path = args[0];
                    let path_roots = pyre_object::gc_roots::push_roots();
                    let path_base = path_roots.pin_roots(&[w_path]);
                    let path = crate::gateway::fsencode_path_or_fd_w(
                        path_roots.get(path_base),
                        "chdir",
                        HAVE_FCHDIR,
                    );
                    w_path = path_roots.get(path_base);
                    let path = path?;
                    if path.is_fd {
                        // `rposix.c_fchdir` releases the GIL and saves errno.
                        // `interp_posix.chdir` dispatches an fd through `os.fchdir`.
                        let ret = pyre_object::with_roots!(w_path => unsafe {
                            majit_rlib::rposix::c_fchdir(path.as_fd)
                        });
                        if ret < 0 {
                            return Err(io_err(
                                std::io::Error::from_raw_os_error(
                                    majit_rlib::rposix::get_saved_errno(),
                                ),
                                "",
                            ));
                        }
                        return Ok(pyre_object::w_none());
                    }
                    let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    // `rposix.c_chdir` releases the GIL and saves errno.
                    let ret = pyre_object::with_roots!(w_path => unsafe {
                        majit_rlib::rposix::c_chdir(c_path.as_ptr())
                    });
                    if ret < 0 {
                        return Err(errno_err_with_filename(
                            majit_rlib::rposix::get_saved_errno(),
                            path.w_path(),
                        ));
                    }
                    Ok(pyre_object::w_none())
                },
                1,
            )
        );

        // os.fchdir(fd)
        crate::__pyre_put_new!(
            ns_slot,
            "fchdir",
            crate::make_builtin_function_with_arity(
                "fchdir",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("fchdir() requires 1 argument"));
                    }
                    // interp_posix.py `fchdir(space, w_fd)` unwraps
                    // through `space.c_filedescriptor_w`, which takes an int or
                    // anything exposing `fileno()`.
                    let mut w_fd = args[0];
                    let fd = pyre_object::with_roots!(w_fd => {
                        crate::baseobjspace::c_filedescriptor_w(w_fd)
                    })?;
                    // `interp_posix.fchdir`: retry on EINTR. `rposix.c_fchdir`
                    // releases the GIL and saves errno.
                    loop {
                        let ret = pyre_object::with_roots!(w_fd => unsafe {
                            majit_rlib::rposix::c_fchdir(fd)
                        });
                        if ret == 0 {
                            break;
                        }
                        let err = std::io::Error::from_raw_os_error(
                            majit_rlib::rposix::get_saved_errno(),
                        );
                        crate::builtins::eintr_retry_with(err, |e| io_err(e, ""))?;
                    }
                    Ok(pyre_object::w_none())
                },
                1,
            )
        );

        // PyPy's `_run_forking_function` enters the callback lifecycle
        // immediately. CPython 3.14 checks finalization first, so a refused
        // fork takes no callback lock and signals no thread.
        fn guard_fork_finalization() -> Result<(), crate::PyError> {
            if !crate::module::thread::is_finalizing() {
                return Ok(());
            }
            Err(crate::builtins::finalization_error(Some(
                "can't fork at interpreter shutdown",
            )))
        }

        // os.fork() -> child pid in parent, 0 in child
        crate::__pyre_put_new!(
            ns_slot,
            "fork",
            crate::make_builtin_function_with_arity(
                "fork",
                |_| {
                    crate::module::thread::ensure_thread_atfork();
                    guard_fork_finalization()?;
                    if majit_gc::gc_sync::registered_threads() > 1 {
                        crate::warn::warn_deprecation(
                            "This process is multi-threaded, use of fork() may lead to deadlocks",
                        )?;
                    }
                    let blocked = crate::module::thread::before_external_block();
                    let fork_serial = FORK_SERIALIZER
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    drop(blocked);
                    run_fork_callbacks("before");
                    // pypy/module/imp/moduledef.py:45-47 registers the import
                    // lock's acquire/release/reinit trio in the same ordered
                    // interpreter-level hook lists that contain the app-level
                    // callback dispatcher.  Holding it across the host fork
                    // prevents a child from inheriting a partially initialized
                    // sys.modules entry from another thread.
                    crate::module::imp::interp_imp::before_fork();
                    // A free-threaded fork must snapshot native/Python lock
                    // state while every other mutator is parked.  Enter
                    // through the collector's full request path so no GC
                    // operation can overlap the STW window, and so the child
                    // inherits the quiesce mutex from the thread that survives
                    // rather than from one that vanished.
                    match majit_gc::gc_sync::fork_under_stw(|| {
                        // `rposix.c_fork` is `_nowrapper=True`. The live
                        // errno is read immediately, the way `rposix.fork`
                        // copies it before `gc_thread_after_fork`.
                        let pid = unsafe { majit_rlib::rposix::c_fork() };
                        if pid < 0 {
                            Err(std::io::Error::from_raw_os_error(
                                majit_rlib::rposix::_get_errno(),
                            ))
                        } else {
                            Ok(pid)
                        }
                    }) {
                        Ok(0) => {
                            crate::module::thread::after_fork_child();
                            run_fork_callbacks("child");
                            crate::module::imp::interp_imp::after_fork_child();
                            // CPython's refcounting drops the replaced
                            // `_MainThread` before os.fork() returns, so its
                            // weakref has disappeared from `_dangling` when
                            // child Python code next runs.  A tracing GC needs
                            // an explicit reachability pass for the same
                            // observable result.  Defer `interp_gc.collect`
                            // (`rgc.collect` then `_run_finalizers`) to the
                            // next opcode: collecting here would run while
                            // this native builtin still owns unregistered
                            // Rust-stack temporaries.  A moving full
                            // collection is admissible at that boundary
                            // (`run_failed_attr_finalizers`); a non-moving
                            // oldgen pass is not, because it has no leading
                            // minor and can death-queue a nursery address.
                            crate::executioncontext::PyExecutionContext::schedule_collect_and_run_finalizers();
                            drop(fork_serial);
                            Ok(pyre_object::w_int_new(0))
                        }
                        Ok(pid) => {
                            run_fork_callbacks("parent");
                            crate::module::imp::interp_imp::after_fork_parent()?;
                            drop(fork_serial);
                            Ok(pyre_object::w_int_new(pid as i64))
                        }
                        Err(error) => {
                            run_fork_callbacks("parent");
                            // interp_posix.py:1570-1575 keeps the original
                            // fork OSError if a parent hook also fails.
                            let _ = crate::module::imp::interp_imp::after_fork_parent();
                            drop(fork_serial);
                            Err(io_err(error, ""))
                        }
                    }
                },
                0,
            )
        );

        // PyPy `interp_posix.forkpty` delegates to `_run_forking_function`
        // with kind `"P"`, so forkpty uses the same before/parent/child hook
        // and thread-reinitialization lifecycle as fork above while returning
        // the master descriptor as its second item.
        #[cfg(not(any(feature = "sandbox", target_os = "redox")))]
        crate::__pyre_put_new!(
            ns_slot,
            "forkpty",
            crate::make_builtin_function_with_arity(
                "forkpty",
                |_| {
                    crate::module::thread::ensure_thread_atfork();
                    guard_fork_finalization()?;
                    if majit_gc::gc_sync::registered_threads() > 1 {
                        crate::warn::warn_deprecation(
                            "This process is multi-threaded, use of forkpty() may lead to deadlocks",
                        )?;
                    }
                    let blocked = crate::module::thread::before_external_block();
                    let fork_serial = FORK_SERIALIZER
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    drop(blocked);
                    run_fork_callbacks("before");
                    crate::module::imp::interp_imp::before_fork();
                    match majit_gc::gc_sync::fork_under_stw(|| {
                        // `rposix.c_forkpty` is `_nowrapper=True`. The
                        // master descriptor is the out-parameter;
                        // name/termios/winsize are null in `rposix.forkpty`.
                        let mut master: libc::c_int = -1;
                        let pid = unsafe {
                            majit_rlib::rposix::c_forkpty(
                                &mut master,
                                std::ptr::null_mut(),
                                std::ptr::null_mut::<libc::termios>(),
                                std::ptr::null_mut::<libc::winsize>(),
                            )
                        };
                        if pid < 0 {
                            Err(std::io::Error::from_raw_os_error(
                                majit_rlib::rposix::_get_errno(),
                            ))
                        } else {
                            Ok((pid, master))
                        }
                    }) {
                        Ok((0, master_fd)) => {
                            crate::module::thread::after_fork_child();
                            run_fork_callbacks("child");
                            crate::module::imp::interp_imp::after_fork_child();
                            crate::executioncontext::PyExecutionContext::schedule_collect_and_run_finalizers();
                            drop(fork_serial);
                            let mut fields = pyre_object::gc_roots::RootedItems::new();
                            fields.push(pyre_object::w_int_new(0));
                            fields.push(pyre_object::w_int_new(master_fd as i64));
                            Ok(pyre_object::w_tuple_new(fields.take()))
                        }
                        Ok((pid, master_fd)) => {
                            run_fork_callbacks("parent");
                            crate::module::imp::interp_imp::after_fork_parent()?;
                            drop(fork_serial);
                            let mut fields = pyre_object::gc_roots::RootedItems::new();
                            fields.push(pyre_object::w_int_new(pid as i64));
                            fields.push(pyre_object::w_int_new(master_fd as i64));
                            Ok(pyre_object::w_tuple_new(fields.take()))
                        }
                        Err(error) => {
                            run_fork_callbacks("parent");
                            let _ = crate::module::imp::interp_imp::after_fork_parent();
                            drop(fork_serial);
                            Err(io_err(error, ""))
                        }
                    }
                },
                0,
            )
        );

        // os.getppid() -> int. `rposix.c_getppid` does not release the GIL and
        // saves errno. `rposix.getppid` reports a negative result through
        // `handle_posix_error`.
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "getppid",
            crate::make_builtin_function_with_arity(
                "getppid",
                |_| {
                    let ppid = unsafe { majit_rlib::rposix::c_getppid() };
                    if ppid < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_int_new(ppid as i64))
                },
                0,
            )
        );

        // `interp_posix.getsid` -> `rposix.getsid`. `rposix.c_getsid` saves errno.
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "getsid",
            crate::make_builtin_function_with_arity(
                "getsid",
                |args| {
                    let pid = match args.first() {
                        // interp_posix.py `@unwrap_spec(pid=c_int)`.
                        Some(&obj) => crate::baseobjspace::c_int_w(obj)? as libc::pid_t,
                        None => {
                            return Err(crate::PyError::type_error("getsid() requires 1 argument"));
                        }
                    };
                    let sid = unsafe { majit_rlib::rposix::c_getsid(pid) };
                    if sid < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_int_new(sid as i64))
                },
                1,
            )
        );

        // `rposix.getpgrp`. `GETPGRP_HAVE_ARG` is false, so `rposix.c_getpgrp`
        // takes no argument. It saves errno, and `handle_posix_error` reports
        // a negative result.
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "getpgrp",
            crate::make_builtin_function_with_arity(
                "getpgrp",
                |_| {
                    let pgrp = unsafe { majit_rlib::rposix::c_getpgrp() };
                    if pgrp < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_int_new(pgrp as i64))
                },
                0,
            )
        );

        // `interp_posix.getpgid` — another process's group, which can be one
        // this process may not ask about. `rposix.c_getpgid` saves errno.
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "getpgid",
            crate::make_builtin_function_with_arity(
                "getpgid",
                |args| {
                    let pid = match args.first() {
                        // interp_posix.py `@unwrap_spec(pid=c_int)`.
                        Some(&obj) => crate::baseobjspace::c_int_w(obj)? as libc::pid_t,
                        None => {
                            return Err(crate::PyError::type_error(
                                "getpgid() requires 1 argument",
                            ));
                        }
                    };
                    let pgid = unsafe { majit_rlib::rposix::c_getpgid(pid) };
                    if pgid < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_int_new(pgid as i64))
                },
                1,
            )
        );

        // `interp_posix.setpgid` -> `rposix.setpgid`, which discards
        // `handle_posix_error`'s result and returns None. `rposix.c_setpgid`
        // releases the GIL and saves errno.
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "setpgid",
            crate::make_builtin_function_with_arity(
                "setpgid",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("setpgid() requires 2 arguments"));
                    }
                    // interp_posix.py `@unwrap_spec(pid=c_int, pgrp=c_int)`.
                    let w_pid = args[0];
                    let mut w_pgrp = args[1];
                    let pid = pyre_object::with_roots!(w_pgrp =>
                        crate::baseobjspace::c_int_w(w_pid))?
                        as libc::pid_t;
                    let pgrp = crate::baseobjspace::c_int_w(w_pgrp)? as libc::pid_t;
                    let ret = unsafe { majit_rlib::rposix::c_setpgid(pid, pgrp) };
                    if ret < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            )
        );

        // `interp_posix.setpgrp` -> `rposix.setpgrp`. `SETPGRP_HAVE_ARG` is
        // false, so `rposix.c_setpgrp` takes no argument. The wrapper returns
        // None.
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "setpgrp",
            crate::make_builtin_function_with_arity(
                "setpgrp",
                |_| {
                    let ret = unsafe { majit_rlib::rposix::c_setpgrp() };
                    if ret < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_none())
                },
                0,
            )
        );

        // `interp_posix.setsid` -> `rposix.setsid`. The session id
        // `handle_posix_error` returns is dropped; the builtin answers None.
        // `rposix.c_setsid` releases the GIL and saves errno.
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "setsid",
            crate::make_builtin_function_with_arity(
                "setsid",
                |_| {
                    let sid = unsafe { majit_rlib::rposix::c_setsid() };
                    if sid < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_none())
                },
                0,
            )
        );

        // The six user/group ID setters share the `c_uid_t` conversion.
        // `rposix.c_setuid` and its siblings save errno. `handle_posix_error`
        // reports a negative result and does not retry EINTR.
        #[cfg(not(feature = "sandbox"))]
        fn set_one_id(
            args: &[PyObjectRef],
            name: &str,
            setter: fn(u32) -> libc::c_int,
        ) -> Result<PyObjectRef, crate::PyError> {
            let id = match args.first() {
                Some(&obj) => crate::baseobjspace::c_uid_t_w(obj)?,
                None => {
                    return Err(crate::PyError::type_error(format!(
                        "{name}() requires 1 argument"
                    )));
                }
            };
            if setter(id) < 0 {
                return Err(io_err(
                    std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                    "",
                ));
            }
            Ok(pyre_object::w_none())
        }

        #[cfg(not(feature = "sandbox"))]
        fn set_two_ids(
            args: &[PyObjectRef],
            name: &str,
            setter: fn(u32, u32) -> libc::c_int,
        ) -> Result<PyObjectRef, crate::PyError> {
            if args.len() < 2 {
                return Err(crate::PyError::type_error(format!(
                    "{name}() requires 2 arguments"
                )));
            }
            let mut w_first = args[0];
            let mut w_second = args[1];
            let first = pyre_object::with_roots!(w_first, w_second => crate::baseobjspace::c_uid_t_w(w_first))?;
            let second = crate::baseobjspace::c_uid_t_w(w_second)?;
            if setter(first, second) < 0 {
                return Err(io_err(
                    std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                    "",
                ));
            }
            Ok(pyre_object::w_none())
        }

        #[cfg(not(feature = "sandbox"))]
        fn setuid(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            set_one_id(args, "setuid", |uid| unsafe {
                majit_rlib::rposix::c_setuid(uid as libc::uid_t)
            })
        }

        #[cfg(not(feature = "sandbox"))]
        fn seteuid(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            set_one_id(args, "seteuid", |euid| unsafe {
                majit_rlib::rposix::c_seteuid(euid as libc::uid_t)
            })
        }

        #[cfg(not(feature = "sandbox"))]
        fn setgid(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            set_one_id(args, "setgid", |gid| unsafe {
                majit_rlib::rposix::c_setgid(gid as libc::gid_t)
            })
        }

        #[cfg(not(feature = "sandbox"))]
        fn setegid(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            set_one_id(args, "setegid", |egid| unsafe {
                majit_rlib::rposix::c_setegid(egid as libc::gid_t)
            })
        }

        #[cfg(not(feature = "sandbox"))]
        fn setreuid(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            set_two_ids(args, "setreuid", |ruid, euid| unsafe {
                majit_rlib::rposix::c_setreuid(ruid as libc::uid_t, euid as libc::uid_t)
            })
        }

        #[cfg(not(feature = "sandbox"))]
        fn setregid(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            set_two_ids(args, "setregid", |rgid, egid| unsafe {
                majit_rlib::rposix::c_setregid(rgid as libc::gid_t, egid as libc::gid_t)
            })
        }

        #[cfg(not(feature = "sandbox"))]
        for (name, function, arity) in [
            ("setuid", setuid as crate::gateway::BuiltinCodeFn, 1),
            ("seteuid", seteuid as crate::gateway::BuiltinCodeFn, 1),
            ("setgid", setgid as crate::gateway::BuiltinCodeFn, 1),
            ("setegid", setegid as crate::gateway::BuiltinCodeFn, 1),
            ("setreuid", setreuid as crate::gateway::BuiltinCodeFn, 2),
            ("setregid", setregid as crate::gateway::BuiltinCodeFn, 2),
        ] {
            crate::__pyre_put_new!(
                ns_slot,
                name,
                crate::make_builtin_function_with_arity(name, function, arity)
            );
        }

        // `interp_posix.ctermid` -> `rposix.ctermid`. The call is handed a
        // null pointer and reads the static buffer it answers with. The
        // result is a filename, so it is decoded the way every other name
        // from the host is. `rposix.c_ctermid` releases the GIL and does not
        // save errno.
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "ctermid",
            crate::make_builtin_function_with_arity(
                "ctermid",
                |_| {
                    let name = unsafe { majit_rlib::rposix::c_ctermid(std::ptr::null_mut()) };
                    if name.is_null() {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    let bytes = unsafe { std::ffi::CStr::from_ptr(name) };
                    Ok(crate::gateway::fsdecode_filename_bytes(bytes.to_bytes()))
                },
                0,
            )
        );

        // os.waitpid(pid, options) -> (pid, status)
        crate::__pyre_put_new!(
            ns_slot,
            "waitpid",
            crate::make_builtin_function_with_arity(
                "waitpid",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("waitpid() requires 2 arguments"));
                    }
                    // interp_posix.py `@unwrap_spec(pid=c_int, options=c_int)`.
                    let w_pid = args[0];
                    let mut w_options = args[1];
                    let pid = pyre_object::with_roots!(w_options =>
                        crate::baseobjspace::c_int_w(w_pid))?
                        as libc::pid_t;
                    let options = crate::baseobjspace::c_int_w(w_options)?;
                    let mut status: i32 = 0;
                    // interp_posix.py `waitpid`: retry on EINTR.
                    // `rposix.c_waitpid` releases the GIL and saves errno.
                    // `0` is a successful `WNOHANG` answer.
                    let res = loop {
                        let res =
                            unsafe { majit_rlib::rposix::c_waitpid(pid, &mut status, options) };
                        if res >= 0 {
                            break res;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            |e| io_err(e, ""),
                        )?;
                    };
                    let mut fields = pyre_object::gc_roots::RootedItems::new();
                    fields.push(pyre_object::w_int_new(res as i64));
                    fields.push(pyre_object::w_int_new(status as i64));
                    Ok(pyre_object::w_tuple_new(fields.take()))
                },
                2,
            )
        );

        // os.wait() -> (pid, status)
        crate::__pyre_put_new!(
            ns_slot,
            "wait",
            crate::make_builtin_function_with_arity(
                "wait",
                |_| {
                    let mut status: i32 = 0;
                    // `app_posix.wait` is `posix.waitpid(-1, 0)`, so it inherits
                    // that call's `eintr_retry=True` rather than surfacing the
                    // interruption of its own.
                    // `rposix.c_waitpid` releases the GIL and saves errno.
                    let res = loop {
                        let res = unsafe { majit_rlib::rposix::c_waitpid(-1, &mut status, 0) };
                        if res >= 0 {
                            break res;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            |e| io_err(e, ""),
                        )?;
                    };
                    let mut fields = pyre_object::gc_roots::RootedItems::new();
                    fields.push(pyre_object::w_int_new(res as i64));
                    fields.push(pyre_object::w_int_new(status as i64));
                    Ok(pyre_object::w_tuple_new(fields.take()))
                },
                0,
            )
        );

        // `app_posix.wait3` / `app_posix.wait4` import `_pypy_wait`, which
        // retries EINTR and wraps the rusage as `resource.struct_rusage`.
        fn wait_rusage_to_py(
            ru: rustpython_host_env::resource::RUsage,
        ) -> Result<PyObjectRef, crate::PyError> {
            let resource = crate::importing::get_builtin_module("resource").ok_or_else(|| {
                crate::PyError::runtime_error("resource module is not initialized")
            })?;
            let cls = crate::baseobjspace::getattr_str(resource, "struct_rusage")?;
            // Field boxing and the argument tuple can collect, so the type is
            // pinned before the first allocation and re-read for the call.
            let _roots = pyre_object::gc_roots::push_roots();
            let cls_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(cls);
            let tv_to_f = |tv: libc::timeval| tv.tv_sec as f64 + (tv.tv_usec as f64) * 1e-6;
            let mut fields = pyre_object::gc_roots::RootedItems::new();
            fields.push(pyre_object::floatobject::w_float_new(tv_to_f(ru.ru_utime)));
            fields.push(pyre_object::floatobject::w_float_new(tv_to_f(ru.ru_stime)));
            fields.push(pyre_object::w_int_new(ru.ru_maxrss));
            fields.push(pyre_object::w_int_new(ru.ru_ixrss));
            fields.push(pyre_object::w_int_new(ru.ru_idrss));
            fields.push(pyre_object::w_int_new(ru.ru_isrss));
            fields.push(pyre_object::w_int_new(ru.ru_minflt));
            fields.push(pyre_object::w_int_new(ru.ru_majflt));
            fields.push(pyre_object::w_int_new(ru.ru_nswap));
            fields.push(pyre_object::w_int_new(ru.ru_inblock));
            fields.push(pyre_object::w_int_new(ru.ru_oublock));
            fields.push(pyre_object::w_int_new(ru.ru_msgsnd));
            fields.push(pyre_object::w_int_new(ru.ru_msgrcv));
            fields.push(pyre_object::w_int_new(ru.ru_nsignals));
            fields.push(pyre_object::w_int_new(ru.ru_nvcsw));
            fields.push(pyre_object::w_int_new(ru.ru_nivcsw));
            // `_make_struct_rusage` calls `struct_rusage((...))`, so a
            // rebound type's constructor is observable.
            let tuple = pyre_object::w_tuple_new(fields.take());
            crate::call::call_function_impl_result(
                pyre_object::gc_roots::shadow_stack_get(cls_slot),
                &[tuple],
            )
        }

        fn wait_with_rusage<F>(wait: F) -> Result<PyObjectRef, crate::PyError>
        where
            F: Fn() -> std::io::Result<(libc::pid_t, i32, rustpython_host_env::resource::RUsage)>,
        {
            loop {
                let result = {
                    let _blocked = crate::module::thread::before_external_block();
                    wait()
                };
                match result {
                    Ok((pid, status, ru)) => {
                        let rusage = wait_rusage_to_py(ru)?;
                        let mut fields = pyre_object::gc_roots::RootedItems::new();
                        fields.push(pyre_object::w_int_new(pid as i64));
                        fields.push(pyre_object::w_int_new(status as i64));
                        fields.push(rusage);
                        return Ok(pyre_object::w_tuple_new(fields.take()));
                    }
                    Err(e) => crate::builtins::eintr_retry_with(e, |e| io_err(e, ""))?,
                }
            }
        }

        crate::__pyre_put_new!(
            ns_slot,
            "wait3",
            crate::make_builtin_function_with_arity(
                "wait3",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("wait3() requires 1 argument"));
                    }
                    let options = crate::baseobjspace::c_int_w(args[0])?;
                    wait_with_rusage(|| host_posix::wait3(options))
                },
                1,
            )
        );

        crate::__pyre_put_new!(
            ns_slot,
            "wait4",
            crate::make_builtin_function_with_arity(
                "wait4",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("wait4() requires 2 arguments"));
                    }
                    let w_pid = args[0];
                    let mut w_options = args[1];
                    let pid = pyre_object::with_roots!(w_options =>
                        crate::baseobjspace::c_int_w(w_pid))?
                        as libc::pid_t;
                    let options = crate::baseobjspace::c_int_w(w_options)?;
                    wait_with_rusage(|| host_posix::wait4(pid, options))
                },
                2,
            )
        );

        // os._exit(code) — immediate process exit, no cleanup.
        crate::__pyre_put_new!(
            ns_slot,
            "_exit",
            crate::make_builtin_function_with_arity(
                "_exit",
                |args| {
                    let code = match args.first() {
                        // interp_posix.py `@unwrap_spec(status=c_int)`.
                        Some(&o) => crate::baseobjspace::c_int_w(o)?,
                        None => {
                            return Err(crate::PyError::type_error("_exit() requires 1 argument"));
                        }
                    };
                    // `rposix.c_exit` is `_exit`. It does not return.
                    unsafe { majit_rlib::rposix::c_exit(code) };
                    unreachable!()
                },
                1,
            )
        );

        // Wait-status decoding macros (WIFEXITED/WEXITSTATUS/...): override
        // the noop stubs registered above with the libc bit-math.
        macro_rules! reg_wstatus {
            ($name:literal, |$s:ident| $body:expr) => {crate::__pyre_put_new!(
                ns_slot,
                $name,
                crate::make_builtin_function_with_arity(
                    $name,
                    |args| {
                        let $s = match args.first() {
                            // interp_posix.py `declare_new_w_star`
                            // types every wait macro `@unwrap_spec(status=c_int)`.
                            Some(&o) => crate::baseobjspace::c_int_w(o)?,
                            None => {
                                return Err(crate::PyError::type_error(concat!(
                                    $name,
                                    "() requires 1 argument"
                                )));
                            }
                        };
                        Ok($body)
                    },
                    1,
                )
            );};
        }
        reg_wstatus!("WIFEXITED", |s| pyre_object::w_bool_from(libc::WIFEXITED(
            s
        )));
        reg_wstatus!("WEXITSTATUS", |s| pyre_object::w_int_new(
            libc::WEXITSTATUS(s) as i64
        ));
        reg_wstatus!("WIFSIGNALED", |s| pyre_object::w_bool_from(
            libc::WIFSIGNALED(s)
        ));
        reg_wstatus!("WTERMSIG", |s| pyre_object::w_int_new(
            libc::WTERMSIG(s) as i64
        ));
        reg_wstatus!("WIFSTOPPED", |s| pyre_object::w_bool_from(
            libc::WIFSTOPPED(s)
        ));
        reg_wstatus!("WSTOPSIG", |s| pyre_object::w_int_new(
            libc::WSTOPSIG(s) as i64
        ));

        // Wait option flags — override the `0` placeholders registered above
        // with their real libc values (os.WNOHANG must be non-zero for
        // subprocess.poll()).
        crate::__pyre_put_new!(ns_slot, "WNOHANG", pyre_object::w_int_new(libc::WNOHANG as i64));
        crate::__pyre_put_new!(ns_slot, "WUNTRACED", pyre_object::w_int_new(libc::WUNTRACED as i64));
        crate::__pyre_put_new!(
            ns_slot,
            "WCONTINUED",
            pyre_object::w_int_new(libc::WCONTINUED as i64)
        );
        // The states `waitid` is asked to report on. They were registered above
        // as calls answering `None`, which is neither the number nor a name a
        // caller can tell apart from one.
        for (name, val) in [
            ("WEXITED", libc::WEXITED as i64),
            ("WSTOPPED", libc::WSTOPPED as i64),
            ("WNOWAIT", libc::WNOWAIT as i64),
        ] {
            crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
        }
        // Which process `waitid` is asked about, and what `si_code` says
        // happened to it once it answers.
        for (name, val) in [
            ("P_ALL", libc::P_ALL as i64),
            ("P_PID", libc::P_PID as i64),
            ("P_PGID", libc::P_PGID as i64),
            ("CLD_EXITED", libc::CLD_EXITED as i64),
            ("CLD_KILLED", libc::CLD_KILLED as i64),
            ("CLD_DUMPED", libc::CLD_DUMPED as i64),
            ("CLD_TRAPPED", libc::CLD_TRAPPED as i64),
            ("CLD_STOPPED", libc::CLD_STOPPED as i64),
            ("CLD_CONTINUED", libc::CLD_CONTINUED as i64),
        ] {
            crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(val));
        }

        // os.waitid(idtype, id, options) -> waitid_result | None
        //
        // `os_waitid_impl` — the call reports on a child without reaping it
        // when WNOWAIT is among the options, which is what separates it from
        // `waitpid`. A zero `si_pid` means the options asked about a state no
        // child is in (WNOHANG with nothing to report), and that is `None`
        // rather than a result whose every field is zero.
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(ns_slot, "waitid_result", waitid_result_seq_type());
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "waitid",
            crate::make_builtin_function_with_arity(
                "waitid",
                |args| {
                    if args.len() != 3 {
                        return Err(crate::PyError::type_error(format!(
                            "waitid expected 3 arguments, got {}",
                            args.len(),
                        )));
                    }
                    let w_idtype = args[0];
                    let mut w_id = args[1];
                    let mut w_options = args[2];
                    let idtype = pyre_object::with_roots!(w_id, w_options =>
                        crate::baseobjspace::c_int_w(w_idtype))?
                        as libc::idtype_t;
                    let w_id_index = pyre_object::with_roots!(w_options =>
                        crate::baseobjspace::space_index(w_id))?;
                    let id = pyre_object::with_roots!(w_options =>
                        crate::baseobjspace::int_w(w_id_index))
                    .map_err(|_| {
                        crate::PyError::overflow_error("Python int too large to convert to C long")
                    })? as libc::id_t;
                    let options = crate::baseobjspace::c_int_w(w_options)?;
                    // `si.si_pid = 0` before the call: the field is what the
                    // "nothing to report" answer is read out of, and a call
                    // that reports nothing does not write it.
                    let mut si: libc::siginfo_t = unsafe { std::mem::zeroed() };
                    loop {
                        let (ret, errno) =
                            crate::module::thread::call_external_function(|| unsafe {
                                libc::waitid(idtype, id, &mut si, options)
                            });
                        if ret >= 0 {
                            break;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(errno),
                            |e| errno_err(e.raw_os_error().unwrap_or(0), ""),
                        )?;
                    }
                    // The three that live in the union `siginfo_t` keeps its
                    // process fields in are read through the accessors that
                    // name which arm is meant; `si_signo` and `si_code` are
                    // outside it.
                    let (pid, uid, status) = unsafe { (si.si_pid(), si.si_uid(), si.si_status()) };
                    if pid == 0 {
                        return Ok(pyre_object::w_none());
                    }
                    let mut fields = pyre_object::gc_roots::RootedItems::new();
                    fields.push(pyre_object::w_int_new(pid as i64));
                    fields.push(pyre_object::w_int_new(uid as i64));
                    fields.push(pyre_object::w_int_new(si.si_signo as i64));
                    fields.push(pyre_object::w_int_new(status as i64));
                    fields.push(pyre_object::w_int_new(si.si_code as i64));
                    Ok(crate::_structseq::new_instance(
                        waitid_result_seq_type(),
                        fields.take(),
                    ))
                },
                3,
            )
        );

        // os.dup(fd) -> new_fd
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "dup",
            crate::make_builtin_function_with_arity(
                "dup",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("dup() requires 1 argument"));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int)`.
                    // `interp_posix.dup` is `rposix.dup(fd, inheritable=False)`
                    // → `c_dup_noninheritable`.
                    let mut w_fd = args[0];
                    let fd = pyre_object::with_roots!(w_fd => crate::baseobjspace::c_int_w(w_fd))?;
                    let n = pyre_object::with_roots!(w_fd => unsafe {
                        majit_rlib::rposix::c_dup_noninheritable(fd)
                    });
                    if n < 0 {
                        return Err(errno_err(majit_rlib::rposix::get_saved_errno(), ""));
                    }
                    Ok(pyre_object::w_int_new(n as i64))
                },
                1,
            )
        );

        // os.dup2(fd, fd2, inheritable=True) -> fd2
        //
        // Carries a `Signature`, so `inheritable` binds by name.  Registered
        // raw it did not: the trailing `__pyre_kw__` marker dict was never
        // split off the argument slice, so it landed in the third positional
        // slot and read truthy, and `dup2(fd, fd2, inheritable=False)`
        // returned an *inheritable* descriptor that an exec would carry.
        //
        // The arguments stay `PyObjectRef` and are unwrapped in the body: the
        // macro's bare `i32` binding is a raw `w_int_get_value` cast, which
        // would read a non-int argument's payload instead of reporting it.
        #[cfg(not(feature = "sandbox"))]
        #[crate::pyre_function]
        fn dup2(
            fd: pyre_object::PyObjectRef,
            mut fd2: pyre_object::PyObjectRef,
            mut inheritable: Option<pyre_object::PyObjectRef>,
        ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
            // interp_posix.py `@unwrap_spec(fd=c_int, fd2=c_int, inheritable=bool)`.
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.pin_roots(&[fd2, inheritable.unwrap_or(pyre_object::PY_NULL)]);
            let fd = crate::baseobjspace::c_int_w(fd);
            fd2 = roots.get(base);
            let w = roots.get(base + 1);
            inheritable = if w.is_null() { None } else { Some(w) };
            drop(roots);
            let fd = fd?;
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.pin_roots(&[inheritable.unwrap_or(pyre_object::PY_NULL)]);
            let fd2 = crate::baseobjspace::c_int_w(fd2);
            let w = roots.get(base);
            inheritable = if w.is_null() { None } else { Some(w) };
            drop(roots);
            let fd2 = fd2?;
            let inheritable = match inheritable {
                Some(w) => crate::baseobjspace::is_true(w)?,
                None => true,
            };
            // `rposix.dup2`: inheritable uses `c_dup2`; otherwise
            // `c_dup2_noninheritable`, which returns 0 on the HAVE_DUP3
            // path. `interp_posix.dup2` answers `fd2`.
            let n = if inheritable {
                unsafe { majit_rlib::rposix::c_dup2(fd, fd2) }
            } else {
                unsafe { majit_rlib::rposix::c_dup2_noninheritable(fd, fd2) }
            };
            if n < 0 {
                return Err(errno_err(majit_rlib::rposix::get_saved_errno(), ""));
            }
            Ok(pyre_object::w_int_new(fd2 as i64))
        }

        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "dup2",
            crate::make_builtin_function_with_arity_and_maybe_sig(
                "dup2",
                dup2,
                dup2_pyre_arity(),
                dup2_pyre_sig(),
            )
        );

        // os.fsync(fd)
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "fsync",
            crate::make_builtin_function_with_arity(
                "fsync",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("fsync() requires 1 argument"));
                    }
                    // interp_posix.py `fsync(space, w_fd)` unwraps
                    // through `space.c_filedescriptor_w`.
                    let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                    // `interp_posix.fsync`: retry on EINTR. `rposix.c_fsync`
                    // releases the GIL and saves errno.
                    loop {
                        let r = unsafe { majit_rlib::rposix::c_fsync(fd) };
                        if r >= 0 {
                            break;
                        }
                        let errno = majit_rlib::rposix::get_saved_errno();
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(errno),
                            |e| errno_err(e.raw_os_error().unwrap_or(0), ""),
                        )?;
                    }
                    Ok(pyre_object::w_none())
                },
                1,
            )
        );

        // os.fdatasync(fd). `rposix.c_fdatasync` is `external('fdatasync')`.
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "fdatasync",
            crate::make_builtin_function_with_arity(
                "fdatasync",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "fdatasync() requires 1 argument",
                        ));
                    }
                    // interp_posix.py `fdatasync(space, w_fd)` unwraps
                    // through `space.c_filedescriptor_w`.
                    let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                    // `interp_posix.fdatasync`: retry on EINTR. `rposix.c_fdatasync`
                    // releases the GIL and saves errno.
                    loop {
                        let r = unsafe { majit_rlib::rposix::c_fdatasync(fd) };
                        if r >= 0 {
                            break;
                        }
                        let errno = majit_rlib::rposix::get_saved_errno();
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(errno),
                            |e| errno_err(e.raw_os_error().unwrap_or(0), ""),
                        )?;
                    }
                    Ok(pyre_object::w_none())
                },
                1,
            )
        );

        // interp_posix.py:407-412: retry EINTR, propagate every other OSError.
        // The retry is `eintr_retry=True`, which runs the pending Python signal
        // handlers before going back to the call — a handler that raises ends
        // the loop there, and one that disarms the timer stops the interruption
        // recurring. Retrying on the bare errno would spin without ever giving
        // that handler a turn.
        //
        // Which filename the caller then reports is its own: `os.ftruncate` was
        // given no name to report, while `os.truncate` names the one it opened.
        #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
        fn ftruncate_retry(
            fd: libc::c_int,
            length: libc::off_t,
            wrap: impl Fn(i32) -> crate::PyError,
        ) -> Result<(), crate::PyError> {
            // `rposix.c_ftruncate` is `macro=libc::ftruncate`. It releases
            // the GIL and saves errno.
            loop {
                let result = unsafe { majit_rlib::rposix::c_ftruncate(fd, length) };
                if result >= 0 {
                    return Ok(());
                }
                let errno = majit_rlib::rposix::get_saved_errno();
                crate::builtins::eintr_retry_with(std::io::Error::from_raw_os_error(errno), |e| {
                    wrap(e.raw_os_error().unwrap_or(0))
                })?;
            }
        }

        // os.truncate(path, length) -> None
        //
        // interp_posix.py takes a descriptor as it stands and opens a
        // name write-only, truncates whichever it ended up with, and closes
        // only the one it opened itself. The descriptor form is what
        // HAVE_FTRUNCATE advertises through `os.py:149`.
        #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
        crate::__pyre_put_new!(
            ns_slot,
            "truncate",
            crate::make_builtin_function_with_arity(
                "truncate",
                |args| {
                    if args.len() != 2 {
                        return Err(crate::PyError::type_error(format!(
                            "truncate expected 2 arguments, got {}",
                            args.len(),
                        )));
                    }
                    let w_path = args[0];
                    let mut w_length = args[1];
                    // The path owns a bracket of its own, above this one; this
                    // one stays open until the path is gone.
                    let length_roots = pyre_object::gc_roots::push_roots();
                    let length_base = length_roots.pin_roots(&[w_length]);
                    let path =
                        crate::gateway::fsencode_path_or_fd_w(w_path, "truncate", HAVE_FTRUNCATE);
                    w_length = length_roots.get(length_base);
                    let path = path?;
                    let length = truncate_length_w(w_length)?;
                    if path.is_fd {
                        ftruncate_retry(path.as_fd, length, |e| errno_err(e, ""))?;
                        return Ok(pyre_object::w_none());
                    }
                    let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    // `truncate` opens the name through the module's own `open`
                    // (`interp_posix.py`), so this is that call and not a
                    // bare syscall: `inheritable=False`, the interpreter
                    // released for the duration — a FIFO with no reader waits
                    // here until another thread opens the other end — and an
                    // interrupted open re-issued after the signal handler has
                    // run rather than reported as `InterruptedError`.
                    let fd = loop {
                        // `rposix.c_open` releases the GIL and saves errno.
                        let fd = unsafe {
                            majit_rlib::rposix::c_open(
                                c_path.as_ptr(),
                                libc::O_WRONLY | libc::O_CLOEXEC,
                                0,
                            )
                        };
                        if fd >= 0 {
                            break fd;
                        }
                        let errno = majit_rlib::rposix::get_saved_errno();
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(errno),
                            |e| {
                                errno_err_with_filename(
                                    e.raw_os_error().unwrap_or(0),
                                    path.w_path(),
                                )
                            },
                        )?;
                    };
                    let truncated =
                        ftruncate_retry(fd, length, |e| errno_err_with_filename(e, path.w_path()));
                    // `interp_posix.py:429-431` closes the descriptor it opened
                    // in a `finally`, through the module's own `close` — so a
                    // writeback error the close is the first to see is the
                    // caller's, not something `truncate` reports success over.
                    // The truncation's own failure is the one reported when
                    // both fail, which is the order the `finally` gives them.
                    let closed = unsafe { majit_rlib::rposix::c_close(fd) };
                    let close_errno = (closed < 0).then(majit_rlib::rposix::get_saved_errno);
                    truncated?;
                    if let Some(errno) = close_errno {
                        return Err(errno_err_with_filename(errno, path.w_path()));
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            )
        );

        // rpython/rlib/rposix.py `ftruncate(fd, length)` — this must be a
        // real fd mutation whenever HAVE_FTRUNCATE is advertised.  Shared
        // memory sizes its newly-created object through this call before
        // mapping it.
        #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
        crate::__pyre_put_new!(
            ns_slot,
            "ftruncate",
            crate::make_builtin_function_with_arity(
                "ftruncate",
                |args| {
                    // interp_posix.py `@unwrap_spec(fd=c_int, length=r_longlong)`.
                    // CPython 3.14's clinic gateway accepts `__index__` but not
                    // `__int__`; that newer coercion wins over PyPy's legacy
                    // `gateway_int_w` behavior.
                    if args.len() != 2 {
                        return Err(crate::PyError::type_error(format!(
                            "ftruncate expected 2 arguments, got {}",
                            args.len(),
                        )));
                    }
                    let w_fd = args[0];
                    let mut w_length = args[1];
                    let w_fd = pyre_object::with_roots!(w_length => crate::baseobjspace::space_index(w_fd))?;
                    let fd_value = pyre_object::with_roots!(w_length =>
                        crate::baseobjspace::int_w(w_fd))
                    .map_err(|err| {
                        if err.kind == crate::PyErrorKind::OverflowError {
                            crate::PyError::overflow_error(
                                "Python int too large to convert to C int",
                            )
                        } else {
                            err
                        }
                    })?;
                    let fd = libc::c_int::try_from(fd_value).map_err(|_| {
                        crate::PyError::overflow_error("Python int too large to convert to C int")
                    })?;
                    let length = truncate_length_w(w_length)?;
                    ftruncate_retry(fd, length, |e| errno_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                2,
            )
        );

        // os.lockf(fd, cmd, len) -> None
        //
        // `interp_posix.py` — one `lockf` under the `eintr_retry`
        // loop, so a lock that waits and is interrupted goes back to waiting
        // after the signal handler has run rather than surfacing as
        // `InterruptedError`. F_LOCK blocks, so it is put through the call
        // gate the way every other waiting call here is.
        #[cfg(all(unix, not(feature = "sandbox")))]
        crate::__pyre_put_new!(
            ns_slot,
            "lockf",
            crate::make_builtin_function_with_arity(
                "lockf",
                |args| {
                    if args.len() != 3 {
                        return Err(crate::PyError::type_error(format!(
                            "lockf expected 3 arguments, got {}",
                            args.len(),
                        )));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int, cmd=c_int,
                    // length=r_longlong)` — the length is an offset, so it is
                    // the same conversion `ftruncate` gives one.
                    let mut w_fd = args[0];
                    let mut w_cmd = args[1];
                    let mut w_length = args[2];
                    let fd = pyre_object::with_roots!(w_fd, w_cmd, w_length =>
                        crate::baseobjspace::c_int_w(w_fd))?;
                    let cmd = pyre_object::with_roots!(w_cmd, w_length => crate::baseobjspace::c_int_w(w_cmd))?;
                    let length = truncate_length_w(w_length)?;
                    // `rposix.c_lockf` releases the GIL and saves errno.
                    loop {
                        let ret = unsafe { majit_rlib::rposix::c_lockf(fd, cmd, length) };
                        if ret == 0 {
                            break;
                        }
                        let errno = majit_rlib::rposix::get_saved_errno();
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(errno),
                            |e| errno_err(e.raw_os_error().unwrap_or(0), ""),
                        )?;
                    }
                    // `os_lockf_impl` answers `None`. `interp_posix.py:3012`
                    // answers the `0` the call returns on success, which 3.14
                    // — the oracle the parity suite reads — does not carry.
                    Ok(pyre_object::w_none())
                },
                3,
            )
        );

        // os.mkfifo(path, mode=0o666, *, dir_fd=None) -> None
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "mkfifo",
            crate::make_builtin_function("mkfifo", |args| {
                let (bound, mut kwargs) =
                    bind_path_args(args, "mkfifo", &["path", "mode"], 1, &["dir_fd"])?;
                let roots = pyre_object::gc_roots::push_roots();
                let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
                let path = crate::gateway::fsencode_path_or_fd_w(
                    bound[0].expect("path is required"),
                    "mkfifo",
                    false,
                );
                let w = roots.get(base);
                kwargs = if w.is_null() { None } else { Some(w) };
                let path = path?;
                // interp_posix.py `@unwrap_spec(mode=c_int, ...)`.
                let mode = match bound[1] {
                    Some(value) => {
                        let roots = pyre_object::gc_roots::push_roots();
                        let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
                        let r = crate::baseobjspace::c_int_w(value);
                        let w = roots.get(base);
                        kwargs = if w.is_null() { None } else { Some(w) };
                        drop(roots);
                        r? as libc::mode_t
                    }
                    None => 0o666,
                };
                // `mkfifo` types `dir_fd` as `DirFD(rposix.HAVE_MKFIFOAT)`
                // (`interp_posix.py`).
                let dir_fd = dir_fd_kwarg(kwargs, HAVE_MKFIFOAT)?;
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                // `mkfifoat` resolves the name against the descriptor
                // (`rposix.py`). The no-descriptor call is `rposix.c_mkfifo`,
                // which releases the GIL and saves errno.
                // interp_posix.py `mkfifo`: retry on EINTR.
                loop {
                    let (r, errno) = match dir_fd {
                        Some(dir_fd) => {
                            let r = unsafe {
                                majit_rlib::rposix::c_mkfifoat(dir_fd, c_path.as_ptr(), mode)
                            };
                            (r, majit_rlib::rposix::get_saved_errno())
                        }
                        None => {
                            let r = unsafe { majit_rlib::rposix::c_mkfifo(c_path.as_ptr(), mode) };
                            (r, majit_rlib::rposix::get_saved_errno())
                        }
                    };
                    if r >= 0 {
                        break;
                    }
                    crate::builtins::eintr_retry_with(
                        std::io::Error::from_raw_os_error(errno),
                        |e| io_err_with_filename(e, path.w_path()),
                    )?;
                }
                Ok(pyre_object::w_none())
            })
        );

        // os.mknod(path, mode=0o600, device=0, *, dir_fd=None) -> None
        // The node's kind is carried in `mode` alongside its permissions, so
        // an unadorned `mode` asks for a regular file — which is why the plain
        // call is the one a non-root process cannot make. `moduledef.py:160`
        // registers this only where the host has `mknod` at all.
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "mknod",
            crate::make_builtin_function("mknod", |args| {
                let (bound, mut kwargs) =
                    bind_path_args(args, "mknod", &["path", "mode", "device"], 1, &["dir_fd"])?;
                let roots = pyre_object::gc_roots::push_roots();
                let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
                let path = crate::gateway::fsencode_path_or_fd_w(
                    bound[0].expect("path is required"),
                    "mknod",
                    false,
                );
                let w = roots.get(base);
                kwargs = if w.is_null() { None } else { Some(w) };
                let path = path?;
                // interp_posix.py `@unwrap_spec(mode=c_int, device=c_int,
                // ...)`.
                let mode = match bound[1] {
                    Some(value) => {
                        let roots = pyre_object::gc_roots::push_roots();
                        let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
                        let r = crate::baseobjspace::c_int_w(value);
                        let w = roots.get(base);
                        kwargs = if w.is_null() { None } else { Some(w) };
                        drop(roots);
                        r? as libc::mode_t
                    }
                    None => 0o600,
                };
                // interp_posix.py `@unwrap_spec(..., device=c_int)`.
                // `rposix.c_mknodat` takes `rffi.INT`; `rposix.c_mknod`
                // spells the same parameter `rffi.INT` and the libc
                // declaration widens it to `dev_t`.
                let device = match bound[2] {
                    Some(value) => {
                        let roots = pyre_object::gc_roots::push_roots();
                        let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
                        let r = crate::baseobjspace::c_int_w(value);
                        let w = roots.get(base);
                        kwargs = if w.is_null() { None } else { Some(w) };
                        drop(roots);
                        r?
                    }
                    None => 0,
                };
                // `mknod` types `dir_fd` as `DirFD(rposix.HAVE_MKNODAT)`
                // (`interp_posix.py`).
                let dir_fd = dir_fd_kwarg(kwargs, HAVE_MKNODAT)?;
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                // `mknodat` resolves the name against the descriptor
                // (`rposix.py`). The no-descriptor call is `rposix.c_mknod`,
                // which releases the GIL and saves errno.
                // interp_posix.py `mknod`: retry on EINTR.
                loop {
                    let (r, errno) = match dir_fd {
                        Some(dir_fd) => {
                            let r = unsafe {
                                majit_rlib::rposix::c_mknodat(dir_fd, c_path.as_ptr(), mode, device)
                            };
                            (r, majit_rlib::rposix::get_saved_errno())
                        }
                        None => {
                            let r = unsafe {
                                majit_rlib::rposix::c_mknod(
                                    c_path.as_ptr(),
                                    mode,
                                    device as libc::dev_t,
                                )
                            };
                            (r, majit_rlib::rposix::get_saved_errno())
                        }
                    };
                    if r >= 0 {
                        break;
                    }
                    crate::builtins::eintr_retry_with(
                        std::io::Error::from_raw_os_error(errno),
                        |e| io_err_with_filename(e, path.w_path()),
                    )?;
                }
                Ok(pyre_object::w_none())
            })
        );

        // os.chflags(path, flags, follow_symlinks=True) -> None
        // os.lchflags(path, flags) -> None
        //
        // One call whose `follow_symlinks=False` arm is `lchflags` under its
        // own name, which is what `os.py` reads `HAVE_LCHFLAGS` as. Only
        // the hosts that carry the pair are given the names at all: `chflags`
        // is a BSD interface, and `shutil.copystat` (`shutil.py`) reaches
        // for it through `lookup("chflags")`, which answers `_nop` where the
        // name is absent — so a name that exists has to work.
        //
        // Neither takes a `dir_fd`, so neither has a keyword-only tail, and a
        // surplus argument is counted the way `bind_path_args` counts one
        // without: `os.lchflags(p, 0, follow_symlinks=False)` is over the limit
        // rather than an unknown keyword.
        #[cfg(all(
            not(feature = "sandbox"),
            any(
                target_os = "macos",
                target_os = "ios",
                target_os = "freebsd",
                target_os = "netbsd",
                target_os = "openbsd",
                target_os = "dragonfly",
            )
        ))]
        {
            fn chflags_entry(
                args: &[pyre_object::PyObjectRef],
                name: &str,
                default_follow: bool,
            ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
                let params: &[&'static str] = if default_follow {
                    &["path", "flags", "follow_symlinks"]
                } else {
                    &["path", "flags"]
                };
                let (bound, _) = bind_path_args(args, name, params, 2, &[])?;
                let path = crate::gateway::fsencode_path_or_fd_w(
                    bound[0].expect("path is required"),
                    name,
                    false,
                )?;
                // The flag word is read as a bit pattern rather than a number:
                // `SF_SETTABLE` does not fit a C int, and a negative value is
                // the mask it spells rather than an error.
                let flags =
                    crate::baseobjspace::int_w(bound[1].expect("flags is required"))? as u64;
                let follow = match bound.get(2).copied().flatten() {
                    Some(value) => crate::baseobjspace::is_true(value)?,
                    None => default_follow,
                };
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                let mut w_path = path.w_path();
                // `rposix.c_chflags` / `c_lchflags` release the GIL and save errno.
                let r = pyre_object::with_roots!(w_path => unsafe {
                    if follow {
                        majit_rlib::rposix::c_chflags(c_path.as_ptr(), flags as _)
                    } else {
                        majit_rlib::rposix::c_lchflags(c_path.as_ptr(), flags as _)
                    }
                });
                if r < 0 {
                    return Err(io_err_with_filename(
                        std::io::Error::from_raw_os_error(
                            majit_rlib::rposix::get_saved_errno(),
                        ),
                        w_path,
                    ));
                }
                Ok(pyre_object::w_none())
            }
            crate::__pyre_put_new!(
                ns_slot,
                "chflags",
                crate::make_builtin_function("chflags", |args| {
                    chflags_entry(args, "chflags", true)
                })
            );
            crate::__pyre_put_new!(
                ns_slot,
                "lchflags",
                crate::make_builtin_function("lchflags", |args| {
                    chflags_entry(args, "lchflags", false)
                })
            );
        }

        // [3.14-spec] interp_posix.py `abort` sends SIGABRT with
        // `rposix.kill`, so a process holding a SIGABRT handler runs it and
        // the call returns.  `os_abort_impl` calls `abort()`, whose contract
        // is that it never returns: the signal is unblocked, and the default
        // disposition is restored and re-raised if a handler does return.
        // `os.abort` is documented as terminating, so follow that contract.
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "abort",
            crate::make_builtin_function_with_arity("abort", |_| unsafe { libc::abort() }, 0)
        );

        // os.kill(pid, sig) / os.killpg(pgid, sig)
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "kill",
            crate::make_builtin_function_with_arity(
                "kill",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("kill() requires 2 arguments"));
                    }
                    // interp_posix.py `@unwrap_spec(pid=c_int, signal=c_int)`.
                    // Both arguments go through `c_int_w`: a raw payload read
                    // would accept any object, and `is_int` is an exact-type
                    // check, so an `int` subclass instance — `signal.SIGHUP` is
                    // an `IntEnum` member — never reaches the checked path.
                    let mut w_pid = args[0];
                    let mut w_sig = args[1];
                    let pid = pyre_object::with_roots!(w_pid, w_sig => crate::baseobjspace::c_int_w(w_pid))?
                        as libc::pid_t;
                    let sig = crate::baseobjspace::c_int_w(w_sig)? as libc::c_int;
                    // `rposix.c_kill` releases the GIL and saves errno.
                    // `rposix.kill` reports a negative result through
                    // `handle_posix_error` and does not retry EINTR.
                    let r = unsafe { majit_rlib::rposix::c_kill(pid, sig) };
                    if r < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            )
        );
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "killpg",
            crate::make_builtin_function_with_arity(
                "killpg",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("killpg() requires 2 arguments"));
                    }
                    // interp_posix.py `@unwrap_spec(pgid=c_int, signal=c_int)`.
                    let w_pgid = args[0];
                    let mut w_sig = args[1];
                    let pgid = pyre_object::with_roots!(w_sig =>
                        crate::baseobjspace::c_int_w(w_pgid))?
                        as libc::c_int;
                    let sig = crate::baseobjspace::c_int_w(w_sig)? as libc::c_int;
                    // `rposix.c_killpg` takes the group as `INT`. It releases
                    // the GIL and saves errno. `rposix.killpg` does not retry
                    // EINTR.
                    let r = unsafe { majit_rlib::rposix::c_killpg(pgid, sig) };
                    if r < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            )
        );

        // os.statvfs(path) / os.fstatvfs(fd) -> statvfs_result
        #[cfg(not(target_os = "redox"))]
        crate::__pyre_put_new!(ns_slot, "statvfs_result", super::statvfs_result_seq_type());

        #[cfg(not(target_os = "redox"))]
        fn statvfs_info_from_raw(st: libc::statvfs) -> rustpython_host_env::posix::StatVfsInfo {
            // Darwin `f_fsid` is `fsid_t`, not `c_ulong`. Copy native-endian
            // bytes the way `host_env::posix::statvfs_info_from_raw` does.
            let f_fsid = {
                let ptr = core::ptr::addr_of!(st.f_fsid) as *const u8;
                let size = core::mem::size_of_val(&st.f_fsid);
                if size >= 8 {
                    let bytes = unsafe { core::slice::from_raw_parts(ptr, 8) };
                    u64::from_ne_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ]) as libc::c_ulong
                } else if size >= 4 {
                    let bytes = unsafe { core::slice::from_raw_parts(ptr, 4) };
                    u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as libc::c_ulong
                } else {
                    0
                }
            };
            rustpython_host_env::posix::StatVfsInfo {
                f_bsize: st.f_bsize,
                f_frsize: st.f_frsize,
                f_blocks: st.f_blocks,
                f_bfree: st.f_bfree,
                f_bavail: st.f_bavail,
                f_files: st.f_files,
                f_ffree: st.f_ffree,
                f_favail: st.f_favail,
                f_flag: st.f_flag,
                f_namemax: st.f_namemax,
                f_fsid,
            }
        }
        #[cfg(not(target_os = "redox"))]
        fn statvfs_to_obj(
            info: rustpython_host_env::posix::StatVfsInfo,
        ) -> pyre_object::PyObjectRef {
            let _roots = pyre_object::gc_roots::push_roots();
            let fsid_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(info.f_fsid as i64));
            let mut fields = pyre_object::gc_roots::RootedItems::new();
            fields.push(pyre_object::w_int_new(info.f_bsize as i64));
            fields.push(pyre_object::w_int_new(info.f_frsize as i64));
            fields.push(pyre_object::w_int_new(info.f_blocks as i64));
            fields.push(pyre_object::w_int_new(info.f_bfree as i64));
            fields.push(pyre_object::w_int_new(info.f_bavail as i64));
            fields.push(pyre_object::w_int_new(info.f_files as i64));
            fields.push(pyre_object::w_int_new(info.f_ffree as i64));
            fields.push(pyre_object::w_int_new(info.f_favail as i64));
            fields.push(pyre_object::w_int_new(info.f_flag as i64));
            fields.push(pyre_object::w_int_new(info.f_namemax as i64));
            crate::_structseq::new_instance_with_extra(
                super::statvfs_result_seq_type(),
                fields.take(),
                vec![("f_fsid", pyre_object::gc_roots::shadow_stack_get(fsid_slot))],
            )
        }
        #[cfg(not(target_os = "redox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "statvfs",
            crate::make_builtin_function_with_arity(
                "statvfs",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("statvfs() requires 1 argument"));
                    }
                    // interp_posix.py `statvfs` uses `dispatch_filename(
                    // rposix_stat.statvfs, allow_fd_fn=rposix_stat.fstatvfs)`
                    // with `eintr_retry=False`. `rposix_stat.c_statvfs` /
                    // `c_fstatvfs` release the GIL and save errno.
                    let mut w_path = args[0];
                    // The path owns a bracket of its own, above this one; this
                    // one stays open until the path is gone.
                    let path_roots = pyre_object::gc_roots::push_roots();
                    let path_base = path_roots.pin_roots(&[w_path]);
                    let path = crate::gateway::fsencode_path_or_fd_w(
                        path_roots.get(path_base),
                        "statvfs",
                        HAVE_FSTATVFS,
                    );
                    w_path = path_roots.get(path_base);
                    let path = path?;
                    if path.is_fd {
                        let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
                        let ret = pyre_object::with_roots!(w_path => unsafe {
                            majit_rlib::rposix::c_fstatvfs(path.as_fd, &mut st)
                        });
                        if ret < 0 {
                            return Err(io_err(
                                std::io::Error::from_raw_os_error(
                                    majit_rlib::rposix::get_saved_errno(),
                                ),
                                "",
                            ));
                        }
                        return Ok(pyre_object::with_roots!(w_path => {
                            statvfs_to_obj(statvfs_info_from_raw(st))
                        }));
                    }
                    let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
                    let ret = pyre_object::with_roots!(w_path => unsafe {
                        majit_rlib::rposix::c_statvfs(c_path.as_ptr(), &mut st)
                    });
                    if ret < 0 {
                        return Err(io_err_with_filename(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            path.w_path(),
                        ));
                    }
                    Ok(pyre_object::with_roots!(w_path => {
                        statvfs_to_obj(statvfs_info_from_raw(st))
                    }))
                },
                1,
            )
        );
        #[cfg(not(target_os = "redox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "fstatvfs",
            crate::make_builtin_function_with_arity(
                "fstatvfs",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("fstatvfs() requires 1 argument"));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int)`.
                    let mut w_fd = args[0];
                    let fd = pyre_object::with_roots!(w_fd => crate::baseobjspace::c_int_w(w_fd))?;
                    // interp_posix.py `fstatvfs`: retry on EINTR.
                    // `rposix_stat.c_fstatvfs` releases the GIL and saves errno.
                    let info = loop {
                        let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
                        let ret = pyre_object::with_roots!(w_fd => unsafe {
                            majit_rlib::rposix::c_fstatvfs(fd, &mut st)
                        });
                        if ret == 0 {
                            break statvfs_info_from_raw(st);
                        }
                        pyre_object::with_roots!(w_fd => {
                            crate::builtins::eintr_retry_with(
                                std::io::Error::from_raw_os_error(
                                    majit_rlib::rposix::get_saved_errno(),
                                ),
                                |e| io_err(e, ""),
                            )
                        })?;
                    };
                    Ok(pyre_object::with_roots!(w_fd => statvfs_to_obj(info)))
                },
                1,
            )
        );

        // os.cpu_count() -> int | None — `interp_posix.cpu_count` answers
        // None when `rposix._cpu_count() <= 0`.
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "cpu_count",
            crate::make_builtin_function_with_arity(
                "cpu_count",
                |_| {
                    let n = unsafe { majit_rlib::rposix::_cpu_count() };
                    if n <= 0 {
                        Ok(pyre_object::w_none())
                    } else {
                        Ok(pyre_object::w_int_new(n as i64))
                    }
                },
                0,
            )
        );
        // _cpu_count alias — newer CPython exposes both.
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "_cpu_count",
            crate::make_builtin_function_with_arity(
                "_cpu_count",
                |_| {
                    let n = unsafe { majit_rlib::rposix::_cpu_count() };
                    if n <= 0 {
                        Ok(pyre_object::w_none())
                    } else {
                        Ok(pyre_object::w_int_new(n as i64))
                    }
                },
                0,
            )
        );

        // os.symlink(src, dst, target_is_directory=False) -> None
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "symlink",
            crate::make_builtin_function("symlink", |args| {
                let (bound, mut kwargs) = bind_path_args(
                    args,
                    "symlink",
                    &["src", "dst", "target_is_directory"],
                    2,
                    &["dir_fd"],
                )?;
                // `target_is_directory` selects between the two Windows link
                // kinds and is ignored everywhere else (`os_symlink_impl`).
                // Bound rather than dropped so a fourth positional is the
                // `dir_fd` error it is, not a silently created link.
                let _target_is_directory = match bound[2] {
                    Some(value) => {
                        let roots = pyre_object::gc_roots::push_roots();
                        let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
                        let r = crate::baseobjspace::is_true(value);
                        let w = roots.get(base);
                        kwargs = if w.is_null() { None } else { Some(w) };
                        drop(roots);
                        r?
                    }
                    None => false,
                };
                // `symlink` types `dir_fd` as `DirFD(rposix.HAVE_SYMLINKAT)`.
                let _dir_fd = dir_fd_kwarg(kwargs, HAVE_SYMLINKAT)?;
                let src = crate::gateway::fsencode_path_named_w(
                    bound[0].expect("src is required"),
                    "symlink",
                    "src",
                )?;
                let dst = crate::gateway::fsencode_path_named_w(
                    bound[1].expect("dst is required"),
                    "symlink",
                    "dst",
                )?;
                let c_src = std::ffi::CString::new(src.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in src"))?;
                let c_dst = std::ffi::CString::new(dst.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in dst"))?;
                // `rposix.c_symlink` is the no-descriptor call.
                // `rposix.c_symlinkat` resolves the name against `dir_fd`.
                let ret = match _dir_fd {
                    Some(dir_fd) => unsafe {
                        majit_rlib::rposix::c_symlinkat(c_src.as_ptr(), dir_fd, c_dst.as_ptr())
                    },
                    None => unsafe {
                        majit_rlib::rposix::c_symlink(c_src.as_ptr(), c_dst.as_ptr())
                    },
                };
                if ret < 0 {
                    // `os_symlink_impl` reports through `path_error2`, so the
                    // failure carries the name it was asked to link to as well
                    // as the one it could not create.
                    return Err(fs_err_with_filename2(
                        std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                        0,
                        src.w_path(),
                        dst.w_path(),
                    ));
                }
                Ok(pyre_object::w_none())
            })
        );

        // os.link(src, dst) -> None — a second name for the file `src` names,
        // both of which the failure reports.
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "link",
            crate::make_builtin_function("link", |args| {
                let (args, mut kwargs) = crate::builtins::split_builtin_kwargs(args);
                crate::builtins::kwarg_reject_unknown(
                    kwargs,
                    &["src_dir_fd", "dst_dir_fd", "follow_symlinks"],
                    "link",
                )?;
                link_positional(args)?;
                let n_args = args.len();
                let roots = pyre_object::gc_roots::push_roots();
                let base = roots.publish(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
                let args_base = roots.publish(args);
                roots.normalize(base, 1 + n_args);
                let src =
                    crate::gateway::fsencode_path_named_w(roots.get(args_base), "link", "src")?;
                let w = roots.get(base);
                kwargs = if w.is_null() { None } else { Some(w) };
                let dst =
                    crate::gateway::fsencode_path_named_w(roots.get(args_base + 1), "link", "dst")?;
                let w = roots.get(base);
                kwargs = if w.is_null() { None } else { Some(w) };
                let c_src = std::ffi::CString::new(src.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in src"))?;
                let c_dst = std::ffi::CString::new(dst.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in dst"))?;
                // `link` takes `DirFD(rposix.HAVE_LINKAT)` for both ends: a
                // descriptor the platform can honour resolves the name against
                // it, and one it cannot is refused rather than silently
                // resolved against the process's own directory.
                let dir_fd =
                    |kwargs: Option<PyObjectRef>, name: &str| -> Result<i32, crate::PyError> {
                        match crate::builtins::kwarg_get(kwargs, name)
                            .filter(|&w| !unsafe { pyre_object::is_none(w) })
                        {
                            Some(w) => unwrap_fd(w, "integer or None"),
                            None => Ok(libc::AT_FDCWD),
                        }
                    };
                let roots = pyre_object::gc_roots::push_roots();
                let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
                let src_dir_fd = dir_fd(kwargs, "src_dir_fd");
                let w = roots.get(base);
                kwargs = if w.is_null() { None } else { Some(w) };
                drop(roots);
                let src_dir_fd = src_dir_fd?;
                let roots = pyre_object::gc_roots::push_roots();
                let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
                let dst_dir_fd = dir_fd(kwargs, "dst_dir_fd");
                let w = roots.get(base);
                kwargs = if w.is_null() { None } else { Some(w) };
                drop(roots);
                let dst_dir_fd = dst_dir_fd?;
                // `os_link_impl` follows the final symlink of `src` by
                // default, which is `AT_SYMLINK_FOLLOW`.
                let follow = match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
                    Some(w) => crate::baseobjspace::is_true(w)?,
                    None => true,
                };
                // `interp_posix.link` calls `rposix.link` when both directory
                // descriptors are absent and `follow_symlinks` stays true.
                // Anything else is `rposix.linkat`.
                let plain = follow && src_dir_fd == libc::AT_FDCWD && dst_dir_fd == libc::AT_FDCWD;
                let (ret, err) = if plain {
                    let ret = unsafe { majit_rlib::rposix::c_link(c_src.as_ptr(), c_dst.as_ptr()) };
                    (
                        ret,
                        std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                    )
                } else {
                    let flags = if follow { libc::AT_SYMLINK_FOLLOW } else { 0 };
                    let ret = unsafe {
                        majit_rlib::rposix::c_linkat(
                            src_dir_fd,
                            c_src.as_ptr(),
                            dst_dir_fd,
                            c_dst.as_ptr(),
                            flags,
                        )
                    };
                    (
                        ret,
                        std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                    )
                };
                if ret < 0 {
                    return Err(fs_err_with_filename2(err, 0, src.w_path(), dst.w_path()));
                }
                Ok(pyre_object::w_none())
            })
        );

        // os.chmod(path, mode, *, dir_fd=None, follow_symlinks=True) -> None
        #[cfg(not(feature = "sandbox"))]
        fn chmod_entry(
            args: &[pyre_object::PyObjectRef],
            name: &str,
            default_follow: bool,
        ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
            let (pos, mut kwargs) = crate::builtins::split_builtin_kwargs(args);
            // `lchmod(path, mode)` is `chmod(path, mode,
            // follow_symlinks=False)` under another name and declares no
            // keyword of its own.
            let allowed: &[&str] = if default_follow {
                &["path", "mode", "dir_fd", "follow_symlinks"]
            } else {
                &["path", "mode"]
            };
            crate::builtins::kwarg_reject_unknown(kwargs, allowed, name)?;
            if pos.len() > 2 {
                let surplus = if default_follow {
                    format!(
                        "{name}() takes exactly 2 positional arguments ({} given)",
                        pos.len()
                    )
                } else {
                    format!("{name}() takes at most 2 arguments ({} given)", pos.len())
                };
                return Err(crate::PyError::type_error(surplus));
            }
            let arg = |index: usize, key: &'static str| -> Result<PyObjectRef, crate::PyError> {
                match crate::builtins::bind_pos_or_kw(pos, kwargs, index, key, name, index + 1)? {
                    Some(value) => Ok(value),
                    None => Err(crate::PyError::type_error(format!(
                        "{name}() missing required argument '{key}' (pos {})",
                        index + 1
                    ))),
                }
            };
            let (path_obj, mut mode_obj) = (arg(0, "path")?, arg(1, "mode")?);
            // interp_posix.py reads a `chmod` whose path did not
            // fsencode as a descriptor and answers it with `os.fchmod`.
            // `lchmod` names no descriptor form.
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.pin_roots(&[mode_obj, kwargs.unwrap_or(pyre_object::PY_NULL)]);
            let path = crate::gateway::fsencode_path_or_fd_w(
                path_obj,
                name,
                default_follow && HAVE_FCHMOD,
            );
            mode_obj = roots.get(base);
            let w = roots.get(base + 1);
            kwargs = if w.is_null() { None } else { Some(w) };
            let path = path?;
            // `posix.chmod` unwraps `mode` as `c_int`, so a non-integer raises
            // TypeError instead of reinterpreting its layout.
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
            let mode = crate::baseobjspace::c_int_w(mode_obj);
            let w = roots.get(base);
            kwargs = if w.is_null() { None } else { Some(w) };
            drop(roots);
            let mode = mode? as u32;
            // `chmod` types `dir_fd` as `DirFD(rposix.HAVE_FCHMODAT)`
            // (`interp_posix.py`).
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
            let dir_fd = dir_fd_kwarg(kwargs, HAVE_FCHMODAT);
            let w = roots.get(base);
            kwargs = if w.is_null() { None } else { Some(w) };
            drop(roots);
            let dir_fd = dir_fd?;
            let follow_symlinks = match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
                Some(v) => crate::baseobjspace::is_true(v)?,
                None => default_follow,
            };
            // interp_posix.py `chmod`: retry the selected syscall on EINTR.
            if path.is_fd {
                // A descriptor answers before either modifier is consulted
                // (`interp_posix.py:1233-1242`), so neither is an error here —
                // unlike `chown`, which turns both away (`:2481-2486`). The
                // descriptor already names the file, and `fchmod` is what
                // `os.chmod(fd, …)` means.
                // `rposix.c_fchmod` releases the GIL and saves errno.
                loop {
                    let ret =
                        unsafe { majit_rlib::rposix::c_fchmod(path.as_fd, mode as libc::mode_t) };
                    if ret >= 0 {
                        break;
                    }
                    crate::builtins::eintr_retry_with(
                        std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                        |e| io_err(e, ""),
                    )?;
                }
                return Ok(pyre_object::w_none());
            }
            let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
            // `_chmod_path` (`interp_posix.py`) keeps the plain
            // `chmod` for the unmodified call and reaches for `fchmodat` only
            // where a name has to be resolved against something else or the
            // final symlink must not be followed (`rposix.py`). The plain
            // call is `rposix.c_chmod`, which releases the GIL and saves
            // errno. `c_fchmodat` releases the GIL and saves errno.
            // The mode argument is `mode_t`.
            let use_at = dir_fd.is_some() || !follow_symlinks;
            loop {
                let ret = if use_at {
                    let flag = if follow_symlinks {
                        0
                    } else {
                        libc::AT_SYMLINK_NOFOLLOW
                    };
                    unsafe {
                        majit_rlib::rposix::c_fchmodat(
                            dir_fd.unwrap_or(libc::AT_FDCWD),
                            c_path.as_ptr(),
                            mode as libc::mode_t,
                            flag,
                        )
                    }
                } else {
                    unsafe { majit_rlib::rposix::c_chmod(c_path.as_ptr(), mode as libc::mode_t) }
                };
                if ret >= 0 {
                    break;
                }
                let err = std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno());
                // A host can accept `AT_SYMLINK_NOFOLLOW` and not implement it,
                // reporting so by refusing the call rather than by lacking
                // `fchmodat` — which is why `HAVE_LCHMOD` is a narrower bit than
                // `HAVE_FCHMODAT`. `interp_posix.py:1247-1251` reads that refusal
                // as the modifier being unavailable rather than as an OS error,
                // and reads it ahead of the retry, so an unimplemented modifier
                // is never mistaken for an interruption.
                if !follow_symlinks {
                    let errno = crate::builtins::io_error_posix_errno(&err, 0);
                    if errno == libc::ENOTSUP || errno == libc::EOPNOTSUPP {
                        return Err(argument_unavailable(name, "follow_symlinks"));
                    }
                }
                crate::builtins::eintr_retry_with(err, |e| io_err_with_filename(e, path.w_path()))?;
            }
            Ok(pyre_object::w_none())
        }
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "chmod",
            crate::make_builtin_function("chmod", |args| chmod_entry(args, "chmod", true))
        );
        // `os.lchmod` exists only where the host has a working one — os.py:159
        // records that some platforms carry a stub returning ENOTSUP, and that
        // `fchmodat`'s `AT_SYMLINK_NOFOLLOW` does not work either where that is
        // so. It is the same call the `follow_symlinks=False` arm above makes.
        #[cfg(all(
            not(feature = "sandbox"),
            any(
                target_os = "macos",
                target_os = "ios",
                target_os = "freebsd",
                target_os = "netbsd",
                target_os = "openbsd",
                target_os = "dragonfly",
            )
        ))]
        crate::__pyre_put_new!(
            ns_slot,
            "lchmod",
            crate::make_builtin_function("lchmod", |args| chmod_entry(args, "lchmod", false))
        );

        // os.fchmod(fd, mode) -> None
        crate::__pyre_put_new!(
            ns_slot,
            "fchmod",
            crate::make_builtin_function_with_arity(
                "fchmod",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("fchmod() requires 2 arguments"));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int, mode=c_int)`.
                    let mut w_fd = args[0];
                    let mut w_mode = args[1];
                    let fd = pyre_object::with_roots!(w_fd, w_mode => crate::baseobjspace::c_int_w(w_fd))?;
                    let mode = crate::baseobjspace::c_int_w(w_mode)? as u32;
                    // `rposix.c_fchmod` releases the GIL and saves errno.
                    // interp_posix.py `fchmod`: retry on EINTR.
                    loop {
                        let ret = unsafe { majit_rlib::rposix::c_fchmod(fd, mode as libc::mode_t) };
                        if ret >= 0 {
                            break;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            |e| io_err(e, ""),
                        )?;
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            )
        );

        // os.chown(path, uid, gid, *, dir_fd=None, follow_symlinks=True) -> None
        // os.lchown(path, uid, gid) -> None
        // `uid`/`gid` of -1 means "leave unchanged", as for fchown.
        fn chown_entry(
            args: &[pyre_object::PyObjectRef],
            name: &str,
            default_follow: bool,
        ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
            let (pos, mut kwargs) = crate::builtins::split_builtin_kwargs(args);
            let allowed: &[&str] = if default_follow {
                &["path", "uid", "gid", "dir_fd", "follow_symlinks"]
            } else {
                &["path", "uid", "gid"]
            };
            crate::builtins::kwarg_reject_unknown(kwargs, allowed, name)?;
            if pos.len() > 3 {
                // `chown` declares keyword-only `dir_fd`/`follow_symlinks`, so
                // its surplus-positional report is the "positional arguments"
                // form; `lchown` takes no keywords at all and reports the
                // plain "arguments" form.
                let surplus = if default_follow {
                    format!(
                        "{name}() takes exactly 3 positional arguments ({} given)",
                        pos.len()
                    )
                } else {
                    format!("{name}() takes at most 3 arguments ({} given)", pos.len())
                };
                return Err(crate::PyError::type_error(surplus));
            }
            // Every parameter is bound duplicate-aware, so a call that supplies
            // one both ways raises before the ownership syscall runs.
            let arg = |index: usize, key: &'static str| -> Result<PyObjectRef, crate::PyError> {
                match crate::builtins::bind_pos_or_kw(pos, kwargs, index, key, name, index + 1)? {
                    Some(value) => Ok(value),
                    None => Err(crate::PyError::type_error(format!(
                        "{name}() missing required argument '{key}' (pos {})",
                        index + 1
                    ))),
                }
            };
            let (path_obj, mut uid_obj, mut gid_obj) =
                (arg(0, "path")?, arg(1, "uid")?, arg(2, "gid")?);
            // `posixmodule.c path_converter` calls `__fspath__` and lets what it
            // raises out: a `RuntimeError` from a user `__fspath__` is that
            // object's error, not a statement that the argument was the wrong
            // type.  Rewriting every failure into a `TypeError` here would also
            // swallow the `UnicodeEncodeError` a lone surrogate produces.
            // The path owns a bracket of its own, above this one; this one
            // stays open until the path is gone.
            let has_kwargs = kwargs.is_some();
            let id_roots = pyre_object::gc_roots::push_roots();
            let id_base = id_roots.publish(&[gid_obj, uid_obj]);
            let kwargs_slot = id_roots.publish(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
            id_roots.normalize(id_base, 3);
            let path = crate::gateway::fsencode_path_or_fd_w(
                path_obj,
                name,
                // `lchown` is `path_t(allow_fd=0)` — only `chown` reads an
                // integer as a descriptor (`interp_posix.py`).
                default_follow && HAVE_FCHOWN,
            );
            gid_obj = id_roots.get(id_base);
            uid_obj = id_roots.get(id_base + 1);
            kwargs = has_kwargs.then(|| id_roots.get(kwargs_slot));
            let path = path?;
            // `_Py_Uid_Converter` / `_Py_Gid_Converter`: `uid_t` is unsigned, yet
            // -1 is always accepted as the "leave unchanged" sentinel.  Only
            // that one value means unchanged; every other id is judged by
            // round-tripping through `uid_t`, so nothing is silently wrapped —
            // 2**32 truncates to 0 and would otherwise request uid 0.
            //
            // The two range reports follow the C converter's own split: a value
            // that still fits a C long but fails the round trip is "less than
            // minimum" (including 2**32, whose truncation reads as underflow),
            // while one too wide for a long is "greater than maximum".
            let id_of =
                |w: pyre_object::PyObjectRef, what: &str| -> Result<Option<u32>, crate::PyError> {
                    if !unsafe { crate::builtins::index_check(w) } {
                        return Err(crate::PyError::type_error(format!(
                            "{what} should be integer, not {}",
                            crate::type_methods::arg_type_name(w)
                        )));
                    }
                    let w_index = crate::baseobjspace::space_index(w)?;
                    let raw = crate::baseobjspace::int_w(w_index).map_err(|_| {
                        crate::PyError::overflow_error(format!("{what} is greater than maximum"))
                    })?;
                    if raw == -1 {
                        return Ok(None);
                    }
                    let narrowed = raw as u32;
                    if i64::from(narrowed) != raw {
                        return Err(crate::PyError::overflow_error(format!(
                            "{what} is less than minimum"
                        )));
                    }
                    Ok(Some(narrowed))
                };
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.pin_roots(&[gid_obj, kwargs.unwrap_or(pyre_object::PY_NULL)]);
            let uid = id_of(uid_obj, "uid");
            gid_obj = roots.get(base);
            let w = roots.get(base + 1);
            kwargs = if w.is_null() { None } else { Some(w) };
            drop(roots);
            let uid = uid?;
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
            let gid = id_of(gid_obj, "gid");
            let w = roots.get(base);
            kwargs = if w.is_null() { None } else { Some(w) };
            drop(roots);
            let gid = gid?;
            // `chown` types `dir_fd` as `DirFD(rposix.HAVE_FCHOWNAT)`
            // (`interp_posix.py`); `lchown` declares no keyword at
            // all, so `allowed` above has already rejected it.
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
            let dir_fd = dir_fd_kwarg(kwargs, HAVE_FCHOWNAT);
            let w = roots.get(base);
            kwargs = if w.is_null() { None } else { Some(w) };
            drop(roots);
            let dir_fd = dir_fd?;
            let follow_symlinks = match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
                Some(v) => crate::baseobjspace::is_true(v)?,
                None => default_follow,
            };
            // interp_posix.py `chown`: retry the selected syscall on EINTR.
            // `-1` is the unchanged id `rposix.c_chown` takes.
            let uid_arg = match uid {
                Some(uid) => uid as libc::c_int,
                None => -1,
            };
            let gid_arg = match gid {
                Some(gid) => gid as libc::c_int,
                None => -1,
            };
            if path.is_fd {
                // interp_posix.py:2481-2486 — a descriptor already names the
                // file, so neither modifier, which each reinterpret a name, can
                // apply. Upstream spells the second "cannnot"; 3.14, which the
                // parity suite reads as the oracle, spells it "cannot".
                if dir_fd.is_some() {
                    return Err(crate::PyError::value_error(format!(
                        "{name}: can't specify both dir_fd and fd"
                    )));
                }
                if !follow_symlinks {
                    return Err(crate::PyError::value_error(format!(
                        "{name}: cannot use fd and follow_symlinks together"
                    )));
                }
                // `rposix.c_fchown` releases the GIL and saves errno.
                loop {
                    let ret = unsafe { majit_rlib::rposix::c_fchown(path.as_fd, uid_arg, gid_arg) };
                    if ret >= 0 {
                        break;
                    }
                    crate::builtins::eintr_retry_with(
                        std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                        |e| io_err(e, ""),
                    )?;
                }
                return Ok(pyre_object::w_none());
            }
            // `interp_posix.chown` calls `rposix.lchown` when
            // `follow_symlinks` is false and no directory descriptor is set,
            // and `rposix.chown` when the call follows and no descriptor is
            // set. A directory descriptor selects `rposix.fchownat`.
            // `rposix.c_chown` and `rposix.c_lchown` release the GIL and save
            // errno.
            if dir_fd.is_none() {
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                let invoke = || {
                    if follow_symlinks {
                        unsafe { majit_rlib::rposix::c_chown(c_path.as_ptr(), uid_arg, gid_arg) }
                    } else {
                        unsafe { majit_rlib::rposix::c_lchown(c_path.as_ptr(), uid_arg, gid_arg) }
                    }
                };
                if name == "chown" {
                    loop {
                        if invoke() >= 0 {
                            break;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            |e| io_err_with_filename(e, path.w_path()),
                        )?;
                    }
                } else {
                    // `interp_posix.lchown` does not retry EINTR.
                    if invoke() < 0 {
                        return Err(io_err_with_filename(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            path.w_path(),
                        ));
                    }
                }
                return Ok(pyre_object::w_none());
            }
            // `lchown` rejects a directory descriptor above, so this arm is
            // `chown` and retries EINTR.
            // `rposix.c_fchownat` releases the GIL and saves errno. Owner
            // and group are the `c_int` ids above, and `-1` leaves one
            // unchanged. `fd_borrow` refuses `-1` before the call.
            let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
            let raw = {
                use std::os::fd::AsRawFd;
                fd_borrow(dir_fd.unwrap())?.as_raw_fd()
            };
            let flag = if follow_symlinks {
                0
            } else {
                libc::AT_SYMLINK_NOFOLLOW
            };
            loop {
                let ret = unsafe {
                    majit_rlib::rposix::c_fchownat(raw, c_path.as_ptr(), uid_arg, gid_arg, flag)
                };
                if ret >= 0 {
                    break;
                }
                crate::builtins::eintr_retry_with(
                    std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                    |e| io_err_with_filename(e, path.w_path()),
                )?;
            }
            Ok(pyre_object::w_none())
        }
        crate::__pyre_put_new!(
            ns_slot,
            "chown",
            crate::make_builtin_function("chown", |args| chown_entry(args, "chown", true))
        );
        crate::__pyre_put_new!(
            ns_slot,
            "lchown",
            crate::make_builtin_function("lchown", |args| chown_entry(args, "lchown", false))
        );

        // os.fchown(fd, uid, gid) -> None  (uid/gid of -1 means "leave unchanged")
        crate::__pyre_put_new!(
            ns_slot,
            "fchown",
            crate::make_builtin_function_with_arity(
                "fchown",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error("fchown() requires 3 arguments"));
                    }
                    // interp_posix.py `@unwrap_spec(uid=c_uid_t,
                    // gid=c_gid_t)` with the descriptor taken by
                    // `space.c_filedescriptor_w`. The spec is applied by the
                    // gateway before the body runs, so a bad uid/gid is
                    // reported ahead of a bad descriptor.
                    //
                    // `c_uid_t_w` turns -1 into `u32::MAX`. Those bits are
                    // the `(uid_t)-1` sentinel `rposix.c_fchown` takes.
                    let mut w_fd = args[0];
                    let w_uid = args[1];
                    let mut w_gid = args[2];
                    let uid = pyre_object::with_roots!(w_fd, w_gid =>
                        crate::baseobjspace::c_uid_t_w(w_uid))?
                        as libc::c_int;
                    let gid = pyre_object::with_roots!(w_fd =>
                        crate::baseobjspace::c_uid_t_w(w_gid))?
                        as libc::c_int;
                    let fd = crate::baseobjspace::c_filedescriptor_w(w_fd)?;
                    // `rposix.c_fchown` releases the GIL and saves errno.
                    // interp_posix.py `fchown`: retry on EINTR.
                    loop {
                        let ret = unsafe { majit_rlib::rposix::c_fchown(fd, uid, gid) };
                        if ret >= 0 {
                            break;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            |e| io_err(e, ""),
                        )?;
                    }
                    Ok(pyre_object::w_none())
                },
                3,
            )
        );

        // os.get_inheritable(fd) -> bool. `interp_posix.get_inheritable`:
        // `eintr_retry=False`.
        crate::__pyre_put_new!(
            ns_slot,
            "get_inheritable",
            crate::make_builtin_function_with_arity(
                "get_inheritable",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "get_inheritable() requires 1 argument",
                        ));
                    }
                    let mut w_fd = args[0];
                    let fd = pyre_object::with_roots!(w_fd => crate::baseobjspace::c_int_w(w_fd))?;
                    let res = pyre_object::with_roots!(w_fd => unsafe {
                        majit_rlib::rposix::_c_get_inheritable(fd)
                    });
                    if res < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_bool_from(res != 0))
                },
                1,
            )
        );

        // os.set_inheritable(fd, inheritable) -> None
        crate::__pyre_put_new!(
            ns_slot,
            "set_inheritable",
            crate::make_builtin_function_with_arity(
                "set_inheritable",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "set_inheritable() requires 2 arguments",
                        ));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int, inheritable=int)`.
                    let mut w_fd = args[0];
                    let mut w_inherit = args[1];
                    let fd = pyre_object::with_roots!(w_fd, w_inherit => crate::baseobjspace::c_int_w(w_fd))?;
                    let inherit = pyre_object::with_roots!(w_fd, w_inherit =>
                        crate::baseobjspace::int_w(w_inherit))?
                        != 0;
                    let res = pyre_object::with_roots!(w_fd, w_inherit => unsafe {
                        majit_rlib::rposix::_c_set_inheritable(fd, inherit as libc::c_int)
                    });
                    if res < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            )
        );

        // os.access(path, mode, *, dir_fd=None, effective_ids=False,
        //           follow_symlinks=True) -> bool
        crate::__pyre_put_new!(
            ns_slot,
            "access",
            crate::make_builtin_function("access", |args| {
                // `access` names three keyword-only modifiers, so a third
                // positional is an error rather than a `dir_fd`.
                let (bound, mut kwargs) = bind_path_args(
                    args,
                    "access",
                    &["path", "mode"],
                    2,
                    &["dir_fd", "effective_ids", "follow_symlinks"],
                )?;
                // The parameters convert in declaration order, and every one of
                // them can raise, so the order is observable: `path` reports
                // before `mode`, `mode` before `dir_fd`, and both before either
                // flag's `__bool__` is called at all.
                let roots = pyre_object::gc_roots::push_roots();
                let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
                let path = crate::gateway::fsencode_path_named_w(
                    bound[0].expect("path is required"),
                    "access",
                    "path",
                );
                let w = roots.get(base);
                kwargs = if w.is_null() { None } else { Some(w) };
                let path = path?.as_bytes;
                // interp_posix.py `@unwrap_spec(mode=c_int, ...)`.
                let roots = pyre_object::gc_roots::push_roots();
                let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
                let mode = crate::baseobjspace::c_int_w(bound[1].expect("mode is required"));
                let w = roots.get(base);
                kwargs = if w.is_null() { None } else { Some(w) };
                drop(roots);
                let mode = mode?;
                // interp_posix.py:745 types `dir_fd` as
                // `DirFD(rposix.HAVE_FACCESSAT)`, so a host with no `faccessat`
                // turns the descriptor away instead of resolving the name
                // against the working directory as though none had been given.
                let roots = pyre_object::gc_roots::push_roots();
                let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
                let dir_fd = dir_fd_kwarg(kwargs, HAVE_FACCESSAT);
                let w = roots.get(base);
                kwargs = if w.is_null() { None } else { Some(w) };
                drop(roots);
                let dir_fd = dir_fd?;
                let effective_ids = match crate::builtins::kwarg_get(kwargs, "effective_ids") {
                    Some(v) => {
                        let roots = pyre_object::gc_roots::push_roots();
                        let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
                        let r = crate::baseobjspace::is_true(v);
                        let w = roots.get(base);
                        kwargs = if w.is_null() { None } else { Some(w) };
                        drop(roots);
                        r?
                    }
                    None => false,
                };
                let follow_symlinks = match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
                    Some(v) => crate::baseobjspace::is_true(v)?,
                    None => true,
                };
                // interp_posix.py:771-775 — the two flag modifiers have no other
                // call to reach, so without `faccessat` they are refused rather
                // than answered as though they had been applied.
                if !HAVE_FACCESSAT {
                    if !follow_symlinks {
                        return Err(argument_unavailable("access", "follow_symlinks"));
                    }
                    if effective_ids {
                        return Err(argument_unavailable("access", "effective_ids"));
                    }
                }
                #[cfg(feature = "sandbox")]
                {
                    // `HAVE_FACCESSAT` is false here, so the three modifiers
                    // have already been turned away and only the plain form is
                    // left to serve.
                    let _ = dir_fd;
                    return Ok(pyre_object::w_bool_from(
                        crate::host_seam::ops::access(&path, mode).unwrap_or(false),
                    ));
                }
                #[cfg(not(feature = "sandbox"))]
                {
                    let c_path = std::ffi::CString::new(path.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null character"))?;
                    // interp_posix.py keeps the plain `access` for the
                    // unmodified call and reaches for `faccessat` only where the
                    // name resolves against a descriptor, the final symlink must
                    // not be followed, or the effective ids are the ones to ask
                    // about. `rposix.py` is the flag mapping.
                    let ret = if dir_fd.is_some() || !follow_symlinks || effective_ids {
                        let mut flags = 0;
                        if !follow_symlinks {
                            flags |= libc::AT_SYMLINK_NOFOLLOW;
                        }
                        if effective_ids {
                            flags |= libc::AT_EACCESS;
                        }
                        unsafe {
                            majit_rlib::rposix::c_faccessat(
                                dir_fd.unwrap_or(libc::AT_FDCWD),
                                c_path.as_ptr(),
                                mode,
                                flags,
                            )
                        }
                    } else {
                        unsafe { majit_rlib::rposix::c_access(c_path.as_ptr(), mode) }
                    };
                    // `rposix.access` and `rposix.faccessat` both answer
                    // `error == 0` without `handle_posix_error`, so a refused
                    // call is False and not an `OSError` — including the EINVAL
                    // a mode outside `R_OK | W_OK | X_OK` can draw.
                    Ok(pyre_object::w_bool_from(ret == 0))
                }
            })
        );

        // os.chroot(path) -> None
        crate::__pyre_put_new!(
            ns_slot,
            "chroot",
            crate::make_builtin_function_with_arity(
                "chroot",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("chroot() requires 1 argument"));
                    }
                    let path = crate::gateway::fsencode_path_named_w(args[0], "chroot", "path")?;
                    let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    // `rposix.c_chroot` releases the GIL and saves errno.
                    let ret = unsafe { majit_rlib::rposix::c_chroot(c_path.as_ptr()) };
                    if ret < 0 {
                        return Err(io_err_with_filename(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            path.w_path(),
                        ));
                    }
                    Ok(pyre_object::w_none())
                },
                1,
            )
        );

        // os.getloadavg() -> (1m, 5m, 15m)
        crate::__pyre_put_new!(
            ns_slot,
            "getloadavg",
            crate::make_builtin_function_with_arity(
                "getloadavg",
                |_| {
                    // `rposix.c_getloadavg` does not save errno.
                    // `rposix.getloadavg` raises a bare `OSError` when the
                    // count is not 3, and `interp_posix.getloadavg` turns
                    // that into `OSError("Load averages are unobtainable")`.
                    #[cfg(not(any(target_os = "android", target_os = "redox")))]
                    let [l1, l5, l15] = {
                        let mut loads = [0.0f64; 3];
                        let n = unsafe { majit_rlib::rposix::c_getloadavg(loads.as_mut_ptr(), 3) };
                        if n != 3 {
                            return Err(crate::PyError::os_error("Load averages are unobtainable"));
                        }
                        loads
                    };
                    #[cfg(any(target_os = "android", target_os = "redox"))]
                    let [l1, l5, l15] =
                        rustpython_host_env::time::getloadavg().map_err(|e| io_err(e, ""))?;
                    let mut fields = pyre_object::gc_roots::RootedItems::new();
                    fields.push(pyre_object::w_float_new(l1));
                    fields.push(pyre_object::w_float_new(l5));
                    fields.push(pyre_object::w_float_new(l15));
                    Ok(pyre_object::w_tuple_new(fields.take()))
                },
                0,
            )
        );

        // os.times() -> posix.times_result(user, system, children_user,
        //                                  children_system, elapsed)
        crate::__pyre_put_new!(
            ns_slot,
            "times",
            crate::make_builtin_function_with_arity(
                "times",
                |_| {
                    // `rposix.c_times` uses `RFFI_FULL_ERRNO_ZERO` because a
                    // clock_t of -1 is also a successful elapsed count.
                    let mut tms = std::mem::MaybeUninit::<libc::tms>::zeroed();
                    let ticks = unsafe { majit_rlib::rposix::c_times(tms.as_mut_ptr()) } as i64;
                    if ticks == -1 {
                        let err = majit_rlib::rposix::get_saved_errno();
                        if err != 0 {
                            return Err(io_err(std::io::Error::from_raw_os_error(err), ""));
                        }
                    }
                    let tms = unsafe { tms.assume_init() };
                    let clk = unsafe { majit_rlib::rposix::c_sysconf(libc::_SC_CLK_TCK) } as f64;
                    if clk <= 0.0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    let mut fields = pyre_object::gc_roots::RootedItems::new();
                    fields.push(pyre_object::w_float_new(tms.tms_utime as f64 / clk));
                    fields.push(pyre_object::w_float_new(tms.tms_stime as f64 / clk));
                    fields.push(pyre_object::w_float_new(tms.tms_cutime as f64 / clk));
                    fields.push(pyre_object::w_float_new(tms.tms_cstime as f64 / clk));
                    fields.push(pyre_object::w_float_new(ticks as f64 / clk));
                    Ok(crate::_structseq::new_instance(
                        super::times_result_seq_type(),
                        fields.take(),
                    ))
                },
                0,
            )
        );

        // os.waitstatus_to_exitcode(status) -> int
        crate::__pyre_put_new!(
            ns_slot,
            "waitstatus_to_exitcode",
            crate::make_builtin_function_with_arity(
                "waitstatus_to_exitcode",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "waitstatus_to_exitcode() requires 1 argument",
                        ));
                    }
                    // app_posix.py waitstatus_to_exitcode is app-level and reaches the status
                    // through `posix.WIFEXITED`/`WEXITSTATUS`, each of which is
                    // `@unwrap_spec(status=c_int)`.
                    let status = crate::baseobjspace::c_int_w(args[0])?;
                    match rustpython_host_env::time::waitstatus_to_exitcode(status) {
                        Some(code) => Ok(pyre_object::w_int_new(code as i64)),
                        None => Err(crate::PyError::value_error(
                            "waitstatus_to_exitcode: invalid status",
                        )),
                    }
                },
                1,
            )
        );

        // os.system(command) -> exit_status
        crate::__pyre_put_new!(
            ns_slot,
            "system",
            crate::make_builtin_function_with_arity(
                "system",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("system() requires 1 argument"));
                    }
                    // `interp_posix.py command='fsencode'`, which
                    // `gateway.py visit_fsencode` unwraps with `space.fsencode_w`: the
                    // shell gets the filesystem bytes, so a command naming a
                    // byte with no UTF-8 spelling survives instead of being
                    // refused, and `bytes` / `__fspath__` are accepted as the
                    // converter accepts them.
                    let cmd = crate::gateway::fsencode_bytes_w(args[0])?;
                    let c_cmd = std::ffi::CString::new(cmd)
                        .map_err(|_| crate::PyError::value_error("embedded null in command"))?;
                    // `rposix.c_system` does not save errno. The result is
                    // the wait status.
                    let rc = unsafe { majit_rlib::rposix::c_system(c_cmd.as_ptr()) };
                    Ok(pyre_object::w_int_new(rc as i64))
                },
                1,
            )
        );

        // os.sendfile(out_fd, in_fd, offset, count) -> bytes_sent
        //
        // Ported from pypy/module/posix/interp_posix.py:
        //   * 4 positional args: out_fd, in_fd (called "in_" in PyPy because
        //     "in" is reserved), offset, count.
        //   * offset == None: linux-only "no-offset" path (NULL pointer);
        //     non-linux raises TypeError("an integer is required (got None)")
        //     verbatim from PyPy.
        //   * offset == int: read as i64 (`space.gateway_r_longlong_w`)
        //     and routed through `rposix.c_sendfile` (linux) or, on Darwin,
        //     `rposix.sendfile` when headers/trailers are absent and flags
        //     is 0, else `host_posix::sendfile` for the 3.14 header/trailer
        //     arguments.
        //   * Returns bytes-sent as int (PyPy: space.newint(res)).
        //
        // Both arms of `interp_posix.py` sit in a
        // `while True: ... except OSError: wrap_oserror(..., eintr_retry=True)`,
        // so an interrupted transfer runs the pending Python signal handlers and
        // then goes back to the call. The three below do the same through
        // `builtins::eintr_retry_with`.
        //
        // The BSD arm discards a partial `sbytes` on EINTR rather than reporting
        // it: `rposix.py` rescues a partial transfer for `EAGAIN` and
        // `EBUSY` alone, and EINTR falls through to `handle_posix_error`, which
        // raises. The loop then re-runs the whole call with the same `offset` and
        // `count` — both are loop-invariant in `interp_posix.py`, and `rposix`
        // never sees the retry — so the transfer restarts from the range the
        // caller asked for, not from where it had got to.
        #[cfg(all(
            any(target_os = "linux", target_os = "android", target_os = "macos"),
            not(feature = "sandbox")
        ))]
        crate::__pyre_put_new!(
            ns_slot,
            "sendfile",
            crate::make_builtin_function("sendfile", |args| {
                #[cfg(target_os = "macos")]
                use std::os::fd::BorrowedFd;
                // Every parameter is positional-or-keyword. `headers`,
                // `trailers` and `flags` are the BSD `sendfile(2)` tail.
                // Darwin with no headers/trailers and flags=0 uses
                // `rposix.c_sendfile`; headers or trailers keep
                // `host_posix::sendfile`. Listing the BSD-only parameters
                // makes unknown keywords fail during argument binding on
                // every platform.
                // interp_posix.py `@unwrap_spec(out_fd=c_int, count=int)`,
                // with `in_ = space.c_int_w(w_in_fd)` in the body (`sendfile`).
                // The spec runs in the gateway, so the count is converted
                // before the descriptor argument that follows it here.
                // Named locals replace `bound` so the `Vec<Option<PyObjectRef>>`
                // is not live across `rposix.c_sendfile`.
                #[cfg(any(target_os = "linux", target_os = "android"))]
                let (mut w_out_fd, mut w_in_fd, mut w_offset, mut w_count) = {
                    let (bound, _kwargs) = bind_path_args(
                        args,
                        "sendfile",
                        &[
                            "out_fd", "in_fd", "offset", "count", "headers", "trailers", "flags",
                        ],
                        4,
                        &[],
                    )?;
                    (
                        bound[0].expect("out_fd is required"),
                        bound[1].expect("in_fd is required"),
                        bound[2].expect("offset is required"),
                        bound[3].expect("count is required"),
                    )
                };
                #[cfg(target_os = "macos")]
                let (
                    mut w_out_fd,
                    mut w_in_fd,
                    mut w_offset,
                    mut w_count,
                    mut w_headers,
                    mut w_trailers,
                    mut w_flags,
                ) = {
                    let (bound, _kwargs) = bind_path_args(
                        args,
                        "sendfile",
                        &[
                            "out_fd", "in_fd", "offset", "count", "headers", "trailers", "flags",
                        ],
                        4,
                        &[],
                    )?;
                    (
                        bound[0].expect("out_fd is required"),
                        bound[1].expect("in_fd is required"),
                        bound[2].expect("offset is required"),
                        bound[3].expect("count is required"),
                        bound[4].unwrap_or(pyre_object::PY_NULL),
                        bound[5].unwrap_or(pyre_object::PY_NULL),
                        bound[6].unwrap_or(pyre_object::PY_NULL),
                    )
                };
                #[cfg(any(target_os = "linux", target_os = "android"))]
                let out_fd = pyre_object::with_roots!(w_out_fd, w_in_fd, w_offset, w_count => {
                    crate::baseobjspace::c_int_w(w_out_fd)
                })?;
                #[cfg(target_os = "macos")]
                let out_fd = pyre_object::with_roots!(
                    w_out_fd, w_in_fd, w_offset, w_count, w_headers, w_trailers, w_flags => {
                    crate::baseobjspace::c_int_w(w_out_fd)
                })?;
                #[cfg(any(target_os = "linux", target_os = "android"))]
                let count_raw = pyre_object::with_roots!(w_out_fd, w_in_fd, w_offset, w_count => {
                    crate::baseobjspace::int_w(w_count)
                })?;
                #[cfg(target_os = "macos")]
                let count_raw = pyre_object::with_roots!(
                    w_out_fd, w_in_fd, w_offset, w_count, w_headers, w_trailers, w_flags => {
                    crate::baseobjspace::int_w(w_count)
                })?;
                #[cfg(any(target_os = "linux", target_os = "android"))]
                let in_fd = pyre_object::with_roots!(w_out_fd, w_in_fd, w_offset, w_count => {
                    crate::baseobjspace::c_int_w(w_in_fd)
                })?;
                #[cfg(target_os = "macos")]
                let in_fd = pyre_object::with_roots!(
                    w_out_fd, w_in_fd, w_offset, w_count, w_headers, w_trailers, w_flags => {
                    crate::baseobjspace::c_int_w(w_in_fd)
                })?;
                if unsafe { pyre_object::is_none(w_offset) } {
                    // linux-only no-offset path; non-linux raises TypeError
                    // matching interp_posix.sendfile.
                    #[cfg(not(any(target_os = "linux", target_os = "android")))]
                    {
                        let _ = (out_fd, in_fd, count_raw);
                        return Err(crate::PyError::type_error(
                            "an integer is required (got None)",
                        ));
                    }
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    {
                        // `rposix.sendfile_no_offset`: a null pointer, so the
                        // kernel uses the input descriptor's live position.
                        // `rposix.c_sendfile` releases the GIL and saves errno.
                        let count = count_raw as majit_rlib::rffi::SIZE_T;
                        loop {
                            let res = pyre_object::with_roots!(
                                w_out_fd, w_in_fd, w_offset, w_count => unsafe {
                                majit_rlib::rposix::c_sendfile(
                                    out_fd,
                                    in_fd,
                                    core::ptr::null_mut(),
                                    count,
                                )
                            });
                            if res >= 0 {
                                return Ok(pyre_object::with_roots!(
                                    w_out_fd, w_in_fd, w_offset, w_count => {
                                    pyre_object::w_int_new(res as i64)
                                }));
                            }
                            pyre_object::with_roots!(w_out_fd, w_in_fd, w_offset, w_count => {
                                crate::builtins::eintr_retry_with(
                                    std::io::Error::from_raw_os_error(
                                        majit_rlib::rposix::get_saved_errno(),
                                    ),
                                    |e| io_err(e, ""),
                                )
                            })?;
                        }
                    }
                }
                // interp_posix.py `space.gateway_r_longlong_w(w_offset)`.
                #[cfg(any(target_os = "linux", target_os = "android"))]
                let offset_i64 = pyre_object::with_roots!(w_out_fd, w_in_fd, w_offset, w_count => {
                    crate::baseobjspace::int_w(w_offset)
                })?;
                #[cfg(target_os = "macos")]
                let offset_i64 = pyre_object::with_roots!(
                    w_out_fd, w_in_fd, w_offset, w_count, w_headers, w_trailers, w_flags => {
                    crate::baseobjspace::int_w(w_offset)
                })?;
                #[cfg(any(target_os = "linux", target_os = "android"))]
                {
                    let count = count_raw as majit_rlib::rffi::SIZE_T;
                    loop {
                        // Seeded from the caller's value on every attempt.
                        // `rposix.sendfile` writes the offset into a fresh
                        // cell each call from the argument it was passed, and
                        // the retry sits above it holding that argument
                        // unchanged, so what a failed call left behind here is
                        // not what the next one starts from.
                        let mut offset: libc::off_t = offset_i64 as libc::off_t;
                        let res = pyre_object::with_roots!(
                            w_out_fd, w_in_fd, w_offset, w_count => unsafe {
                            majit_rlib::rposix::c_sendfile(out_fd, in_fd, &mut offset, count)
                        });
                        if res >= 0 {
                            return Ok(pyre_object::with_roots!(
                                w_out_fd, w_in_fd, w_offset, w_count => {
                                pyre_object::w_int_new(res as i64)
                            }));
                        }
                        pyre_object::with_roots!(w_out_fd, w_in_fd, w_offset, w_count => {
                            crate::builtins::eintr_retry_with(
                                std::io::Error::from_raw_os_error(
                                    majit_rlib::rposix::get_saved_errno(),
                                ),
                                |e| io_err(e, ""),
                            )
                        })?;
                    }
                }
                #[cfg(target_os = "macos")]
                {
                    let flags = if w_flags.is_null() || unsafe { pyre_object::is_none(w_flags) } {
                        0
                    } else {
                        pyre_object::with_roots!(
                            w_out_fd, w_in_fd, w_offset, w_count,
                            w_headers, w_trailers, w_flags => {
                            crate::baseobjspace::c_int_w(w_flags)
                        })?
                    };
                    // Both Python sequences and all of their buffer exports are
                    // consumed before entering the EINTR retry loop. The retry
                    // therefore reuses only Rust-owned bytes.
                    let (header_buffers, trailer_buffers) = {
                        let _roots = pyre_object::gc_roots::push_roots();
                        let base = pyre_object::gc_roots::pin_roots(&[
                            w_out_fd, w_in_fd, w_offset, w_count, w_headers, w_trailers, w_flags,
                        ]);
                        w_out_fd = pyre_object::gc_roots::shadow_stack_get(base);
                        w_in_fd = pyre_object::gc_roots::shadow_stack_get(base + 1);
                        w_offset = pyre_object::gc_roots::shadow_stack_get(base + 2);
                        w_count = pyre_object::gc_roots::shadow_stack_get(base + 3);
                        w_headers = pyre_object::gc_roots::shadow_stack_get(base + 4);
                        w_trailers = pyre_object::gc_roots::shadow_stack_get(base + 5);
                        w_flags = pyre_object::gc_roots::shadow_stack_get(base + 6);
                        let header_slot = if w_headers.is_null() {
                            None
                        } else {
                            Some(base + 4)
                        };
                        let trailer_slot = if w_trailers.is_null() {
                            None
                        } else {
                            Some(base + 5)
                        };
                        let collect_buffers = |slot: Option<usize>, name: &str| {
                            let Some(slot) = slot else {
                                return Ok(None);
                            };
                            let mut value = pyre_object::gc_roots::shadow_stack_get(slot);
                            if unsafe { pyre_object::is_none(value) } {
                                return Ok(None);
                            }
                            // Indexed header/trailer vectors require a sequence;
                            // consuming an iterator or mapping keys would change
                            // the accepted `sendfile` argument protocol.
                            if !pyre_object::with_roots!(value => crate::baseobjspace::issequence_w(value))
                            {
                                return Err(crate::PyError::type_error(format!(
                                    "sendfile() {name} must be a sequence"
                                )));
                            }
                            let items = crate::baseobjspace::unpackiterable(value, -1)?;
                            let items_base = pyre_object::gc_roots::pin_roots(&items);
                            let mut buffers = Vec::with_capacity(items.len());
                            for index in 0..items.len() {
                                let item =
                                    pyre_object::gc_roots::shadow_stack_get(items_base + index);
                                let Some(buffer) = crate::baseobjspace::simple_buffer_bytes(item)?
                                else {
                                    return Err(crate::PyError::type_error(format!(
                                        "sendfile() {name} items must be bytes-like"
                                    )));
                                };
                                buffers.push(buffer.as_bytes().to_vec());
                                buffer.release();
                            }
                            if buffers.is_empty() {
                                Ok(None)
                            } else {
                                Ok(Some(buffers))
                            }
                        };
                        let buffers = (
                            collect_buffers(header_slot, "headers")?,
                            collect_buffers(trailer_slot, "trailers")?,
                        );
                        w_out_fd = pyre_object::gc_roots::shadow_stack_get(base);
                        w_in_fd = pyre_object::gc_roots::shadow_stack_get(base + 1);
                        w_offset = pyre_object::gc_roots::shadow_stack_get(base + 2);
                        w_count = pyre_object::gc_roots::shadow_stack_get(base + 3);
                        w_headers = pyre_object::gc_roots::shadow_stack_get(base + 4);
                        w_trailers = pyre_object::gc_roots::shadow_stack_get(base + 5);
                        w_flags = pyre_object::gc_roots::shadow_stack_get(base + 6);
                        buffers
                    };
                    if header_buffers.is_none() && trailer_buffers.is_none() && flags == 0 {
                        // `rposix.sendfile`: `c_sendfile(in_fd, out_fd, offset,
                        // p_len, NULL, 0)` then the EAGAIN/EBUSY sbytes rescue.
                        // `interp_posix.sendfile` uses `eintr_retry=True`.
                        loop {
                            match pyre_object::with_roots!(
                                w_out_fd, w_in_fd, w_offset, w_count,
                                w_headers, w_trailers, w_flags => {
                                majit_rlib::rposix::sendfile(
                                    out_fd,
                                    in_fd,
                                    offset_i64 as libc::off_t,
                                    count_raw as libc::off_t,
                                )
                            }) {
                                Ok(n) => {
                                    return Ok(pyre_object::with_roots!(
                                        w_out_fd, w_in_fd, w_offset, w_count,
                                        w_headers, w_trailers, w_flags => {
                                        pyre_object::w_int_new(n as i64)
                                    }));
                                }
                                Err(errno) => {
                                    pyre_object::with_roots!(
                                        w_out_fd, w_in_fd, w_offset, w_count,
                                        w_headers, w_trailers, w_flags => {
                                        crate::builtins::eintr_retry_with(
                                            std::io::Error::from_raw_os_error(errno),
                                            |e| io_err(e, ""),
                                        )
                                    })?;
                                }
                            }
                        }
                    }
                    let out_b = pyre_object::with_roots!(
                        w_out_fd, w_in_fd, w_offset, w_count,
                        w_headers, w_trailers, w_flags => fd_borrow(out_fd)
                    )?;
                    let in_b = pyre_object::with_roots!(
                        w_out_fd, w_in_fd, w_offset, w_count,
                        w_headers, w_trailers, w_flags => fd_borrow(in_fd)
                    )?;
                    // An empty sequence is indistinguishable from an absent
                    // one at the syscall boundary, independently for headers
                    // and trailers.
                    let header_slices = header_buffers
                        .as_ref()
                        .map(|buffers| buffers.iter().map(Vec::as_slice).collect::<Vec<&[u8]>>());
                    let trailer_slices = trailer_buffers
                        .as_ref()
                        .map(|buffers| buffers.iter().map(Vec::as_slice).collect::<Vec<&[u8]>>());
                    // `sendfile(2)` on this host spends the length cell on the
                    // header and the file together — "the value of len argument
                    // indicates the maximum number of bytes in the header
                    // and/or file to be sent" — so a caller asking for `count`
                    // bytes of the file has to be given room for its headers on
                    // top, or the headers eat into the range it asked for. The
                    // trailer is outside the budget and is always sent whole.
                    // A count of 0 already asks for everything and stays 0.
                    let count = match header_buffers.as_ref() {
                        Some(buffers) if count_raw != 0 => {
                            buffers.iter().try_fold(count_raw, |count, buffer| {
                                count.checked_add(buffer.len() as i64).ok_or_else(|| {
                                    crate::PyError::overflow_error("sendfile() count is too large")
                                })
                            })?
                        }
                        _ => count_raw,
                    };
                    loop {
                        let (res, written) = pyre_object::with_roots!(
                            w_out_fd, w_in_fd, w_offset, w_count,
                            w_headers, w_trailers, w_flags => {
                            let _blocked = crate::module::thread::before_external_block();
                            host_posix::sendfile(
                                in_b,
                                out_b,
                                offset_i64 as rustpython_host_env::crt_fd::Offset,
                                count,
                                header_slices.as_deref(),
                                trailer_slices.as_deref(),
                            )
                        });
                        match res {
                            Ok(_) => {
                                return Ok(pyre_object::with_roots!(
                                    w_out_fd, w_in_fd, w_offset, w_count,
                                    w_headers, w_trailers, w_flags => {
                                    pyre_object::w_int_new(written)
                                }));
                            }
                            Err(error) => {
                                // rposix.sendfile: BSD sendfile reports a
                                // partial transfer through sbytes even when the
                                // syscall result is EAGAIN/EBUSY. Return that
                                // progress so asyncio advances its file offset
                                // instead of resending the same range. EINTR is
                                // not in that set, so a partial transfer a signal
                                // interrupted goes to the retry below, which asks
                                // for the caller's original range again.
                                if written != 0
                                    && matches!(
                                        error.raw_os_error(),
                                        Some(libc::EAGAIN) | Some(libc::EBUSY)
                                    )
                                {
                                    return Ok(pyre_object::with_roots!(
                                        w_out_fd, w_in_fd, w_offset, w_count,
                                        w_headers, w_trailers, w_flags => {
                                        pyre_object::w_int_new(written)
                                    }));
                                }
                                pyre_object::with_roots!(
                                    w_out_fd, w_in_fd, w_offset, w_count,
                                    w_headers, w_trailers, w_flags => {
                                    crate::builtins::eintr_retry_with(error, |e| io_err(e, ""))
                                })?;
                            }
                        }
                    }
                }
            })
        );

        // os.posix_spawn(path, argv, env, *, file_actions=None, setpgroup=None,
        // resetids=False, setsid=False, setsigmask=(), setsigdef=(),
        // scheduler=None) -> pid
        // os.posix_spawnp(file, argv, env, *, file_actions=None, ...) -> pid
        #[cfg(all(
            any(target_os = "linux", target_os = "freebsd", target_os = "macos"),
            not(feature = "sandbox")
        ))]
        {
            #[cfg(all(
                any(target_os = "linux", target_os = "freebsd"),
                not(target_env = "musl")
            ))]
            use rustpython_host_env::posix::PosixSpawnScheduler as SpawnScheduler;
            #[cfg(not(all(
                any(target_os = "linux", target_os = "freebsd"),
                not(target_env = "musl")
            )))]
            type SpawnScheduler = ();

            fn build_posix_spawn(
                args: &[pyre_object::PyObjectRef],
                spawnp: bool,
            ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
                // The two entry points share this body and have an argument
                // clinic declaration each, so the one the caller reached is the
                // name its rejected path reports.
                let func = if spawnp {
                    "posix_spawnp"
                } else {
                    "posix_spawn"
                };
                // `(path, argv, env, /, *, file_actions=(), ...)` — the three
                // names are positional-only and everything else is
                // keyword-only, so there is no positional-or-keyword slot at
                // all.
                let (bound, kwargs) = bind_posonly_args(
                    args,
                    func,
                    func,
                    3,
                    3,
                    &[
                        "file_actions",
                        "setpgroup",
                        "resetids",
                        "setsid",
                        "setsigmask",
                        "setsigdef",
                        "scheduler",
                    ],
                )?;
                // Every conversion below reaches app-level code -- `__fspath__`,
                // a mapping's `keys()`, `__index__` -- and each one can collect
                // and move the arguments that are still unread, including the
                // keyword dictionary the remaining options are looked up in.
                // Publish them once and read each back at its use.
                let _roots = pyre_object::gc_roots::push_roots();
                let positional_base = pyre_object::gc_roots::publish_roots(&[
                    bound[0].expect("path is required"),
                    bound[1].expect("argv is required"),
                    bound[2].expect("env is required"),
                ]);
                let kwargs_slot =
                    kwargs.map(|kwargs| pyre_object::gc_roots::publish_roots(&[kwargs]));
                pyre_object::gc_roots::normalize_roots(
                    positional_base,
                    3 + usize::from(kwargs_slot.is_some()),
                );
                let positional =
                    |index: usize| pyre_object::gc_roots::shadow_stack_get(positional_base + index);
                let kwargs = || kwargs_slot.map(pyre_object::gc_roots::shadow_stack_get);
                let path = crate::gateway::fsencode_path_named_w(positional(0), func, "path")?;
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice()).map_err(|_| {
                    crate::PyError::value_error("posix_spawn: embedded null in path")
                })?;
                let argv = collect_cstring_seq(positional(1), func, "argv")?;
                // posixmodule.c parses `env` through the same keys/values
                // snapshot used by execve, then filesystem-encodes paired
                // elements into `key=value`.
                let env = collect_spawn_env(positional(2))?;
                let setpgroup = match crate::builtins::kwarg_get(kwargs(), "setpgroup") {
                    Some(value) if !unsafe { pyre_object::is_none(value) } => {
                        let value = crate::baseobjspace::space_index(value)?;
                        let value = crate::baseobjspace::int_w(value)?;
                        Some(libc::pid_t::try_from(value).map_err(|_| {
                            crate::PyError::overflow_error(
                                "Python int too large to convert to C pid_t",
                            )
                        })?)
                    }
                    _ => None,
                };
                let resetids = crate::builtins::kwarg_get(kwargs(), "resetids")
                    .map(crate::baseobjspace::is_true)
                    .transpose()?
                    .unwrap_or(false);
                let setsid = crate::builtins::kwarg_get(kwargs(), "setsid")
                    .map(crate::baseobjspace::is_true)
                    .transpose()?
                    .unwrap_or(false);
                if setsid && !host_posix::supports_posix_spawn_setsid() {
                    return Err(argument_unavailable(func, "setsid"));
                }
                let setsigmask = match crate::builtins::kwarg_get(kwargs(), "setsigmask") {
                    Some(value) => Some(sigset_arg(value)?),
                    None => None,
                };
                let setsigdef = match crate::builtins::kwarg_get(kwargs(), "setsigdef") {
                    Some(value) => Some(sigset_arg(value)?),
                    None => None,
                };
                let _scheduler = parse_spawn_scheduler(func, kwargs())?;
                let file_actions_obj = crate::builtins::kwarg_get(kwargs(), "file_actions");
                let actions: Vec<rustpython_host_env::posix::PosixSpawnFileAction> =
                    if let Some(fa) = file_actions_obj {
                        if unsafe { pyre_object::is_none(fa) } {
                            Vec::new()
                        } else {
                            decode_file_actions(fa)?
                        }
                    } else {
                        Vec::new()
                    };
                let config = host_posix::PosixSpawnConfig {
                    path: c_path.as_c_str(),
                    args: &argv,
                    env: &env,
                    file_actions: &actions,
                    setsigdef: setsigdef.as_deref(),
                    setpgroup,
                    resetids,
                    setsid,
                    setsigmask: setsigmask.as_deref(),
                    #[cfg(all(
                        any(target_os = "linux", target_os = "freebsd"),
                        not(target_env = "musl")
                    ))]
                    scheduler: _scheduler,
                    spawnp,
                };
                let result = {
                    let _blocked = crate::module::thread::before_external_block();
                    host_posix::posix_spawn(config)
                };
                let pid = result.map_err(|e| io_err_with_filename(e, path.w_path()))?;
                Ok(pyre_object::w_int_new(pid as i64))
            }
            fn collect_spawn_env(
                mut mapping: pyre_object::PyObjectRef,
            ) -> Result<Vec<std::ffi::CString>, crate::PyError> {
                // Inherit snapshot is `rposix_environ.envitems_llimpl` so it
                // shares the keepalive mutex with `putenv_llimpl`.
                let entries = if unsafe { pyre_object::is_none(mapping) } {
                    pyre_object::with_roots!(mapping => {
                        majit_rlib::rposix_environ::envitems_llimpl()
                    })
                    .into_iter()
                    .map(|(key, value)| {
                        let mut entry = Vec::with_capacity(key.len() + 1 + value.len());
                        entry.extend_from_slice(&key);
                        entry.push(b'=');
                        entry.extend_from_slice(&value);
                        entry
                    })
                    .collect()
                } else {
                    pyre_object::with_roots!(mapping => {
                        collect_env_entries(mapping, "posix_spawn", true)
                    })?
                };
                entries
                    .into_iter()
                    .map(|entry| {
                        std::ffi::CString::new(entry).map_err(|_| {
                            crate::PyError::value_error(
                                "posix_spawn() environment contains an embedded null byte",
                            )
                        })
                    })
                    .collect()
            }
            fn collect_cstring_seq(
                obj: pyre_object::PyObjectRef,
                fn_name: &str,
                arg_name: &str,
            ) -> Result<Vec<std::ffi::CString>, crate::PyError> {
                let items: Vec<pyre_object::PyObjectRef> = if unsafe { pyre_object::is_list(obj) } {
                    let n = unsafe { pyre_object::w_list_len(obj) };
                    (0..n)
                        .filter_map(|i| unsafe { pyre_object::w_list_getitem(obj, i as i64) })
                        .collect()
                } else if unsafe { pyre_object::is_tuple(obj) } {
                    let n = unsafe { pyre_object::w_tuple_len(obj) };
                    (0..n)
                        .filter_map(|i| unsafe { pyre_object::w_tuple_getitem(obj, i as i64) })
                        .collect()
                } else {
                    return Err(crate::PyError::type_error(format!(
                        "{fn_name}(): {arg_name} must be a list or tuple",
                    )));
                };
                // `interp_posix.py:1742 args = [space.fsencode_w(w_arg)
                // for w_arg in args_w]`: every argv/envp entry crosses
                // to the new process as filesystem bytes, and the same
                // converter decides what an entry may be.
                // `fsencode_bytes_w` reaches `__fspath__` for a non-str, non-bytes entry, so
                // encoding one entry can collect and move the entries not yet converted.
                // Publish the sequence once and read each entry back per iteration.
                let _seq_roots = pyre_object::gc_roots::push_roots();
                let items_base = pyre_object::gc_roots::pin_roots(&items);
                let mut out = Vec::with_capacity(items.len());
                for i in 0..items.len() {
                    let bytes = crate::gateway::fsencode_bytes_w(
                        pyre_object::gc_roots::shadow_stack_get(items_base + i),
                    )?;
                    out.push(std::ffi::CString::new(bytes).map_err(|_| {
                        crate::PyError::value_error(format!(
                            "{fn_name}(): embedded null in {arg_name}",
                        ))
                    })?);
                }
                Ok(out)
            }
            fn decode_file_actions(
                obj: pyre_object::PyObjectRef,
            ) -> Result<Vec<rustpython_host_env::posix::PosixSpawnFileAction>, crate::PyError>
            {
                use rustpython_host_env::posix::PosixSpawnFileAction;
                let items =
                    crate::builtins::sequence_fast(obj, "file_actions must be a sequence or None")?;
                // Every field of a `file_actions` entry is an `int` argument of
                // `os.posix_spawn`, so it is converted rather than read as a
                // payload: the caller controls the tuple's contents, and an
                // `int` subclass or a plain non-int would otherwise be
                // reinterpreted as a descriptor, flag set or mode.
                //
                // That conversion reaches `__index__`, and an OPEN path reaches
                // `__fspath__`, so reading one field can collect and move the
                // entries not yet read. The sequence is published once and each
                // entry read back out of its slot at every use, the way
                // `collect_cstring_seq` above does.
                let field = |slot: usize, index: i64| -> Result<i32, crate::PyError> {
                    let entry = pyre_object::gc_roots::shadow_stack_get(slot);
                    let value =
                        unsafe { pyre_object::w_tuple_getitem(entry, index) }.ok_or_else(|| {
                            crate::PyError::type_error(
                                "Each file_actions element must be a non-empty tuple",
                            )
                        })?;
                    crate::baseobjspace::c_int_w(value)
                };
                let _seq_roots = pyre_object::gc_roots::push_roots();
                let items_base = pyre_object::gc_roots::pin_roots(&items);
                let mut out = Vec::with_capacity(items.len());
                for offset in 0..items.len() {
                    let slot = items_base + offset;
                    let entry = pyre_object::gc_roots::shadow_stack_get(slot);
                    let tlen = if unsafe { pyre_object::is_tuple(entry) } {
                        unsafe { pyre_object::w_tuple_len(entry) }
                    } else {
                        return Err(crate::PyError::type_error(
                            "Each file_actions element must be a non-empty tuple",
                        ));
                    };
                    if tlen == 0 {
                        return Err(crate::PyError::type_error(
                            "Each file_actions element must be a non-empty tuple",
                        ));
                    }
                    let op = field(slot, 0)?;
                    match op {
                        0 => {
                            // POSIX_SPAWN_OPEN: (op, fd, path, flags, mode)
                            if tlen != 5 {
                                return Err(crate::PyError::type_error(
                                    "A open file_action tuple must have 5 elements",
                                ));
                            }
                            let fd = field(slot, 1)?;
                            let path_obj = unsafe {
                                pyre_object::w_tuple_getitem(
                                    pyre_object::gc_roots::shadow_stack_get(slot),
                                    2,
                                )
                                .unwrap()
                            };
                            let path = extract_path(path_obj)?;
                            let cpath = std::ffi::CString::new(path).map_err(|_| {
                                crate::PyError::value_error(
                                    "posix_spawn: embedded null in OPEN path",
                                )
                            })?;
                            let oflag = field(slot, 3)?;
                            let mode = field(slot, 4)? as u32;
                            out.push(PosixSpawnFileAction::Open {
                                fd,
                                path: cpath,
                                oflag,
                                mode,
                            });
                        }
                        1 => {
                            // POSIX_SPAWN_CLOSE: (op, fd)
                            if tlen != 2 {
                                return Err(crate::PyError::type_error(
                                    "A close file_action tuple must have 2 elements",
                                ));
                            }
                            let fd = field(slot, 1)?;
                            out.push(PosixSpawnFileAction::Close { fd });
                        }
                        2 => {
                            // POSIX_SPAWN_DUP2: (op, fd, newfd)
                            if tlen != 3 {
                                return Err(crate::PyError::type_error(
                                    "A dup2 file_action tuple must have 3 elements",
                                ));
                            }
                            let fd = field(slot, 1)?;
                            let newfd = field(slot, 2)?;
                            out.push(PosixSpawnFileAction::Dup2 { fd, newfd });
                        }
                        _ => {
                            return Err(crate::PyError::type_error(
                                "Unknown file_actions identifier",
                            ));
                        }
                    }
                }
                Ok(out)
            }

            fn sigset_arg(value: PyObjectRef) -> Result<Vec<i32>, crate::PyError> {
                let items = crate::builtins::collect_iterable(value)?;
                // `space_index` runs `__index__`, so reading one element can
                // collect and move the elements not yet read.  Publish the
                // sequence once and read each element back per iteration.
                let _seq_roots = pyre_object::gc_roots::push_roots();
                let items_base = pyre_object::gc_roots::pin_roots(&items);
                let mut sigs = Vec::with_capacity(items.len());
                for offset in 0..items.len() {
                    let item = pyre_object::gc_roots::shadow_stack_get(items_base + offset);
                    let item = crate::baseobjspace::space_index(item)?;
                    let signum = crate::baseobjspace::int_w(item)?;
                    if !(1..crate::module::signal::signalstate::NSIG as i64).contains(&signum) {
                        return Err(crate::PyError::value_error(format!(
                            "signal number {signum} out of range [1; {}]",
                            crate::module::signal::signalstate::NSIG - 1
                        )));
                    }
                    sigs.push(signum as i32);
                }
                Ok(sigs)
            }

            fn parse_spawn_scheduler(
                func: &str,
                kwargs: Option<PyObjectRef>,
            ) -> Result<Option<SpawnScheduler>, crate::PyError> {
                let Some(value) = crate::builtins::kwarg_get(kwargs, "scheduler") else {
                    return Ok(None);
                };
                if unsafe { pyre_object::is_none(value) } {
                    return Ok(None);
                }
                if unsafe { !pyre_object::is_tuple(value) } {
                    return Err(crate::PyError::type_error(format!(
                        "{func}: scheduler must be a tuple or None"
                    )));
                }
                if unsafe { pyre_object::w_tuple_len(value) } != 2 {
                    return Err(crate::PyError::type_error(
                        "A scheduler tuple must have two elements",
                    ));
                }

                #[cfg(all(target_os = "linux", not(target_env = "musl")))]
                {
                    // `sched_priority_w` reaches `__index__`, so the collection
                    // it can trigger forwards the tuple's own slots while a raw
                    // element read before it goes stale.  Root the tuple and
                    // take the policy out of it afterwards.
                    let _roots = pyre_object::gc_roots::push_roots();
                    let tuple_slot = pyre_object::gc_roots::shadow_stack_len();
                    let value = pyre_object::gc_roots::pin_root(value);
                    let param_obj = unsafe { pyre_object::w_tuple_getitem(value, 1).unwrap() };
                    let priority = sched_priority_w(param_obj)?;
                    let value = pyre_object::gc_roots::shadow_stack_get(tuple_slot);
                    let policy_obj = unsafe { pyre_object::w_tuple_getitem(value, 0).unwrap() };
                    let mut param: libc::sched_param =
                        unsafe { core::mem::zeroed::<libc::sched_param>() };
                    param.sched_priority = priority;
                    let policy = if unsafe { pyre_object::is_none(policy_obj) } {
                        None
                    } else {
                        let policy = crate::baseobjspace::space_index(policy_obj)?;
                        let policy = crate::baseobjspace::int_w(policy)?;
                        Some(libc::c_int::try_from(policy).map_err(|_| {
                            crate::PyError::overflow_error(
                                "Python int too large to convert to C int",
                            )
                        })?)
                    };
                    Ok(Some(SpawnScheduler { policy, param }))
                }

                #[cfg(any(not(target_os = "linux"), target_env = "musl"))]
                {
                    Err(crate::PyError::not_implemented(
                        "The scheduler option is not supported in this system.",
                    ))
                }
            }

            crate::__pyre_put_new!(
                ns_slot,
                "posix_spawn",
                crate::make_builtin_function("posix_spawn", |args| {
                    build_posix_spawn(args, false)
                })
            );
            crate::__pyre_put_new!(
                ns_slot,
                "posix_spawnp",
                crate::make_builtin_function("posix_spawnp", |args| {
                    build_posix_spawn(args, true)
                })
            );
            crate::__pyre_put_new!(ns_slot, "POSIX_SPAWN_OPEN", pyre_object::w_int_new(0));
            crate::__pyre_put_new!(ns_slot, "POSIX_SPAWN_CLOSE", pyre_object::w_int_new(1));
            crate::__pyre_put_new!(ns_slot, "POSIX_SPAWN_DUP2", pyre_object::w_int_new(2));
        }

        // os.ttyname(fd) -> str
        crate::__pyre_put_new!(
            ns_slot,
            "ttyname",
            crate::make_builtin_function_with_arity(
                "ttyname",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("ttyname() requires fd"));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int)`.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    // `-1` is EBADF before the call. `rposix.c_ttyname` is
                    // `releasegil=False` and saves errno.
                    fd_borrow(fd)?;
                    let name = unsafe { majit_rlib::rposix::c_ttyname(fd) };
                    if name.is_null() {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    let bytes = unsafe { std::ffi::CStr::from_ptr(name) }.to_bytes();
                    Ok(crate::gateway::fsdecode_filename_bytes(bytes))
                },
                1,
            )
        );

        // os.tcgetpgrp(fd) -> pgid. `rposix.c_tcgetpgrp` releases the GIL and
        // saves errno.
        crate::__pyre_put_new!(
            ns_slot,
            "tcgetpgrp",
            crate::make_builtin_function_with_arity(
                "tcgetpgrp",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("tcgetpgrp() requires fd"));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int)`.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let pgid = unsafe { majit_rlib::rposix::c_tcgetpgrp(fd) };
                    if pgid < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_int_new(pgid as i64))
                },
                1,
            )
        );

        // os.tcsetpgrp(fd, pgid) -> None. `rposix.c_tcsetpgrp` releases the
        // GIL and saves errno. `rposix.tcsetpgrp` discards
        // `handle_posix_error`'s result.
        crate::__pyre_put_new!(
            ns_slot,
            "tcsetpgrp",
            crate::make_builtin_function_with_arity(
                "tcsetpgrp",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("tcsetpgrp() requires fd, pgid"));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int, pgid=c_gid_t)`.
                    let mut w_fd = args[0];
                    let mut w_pgid = args[1];
                    let fd = pyre_object::with_roots!(w_fd, w_pgid => crate::baseobjspace::c_int_w(w_fd))?;
                    let pgid = crate::baseobjspace::c_uid_t_w(w_pgid)? as libc::pid_t;
                    let ret = unsafe { majit_rlib::rposix::c_tcsetpgrp(fd, pgid) };
                    if ret < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            )
        );

        // os.getpriority(which, who) -> int
        crate::__pyre_put_new!(
            ns_slot,
            "getpriority",
            crate::make_builtin_function_with_arity(
                "getpriority",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "getpriority() requires which, who",
                        ));
                    }
                    // interp_posix.py `@unwrap_spec(which=int, who=int)`.
                    let w_which = args[0];
                    let mut w_who = args[1];
                    // The host types keep the truncation `getpriority` used
                    // to apply, then the declaration's `c_int` / `id_t`.
                    // `rposix.c_getpriority` uses `RFFI_FULL_ERRNO_ZERO`: `-1`
                    // is a successful priority when the saved errno stays 0.
                    let which = pyre_object::with_roots!(w_who => crate::baseobjspace::int_w(w_which))?
                        as host_posix::PriorityWhichType
                        as libc::c_int;
                    let who = crate::baseobjspace::int_w(w_who)? as host_posix::PriorityWhoType
                        as libc::id_t;
                    let prio = unsafe { majit_rlib::rposix::c_getpriority(which, who) };
                    let errno = majit_rlib::rposix::get_saved_errno();
                    if errno != 0 {
                        return Err(io_err(std::io::Error::from_raw_os_error(errno), ""));
                    }
                    Ok(pyre_object::w_int_new(prio as i64))
                },
                2,
            )
        );

        // os.setpriority(which, who, priority) -> None
        crate::__pyre_put_new!(
            ns_slot,
            "setpriority",
            crate::make_builtin_function_with_arity(
                "setpriority",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error(
                            "setpriority() requires which, who, priority",
                        ));
                    }
                    // interp_posix.py:2352 `@unwrap_spec(which=int, who=int,
                    // priority=int)`.
                    let w_which = args[0];
                    let mut w_who = args[1];
                    let mut w_prio = args[2];
                    let which = pyre_object::with_roots!(w_who, w_prio =>
                        crate::baseobjspace::int_w(w_which))?
                        as host_posix::PriorityWhichType
                        as libc::c_int;
                    let who = pyre_object::with_roots!(w_prio => crate::baseobjspace::int_w(w_who))?
                        as host_posix::PriorityWhoType as libc::id_t;
                    let prio = crate::baseobjspace::int_w(w_prio)? as i32;
                    // `rposix.c_setpriority` releases the GIL and saves errno.
                    // `-1` is the failure.
                    let ret = unsafe { majit_rlib::rposix::c_setpriority(which, who, prio) };
                    if ret < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_none())
                },
                3,
            )
        );

        crate::__pyre_put_new!(
            ns_slot,
            "PRIO_PROCESS",
            pyre_object::w_int_new(libc::PRIO_PROCESS as i64)
        );
        crate::__pyre_put_new!(ns_slot, "PRIO_PGRP", pyre_object::w_int_new(libc::PRIO_PGRP as i64));
        crate::__pyre_put_new!(ns_slot, "PRIO_USER", pyre_object::w_int_new(libc::PRIO_USER as i64));

        // `posixmodule.c` `pathconf_names` — the `_PC_*` table
        // `conv_path_confname` resolves a string `name` argument through.
        // `libc` exports the constants only on the BSD family, so the glibc
        // values (`bits/confname.h`) are spelled out for Linux.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        const PATHCONF_NAMES: &[(&str, i32)] = &[
            ("PC_ALLOC_SIZE_MIN", libc::_PC_ALLOC_SIZE_MIN),
            ("PC_ASYNC_IO", libc::_PC_ASYNC_IO),
            ("PC_CHOWN_RESTRICTED", host_posix::_PC_CHOWN_RESTRICTED),
            ("PC_FILESIZEBITS", libc::_PC_FILESIZEBITS),
            ("PC_LINK_MAX", host_posix::_PC_LINK_MAX),
            ("PC_MAX_CANON", host_posix::_PC_MAX_CANON),
            ("PC_MAX_INPUT", host_posix::_PC_MAX_INPUT),
            ("PC_MIN_HOLE_SIZE", libc::_PC_MIN_HOLE_SIZE),
            ("PC_NAME_MAX", host_posix::_PC_NAME_MAX),
            ("PC_NO_TRUNC", host_posix::_PC_NO_TRUNC),
            ("PC_PATH_MAX", host_posix::_PC_PATH_MAX),
            ("PC_PIPE_BUF", host_posix::_PC_PIPE_BUF),
            ("PC_PRIO_IO", libc::_PC_PRIO_IO),
            ("PC_REC_INCR_XFER_SIZE", libc::_PC_REC_INCR_XFER_SIZE),
            ("PC_REC_MAX_XFER_SIZE", libc::_PC_REC_MAX_XFER_SIZE),
            ("PC_REC_MIN_XFER_SIZE", libc::_PC_REC_MIN_XFER_SIZE),
            ("PC_REC_XFER_ALIGN", libc::_PC_REC_XFER_ALIGN),
            ("PC_SYMLINK_MAX", libc::_PC_SYMLINK_MAX),
            ("PC_SYNC_IO", libc::_PC_SYNC_IO),
            ("PC_VDISABLE", host_posix::_PC_VDISABLE),
        ];
        #[cfg(target_os = "linux")]
        const PATHCONF_NAMES: &[(&str, i32)] = &[
            ("PC_2_SYMLINKS", host_posix::_PC_2_SYMLINKS),
            ("PC_ALLOC_SIZE_MIN", host_posix::_PC_ALLOC_SIZE_MIN),
            ("PC_ASYNC_IO", host_posix::_PC_ASYNC_IO),
            ("PC_CHOWN_RESTRICTED", host_posix::_PC_CHOWN_RESTRICTED),
            ("PC_FILESIZEBITS", host_posix::_PC_FILESIZEBITS),
            ("PC_LINK_MAX", host_posix::_PC_LINK_MAX),
            ("PC_MAX_CANON", host_posix::_PC_MAX_CANON),
            ("PC_MAX_INPUT", host_posix::_PC_MAX_INPUT),
            ("PC_NAME_MAX", host_posix::_PC_NAME_MAX),
            ("PC_NO_TRUNC", host_posix::_PC_NO_TRUNC),
            ("PC_PATH_MAX", host_posix::_PC_PATH_MAX),
            ("PC_PIPE_BUF", host_posix::_PC_PIPE_BUF),
            ("PC_PRIO_IO", host_posix::_PC_PRIO_IO),
            ("PC_REC_INCR_XFER_SIZE", host_posix::_PC_REC_INCR_XFER_SIZE),
            ("PC_REC_MAX_XFER_SIZE", host_posix::_PC_REC_MAX_XFER_SIZE),
            ("PC_REC_MIN_XFER_SIZE", host_posix::_PC_REC_MIN_XFER_SIZE),
            ("PC_REC_XFER_ALIGN", host_posix::_PC_REC_XFER_ALIGN),
            ("PC_SOCK_MAXBUF", 12),
            ("PC_SYMLINK_MAX", host_posix::_PC_SYMLINK_MAX),
            ("PC_SYNC_IO", host_posix::_PC_SYNC_IO),
            ("PC_VDISABLE", host_posix::_PC_VDISABLE),
        ];
        #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "linux")))]
        const PATHCONF_NAMES: &[(&str, i32)] = &[];
        /// The dict a `conv_confname` table is published as — `pathconf_names`
        /// for `pathconf`, `confstr_names` for `confstr`.
        fn store_names_dict(ns: PyObjectRef, key: &str, table: &[(&str, i32)]) {
            let _root_scope = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ns = pyre_object::gc_roots::pin_root(ns);
            let _names_roots = pyre_object::gc_roots::push_roots();
            let names_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_dict_new());
            for (name, value) in table {
                // The value is allocated before the store, and the dict is
                // reloaded from its root slot every iteration because the
                // insert itself can grow — and so relocate — the dict.
                let _value_roots = pyre_object::gc_roots::push_roots();
                let value_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(*value as i64));
                unsafe {
                    pyre_object::w_dict_setitem_str(
                        pyre_object::gc_roots::shadow_stack_get(names_slot),
                        name,
                        pyre_object::gc_roots::shadow_stack_get(value_slot),
                    )
                };
            }
            crate::__pyre_put_new!(
                ns_slot,
                key,
                pyre_object::gc_roots::shadow_stack_get(names_slot)
            );
        }
        store_names_dict(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            "pathconf_names",
            PATHCONF_NAMES,
        );

        /// A limit the host has no determinate answer for. `pathconf` reports
        /// it as `-1` with the errno left alone, which the host wrapper spells
        /// `None` — but `interp_posix.py` hands whatever `pathconf`
        /// returned straight to `space.newint`, so what the caller sees is the
        /// number `-1`. `PC_ASYNC_IO` and `PC_SYMLINK_MAX` answer this way on
        /// hosts that do not implement them, and `None` is neither the value
        /// nor the type the caller can compare against a limit.
        fn indeterminate_limit(limit: Option<libc::c_long>) -> i64 {
            limit.map_or(-1, |v| v)
        }

        /// `-1` with errno 0 is the indeterminate answer `pathconf` spells
        /// `None`. `rposix.c_pathconf` and `rposix.c_fpathconf` use
        /// `RFFI_FULL_ERRNO_ZERO`, so the saved errno is this call's.
        fn limit_or_errno(raw: libc::c_long) -> std::io::Result<Option<libc::c_long>> {
            if raw == -1 {
                let errno = majit_rlib::rposix::get_saved_errno();
                if errno != 0 {
                    return Err(std::io::Error::from_raw_os_error(errno));
                }
                return Ok(None);
            }
            Ok(Some(raw))
        }

        /// `posixmodule.c conv_confname`: an `int` passes through, a `str` is
        /// resolved through the table the caller's entry point publishes.
        fn confname_arg(w: PyObjectRef, table: &[(&str, i32)]) -> Result<i32, crate::PyError> {
            if unsafe { pyre_object::is_str(w) } {
                // A str carrying a lone surrogate has no `&str` view.  It simply
                // matches no known name, which is the ValueError below — not an
                // interpreter abort, which is what reading the value unchecked
                // would produce.
                let name = unsafe { pyre_object::w_str_get_value_opt(w) };
                return name
                    .and_then(|name| {
                        table
                            .iter()
                            .find(|(known, _)| *known == name)
                            .map(|(_, value)| *value)
                    })
                    .ok_or_else(|| crate::PyError::value_error("unrecognized configuration name"));
            }
            // `conv_confname` gates on `PyIndex_Check` before converting, so an
            // object that is neither a str nor index-able is this TypeError,
            // while an `__index__` that raises propagates its own exception.
            if !unsafe { crate::builtins::index_check(w) } {
                return Err(crate::PyError::type_error(
                    "configuration names must be strings or integers",
                ));
            }
            // `conv_confname` narrows to a C `int` and reports a value that does
            // not fit rather than truncating it. A truncated name reaches the
            // syscall as an unrelated one — `2**40` narrows to 0 — and comes
            // back EINVAL, which reads as "no such configuration option" for a
            // name the caller never asked about. The object is an int here, so
            // an `int_w` that fails did so on width.
            let too_large =
                || crate::PyError::overflow_error("Python int too large to convert to C int");
            let value = crate::baseobjspace::int_w(crate::baseobjspace::space_index(w)?)
                .map_err(|_| too_large())?;
            i32::try_from(value).map_err(|_| too_large())
        }

        // os.pathconf(path, name) -> int | None
        crate::__pyre_put_new!(
            ns_slot,
            "pathconf",
            crate::make_builtin_function_with_arity(
                "pathconf",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("pathconf() requires path, name"));
                    }
                    // interp_posix.py `path_or_fd(allow_fd=hasattr(os,
                    // 'fpathconf'))`, whose body converts the name before it
                    // reads the path/fd discriminant.
                    let w_path = args[0];
                    let mut w_name = args[1];
                    // The path owns a bracket of its own, above this one; this
                    // one stays open until the path is gone.
                    let name_roots = pyre_object::gc_roots::push_roots();
                    let name_base = name_roots.pin_roots(&[w_name]);
                    let path =
                        crate::gateway::fsencode_path_or_fd_w(w_path, "pathconf", HAVE_FPATHCONF);
                    w_name = name_roots.get(name_base);
                    let path = path?;
                    let name = confname_arg(w_name, PATHCONF_NAMES)?;
                    let limit = if path.is_fd {
                        let raw = unsafe { majit_rlib::rposix::c_fpathconf(path.as_fd, name) };
                        limit_or_errno(raw).map_err(|e| io_err(e, ""))?
                    } else {
                        let cpath =
                            std::ffi::CString::new(path.as_bytes.as_slice()).map_err(|_| {
                                crate::PyError::value_error("pathconf: embedded null in path")
                            })?;
                        let raw = unsafe { majit_rlib::rposix::c_pathconf(cpath.as_ptr(), name) };
                        limit_or_errno(raw).map_err(|e| io_err_with_filename(e, path.w_path()))?
                    };
                    Ok(pyre_object::w_int_new(indeterminate_limit(limit)))
                },
                2,
            )
        );

        // os.fpathconf(fd, name) -> int | None
        crate::__pyre_put_new!(
            ns_slot,
            "fpathconf",
            crate::make_builtin_function_with_arity(
                "fpathconf",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("fpathconf() requires fd, name"));
                    }
                    // interp_posix.py:2411 descriptor argument: accept an int
                    // or a `fileno()` object through `space.c_filedescriptor_w`;
                    // that boundary also raises the bool file descriptor warning.
                    let w_fd = args[0];
                    let mut w_name = args[1];
                    let fd = pyre_object::with_roots!(w_name =>
                        crate::baseobjspace::c_filedescriptor_w(w_fd))?;
                    let name = confname_arg(w_name, PATHCONF_NAMES)?;
                    let raw = unsafe { majit_rlib::rposix::c_fpathconf(fd, name) };
                    let limit = limit_or_errno(raw).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(indeterminate_limit(limit)))
                },
                2,
            )
        );

        // os.sysconf(name) -> int
        crate::__pyre_put_new!(
            ns_slot,
            "sysconf",
            crate::make_builtin_function_with_arity(
                "sysconf",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("sysconf() requires name"));
                    }
                    let name = confname_arg(args[0], sysconf_names())?;
                    // `rposix.c_sysconf` uses `RFFI_FULL_ERRNO_ZERO`. `-1`
                    // with errno 0 is a published answer.
                    let v = unsafe { majit_rlib::rposix::c_sysconf(name) };
                    if v == -1 {
                        let errno = majit_rlib::rposix::get_saved_errno();
                        if errno != 0 {
                            return Err(io_err(std::io::Error::from_raw_os_error(errno), ""));
                        }
                    }
                    Ok(pyre_object::w_int_new(v as i64))
                },
                1,
            )
        );
        let w_sysconf_names = pyre_object::w_dict_new();
        let _sysconf_names_root = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(w_sysconf_names);
        let names_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        for (name, value) in sysconf_names() {
            // Build the value first: call arguments evaluate left to right, so
            // reading the rooted dict inline would take its address before
            // `w_int_new` allocates, and a collection there would leave the
            // store writing through the pre-move address.
            let w_value = pyre_object::w_int_new(*value as i64);
            unsafe {
                pyre_object::w_dict_setitem_str(
                    pyre_object::gc_roots::shadow_stack_get(names_slot),
                    name,
                    w_value,
                );
            }
        }
        crate::__pyre_put_new!(
            ns_slot,
            "sysconf_names",
            pyre_object::gc_roots::shadow_stack_get(names_slot)
        );

        // `posixmodule.c` `posix_constants_confstr` — the `_CS_*` table
        // `conv_confstr_confname` resolves a string `name` argument through,
        // and the same candidate set `rposix.py:2248-2300` names. Every entry
        // there is `#ifdef`-guarded, so a host publishes exactly the names its
        // own `<unistd.h>` defines; `libc` carries `_CS_PATH` alone, and the
        // two numberings disagree from that first entry on — it is 1 on the
        // Apple targets and 0 in glibc's `bits/confname.h` enum. The ten names
        // the candidate set carries for the System V hosts (`CS_ARCHITECTURE`,
        // `CS_HOSTNAME`, `CS_HW_PROVIDER`, `CS_HW_SERIAL`, `CS_INITTAB_NAME`,
        // `CS_MACHINE`, `CS_RELEASE`, `CS_SRPC_DOMAIN`, `CS_SYSNAME`,
        // `CS_VERSION`) are defined by neither header, so neither table has
        // them.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        const CONFSTR_NAMES: &[(&str, i32)] = &[
            ("CS_PATH", 1),
            ("CS_XBS5_ILP32_OFF32_CFLAGS", 20),
            ("CS_XBS5_ILP32_OFF32_LDFLAGS", 21),
            ("CS_XBS5_ILP32_OFF32_LIBS", 22),
            ("CS_XBS5_ILP32_OFF32_LINTFLAGS", 23),
            ("CS_XBS5_ILP32_OFFBIG_CFLAGS", 24),
            ("CS_XBS5_ILP32_OFFBIG_LDFLAGS", 25),
            ("CS_XBS5_ILP32_OFFBIG_LIBS", 26),
            ("CS_XBS5_ILP32_OFFBIG_LINTFLAGS", 27),
            ("CS_XBS5_LP64_OFF64_CFLAGS", 28),
            ("CS_XBS5_LP64_OFF64_LDFLAGS", 29),
            ("CS_XBS5_LP64_OFF64_LIBS", 30),
            ("CS_XBS5_LP64_OFF64_LINTFLAGS", 31),
            ("CS_XBS5_LPBIG_OFFBIG_CFLAGS", 32),
            ("CS_XBS5_LPBIG_OFFBIG_LDFLAGS", 33),
            ("CS_XBS5_LPBIG_OFFBIG_LIBS", 34),
            ("CS_XBS5_LPBIG_OFFBIG_LINTFLAGS", 35),
        ];
        // glibc numbers the enum from zero and restarts it twice, at 1000 for
        // the large-file names and at 1100 for the XBS5 ones.
        #[cfg(target_os = "linux")]
        const CONFSTR_NAMES: &[(&str, i32)] = &[
            ("CS_PATH", 0),
            ("CS_GNU_LIBC_VERSION", 2),
            ("CS_GNU_LIBPTHREAD_VERSION", 3),
            ("CS_LFS_CFLAGS", 1000),
            ("CS_LFS_LDFLAGS", 1001),
            ("CS_LFS_LIBS", 1002),
            ("CS_LFS_LINTFLAGS", 1003),
            ("CS_LFS64_CFLAGS", 1004),
            ("CS_LFS64_LDFLAGS", 1005),
            ("CS_LFS64_LIBS", 1006),
            ("CS_LFS64_LINTFLAGS", 1007),
            ("CS_XBS5_ILP32_OFF32_CFLAGS", 1100),
            ("CS_XBS5_ILP32_OFF32_LDFLAGS", 1101),
            ("CS_XBS5_ILP32_OFF32_LIBS", 1102),
            ("CS_XBS5_ILP32_OFF32_LINTFLAGS", 1103),
            ("CS_XBS5_ILP32_OFFBIG_CFLAGS", 1104),
            ("CS_XBS5_ILP32_OFFBIG_LDFLAGS", 1105),
            ("CS_XBS5_ILP32_OFFBIG_LIBS", 1106),
            ("CS_XBS5_ILP32_OFFBIG_LINTFLAGS", 1107),
            ("CS_XBS5_LP64_OFF64_CFLAGS", 1108),
            ("CS_XBS5_LP64_OFF64_LDFLAGS", 1109),
            ("CS_XBS5_LP64_OFF64_LIBS", 1110),
            ("CS_XBS5_LP64_OFF64_LINTFLAGS", 1111),
            ("CS_XBS5_LPBIG_OFFBIG_CFLAGS", 1112),
            ("CS_XBS5_LPBIG_OFFBIG_LDFLAGS", 1113),
            ("CS_XBS5_LPBIG_OFFBIG_LIBS", 1114),
            ("CS_XBS5_LPBIG_OFFBIG_LINTFLAGS", 1115),
        ];
        #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "linux")))]
        const CONFSTR_NAMES: &[(&str, i32)] = &[];
        store_names_dict(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            "confstr_names",
            CONFSTR_NAMES,
        );

        // os.confstr(name) -> str | None
        #[cfg(not(feature = "sandbox"))]
        crate::__pyre_put_new!(
            ns_slot,
            "confstr",
            crate::make_builtin_function_with_arity(
                "confstr",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("confstr() requires name"));
                    }
                    let name = confname_arg(args[0], CONFSTR_NAMES)?;
                    // `rposix.confstr` (`rposix.py`) asks for the
                    // length first and fills a buffer of exactly that size on
                    // the second call. A zero length is either a name this host
                    // has no string for, which is `None`, or a name it does not
                    // know at all, which is the errno it set. `c_confstr` uses
                    // `RFFI_FULL_ERRNO_ZERO`, so errno is cleared before the
                    // question is put. The length call's errno is read before
                    // the second call can replace it.
                    let len =
                        unsafe { majit_rlib::rposix::c_confstr(name, std::ptr::null_mut(), 0) };
                    if len == 0 {
                        let errno = majit_rlib::rposix::get_saved_errno();
                        if errno != 0 {
                            return Err(errno_err(errno, ""));
                        }
                        return Ok(pyre_object::w_none());
                    }
                    let mut buf = vec![0u8; len];
                    unsafe {
                        majit_rlib::rposix::c_confstr(
                            name,
                            buf.as_mut_ptr() as *mut libc::c_char,
                            len,
                        )
                    };
                    // The length counts the terminator, which is not part of
                    // the string — `os_confstr_impl` decodes `len - 1` bytes.
                    // (`rffi.charp2strn(buf, n)` keeps it, so upstream's
                    // `space.newtext` carries a trailing NUL.)
                    buf.truncate(len - 1);
                    // The value can be a search path, so it is decoded the way
                    // every other name from the host is.
                    Ok(crate::gateway::fsdecode_filename_bytes(&buf))
                },
                1,
            )
        );

        // os.initgroups(username, gid) -> None
        #[cfg(any(
            target_os = "freebsd",
            target_os = "linux",
            target_os = "openbsd",
            target_os = "macos",
            target_os = "ios"
        ))]
        crate::__pyre_put_new!(
            ns_slot,
            "initgroups",
            crate::make_builtin_function_with_arity(
                "initgroups",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "initgroups() requires username, gid",
                        ));
                    }
                    let w_user = args[0];
                    let mut w_gid = args[1];
                    let user = unsafe {
                        if pyre_object::is_str(w_user) {
                            pyre_object::with_roots!(w_gid => crate::baseobjspace::str_utf8_w(w_user))?
                                .to_string()
                        } else {
                            return Err(crate::PyError::type_error(
                                "initgroups(): username must be str",
                            ));
                        }
                    };
                    let cuser = std::ffi::CString::new(user.as_bytes()).map_err(|_| {
                        crate::PyError::value_error("initgroups: embedded null in username")
                    })?;
                    // interp_posix.py `@unwrap_spec(username='text', gid=c_gid_t)`.
                    // Darwin's `initgroups` takes `int`; other unix hosts take `gid_t`.
                    let gid = crate::baseobjspace::c_uid_t_w(w_gid)?;
                    #[cfg(any(target_os = "macos", target_os = "ios"))]
                    let gid = gid as libc::c_int;
                    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                    let gid = gid as libc::gid_t;
                    // `rposix.c_initgroups` releases the GIL and saves errno.
                    let ret = unsafe { majit_rlib::rposix::c_initgroups(cuser.as_ptr(), gid) };
                    if ret < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            )
        );

        // os.openpty() -> (master_fd, slave_fd)
        crate::__pyre_put_new!(
            ns_slot,
            "openpty",
            crate::make_builtin_function_with_arity(
                "openpty",
                |_| {
                    // `rposix.c_openpty` releases the GIL and saves errno.
                    // `interp_posix.openpty` then `rposix.set_inheritable`
                    // (`_c_set_inheritable`); a failure there closes both ends.
                    let mut master = 0;
                    let mut slave = 0;
                    let ret = unsafe {
                        majit_rlib::rposix::c_openpty(
                            &mut master,
                            &mut slave,
                            std::ptr::null_mut::<libc::c_char>(),
                            std::ptr::null_mut::<libc::termios>(),
                            std::ptr::null_mut::<libc::winsize>(),
                        )
                    };
                    if ret < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    if unsafe { majit_rlib::rposix::_c_set_inheritable(master, 0) } < 0
                        || unsafe { majit_rlib::rposix::_c_set_inheritable(slave, 0) } < 0
                    {
                        let err = majit_rlib::rposix::get_saved_errno();
                        unsafe {
                            let _ = majit_rlib::rposix::c_close(master);
                            let _ = majit_rlib::rposix::c_close(slave);
                        }
                        return Err(io_err(std::io::Error::from_raw_os_error(err), ""));
                    }
                    let mut fields = pyre_object::gc_roots::RootedItems::new();
                    fields.push(pyre_object::w_int_new(master as i64));
                    fields.push(pyre_object::w_int_new(slave as i64));
                    Ok(pyre_object::w_tuple_new(fields.take()))
                },
                0,
            )
        );

        // os.getresuid() -> (ruid, euid, suid)
        #[cfg(any(target_os = "android", target_os = "linux", target_os = "openbsd"))]
        crate::__pyre_put_new!(
            ns_slot,
            "getresuid",
            crate::make_builtin_function_with_arity(
                "getresuid",
                |_| {
                    // `rposix.c_getresuid` releases the GIL and saves errno.
                    let mut r: libc::uid_t = 0;
                    let mut e: libc::uid_t = 0;
                    let mut s: libc::uid_t = 0;
                    let ret = unsafe { majit_rlib::rposix::c_getresuid(&mut r, &mut e, &mut s) };
                    if ret < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    let mut fields = pyre_object::gc_roots::RootedItems::new();
                    fields.push(pyre_object::w_int_new(r as i64));
                    fields.push(pyre_object::w_int_new(e as i64));
                    fields.push(pyre_object::w_int_new(s as i64));
                    Ok(pyre_object::w_tuple_new(fields.take()))
                },
                0,
            )
        );

        // os.getresgid() -> (rgid, egid, sgid)
        #[cfg(any(target_os = "android", target_os = "linux", target_os = "openbsd"))]
        crate::__pyre_put_new!(
            ns_slot,
            "getresgid",
            crate::make_builtin_function_with_arity(
                "getresgid",
                |_| {
                    // `rposix.c_getresgid` releases the GIL and saves errno.
                    let mut r: libc::gid_t = 0;
                    let mut e: libc::gid_t = 0;
                    let mut s: libc::gid_t = 0;
                    let ret = unsafe { majit_rlib::rposix::c_getresgid(&mut r, &mut e, &mut s) };
                    if ret < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    let mut fields = pyre_object::gc_roots::RootedItems::new();
                    fields.push(pyre_object::w_int_new(r as i64));
                    fields.push(pyre_object::w_int_new(e as i64));
                    fields.push(pyre_object::w_int_new(s as i64));
                    Ok(pyre_object::w_tuple_new(fields.take()))
                },
                0,
            )
        );

        // os.setresuid(ruid, euid, suid) -> None
        #[cfg(any(
            target_os = "android",
            target_os = "freebsd",
            target_os = "linux",
            target_os = "openbsd"
        ))]
        crate::__pyre_put_new!(
            ns_slot,
            "setresuid",
            crate::make_builtin_function_with_arity(
                "setresuid",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error(
                            "setresuid() requires ruid, euid, suid",
                        ));
                    }
                    // interp_posix.py `@unwrap_spec(ruid=c_uid_t,
                    // euid=c_uid_t, suid=c_uid_t)`.
                    let mut w_ruid = args[0];
                    let mut w_euid = args[1];
                    let mut w_suid = args[2];
                    let r = pyre_object::with_roots!(w_ruid, w_euid, w_suid => {
                        crate::baseobjspace::c_uid_t_w(w_ruid)
                    })?;
                    let e = pyre_object::with_roots!(w_euid, w_suid => crate::baseobjspace::c_uid_t_w(w_euid))?;
                    let s = crate::baseobjspace::c_uid_t_w(w_suid)?;
                    // `rposix.c_setresuid` releases the GIL and saves errno.
                    let ret = unsafe {
                        majit_rlib::rposix::c_setresuid(
                            r as libc::uid_t,
                            e as libc::uid_t,
                            s as libc::uid_t,
                        )
                    };
                    if ret < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_none())
                },
                3,
            )
        );

        // os.setresgid(rgid, egid, sgid) -> None
        #[cfg(any(target_os = "freebsd", target_os = "linux", target_os = "openbsd"))]
        crate::__pyre_put_new!(
            ns_slot,
            "setresgid",
            crate::make_builtin_function_with_arity(
                "setresgid",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error(
                            "setresgid() requires rgid, egid, sgid",
                        ));
                    }
                    // interp_posix.py `@unwrap_spec(rgid=c_gid_t,
                    // egid=c_gid_t, sgid=c_gid_t)`.
                    let mut w_rgid = args[0];
                    let mut w_egid = args[1];
                    let mut w_sgid = args[2];
                    let r = pyre_object::with_roots!(w_rgid, w_egid, w_sgid => {
                        crate::baseobjspace::c_uid_t_w(w_rgid)
                    })?;
                    let e = pyre_object::with_roots!(w_egid, w_sgid => crate::baseobjspace::c_uid_t_w(w_egid))?;
                    let s = crate::baseobjspace::c_uid_t_w(w_sgid)?;
                    // `rposix.c_setresgid` releases the GIL and saves errno.
                    let ret = unsafe {
                        majit_rlib::rposix::c_setresgid(
                            r as libc::gid_t,
                            e as libc::gid_t,
                            s as libc::gid_t,
                        )
                    };
                    if ret < 0 {
                        return Err(io_err(
                            std::io::Error::from_raw_os_error(majit_rlib::rposix::get_saved_errno()),
                            "",
                        ));
                    }
                    Ok(pyre_object::w_none())
                },
                3,
            )
        );
    }

    // ── the descriptor calls Windows serves through the C runtime (the same
    //    noop placeholders, overridden) ───────────────────────────────────
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    {
        use rustpython_host_env::os::ErrorExt;
        use rustpython_host_env::{crt_fd, nt as host_nt};

        /// A failed C runtime call carries its errno as the error's payload
        /// rather than as a raw OS code, which is what `posix_errno` reads.
        fn crt_errno_of(e: &std::io::Error) -> i32 {
            e.posix_errno()
        }

        /// The descriptor an fd argument names, for the runtime calls that
        /// take one. `-1` is the sentinel for "no descriptor", never a real
        /// one, so it is refused before the call sees it.
        fn borrow_raw_fd(fd: i32) -> Result<crt_fd::Borrowed<'static>, crate::PyError> {
            unsafe { crt_fd::Borrowed::try_borrow_raw(fd) }
                .map_err(|e| errno_err(crt_errno_of(&e), ""))
        }

        /// The `fd: int` spelling of that argument, which is `_PyLong_AsInt`
        /// and nothing more.
        fn borrowed_fd(w_fd: PyObjectRef) -> Result<crt_fd::Borrowed<'static>, crate::PyError> {
            borrow_raw_fd(crate::baseobjspace::c_int_w(w_fd)?)
        }

        fn crt_result(result: std::io::Result<()>) -> Result<PyObjectRef, crate::PyError> {
            match result {
                Ok(()) => Ok(pyre_object::w_none()),
                Err(e) => Err(errno_err(crt_errno_of(&e), "")),
            }
        }

        /// The Win32 error a handle call failed with.  A descriptor that names
        /// no handle never reaches such a call, and `ERROR_INVALID_HANDLE` is
        /// what the caller reports in its place.
        fn handle_err(e: &std::io::Error) -> crate::PyError {
            let winerror = e
                .raw_os_error()
                .unwrap_or(rustpython_host_env::nt::ERROR_INVALID_HANDLE_I32);
            crate::PyError::os_error_win32_syscall2(
                winerror,
                pyre_object::PY_NULL,
                pyre_object::PY_NULL,
            )
        }

        /// The descriptor's handle, or `None` when it names none.
        fn fd_handle(fd: i32) -> Option<host_nt::Handle> {
            let handle = host_nt::handle_from_fd(fd);
            (!host_nt::is_invalid_handle(handle)).then_some(handle)
        }

        // os.dup(fd) -> new_fd.  `_Py_dup` makes the copy non-inheritable, so
        // it does not leak into a child the way the CRT's own copy would.
        crate::__pyre_put_new!(
            ns_slot,
            "dup",
            crate::make_builtin_function_with_arity(
                "dup",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("dup() requires 1 argument"));
                    }
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    // `_Py_dup` leaves the interpreter for the duplication.
                    let duplicated = {
                        let _blocked = crate::module::thread::before_external_block();
                        host_nt::dup(fd)
                    };
                    match duplicated {
                        Ok(n) => Ok(pyre_object::w_int_new(n as i64)),
                        Err(e) => Err(errno_err(crt_errno_of(&e), "")),
                    }
                },
                1,
            )
        );

        // os.dup2(fd, fd2, inheritable=True) -> fd2 — the `Signature`-bearing
        // twin of the unix registration, and defective in the same way while
        // it was registered raw.
        #[crate::pyre_function]
        fn dup2(
            fd: pyre_object::PyObjectRef,
            fd2: pyre_object::PyObjectRef,
            inheritable: Option<pyre_object::PyObjectRef>,
        ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
            let fd = crate::baseobjspace::c_int_w(fd)?;
            let fd2 = crate::baseobjspace::c_int_w(fd2)?;
            let inheritable = match inheritable {
                Some(w) => crate::baseobjspace::is_true(w)?,
                None => true,
            };
            // `_Py_dup2`, like `_Py_dup`, runs outside the interpreter.
            let duplicated = {
                let _blocked = crate::module::thread::before_external_block();
                host_nt::dup2(fd, fd2, inheritable)
            };
            match duplicated {
                Ok(n) => Ok(pyre_object::w_int_new(n as i64)),
                Err(e) => Err(errno_err(crt_errno_of(&e), "")),
            }
        }

        crate::__pyre_put_new!(
            ns_slot,
            "dup2",
            crate::make_builtin_function_with_arity_and_maybe_sig(
                "dup2",
                dup2,
                dup2_pyre_arity(),
                dup2_pyre_sig(),
            )
        );

        // os.fsync(fd) — `_commit`, the runtime's flush-to-disk.
        crate::__pyre_put_new!(
            ns_slot,
            "fsync",
            crate::make_builtin_function_with_arity(
                "fsync",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("fsync() requires 1 argument"));
                    }
                    // `fd: fildes`, which is `PyObject_AsFileDescriptor`:
                    // it takes an object with a `fileno()` and warns that a
                    // bool is being used as a descriptor. It is the only
                    // argument spelled that way in this half of the module.
                    let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                    crt_result(crt_fd::fsync(borrow_raw_fd(fd)?))
                },
                1,
            )
        );

        // os.ftruncate(fd, length) — `_chsize_s`.
        crate::__pyre_put_new!(
            ns_slot,
            "ftruncate",
            crate::make_builtin_function_with_arity(
                "ftruncate",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "ftruncate() requires 2 arguments",
                        ));
                    }
                    let length = truncate_length_w(args[1])?;
                    crt_result(crt_fd::ftruncate(borrowed_fd(args[0])?, length))
                },
                2,
            )
        );

        // os.truncate(path, length) — `_wopen` then the same `_chsize_s`
        // (`os_truncate_impl`).  Its path is `path_t(allow_fd=…)`, so an
        // integer names an open descriptor and the call is `ftruncate` on it,
        // with no name to report the failure with.
        crate::__pyre_put_new!(
            ns_slot,
            "truncate",
            crate::make_builtin_function_with_arity(
                "truncate",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "truncate() requires 2 arguments",
                        ));
                    }
                    let path = crate::gateway::fsencode_path_or_fd_w(args[0], "truncate", true)?;
                    let length = truncate_length_w(args[1])?;
                    if path.is_fd {
                        let bfd = unsafe { crt_fd::Borrowed::borrow_raw(path.as_fd) };
                        return crt_result(crt_fd::ftruncate(bfd, length));
                    }
                    let name = |e: &std::io::Error| {
                        errno_err_with_filename(crt_errno_of(e), path.w_path())
                    };
                    let wide = wide_path(&path.as_bytes)?;
                    let flags = libc::O_WRONLY | libc::O_BINARY | libc::O_NOINHERIT;
                    let fd = crt_fd::wopen(&wide, flags, 0).map_err(|e| name(&e))?;
                    let result = crt_fd::ftruncate(fd.borrow(), length);
                    let closed = crt_fd::close(fd);
                    result.and(closed).map_err(|e| name(&e))?;
                    Ok(pyre_object::w_none())
                },
                2,
            )
        );

        /// `win32_wchdir` -- `SetCurrentDirectoryW`, then the directory read
        /// back and published for the drive it is on.
        fn win32_wchdir(path: &std::path::Path) -> std::io::Result<()> {
            host_os::set_current_dir(path)
        }

        // os.chdir(path)
        crate::__pyre_put_new!(
            ns_slot,
            "chdir",
            crate::make_builtin_function_with_arity(
                "chdir",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("chdir() requires 1 argument"));
                    }
                    // The POSIX `chdir` names `integer` in its allowed types
                    // because it can `fchdir`; there is none here, so the list
                    // this one shows is the path-only one.
                    let path = crate::gateway::fsencode_path_named_w(args[0], "chdir", "path")?;
                    win32_wchdir(path_from_bytes(&path.as_bytes).as_ref())
                        .map_err(|e| fs_err_with_filename(e, path.w_path()))?;
                    Ok(pyre_object::w_none())
                },
                1,
            )
        );

        // os.access(path, mode, *, dir_fd=None, effective_ids=False,
        //           follow_symlinks=True) -> bool
        //
        // Windows has no permission bits to consult beyond the read-only
        // attribute, so `W_OK` is the only mode that can answer False for a
        // name that exists (`os_access_impl`).  None of the three modifiers
        // has a call to reach: `dir_fd` types as `dir_fd(requires='faccessat')`
        // and the other two are the pair `os_access_impl` turns away without
        // `faccessat`, so each is refused rather than answered as though it
        // had been applied.
        crate::__pyre_put_new!(
            ns_slot,
            "access",
            crate::make_builtin_function("access", |args| {
                // The three modifiers are keyword-only, so a third positional
                // is an error rather than a `dir_fd`.
                let (bound, kwargs) = bind_path_args(
                    args,
                    "access",
                    &["path", "mode"],
                    2,
                    &["dir_fd", "effective_ids", "follow_symlinks"],
                )?;
                // The parameters convert in declaration order and each of them
                // can raise, so the order is observable: `path` reports before
                // `mode`, and both before either flag's `__bool__` is called.
                let path = crate::gateway::fsencode_path_named_w(
                    bound[0].expect("path is required"),
                    "access",
                    "path",
                )?;
                // Only `W_OK` is read, so the byte holding it is the whole of
                // the mode as far as the answer goes.
                let mode = crate::baseobjspace::c_int_w(bound[1].expect("mode is required"))? as u8;
                dir_fd_kwarg(kwargs, false)?;
                if let Some(v) = crate::builtins::kwarg_get(kwargs, "effective_ids")
                    && crate::baseobjspace::is_true(v)?
                {
                    return Err(argument_unavailable("access", "effective_ids"));
                }
                if let Some(v) = crate::builtins::kwarg_get(kwargs, "follow_symlinks")
                    && !crate::baseobjspace::is_true(v)?
                {
                    return Err(argument_unavailable("access", "follow_symlinks"));
                }
                // `os_access_impl` leaves the interpreter for
                // `GetFileAttributesW`, which blocks on a network path.
                let allowed = {
                    let _blocked = crate::module::thread::before_external_block();
                    host_nt::access(path_from_bytes(&path.as_bytes).as_ref(), mode)
                };
                Ok(pyre_object::w_bool_from(allowed))
            })
        );

        // os.execv(path, argv) / os.execve(path, argv, env)
        //
        // `_wexecv` / `_wexecve` are the wide forms `os_execv_impl` reaches
        // for; they return only on failure, because on success the calling
        // process is gone by the time they would.
        fn exec_argv_wide(
            w_argv: PyObjectRef,
            function: &str,
        ) -> Result<Vec<widestring::WideCString>, crate::PyError> {
            let items = crate::baseobjspace::unpackiterable(w_argv, -1).map_err(|error| {
                if error.kind == crate::PyErrorKind::TypeError {
                    crate::PyError::type_error(format!(
                        "{function}() arg 2 must be an iterable of strings"
                    ))
                } else {
                    error
                }
            })?;
            if items.is_empty() {
                return Err(crate::PyError::value_error(format!(
                    "{function}() arg 2 must not be empty"
                )));
            }
            let mut argv = Vec::with_capacity(items.len());
            for item in items {
                // An element is converted on the sequence's behalf, not as an
                // argument of the call, so the caller-less message is the one
                // it reports — the same for the environment below.
                let value = extract_path(item)?;
                argv.push(
                    widestring::WideCString::from_os_str(&*os_str_from_bytes(&value)).map_err(
                        |_| {
                            crate::PyError::value_error(format!(
                                "{function}() arg 2 contains an embedded null byte"
                            ))
                        },
                    )?,
                );
            }
            if argv[0].is_empty() {
                return Err(crate::PyError::value_error(format!(
                    "{function}() arg 2 first element cannot be empty"
                )));
            }
            Ok(argv)
        }

        fn exec_pointer_array_wide(values: &[widestring::WideCString]) -> Vec<*const u16> {
            let mut pointers: Vec<_> = values.iter().map(|value| value.as_ptr()).collect();
            pointers.push(std::ptr::null());
            pointers
        }

        crate::__pyre_put_new!(
            ns_slot,
            "execv",
            crate::make_builtin_function_with_arity(
                "execv",
                |args| {
                    // The path names itself; the argv entries do not, because
                    // each of those is converted on the sequence's behalf
                    // rather than as an argument of its own.
                    let mut w_path = args[0];
                    let mut w_argv = args[1];
                    let path = pyre_object::with_roots!(w_path, w_argv =>
                        crate::gateway::fsencode_path_named_w(w_path, "execv", "path"))?;
                    let command_w = wide_path(&path.as_bytes)?;
                    let argv =
                        pyre_object::with_roots!(w_path, w_argv => exec_argv_wide(w_argv, "execv"))?;
                    let argv_ptrs = exec_pointer_array_wide(&argv);
                    rustpython_host_env::os::ensure_drive_current_directory();
                    // The runtime's invalid parameter handler is silenced
                    // around the call: an empty path reaches it, and its
                    // default action ends the process where the call is
                    // supposed to return -1 with `EINVAL` in `errno`.
                    crate::builtins::crt_call!(libc::wexecv(
                        command_w.as_ptr(),
                        argv_ptrs.as_ptr()
                    ));
                    // `os_execv_impl` reports through `path_error`, so the
                    // path this failed on is the error's filename.
                    Err(errno_err_with_filename(
                        crate::builtins::crt_errno(),
                        path.w_path(),
                    ))
                },
                2,
            )
        );

        crate::__pyre_put_new!(
            ns_slot,
            "execve",
            crate::make_builtin_function_with_arity(
                "execve",
                |args| {
                    let mut w_path = args[0];
                    let mut w_argv = args[1];
                    let mut w_env = args[2];
                    let path = pyre_object::with_roots!(w_path, w_argv, w_env =>
                        crate::gateway::fsencode_path_named_w(w_path, "execve", "path"))?;
                    let command_w = wide_path(&path.as_bytes)?;
                    let argv = pyre_object::with_roots!(w_path, w_argv, w_env =>
                        exec_argv_wide(w_argv, "execve"))?;
                    let argv_ptrs = exec_pointer_array_wide(&argv);

                    let env = pyre_object::with_roots!(w_path, w_argv, w_env => {
                        collect_env_entries(w_env, "execve", false)
                    })?
                        .into_iter()
                        .map(|entry| {
                            widestring::WideCString::from_os_str(&*os_str_from_bytes(&entry))
                                .map_err(|_| {
                                    crate::PyError::value_error(
                                        "execve() environment contains an embedded null byte",
                                    )
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let env_ptrs = exec_pointer_array_wide(&env);
                    rustpython_host_env::os::ensure_drive_current_directory();
                    crate::builtins::crt_call!(libc::wexecve(
                        command_w.as_ptr(),
                        argv_ptrs.as_ptr(),
                        env_ptrs.as_ptr()
                    ));
                    Err(errno_err_with_filename(
                        crate::builtins::crt_errno(),
                        path.w_path(),
                    ))
                },
                3,
            )
        );

        // os.spawnv(mode, path, argv) / os.spawnve(mode, path, argv, env)
        //
        // `_wspawnv` / `_wspawnve`. `P_WAIT` answers with the child's exit
        // status and the other modes with the handle the child was started
        // under, which is what `os.waitpid` takes.
        //
        // `os_spawnv_impl` takes a list or a tuple and nothing else -- an
        // iterable of its own is not one -- and reads the shape of `argv`
        // before anything in it, which is why the two halves are separate.
        fn spawn_argv_shape(
            w_argv: PyObjectRef,
            function: &str,
        ) -> Result<Vec<PyObjectRef>, crate::PyError> {
            if unsafe { !pyre_object::is_list(w_argv) && !pyre_object::is_tuple(w_argv) } {
                return Err(crate::PyError::type_error(format!(
                    "{function}() arg 2 must be a tuple or list"
                )));
            }
            let items = crate::baseobjspace::unpackiterable(w_argv, -1)?;
            if items.is_empty() {
                return Err(crate::PyError::value_error(format!(
                    "{function}() arg 2 cannot be empty"
                )));
            }
            Ok(items)
        }

        /// The same elements as wide strings. `fsconvert_strdup` reports for
        /// itself, and only `spawnv` replaces what it says with a message of
        /// its own -- the rejected type and the embedded null alike, since
        /// `os_spawnv_impl` sets its own error over whichever one came back.
        fn spawn_argv_wide(
            items: Vec<PyObjectRef>,
            function: &str,
            name_element_errors: bool,
        ) -> Result<Vec<widestring::WideCString>, crate::PyError> {
            let mut argv = Vec::with_capacity(items.len());
            for (index, item) in items.into_iter().enumerate() {
                let converted = extract_path(item).and_then(|value| {
                    widestring::WideCString::from_os_str(&*os_str_from_bytes(&value))
                        .map_err(|_| crate::PyError::value_error("embedded null character"))
                });
                let wide = converted.map_err(|error| {
                    if name_element_errors {
                        crate::PyError::type_error(format!(
                            "{function}() arg 2 must contain only strings"
                        ))
                    } else {
                        error
                    }
                })?;
                // The first element is judged as soon as it is converted, so a
                // later element that cannot be converted at all does not
                // report ahead of it. `os_spawnve_impl` names `spawnv` here
                // whichever of the two is running.
                if index == 0 && wide.is_empty() {
                    return Err(crate::PyError::value_error(
                        "spawnv() arg 2 first element cannot be empty",
                    ));
                }
                argv.push(wide);
            }
            Ok(argv)
        }

        fn spawn_wide_refs(values: &[widestring::WideCString]) -> Vec<&widestring::WideCStr> {
            values.iter().map(|value| value.as_ucstr()).collect()
        }

        crate::__pyre_put_new!(
            ns_slot,
            "spawnv",
            crate::make_builtin_function_with_arity(
                "spawnv",
                |args| {
                    let mode = crate::baseobjspace::c_int_w(args[0])?;
                    let path = crate::gateway::fsencode_path_named_w(args[1], "spawnv", "path")?;
                    let command_w = wide_path(&path.as_bytes)?;
                    let items = spawn_argv_shape(args[2], "spawnv")?;
                    let argv = spawn_argv_wide(items, "spawnv", true)?;
                    let spawned = {
                        let _blocked = crate::module::thread::before_external_block();
                        host_nt::spawnv(mode, &command_w, &spawn_wide_refs(&argv))
                    };
                    match spawned {
                        Ok(value) => Ok(pyre_object::w_int_new(value as i64)),
                        // `posix_error` -- the runtime's own errno, which is
                        // where `_wspawnv` reports a name it cannot start.
                        Err(error) => Err(errno_err(crt_errno_of(&error), "")),
                    }
                },
                3,
            )
        );

        crate::__pyre_put_new!(
            ns_slot,
            "spawnve",
            crate::make_builtin_function_with_arity(
                "spawnve",
                |args| {
                    let mode = crate::baseobjspace::c_int_w(args[0])?;
                    let path = crate::gateway::fsencode_path_named_w(args[1], "spawnve", "path")?;
                    let command_w = wide_path(&path.as_bytes)?;
                    let items = spawn_argv_shape(args[2], "spawnve")?;
                    // The mapping is judged between the shape of `argv` and
                    // its contents, and by its own wording rather than the
                    // `execve` one `collect_env_entries` would give it.
                    if !crate::baseobjspace::py_mapping_check(args[3]) {
                        return Err(crate::PyError::type_error(
                            "spawnve() arg 3 must be a mapping object",
                        ));
                    }
                    let argv = spawn_argv_wide(items, "spawnve", false)?;
                    let env = collect_env_entries(args[3], "spawnve", false)?
                        .into_iter()
                        .map(|entry| {
                            widestring::WideCString::from_os_str(&*os_str_from_bytes(&entry))
                                .map_err(|_| crate::PyError::value_error("embedded null character"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    rustpython_host_env::os::ensure_drive_current_directory();
                    let spawned = {
                        let _blocked = crate::module::thread::before_external_block();
                        host_nt::spawnve(
                            mode,
                            &command_w,
                            &spawn_wide_refs(&argv),
                            &spawn_wide_refs(&env),
                        )
                    };
                    match spawned {
                        Ok(value) => Ok(pyre_object::w_int_new(value as i64)),
                        Err(error) => Err(errno_err(crt_errno_of(&error), "")),
                    }
                },
                4,
            )
        );

        // os.kill(pid, sig)
        //
        // `os_kill_impl` under `MS_WINDOWS`: the two console control events
        // are delivered to the process group with `GenerateConsoleCtrlEvent`,
        // and any other number is the exit code `TerminateProcess` stamps on
        // the process it ends — there are no signals to send one.
        crate::__pyre_put_new!(
            ns_slot,
            "kill",
            crate::make_builtin_function_with_arity(
                "kill",
                |args| {
                    let pid = crate::baseobjspace::c_int_w(args[0])?;
                    let sig = crate::baseobjspace::c_int_w(args[1])?;
                    let result = {
                        let _blocked = crate::module::thread::before_external_block();
                        host_nt::kill(pid as u32, sig as u32)
                    };
                    result.map_err(|error| fs_err_with_filename(error, pyre_object::PY_NULL))?;
                    Ok(pyre_object::w_none())
                },
                2,
            )
        );

        /// The mode bit Windows keeps: with the owner's write bit the
        /// read-only attribute comes off, without it goes on.
        const S_IWRITE: u32 = 0o200;

        // os.chmod(path, mode, *, dir_fd=None, follow_symlinks=True) -> None
        //
        // `MS_WINDOWS` is what puts this name in `supports_fd` (`os.py:143`)
        // and in `supports_follow_symlinks` (`os.py`), so all three forms
        // are served here: a descriptor through the handle it wraps, a name
        // through the file the link resolves to, and `follow_symlinks=False`
        // through the link's own attributes.  `dir_fd` is the one modifier
        // Windows cannot honour — `chmod` types it as
        // `dir_fd(requires='fchmodat')`, which is `_DirFD_Unavailable`.
        crate::__pyre_put_new!(
            ns_slot,
            "chmod",
            crate::make_builtin_function("chmod", |args| {
                let (args, kwargs) = crate::builtins::split_builtin_kwargs(args);
                crate::builtins::kwarg_reject_unknown(
                    kwargs,
                    &["dir_fd", "follow_symlinks"],
                    "chmod",
                )?;
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("chmod() requires 2 arguments"));
                }
                // Both modifiers are keyword-only.
                if args.len() > 2 {
                    return Err(crate::PyError::type_error(format!(
                        "chmod() takes exactly 2 positional arguments ({} given)",
                        args.len()
                    )));
                }
                // `_DirFD_Unavailable.unwrap` (`interp_posix.py`)
                // converts first and reports the platform second, so a wrongly
                // typed value is a TypeError here as well.
                if let Some(w) = crate::builtins::kwarg_get(kwargs, "dir_fd")
                    .filter(|&w| !unsafe { pyre_object::is_none(w) })
                {
                    unwrap_fd(w, "integer or None")?;
                    return Err(dir_fd_unavailable());
                }
                let path = crate::gateway::fsencode_path_or_fd_w(args[0], "chmod", MS_WINDOWS)?;
                // `posix.chmod` unwraps `mode` as `c_int`, so a non-integer
                // raises TypeError instead of reinterpreting its layout.
                let mode = crate::baseobjspace::c_int_w(args[1])? as u32;
                // The descriptor form has no name to resolve, so neither
                // modifier applies to it and it dispatches straight to
                // `os.fchmod` (`interp_posix.py`).
                if path.is_fd {
                    let changed = {
                        let _blocked = crate::module::thread::before_external_block();
                        host_nt::fchmod(path.as_fd, mode, S_IWRITE)
                    };
                    return match changed {
                        Ok(()) => Ok(pyre_object::w_none()),
                        Err(e) => Err(handle_err(&e)),
                    };
                }
                let follow_symlinks = match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
                    Some(w) => crate::baseobjspace::is_true(w)?,
                    // `CHMOD_DEFAULT_FOLLOW_SYMLINKS` is 0 here, which the
                    // clinic spells `follow_symlinks=(os.name != 'nt')`: an
                    // unqualified `chmod` writes the attribute on the name it
                    // was given, link or not.
                    None => false,
                };
                let wide = wide_path(&path.as_bytes)?;
                // `os_chmod_impl` holds one `Py_BEGIN_ALLOW_THREADS` region
                // around the whole attribute read-modify-write, whichever of
                // the three forms the path selected.
                let result = {
                    let _blocked = crate::module::thread::before_external_block();
                    if follow_symlinks {
                        host_nt::chmod_follow(&wide, mode, S_IWRITE)
                    } else {
                        // `SetFileAttributesW` on the name itself, which is
                        // what "modify the link rather than its target" means
                        // where the mode is one attribute bit.
                        host_nt::win32_lchmod(&wide, mode, S_IWRITE)
                    }
                };
                result.map_err(|e| fs_err_with_filename(e, path.w_path()))?;
                Ok(pyre_object::w_none())
            })
        );

        // `os.lchmod` is the named `follow_symlinks=False` operation.  Windows
        // implements that distinction with the reparse point's own file
        // attributes (`rustpython_host_env::nt::win32_lchmod`), so publishing
        // the function is truthful here rather than the unsupported POSIX
        // `lchmod(2)` stub some Unix hosts carry.
        crate::__pyre_put_new!(
            ns_slot,
            "lchmod",
            crate::make_builtin_function_with_arity(
                "lchmod",
                |args| {
                    let path = crate::gateway::fsencode_path_or_fd_w(args[0], "lchmod", false)?;
                    let mode = crate::baseobjspace::c_int_w(args[1])? as u32;
                    let wide = wide_path(&path.as_bytes)?;
                    // `os_lchmod_impl` runs the same call outside the
                    // interpreter that `os_chmod_impl` does.
                    let changed = {
                        let _blocked = crate::module::thread::before_external_block();
                        host_nt::win32_lchmod(&wide, mode, S_IWRITE)
                    };
                    changed.map_err(|error| fs_err_with_filename(error, path.w_path()))?;
                    Ok(pyre_object::w_none())
                },
                2,
            )
        );

        // os.fchmod(fd, mode) -> None
        crate::__pyre_put_new!(
            ns_slot,
            "fchmod",
            crate::make_builtin_function_with_arity(
                "fchmod",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("fchmod() requires 2 arguments"));
                    }
                    let mut w_fd = args[0];
                    let mut w_mode = args[1];
                    let fd = pyre_object::with_roots!(w_fd, w_mode => crate::baseobjspace::c_int_w(w_fd))?;
                    let mode = crate::baseobjspace::c_int_w(w_mode)? as u32;
                    // Every failure here is the handle call's, reported the
                    // Win32 way (`os_fchmod_impl`), which also leaves the
                    // interpreter for it.
                    let changed = {
                        let _blocked = crate::module::thread::before_external_block();
                        host_nt::fchmod(fd, mode, S_IWRITE)
                    };
                    match changed {
                        Ok(()) => Ok(pyre_object::w_none()),
                        Err(e) => Err(handle_err(&e)),
                    }
                },
                2,
            )
        );

        // os.link(src, dst) -> None.  `CreateHardLinkW` names the new link
        // first and the file it points at second.
        crate::__pyre_put_new!(
            ns_slot,
            "link",
            crate::make_builtin_function("link", |args| {
                let (args, kwargs) = crate::builtins::split_builtin_kwargs(args);
                crate::builtins::kwarg_reject_unknown(
                    kwargs,
                    &["src_dir_fd", "dst_dir_fd", "follow_symlinks"],
                    "link",
                )?;
                // No `linkat` here, so a descriptor to resolve either name
                // against is refused rather than ignored — and the message
                // names both arguments, the way `argument_unavailable_error`
                // spells this one.
                if ["src_dir_fd", "dst_dir_fd"].iter().any(|name| {
                    crate::builtins::kwarg_get(kwargs, name)
                        .is_some_and(|w| !unsafe { pyre_object::is_none(w) })
                }) {
                    return Err(crate::PyError::not_implemented(
                        "link: src_dir_fd and dst_dir_fd unavailable on this platform",
                    ));
                }
                // `CreateHardLinkW` links the symlink `src` names rather than
                // what it points at, so asking for the other behaviour is
                // refused.  Leaving the argument out is not asking: the
                // default is the unspecified one.
                if let Some(w) = crate::builtins::kwarg_get(kwargs, "follow_symlinks")
                    && crate::baseobjspace::is_true(w)?
                {
                    return Err(crate::PyError::not_implemented(
                        "link: follow_symlinks=True unavailable on this platform",
                    ));
                }
                link_positional(args)?;
                let src = crate::gateway::fsencode_path_named_w(args[0], "link", "src")?;
                let dst = crate::gateway::fsencode_path_named_w(args[1], "link", "dst")?;
                let (wide_src, wide_dst) = (wide_path(&src.as_bytes)?, wide_path(&dst.as_bytes)?);
                rustpython_host_env::winapi::create_hard_link(&wide_dst, &wide_src)
                    .map_err(|error| fs_err_with_filename2(error, 0, src.w_path(), dst.w_path()))?;
                Ok(pyre_object::w_none())
            })
        );

        // os.symlink(src, dst, target_is_directory=False) -> None.
        // `CreateSymbolicLinkW` names the link first and its target second,
        // and a link to a directory is a different kind of reparse point from
        // a link to a file. `os_symlink_impl` picks the kind from the explicit
        // argument or `host_nt::symlink`'s bounded existing-target probe.
        crate::__pyre_put_new!(
            ns_slot,
            "symlink",
            crate::make_builtin_function("symlink", |args| {
                let (bound, kwargs) = bind_path_args(
                    args,
                    "symlink",
                    &["src", "dst", "target_is_directory"],
                    2,
                    &["dir_fd"],
                )?;
                // `symlink` types `dir_fd` as `DirFD(rposix.HAVE_SYMLINKAT)`,
                // and `CreateSymbolicLinkW` resolves a relative name against
                // the working directory alone.
                dir_fd_kwarg(kwargs, false)?;
                let src = crate::gateway::fsencode_path_named_w(
                    bound[0].expect("src is required"),
                    "symlink",
                    "src",
                )?;
                let dst = crate::gateway::fsencode_path_named_w(
                    bound[1].expect("dst is required"),
                    "symlink",
                    "dst",
                )?;
                let target_is_directory = match bound[2] {
                    Some(w) => crate::baseobjspace::is_true(w)?,
                    None => false,
                };
                let (wide_src, wide_dst) = (wide_path(&src.as_bytes)?, wide_path(&dst.as_bytes)?);
                use std::os::windows::ffi::OsStringExt;
                let src_path = std::ffi::OsString::from_wide(wide_src.as_slice());
                let dst_path = std::ffi::OsString::from_wide(wide_dst.as_slice());
                host_nt::symlink(
                    src_path.as_ref(),
                    dst_path.as_ref(),
                    &wide_src,
                    &wide_dst,
                    target_is_directory,
                )
                .map_err(|e| fs_err_with_filename2(e, 0, src.w_path(), dst.w_path()))?;
                Ok(pyre_object::w_none())
            })
        );

        // os.umask(mask) -> previous mask
        crate::__pyre_put_new!(
            ns_slot,
            "umask",
            crate::make_builtin_function_with_arity(
                "umask",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("umask() requires 1 argument"));
                    }
                    let mask = crate::baseobjspace::c_int_w(args[0])?;
                    match host_nt::umask(mask) {
                        Ok(previous) => Ok(pyre_object::w_int_new(previous as i64)),
                        Err(e) => Err(errno_err(crt_errno_of(&e), "")),
                    }
                },
                1,
            )
        );

        // os.pipe() -> (read_fd, write_fd), both non-inheritable.
        crate::__pyre_put_new!(
            ns_slot,
            "pipe",
            crate::make_builtin_function_with_arity(
                "pipe",
                |_| match host_nt::pipe() {
                    Ok((read_fd, write_fd)) => {
                        let mut fields = pyre_object::gc_roots::RootedItems::new();
                        fields.push(pyre_object::w_int_new(read_fd as i64));
                        fields.push(pyre_object::w_int_new(write_fd as i64));
                        Ok(pyre_object::w_tuple_new(fields.take()))
                    }
                    Err(e) => Err(errno_err(crt_errno_of(&e), "")),
                },
                0,
            )
        );

        // os.getppid() — the parent recorded in the process's own entry, which
        // Windows only offers through a snapshot of the process list.
        crate::__pyre_put_new!(
            ns_slot,
            "getppid",
            crate::make_builtin_function_with_arity(
                "getppid",
                |_| Ok(pyre_object::w_int_new(host_nt::getppid() as i64)),
                0,
            )
        );

        // os._exit(code) — immediate process exit, no cleanup.  `install_noop_stubs`
        // binds the name for every host so os.py finds it, and only the POSIX
        // branch replaced it until now: on Windows `os._exit` returned `None` and
        // the process ran on, which is the one thing the call is documented not to
        // do.  `os__exit_impl` spells it `_exit(status)`, the C runtime's
        // no-cleanup exit rather than `exit`, so a child sharing an inherited
        // stdio buffer with its parent does not flush it a second time.
        crate::__pyre_put_new!(
            ns_slot,
            "_exit",
            crate::make_builtin_function_with_arity(
                "_exit",
                |args| {
                    let code = match args.first() {
                        // interp_posix.py `@unwrap_spec(status=c_int)`.
                        Some(&o) => crate::baseobjspace::c_int_w(o)?,
                        None => {
                            return Err(crate::PyError::type_error("_exit() requires 1 argument"));
                        }
                    };
                    unsafe { libc::_exit(code) }
                },
                1,
            )
        );

        // os.abort() — `os_abort_impl` calls `abort()`, whose contract is that it
        // never returns.  Windows kept the noop placeholder here for the same
        // reason `_exit` did.
        crate::__pyre_put_new!(
            ns_slot,
            "abort",
            crate::make_builtin_function_with_arity("abort", |_| unsafe { libc::abort() }, 0)
        );

        // os.getlogin() -> str
        crate::__pyre_put_new!(
            ns_slot,
            "getlogin",
            crate::make_builtin_function_with_arity(
                "getlogin",
                |_| match host_nt::getlogin() {
                    Ok(name) => Ok(pyre_object::w_str_new_managed(&name)),
                    Err(e) => Err(fs_err_with_filename2(
                        e,
                        0,
                        pyre_object::PY_NULL,
                        pyre_object::PY_NULL,
                    )),
                },
                0,
            )
        );

        // os.startfile(path, operation=None, arguments=None, cwd=None,
        // show_cmd=None) -> None.  `ShellExecuteW` hands the file to whatever
        // program is registered for it, and reports failure by returning 32 or
        // less rather than through a flag.
        crate::__pyre_put_new!(
            ns_slot,
            "startfile",
            crate::make_builtin_function("startfile", |args| {
                // Every optional argument is positional-or-keyword, so the
                // four of them are looked up either way round.
                //
                // `filepath` and `cwd` convert through the caller-less form:
                // this entry point exists only here, so the wording it should
                // name itself with is unmeasured on this host. See the
                // follow-up task.
                let (args, kwargs) = crate::builtins::split_builtin_kwargs(args);
                crate::builtins::kwarg_reject_unknown(
                    kwargs,
                    &["operation", "arguments", "cwd", "show_cmd"],
                    "startfile",
                )?;
                if args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "startfile() missing required argument 'filepath' (pos 1)",
                    ));
                }
                let path = crate::gateway::fsencode_path_w(args[0])?;
                let wide_file = wide_path(&path.as_bytes)?;
                let given = |index: usize, name: &str| -> Option<pyre_object::PyObjectRef> {
                    args.get(index)
                        .copied()
                        .or_else(|| crate::builtins::kwarg_get(kwargs, name))
                        .filter(|&w| !unsafe { pyre_object::is_none(w) })
                };
                // A missing optional argument is spelled `None`, which stands
                // for the null the call takes for "no operation", "no
                // arguments", "the process's own directory".
                let wide_arg = |index: usize, name: &str| -> Result<Option<_>, crate::PyError> {
                    match given(index, name) {
                        Some(w) => {
                            let text = crate::baseobjspace::text_w(w)?;
                            Ok(Some(widestring::WideCString::from_str(text).map_err(
                                |_| crate::PyError::value_error("embedded null character"),
                            )?))
                        }
                        None => Ok(None),
                    }
                };
                let operation = wide_arg(1, "operation")?;
                let arguments = wide_arg(2, "arguments")?;
                let cwd = match given(3, "cwd") {
                    Some(w) => Some(wide_path(&crate::gateway::fsencode_path_w(w)?.as_bytes)?),
                    None => None,
                };
                let show_cmd = match given(4, "show_cmd") {
                    Some(w) => crate::baseobjspace::c_int_w(w)?,
                    None => rustpython_host_env::winapi::SW_SHOWNORMAL,
                };
                rustpython_host_env::winapi::shell_execute_w(
                    &wide_file,
                    operation.as_deref(),
                    arguments.as_deref(),
                    cwd.as_deref(),
                    show_cmd,
                )
                .map_err(|error| fs_err_with_filename(error, path.w_path()))?;
                Ok(pyre_object::w_none())
            })
        );

        // os.cpu_count() -> int | None
        //
        // `rposix.py` reads `GetSystemInfo().dwNumberOfProcessors`
        // here, which counts the processors in the caller's processor group;
        // `available_parallelism` answers the process affinity mask instead, so
        // the two part company on a host that has restricted one. Left as it is
        // because no Windows oracle is reachable from this host to measure
        // which the surface should report — see the follow-up task.
        crate::__pyre_put_new!(
            ns_slot,
            "cpu_count",
            crate::make_builtin_function_with_arity(
                "cpu_count",
                |_| match std::thread::available_parallelism() {
                    Ok(n) => Ok(pyre_object::w_int_new(n.get() as i64)),
                    Err(_) => Ok(pyre_object::w_none()),
                },
                0,
            )
        );

        // os.system(command) -> the command interpreter's exit status.  The
        // wide entry point is the one that can spell every command; the narrow
        // one re-encodes it through the ANSI code page.  Neither reports a
        // failure other than through the status, which `os_system_impl`
        // returns as it is.
        //
        // The POSIX `system` converts its command as a filesystem name and so
        // reports the caller-less message — measured. This one declares text
        // rather than a path, so the message it should report is a different
        // shape entirely and is unmeasured here; it keeps the same conversion
        // meanwhile. See the follow-up task.
        crate::__pyre_put_new!(
            ns_slot,
            "system",
            crate::make_builtin_function_with_arity(
                "system",
                |args| {
                    unsafe extern "C" {
                        fn _wsystem(command: *const u16) -> libc::c_int;
                    }
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("system() requires 1 argument"));
                    }
                    let command = crate::gateway::fsencode_path_w(args[0])?;
                    let wide = wide_path(&command.as_bytes)?;
                    let status = crate::builtins::crt_call!(_wsystem(wide.as_ptr()));
                    Ok(pyre_object::w_int_new(status as i64))
                },
                1,
            )
        );

        // os.waitpid(pid, options) -> (pid, status).  `_cwait` waits for one
        // process by handle; the status it reports is the exit code, which
        // `os_waitpid_impl` shifts into the byte a POSIX wait status keeps it
        // in.
        crate::__pyre_put_new!(
            ns_slot,
            "waitpid",
            crate::make_builtin_function_with_arity(
                "waitpid",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("waitpid() requires 2 arguments"));
                    }
                    let pid = crate::baseobjspace::int_w(args[0])? as isize;
                    let options = crate::baseobjspace::c_int_w(args[1])?;
                    // `os_waitpid_impl` waits outside the interpreter, which is
                    // what lets another thread run while a child is still up.
                    let waited = {
                        let _blocked = crate::module::thread::before_external_block();
                        host_nt::cwait(pid, options)
                    };
                    match waited {
                        // `unsigned long long ustatus = (unsigned int)status`
                        // -- an exit code above `INT_MAX`, which
                        // `ExitProcess` takes, is a positive number here and
                        // not a negative `int`.
                        Ok((pid, status)) => {
                            let mut fields = pyre_object::gc_roots::RootedItems::new();
                            fields.push(pyre_object::w_int_new(pid as i64));
                            fields.push(pyre_object::w_int_new((status as u32 as i64) << 8));
                            Ok(pyre_object::w_tuple_new(fields.take()))
                        }
                        Err(e) => Err(errno_err(crt_errno_of(&e), "")),
                    }
                },
                2,
            )
        );

        // os.waitstatus_to_exitcode(status) -> int
        //
        // `os_waitstatus_to_exitcode_impl` under `MS_WINDOWS`: `os.waitpid`
        // shifted the child's exit code up by a byte to spell it the way a
        // POSIX wait status does, and this shifts it back down.

        /// `PyLong_AsUnsignedLongLong` -- an `int` and nothing else, so an
        /// object carrying `__index__` is turned away rather than converted.
        fn unsigned_long_long_w(value: PyObjectRef) -> Result<u64, crate::PyError> {
            if unsafe { pyre_object::pyobject::is_int(value) } {
                let signed = unsafe { pyre_object::intobject::w_int_get_value(value) };
                return u64::try_from(signed).map_err(|_| {
                    crate::PyError::overflow_error("can't convert negative int to unsigned")
                });
            }
            if unsafe { pyre_object::pyobject::is_long(value) } {
                let big = unsafe { pyre_object::w_long_get_value(value) };
                if big.get_sign() < 0 {
                    return Err(crate::PyError::overflow_error(
                        "can't convert negative int to unsigned",
                    ));
                }
                if pyre_object::longobject::jit_bigint_to_u64_fits(big) == 0 {
                    return Err(crate::PyError::overflow_error("int too big to convert"));
                }
                return Ok(pyre_object::longobject::jit_bigint_to_u64_value(big));
            }
            Err(crate::PyError::type_error("an integer is required"))
        }

        crate::__pyre_put_new!(
            ns_slot,
            "waitstatus_to_exitcode",
            crate::make_builtin_function_with_arity(
                "waitstatus_to_exitcode",
                |args| {
                    let exitcode = unsigned_long_long_w(args[0])? >> 8;
                    // `ExitProcess` takes a `UINT`, so a wider value names no
                    // exit code a child could have left.
                    if exitcode > u64::from(u32::MAX) {
                        return Err(crate::PyError::value_error(format!(
                            "invalid exit code: {exitcode}"
                        )));
                    }
                    Ok(pyre_object::w_int_new(exitcode as i64))
                },
                1,
            )
        );

        // os.times() -> posix.times_result.  Windows keeps the process's own
        // user and kernel time and nothing else, so the three fields that
        // count a child's are zero (`os_times_impl`).
        crate::__pyre_put_new!(
            ns_slot,
            "times",
            crate::make_builtin_function_with_arity(
                "times",
                |_| {
                    let times =
                        rustpython_host_env::time::get_process_times_100ns().ok_or_else(|| {
                            fs_err_with_filename(
                                std::io::Error::last_os_error(),
                                pyre_object::PY_NULL,
                            )
                        })?;
                    // `GetProcessTimes` counts in hundreds of nanoseconds.
                    let seconds = |ticks: u64| pyre_object::w_float_new(ticks as f64 * 1e-7);
                    Ok(crate::_structseq::new_instance(
                        super::times_result_seq_type(),
                        vec![
                            seconds(times.user),
                            seconds(times.system),
                            pyre_object::w_float_new(0.0),
                            pyre_object::w_float_new(0.0),
                            pyre_object::w_float_new(0.0),
                        ],
                    ))
                },
                0,
            )
        );

        // os.listdrives() / os.listvolumes() / os.listmounts(volume) — the
        // names Windows mounts its filesystems under.
        fn name_list(
            names: std::io::Result<Vec<std::ffi::OsString>>,
        ) -> Result<PyObjectRef, crate::PyError> {
            let names = names.map_err(|e| fs_err_with_filename(e, pyre_object::PY_NULL))?;
            // Each name is freshly allocated and the next one allocates again,
            // so they are pinned as they arrive.
            let mut items = pyre_object::gc_roots::RootedItems::new();
            for name in &names {
                items.push(fs_name_obj(false, name.as_encoded_bytes()));
            }
            Ok(pyre_object::w_list_new(items.take()))
        }
        crate::__pyre_put_new!(
            ns_slot,
            "listdrives",
            crate::make_builtin_function_with_arity(
                "listdrives",
                // `os_listdrives_impl` leaves the interpreter for
                // `GetLogicalDriveStringsW`, as the two volume calls below do
                // for theirs.
                |_| {
                    name_list({
                        let _blocked = crate::module::thread::before_external_block();
                        host_nt::listdrives()
                    })
                },
                0,
            )
        );
        crate::__pyre_put_new!(
            ns_slot,
            "listvolumes",
            crate::make_builtin_function_with_arity(
                "listvolumes",
                |_| {
                    name_list({
                        let _blocked = crate::module::thread::before_external_block();
                        host_nt::listvolumes()
                    })
                },
                0,
            )
        );
        // `volume` converts through the caller-less form: 3.14 added this entry
        // point on Windows alone, so what it names itself with is unmeasured on
        // this host. See the follow-up task.
        crate::__pyre_put_new!(
            ns_slot,
            "listmounts",
            crate::make_builtin_function_with_arity(
                "listmounts",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "listmounts() requires 1 argument",
                        ));
                    }
                    let volume = crate::gateway::fsencode_path_w(args[0])?;
                    let wide = wide_path(&volume.as_bytes)?;
                    name_list({
                        let _blocked = crate::module::thread::before_external_block();
                        host_nt::listmounts(&wide)
                    })
                },
                1,
            )
        );

        // os.device_encoding(fd) -> str | None.  `_Py_device_encoding`: only a
        // terminal has one, and a process with no console attached has no code
        // page to name it with.
        crate::__pyre_put_new!(
            ns_slot,
            "device_encoding",
            crate::make_builtin_function_with_arity(
                "device_encoding",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "device_encoding() requires 1 argument",
                        ));
                    }
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    if !host_os::isatty(fd) {
                        return Ok(pyre_object::w_none());
                    }
                    match host_os::device_encoding(fd) {
                        // `GetConsoleCP` answers 0 for a process with no console.
                        Some(name) if name != "cp0" => Ok(pyre_object::w_str_new_managed(&name)),
                        _ => Ok(pyre_object::w_none()),
                    }
                },
                1,
            )
        );

        // os.get_inheritable(fd) / os.set_inheritable(fd, inheritable) — the
        // flag lives on the descriptor's handle (`HANDLE_FLAG_INHERIT`).
        crate::__pyre_put_new!(
            ns_slot,
            "get_inheritable",
            crate::make_builtin_function_with_arity(
                "get_inheritable",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "get_inheritable() requires 1 argument",
                        ));
                    }
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let handle = fd_handle(fd).ok_or_else(|| errno_err(libc::EBADF, ""))?;
                    match host_nt::get_handle_inheritable(handle as _) {
                        Ok(inheritable) => Ok(pyre_object::w_bool_from(inheritable)),
                        Err(e) => Err(handle_err(&e)),
                    }
                },
                1,
            )
        );
        crate::__pyre_put_new!(
            ns_slot,
            "set_inheritable",
            crate::make_builtin_function_with_arity(
                "set_inheritable",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "set_inheritable() requires 2 arguments",
                        ));
                    }
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let inherit = crate::baseobjspace::is_true(args[1])?;
                    let handle = fd_handle(fd).ok_or_else(|| errno_err(libc::EBADF, ""))?;
                    match host_nt::set_handle_inheritable(handle as _, inherit) {
                        Ok(()) => Ok(pyre_object::w_none()),
                        Err(e) => Err(handle_err(&e)),
                    }
                },
                2,
            )
        );
    }

    // The trampoline only mediates the curated ll_os/ll_time surface, so the
    // real impls registered above for process control, fd duplication, host
    // filesystem mutation and privilege changes would otherwise reach libc
    // directly under sandbox.  Overwrite each with a raising stub, mirroring the
    // RPython sandbox where unsupported externals are simply unavailable.  The
    // mediated names (open/read/write/close/lseek/stat/access/getcwd/listdir/
    // getenv/isatty/strerror/get{u,g}id/unlink/mkdir) are intentionally absent
    // here — they stay live through host_seam.
    #[cfg(feature = "sandbox")]
    {
        fn sandbox_unavailable(_: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            Err(crate::host_seam::stub("this OS operation"))
        }
        for name in [
            // process creation / control
            "fork",
            "forkpty",
            "system",
            "execv",
            "execve",
            "execvp",
            "execvpe",
            // Neither the spawn family nor `popen` is an external here: os.py
            // writes both in Python, over fork+exec+waitpid (:881) and over
            // `subprocess` (:1020), and the stubs above already refuse what
            // they reach for. Binding those names would only stop os.py from
            // defining them — and with the spawn family, P_WAIT and P_NOWAIT.
            "posix_spawn",
            "posix_spawnp",
            "abort",
            "_exit",
            "register_at_fork",
            "wait",
            "waitpid",
            "kill",
            "killpg",
            // file-descriptor duplication / pipes / ttys / cross-fd copy +
            // inheritance control (set_inheritable would mutate a real fd).
            "dup",
            "dup2",
            "pipe",
            "openpty",
            "login_tty",
            "sendfile",
            "set_inheritable",
            // host filesystem mutation that bypasses the controller
            "chmod",
            "fchmod",
            "lchmod",
            "chown",
            "fchown",
            "lchown",
            "chroot",
            "chdir",
            "fchdir",
            "link",
            "symlink",
            "truncate",
            "ftruncate",
            "rename",
            "replace",
            "rmdir",
            "mkfifo",
            "mknod",
            // privilege / scheduling
            "setuid",
            "seteuid",
            "setgid",
            "setegid",
            "setreuid",
            "setregid",
            "setresuid",
            "setresgid",
            "setgroups",
            "initgroups",
            "setsid",
            "setpgid",
            "setpgrp",
            "nice",
            "setpriority",
            "sched_get_priority_max",
            "sched_get_priority_min",
            // durability + real process environment mutation
            "sync",
            "fsync",
            "fdatasync",
            "setenv",
            "unsetenv",
            "putenv",
            // host filesystem inspection that bypasses the controller VFS.
            // DirEntry is a type, but its is_dir/is_file/stat/inode methods stat
            // a guest-controlled `path` via host_fs, so neutralise it too (its
            // only producer, scandir, is already stubbed here).
            "readlink",
            "scandir",
            "DirEntry",
            "statvfs",
            "fstatvfs",
            // host process / environment information leaks
            "getpid",
            "getppid",
            "uname",
            "getlogin",
            "getloadavg",
            "getpriority",
            "times",
            "umask",
            "getgroups",
            "getgrouplist",
            "cpu_count",
            "_cpu_count",
            "getresuid",
            "getresgid",
            "getpgrp",
            "getpgid",
            // host system-configuration probes; pathconf consults a
            // guest-controlled path on the real filesystem, and confstr
            // answers with the host's own search path among other strings.
            "pathconf",
            "fpathconf",
            "sysconf",
            "confstr",
            // a lock on a descriptor the controller owns
            "lockf",
            // reports on a child of this process, which the sandbox has none of
            "waitid",
            // terminal / tty inspection + control
            "tcgetpgrp",
            "tcsetpgrp",
            "get_terminal_size",
            "ttyname",
            "ctermid",
        ] {
            crate::__pyre_put_new!(
                ns_slot,
                name,
                crate::make_builtin_function(name, sandbox_unavailable)
            );
        }

        // The same, for the names only some hosts have. Listing them above
        // would not neutralise anything on a host that never registered them —
        // `module_ns_store` writes rather than overwrites, so it would publish
        // a `posix.pipe2` where there is no `pipe2` to refuse.
        #[cfg(any(
            target_os = "android",
            target_os = "dragonfly",
            target_os = "freebsd",
            target_os = "linux",
            target_os = "netbsd",
            target_os = "openbsd"
        ))]
        crate::__pyre_put_new!(
            ns_slot,
            "pipe2",
            crate::make_builtin_function("pipe2", sandbox_unavailable)
        );
        // The policy calls reach the host scheduler; only the setters mutate,
        // but a policy read is a host-process leak in the same way `getpriority`
        // above is. `sched_param` is left alone — it carries no host access.
        #[cfg(any(
            target_os = "android",
            target_os = "freebsd",
            target_os = "linux",
            target_os = "netbsd"
        ))]
        for name in [
            "sched_getscheduler",
            "sched_getparam",
            "sched_rr_get_interval",
        ] {
            crate::__pyre_put_new!(
                ns_slot,
                name,
                crate::make_builtin_function(name, sandbox_unavailable)
            );
        }
        // The affinity mask is the same kind of host-process leak, and carries
        // the narrower gate the pair is published under.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        for name in [
            "sched_getaffinity",
            "sched_setaffinity",
            "memfd_create",
            "posix_fallocate",
            "posix_fadvise",
            "getxattr",
            "setxattr",
            "removexattr",
            "listxattr",
        ] {
            crate::__pyre_put_new!(
                ns_slot,
                name,
                crate::make_builtin_function(name, sandbox_unavailable)
            );
        }
        #[cfg(all(
            not(target_env = "musl"),
            any(
                target_os = "android",
                target_os = "freebsd",
                target_os = "linux",
                target_os = "netbsd"
            )
        ))]
        for name in ["sched_setscheduler", "sched_setparam"] {
            crate::__pyre_put_new!(
                ns_slot,
                name,
                crate::make_builtin_function(name, sandbox_unavailable)
            );
        }
    }

    crate::__pyre_put_new!(ns_slot, "error", crate::typedef::w_object());
    Ok(())
}

#[cfg(test)]
mod split_root_tests {
    use super::split_root;
    use rustpython_wtf8::{CodePoint, Wtf8, Wtf8Buf};

    /// Expectations taken from `ntpath.splitroot`, whose three-way split
    /// joins drive and root into the root this returns.
    #[test]
    fn matches_ntpath_splitroot() {
        let cases: &[(&str, &str, &str)] = &[
            ("", "", ""),
            ("Windows", "", "Windows"),
            ("a/b", "", "a/b"),
            ("\\", "\\", ""),
            ("/", "/", ""),
            ("\\Windows", "\\", "Windows"),
            ("/Windows", "/", "Windows"),
            ("C:", "C:", ""),
            ("C:a", "C:", "a"),
            ("C:\\", "C:\\", ""),
            ("C:/Windows", "C:/", "Windows"),
            (":a", "", ":a"),
            ("\\\\server", "\\\\server", ""),
            ("\\\\server\\share", "\\\\server\\share", ""),
            ("\\\\server\\share\\", "\\\\server\\share\\", ""),
            ("\\\\server\\share\\dir", "\\\\server\\share\\", "dir"),
            ("//server/share/dir", "//server/share/", "dir"),
            ("\\\\.\\device\\x", "\\\\.\\device\\", "x"),
            ("\\\\?\\C:\\x", "\\\\?\\C:\\", "x"),
            (
                "\\\\?\\UNC\\server\\share\\dir",
                "\\\\?\\UNC\\server\\share\\",
                "dir",
            ),
            ("//?/unc/server/share/dir", "//?/unc/server/share/", "dir"),
            // No separator after the share name, so the whole path is root.
            ("\\\\\\a", "\\\\\\a", ""),
            // Offsets address characters: the drive branch is taken here.
            ("\u{e4}:\\x", "\u{e4}:\\", "x"),
            (
                "\\\\s\u{e4}rver\\share\\dir",
                "\\\\s\u{e4}rver\\share\\",
                "dir",
            ),
        ];
        for &(path, root, tail) in cases {
            let (got_root, got_tail) = split_root(Wtf8::new(path));
            assert_eq!(
                (got_root.as_str(), got_tail.as_str()),
                (Ok(root), Ok(tail)),
                "split_root({path:?})"
            );
            assert_eq!(
                format!("{root}{tail}"),
                path,
                "split_root({path:?}) lost characters"
            );
        }
    }

    /// A path carrying a lone surrogate — what `fsdecode` produces for an
    /// undecodable name — splits on the same boundary and keeps the code point.
    #[test]
    fn keeps_a_lone_surrogate() {
        let mut path = Wtf8Buf::from_string("C:\\".to_string());
        path.push(CodePoint::from_u32(0xdcff).unwrap());
        path.push_str("x");
        let (root, tail) = split_root(&path);
        assert_eq!(root.as_str(), Ok("C:\\"));
        assert_eq!(tail.code_points().next().map(|c| c.to_u32()), Some(0xdcff));
        assert_eq!(tail.len(), path.len() - root.len());
    }
}

#[cfg(test)]
mod normpath_tests {
    use super::{NT_SEPS, POSIX_SEPS, PathSeps, normpath_and_size, skiproot, wide_with_nul};
    use rustpython_wtf8::{CodePoint, Wtf8, Wtf8Buf};

    /// The tables spell the lone surrogate `fsdecode` produces for an
    /// undecodable name as U+0001, which a Rust string literal can hold and no
    /// path carries.
    fn w(text: &str) -> Wtf8Buf {
        let mut out = Wtf8Buf::new();
        for ch in text.chars() {
            match ch {
                '\u{1}' => out.push(CodePoint::from_u32(0xdfff).unwrap()),
                _ => out.push_char(ch),
            }
        }
        out
    }

    fn split_ex(path: &Wtf8, seps: PathSeps) -> (Wtf8Buf, Wtf8Buf, Wtf8Buf) {
        let wide: Vec<u16> = path.encode_wide().collect();
        let (drvsize, rootsize) = skiproot(&wide, seps);
        (
            Wtf8Buf::from_wide(&wide[..drvsize]),
            Wtf8Buf::from_wide(&wide[drvsize..drvsize + rootsize]),
            Wtf8Buf::from_wide(&wide[drvsize + rootsize..]),
        )
    }

    fn normpath(path: &Wtf8, seps: PathSeps) -> Wtf8Buf {
        let mut buf = wide_with_nul(path);
        let norm_len = normpath_and_size(&mut buf, seps);
        if norm_len == 0 {
            return Wtf8Buf::from_string(".".to_string());
        }
        Wtf8Buf::from_wide(&buf[..norm_len])
    }

    fn check_split(cases: &[(&str, &str, &str, &str)], seps: PathSeps) {
        for &(path, drive, root, tail) in cases {
            assert_eq!(
                split_ex(&w(path), seps),
                (w(drive), w(root), w(tail)),
                "splitroot({path:?})"
            );
        }
    }

    fn check_normpath(cases: &[(&str, &str)], seps: PathSeps) {
        for &(path, expected) in cases {
            assert_eq!(normpath(&w(path), seps), w(expected), "normpath({path:?})");
        }
    }

    /// Expectations taken from `ntpath.splitroot`, which is `_path_splitroot_ex`
    /// itself on a Windows host.
    #[test]
    fn skiproot_matches_ntpath_splitroot() {
        let cases: &[(&str, &str, &str, &str)] = &[
            ("", "", "", ""),
            (".", "", "", "."),
            ("..", "", "", ".."),
            ("...", "", "", "..."),
            ("/", "", "/", ""),
            ("//", "//", "", ""),
            ("///", "///", "", ""),
            ("////", "///", "/", ""),
            ("\\", "", "\\", ""),
            ("\\\\", "\\\\", "", ""),
            ("\\\\\\", "\\\\\\", "", ""),
            ("foo", "", "", "foo"),
            ("foo/bar", "", "", "foo/bar"),
            ("foo\\bar", "", "", "foo\\bar"),
            ("foo//bar", "", "", "foo//bar"),
            ("foo\\\\bar", "", "", "foo\\\\bar"),
            ("./foo", "", "", "./foo"),
            (".\\foo", "", "", ".\\foo"),
            ("././foo", "", "", "././foo"),
            ("./", "", "", "./"),
            (".\\", "", "", ".\\"),
            ("./.", "", "", "./."),
            ("foo/.", "", "", "foo/."),
            ("foo\\.", "", "", "foo\\."),
            ("foo/..", "", "", "foo/.."),
            ("foo\\..", "", "", "foo\\.."),
            ("foo/../bar", "", "", "foo/../bar"),
            ("foo\\..\\bar", "", "", "foo\\..\\bar"),
            ("../foo", "", "", "../foo"),
            ("..\\foo", "", "", "..\\foo"),
            ("../../foo", "", "", "../../foo"),
            ("..\\..\\foo", "", "", "..\\..\\foo"),
            ("/..", "", "/", ".."),
            ("/../foo", "", "/", "../foo"),
            ("//../foo", "//../foo", "", ""),
            ("///../foo", "///..", "/", "foo"),
            ("\\..\\foo", "", "\\", "..\\foo"),
            ("/foo/../..", "", "/", "foo/../.."),
            ("/foo/../../bar", "", "/", "foo/../../bar"),
            ("foo/../..", "", "", "foo/../.."),
            ("foo/../../bar", "", "", "foo/../../bar"),
            ("foo/bar/../..", "", "", "foo/bar/../.."),
            ("foo/bar/../../..", "", "", "foo/bar/../../.."),
            ("C:", "C:", "", ""),
            ("C:.", "C:", "", "."),
            ("C:..", "C:", "", ".."),
            ("C:foo", "C:", "", "foo"),
            ("C:\\", "C:", "\\", ""),
            ("C:\\.", "C:", "\\", "."),
            ("C:\\..", "C:", "\\", ".."),
            ("C:\\foo", "C:", "\\", "foo"),
            ("C:/foo/../bar", "C:", "/", "foo/../bar"),
            ("C:\\foo\\..\\..\\bar", "C:", "\\", "foo\\..\\..\\bar"),
            ("C:foo\\..\\..\\bar", "C:", "", "foo\\..\\..\\bar"),
            ("c:\\a\\b\\..\\..\\..\\c", "c:", "\\", "a\\b\\..\\..\\..\\c"),
            ("\\\\server\\share", "\\\\server\\share", "", ""),
            ("\\\\server\\share\\", "\\\\server\\share", "\\", ""),
            ("\\\\server\\share\\dir", "\\\\server\\share", "\\", "dir"),
            (
                "\\\\server\\share\\..\\dir",
                "\\\\server\\share",
                "\\",
                "..\\dir",
            ),
            (
                "\\\\server\\share\\dir\\..\\..",
                "\\\\server\\share",
                "\\",
                "dir\\..\\..",
            ),
            (
                "//server/share/dir/../..",
                "//server/share",
                "/",
                "dir/../..",
            ),
            ("//server/share/../..", "//server/share", "/", "../.."),
            ("\\\\?\\C:\\foo\\..\\bar", "\\\\?\\C:", "\\", "foo\\..\\bar"),
            (
                "\\\\?\\UNC\\server\\share\\dir\\..",
                "\\\\?\\UNC\\server\\share",
                "\\",
                "dir\\..",
            ),
            (
                "//?/unc/server/share/dir/..",
                "//?/unc/server/share",
                "/",
                "dir/..",
            ),
            ("\\\\.\\device\\x\\..", "\\\\.\\device", "\\", "x\\.."),
            ("\\\\.\\device", "\\\\.\\device", "", ""),
            ("\\\\", "\\\\", "", ""),
            ("\\\\a", "\\\\a", "", ""),
            ("\\\\a\\", "\\\\a\\", "", ""),
            ("\\\\a\\b", "\\\\a\\b", "", ""),
            ("\\\\a\\b\\", "\\\\a\\b", "\\", ""),
            ("\\\\a\\b\\c", "\\\\a\\b", "\\", "c"),
            (":a", "", "", ":a"),
            ("a:b:c", "a:", "", "b:c"),
            ("\u{e4}:\\x", "\u{e4}:", "\\", "x"),
            ("\u{e4}:\\x\\..", "\u{e4}:", "\\", "x\\.."),
            ("fo\u{0}o", "", "", "fo\u{0}o"),
            ("fo\u{0}o\\..\\bar", "", "", "fo\u{0}o\\..\\bar"),
            ("fo\u{0}o/../bar", "", "", "fo\u{0}o/../bar"),
            ("\u{1}", "", "", "\u{1}"),
            ("\u{1}\\..\\foo", "", "", "\u{1}\\..\\foo"),
            ("\u{1}/../foo", "", "", "\u{1}/../foo"),
            ("\u{1f600}:\\x", "", "", "\u{1f600}:\\x"),
            ("\u{1f600}/../foo", "", "", "\u{1f600}/../foo"),
            ("foo/./bar", "", "", "foo/./bar"),
            ("foo/././bar", "", "", "foo/././bar"),
            ("foo/.bar", "", "", "foo/.bar"),
            ("foo/..bar", "", "", "foo/..bar"),
            ("foo/...", "", "", "foo/..."),
            ("foo/.../bar", "", "", "foo/.../bar"),
            ("/./", "", "/", "./"),
            ("/.", "", "/", "."),
            ("//.", "//.", "", ""),
            ("/././.", "", "/", "././."),
            ("a/b/c/../../../../d", "", "", "a/b/c/../../../../d"),
            ("\\a\\b\\..\\..\\..\\c", "", "\\", "a\\b\\..\\..\\..\\c"),
            ("C:\\\\\\foo", "C:", "\\", "\\\\foo"),
            ("C:////foo", "C:", "/", "///foo"),
            ("//foo", "//foo", "", ""),
            ("///foo", "///foo", "", ""),
            ("//foo/bar", "//foo/bar", "", ""),
            ("foo/bar/", "", "", "foo/bar/"),
            ("foo/bar//", "", "", "foo/bar//"),
            ("foo\\bar\\\\", "", "", "foo\\bar\\\\"),
            ("C:\\foo\\", "C:", "\\", "foo\\"),
        ];
        check_split(cases, NT_SEPS);
    }

    /// Expectations taken from `posixpath.splitroot`.
    #[test]
    fn skiproot_matches_posixpath_splitroot() {
        let cases: &[(&str, &str, &str, &str)] = &[
            ("", "", "", ""),
            (".", "", "", "."),
            ("..", "", "", ".."),
            ("...", "", "", "..."),
            ("/", "", "/", ""),
            ("//", "", "//", ""),
            ("///", "", "/", "//"),
            ("////", "", "/", "///"),
            ("\\", "", "", "\\"),
            ("\\\\", "", "", "\\\\"),
            ("\\\\\\", "", "", "\\\\\\"),
            ("foo", "", "", "foo"),
            ("foo/bar", "", "", "foo/bar"),
            ("foo\\bar", "", "", "foo\\bar"),
            ("foo//bar", "", "", "foo//bar"),
            ("foo\\\\bar", "", "", "foo\\\\bar"),
            ("./foo", "", "", "./foo"),
            (".\\foo", "", "", ".\\foo"),
            ("././foo", "", "", "././foo"),
            ("./", "", "", "./"),
            (".\\", "", "", ".\\"),
            ("./.", "", "", "./."),
            ("foo/.", "", "", "foo/."),
            ("foo\\.", "", "", "foo\\."),
            ("foo/..", "", "", "foo/.."),
            ("foo\\..", "", "", "foo\\.."),
            ("foo/../bar", "", "", "foo/../bar"),
            ("foo\\..\\bar", "", "", "foo\\..\\bar"),
            ("../foo", "", "", "../foo"),
            ("..\\foo", "", "", "..\\foo"),
            ("../../foo", "", "", "../../foo"),
            ("..\\..\\foo", "", "", "..\\..\\foo"),
            ("/..", "", "/", ".."),
            ("/../foo", "", "/", "../foo"),
            ("//../foo", "", "//", "../foo"),
            ("///../foo", "", "/", "//../foo"),
            ("\\..\\foo", "", "", "\\..\\foo"),
            ("/foo/../..", "", "/", "foo/../.."),
            ("/foo/../../bar", "", "/", "foo/../../bar"),
            ("foo/../..", "", "", "foo/../.."),
            ("foo/../../bar", "", "", "foo/../../bar"),
            ("foo/bar/../..", "", "", "foo/bar/../.."),
            ("foo/bar/../../..", "", "", "foo/bar/../../.."),
            ("C:", "", "", "C:"),
            ("C:.", "", "", "C:."),
            ("C:..", "", "", "C:.."),
            ("C:foo", "", "", "C:foo"),
            ("C:\\", "", "", "C:\\"),
            ("C:\\.", "", "", "C:\\."),
            ("C:\\..", "", "", "C:\\.."),
            ("C:\\foo", "", "", "C:\\foo"),
            ("C:/foo/../bar", "", "", "C:/foo/../bar"),
            ("C:\\foo\\..\\..\\bar", "", "", "C:\\foo\\..\\..\\bar"),
            ("C:foo\\..\\..\\bar", "", "", "C:foo\\..\\..\\bar"),
            ("c:\\a\\b\\..\\..\\..\\c", "", "", "c:\\a\\b\\..\\..\\..\\c"),
            ("\\\\server\\share", "", "", "\\\\server\\share"),
            ("\\\\server\\share\\", "", "", "\\\\server\\share\\"),
            ("\\\\server\\share\\dir", "", "", "\\\\server\\share\\dir"),
            (
                "\\\\server\\share\\..\\dir",
                "",
                "",
                "\\\\server\\share\\..\\dir",
            ),
            (
                "\\\\server\\share\\dir\\..\\..",
                "",
                "",
                "\\\\server\\share\\dir\\..\\..",
            ),
            (
                "//server/share/dir/../..",
                "",
                "//",
                "server/share/dir/../..",
            ),
            ("//server/share/../..", "", "//", "server/share/../.."),
            ("\\\\?\\C:\\foo\\..\\bar", "", "", "\\\\?\\C:\\foo\\..\\bar"),
            (
                "\\\\?\\UNC\\server\\share\\dir\\..",
                "",
                "",
                "\\\\?\\UNC\\server\\share\\dir\\..",
            ),
            (
                "//?/unc/server/share/dir/..",
                "",
                "//",
                "?/unc/server/share/dir/..",
            ),
            ("\\\\.\\device\\x\\..", "", "", "\\\\.\\device\\x\\.."),
            ("\\\\.\\device", "", "", "\\\\.\\device"),
            ("\\\\", "", "", "\\\\"),
            ("\\\\a", "", "", "\\\\a"),
            ("\\\\a\\", "", "", "\\\\a\\"),
            ("\\\\a\\b", "", "", "\\\\a\\b"),
            ("\\\\a\\b\\", "", "", "\\\\a\\b\\"),
            ("\\\\a\\b\\c", "", "", "\\\\a\\b\\c"),
            (":a", "", "", ":a"),
            ("a:b:c", "", "", "a:b:c"),
            ("\u{e4}:\\x", "", "", "\u{e4}:\\x"),
            ("\u{e4}:\\x\\..", "", "", "\u{e4}:\\x\\.."),
            ("fo\u{0}o", "", "", "fo\u{0}o"),
            ("fo\u{0}o\\..\\bar", "", "", "fo\u{0}o\\..\\bar"),
            ("fo\u{0}o/../bar", "", "", "fo\u{0}o/../bar"),
            ("\u{1}", "", "", "\u{1}"),
            ("\u{1}\\..\\foo", "", "", "\u{1}\\..\\foo"),
            ("\u{1}/../foo", "", "", "\u{1}/../foo"),
            ("\u{1f600}:\\x", "", "", "\u{1f600}:\\x"),
            ("\u{1f600}/../foo", "", "", "\u{1f600}/../foo"),
            ("foo/./bar", "", "", "foo/./bar"),
            ("foo/././bar", "", "", "foo/././bar"),
            ("foo/.bar", "", "", "foo/.bar"),
            ("foo/..bar", "", "", "foo/..bar"),
            ("foo/...", "", "", "foo/..."),
            ("foo/.../bar", "", "", "foo/.../bar"),
            ("/./", "", "/", "./"),
            ("/.", "", "/", "."),
            ("//.", "", "//", "."),
            ("/././.", "", "/", "././."),
            ("a/b/c/../../../../d", "", "", "a/b/c/../../../../d"),
            ("\\a\\b\\..\\..\\..\\c", "", "", "\\a\\b\\..\\..\\..\\c"),
            ("C:\\\\\\foo", "", "", "C:\\\\\\foo"),
            ("C:////foo", "", "", "C:////foo"),
            ("//foo", "", "//", "foo"),
            ("///foo", "", "/", "//foo"),
            ("//foo/bar", "", "//", "foo/bar"),
            ("foo/bar/", "", "", "foo/bar/"),
            ("foo/bar//", "", "", "foo/bar//"),
            ("foo\\bar\\\\", "", "", "foo\\bar\\\\"),
            ("C:\\foo\\", "", "", "C:\\foo\\"),
        ];
        check_split(cases, POSIX_SEPS);
    }

    /// Expectations taken from `ntpath.normpath`, which is `_path_normpath`
    /// itself on a Windows host.
    #[test]
    fn normpath_matches_ntpath() {
        let cases: &[(&str, &str)] = &[
            ("", "."),
            (".", "."),
            ("..", ".."),
            ("...", "..."),
            ("/", "\\"),
            ("//", "\\\\"),
            ("///", "\\\\\\"),
            ("////", "\\\\\\\\"),
            ("\\", "\\"),
            ("\\\\", "\\\\"),
            ("\\\\\\", "\\\\\\"),
            ("foo", "foo"),
            ("foo/bar", "foo\\bar"),
            ("foo\\bar", "foo\\bar"),
            ("foo//bar", "foo\\bar"),
            ("foo\\\\bar", "foo\\bar"),
            ("./foo", "foo"),
            (".\\foo", "foo"),
            ("././foo", "foo"),
            ("./", "."),
            (".\\", "."),
            ("./.", "."),
            ("foo/.", "foo"),
            ("foo\\.", "foo"),
            ("foo/..", "."),
            ("foo\\..", "."),
            ("foo/../bar", "bar"),
            ("foo\\..\\bar", "bar"),
            ("../foo", "..\\foo"),
            ("..\\foo", "..\\foo"),
            ("../../foo", "..\\..\\foo"),
            ("..\\..\\foo", "..\\..\\foo"),
            ("/..", "\\"),
            ("/../foo", "\\foo"),
            ("//../foo", "\\\\..\\foo"),
            ("///../foo", "\\\\\\..\\foo"),
            ("\\..\\foo", "\\foo"),
            ("/foo/../..", "\\"),
            ("/foo/../../bar", "\\bar"),
            ("foo/../..", ".."),
            ("foo/../../bar", "..\\bar"),
            ("foo/bar/../..", "."),
            ("foo/bar/../../..", ".."),
            ("C:", "C:"),
            ("C:.", "C:"),
            ("C:..", "C:.."),
            ("C:foo", "C:foo"),
            ("C:\\", "C:\\"),
            ("C:\\.", "C:\\"),
            ("C:\\..", "C:\\"),
            ("C:\\foo", "C:\\foo"),
            ("C:/foo/../bar", "C:\\bar"),
            ("C:\\foo\\..\\..\\bar", "C:\\bar"),
            ("C:foo\\..\\..\\bar", "C:..\\bar"),
            ("c:\\a\\b\\..\\..\\..\\c", "c:\\c"),
            ("\\\\server\\share", "\\\\server\\share"),
            ("\\\\server\\share\\", "\\\\server\\share\\"),
            ("\\\\server\\share\\dir", "\\\\server\\share\\dir"),
            ("\\\\server\\share\\..\\dir", "\\\\server\\share\\dir"),
            ("\\\\server\\share\\dir\\..\\..", "\\\\server\\share\\"),
            ("//server/share/dir/../..", "\\\\server\\share\\"),
            ("//server/share/../..", "\\\\server\\share\\"),
            ("\\\\?\\C:\\foo\\..\\bar", "\\\\?\\C:\\bar"),
            (
                "\\\\?\\UNC\\server\\share\\dir\\..",
                "\\\\?\\UNC\\server\\share\\",
            ),
            ("//?/unc/server/share/dir/..", "\\\\?\\unc\\server\\share\\"),
            ("\\\\.\\device\\x\\..", "\\\\.\\device\\"),
            ("\\\\.\\device", "\\\\.\\device"),
            ("\\\\", "\\\\"),
            ("\\\\a", "\\\\a"),
            ("\\\\a\\", "\\\\a\\"),
            ("\\\\a\\b", "\\\\a\\b"),
            ("\\\\a\\b\\", "\\\\a\\b\\"),
            ("\\\\a\\b\\c", "\\\\a\\b\\c"),
            (":a", ":a"),
            ("a:b:c", "a:b:c"),
            ("\u{e4}:\\x", "\u{e4}:\\x"),
            ("\u{e4}:\\x\\..", "\u{e4}:\\"),
            ("fo\u{0}o", "fo\u{0}o"),
            ("fo\u{0}o\\..\\bar", "bar"),
            ("fo\u{0}o/../bar", "bar"),
            ("\u{1}", "\u{1}"),
            ("\u{1}\\..\\foo", "foo"),
            ("\u{1}/../foo", "foo"),
            ("\u{1f600}:\\x", "\u{1f600}:\\x"),
            ("\u{1f600}/../foo", "foo"),
            ("foo/./bar", "foo\\bar"),
            ("foo/././bar", "foo\\bar"),
            ("foo/.bar", "foo\\.bar"),
            ("foo/..bar", "foo\\..bar"),
            ("foo/...", "foo\\..."),
            ("foo/.../bar", "foo\\...\\bar"),
            ("/./", "\\"),
            ("/.", "\\"),
            ("//.", "\\\\."),
            ("/././.", "\\"),
            ("a/b/c/../../../../d", "..\\d"),
            ("\\a\\b\\..\\..\\..\\c", "\\c"),
            ("C:\\\\\\foo", "C:\\foo"),
            ("C:////foo", "C:\\foo"),
            ("//foo", "\\\\foo"),
            ("///foo", "\\\\\\foo"),
            ("//foo/bar", "\\\\foo\\bar"),
            ("foo/bar/", "foo\\bar"),
            ("foo/bar//", "foo\\bar"),
            ("foo\\bar\\\\", "foo\\bar"),
            ("C:\\foo\\", "C:\\foo"),
        ];
        check_normpath(cases, NT_SEPS);
    }

    /// Expectations taken from `posixpath.normpath`.
    #[test]
    fn normpath_matches_posixpath() {
        let cases: &[(&str, &str)] = &[
            ("", "."),
            (".", "."),
            ("..", ".."),
            ("...", "..."),
            ("/", "/"),
            ("//", "//"),
            ("///", "/"),
            ("////", "/"),
            ("\\", "\\"),
            ("\\\\", "\\\\"),
            ("\\\\\\", "\\\\\\"),
            ("foo", "foo"),
            ("foo/bar", "foo/bar"),
            ("foo\\bar", "foo\\bar"),
            ("foo//bar", "foo/bar"),
            ("foo\\\\bar", "foo\\\\bar"),
            ("./foo", "foo"),
            (".\\foo", ".\\foo"),
            ("././foo", "foo"),
            ("./", "."),
            (".\\", ".\\"),
            ("./.", "."),
            ("foo/.", "foo"),
            ("foo\\.", "foo\\."),
            ("foo/..", "."),
            ("foo\\..", "foo\\.."),
            ("foo/../bar", "bar"),
            ("foo\\..\\bar", "foo\\..\\bar"),
            ("../foo", "../foo"),
            ("..\\foo", "..\\foo"),
            ("../../foo", "../../foo"),
            ("..\\..\\foo", "..\\..\\foo"),
            ("/..", "/"),
            ("/../foo", "/foo"),
            ("//../foo", "//foo"),
            ("///../foo", "/foo"),
            ("\\..\\foo", "\\..\\foo"),
            ("/foo/../..", "/"),
            ("/foo/../../bar", "/bar"),
            ("foo/../..", ".."),
            ("foo/../../bar", "../bar"),
            ("foo/bar/../..", "."),
            ("foo/bar/../../..", ".."),
            ("C:", "C:"),
            ("C:.", "C:."),
            ("C:..", "C:.."),
            ("C:foo", "C:foo"),
            ("C:\\", "C:\\"),
            ("C:\\.", "C:\\."),
            ("C:\\..", "C:\\.."),
            ("C:\\foo", "C:\\foo"),
            ("C:/foo/../bar", "C:/bar"),
            ("C:\\foo\\..\\..\\bar", "C:\\foo\\..\\..\\bar"),
            ("C:foo\\..\\..\\bar", "C:foo\\..\\..\\bar"),
            ("c:\\a\\b\\..\\..\\..\\c", "c:\\a\\b\\..\\..\\..\\c"),
            ("\\\\server\\share", "\\\\server\\share"),
            ("\\\\server\\share\\", "\\\\server\\share\\"),
            ("\\\\server\\share\\dir", "\\\\server\\share\\dir"),
            ("\\\\server\\share\\..\\dir", "\\\\server\\share\\..\\dir"),
            (
                "\\\\server\\share\\dir\\..\\..",
                "\\\\server\\share\\dir\\..\\..",
            ),
            ("//server/share/dir/../..", "//server"),
            ("//server/share/../..", "//"),
            ("\\\\?\\C:\\foo\\..\\bar", "\\\\?\\C:\\foo\\..\\bar"),
            (
                "\\\\?\\UNC\\server\\share\\dir\\..",
                "\\\\?\\UNC\\server\\share\\dir\\..",
            ),
            ("//?/unc/server/share/dir/..", "//?/unc/server/share"),
            ("\\\\.\\device\\x\\..", "\\\\.\\device\\x\\.."),
            ("\\\\.\\device", "\\\\.\\device"),
            ("\\\\", "\\\\"),
            ("\\\\a", "\\\\a"),
            ("\\\\a\\", "\\\\a\\"),
            ("\\\\a\\b", "\\\\a\\b"),
            ("\\\\a\\b\\", "\\\\a\\b\\"),
            ("\\\\a\\b\\c", "\\\\a\\b\\c"),
            (":a", ":a"),
            ("a:b:c", "a:b:c"),
            ("\u{e4}:\\x", "\u{e4}:\\x"),
            ("\u{e4}:\\x\\..", "\u{e4}:\\x\\.."),
            ("fo\u{0}o", "fo\u{0}o"),
            ("fo\u{0}o\\..\\bar", "fo\u{0}o\\..\\bar"),
            ("fo\u{0}o/../bar", "bar"),
            ("\u{1}", "\u{1}"),
            ("\u{1}\\..\\foo", "\u{1}\\..\\foo"),
            ("\u{1}/../foo", "foo"),
            ("\u{1f600}:\\x", "\u{1f600}:\\x"),
            ("\u{1f600}/../foo", "foo"),
            ("foo/./bar", "foo/bar"),
            ("foo/././bar", "foo/bar"),
            ("foo/.bar", "foo/.bar"),
            ("foo/..bar", "foo/..bar"),
            ("foo/...", "foo/..."),
            ("foo/.../bar", "foo/.../bar"),
            ("/./", "/"),
            ("/.", "/"),
            ("//.", "//"),
            ("/././.", "/"),
            ("a/b/c/../../../../d", "../d"),
            ("\\a\\b\\..\\..\\..\\c", "\\a\\b\\..\\..\\..\\c"),
            ("C:\\\\\\foo", "C:\\\\\\foo"),
            ("C:////foo", "C:/foo"),
            ("//foo", "//foo"),
            ("///foo", "/foo"),
            ("//foo/bar", "//foo/bar"),
            ("foo/bar/", "foo/bar"),
            ("foo/bar//", "foo/bar"),
            ("foo\\bar\\\\", "foo\\bar\\\\"),
            ("C:\\foo\\", "C:\\foo\\"),
        ];
        check_normpath(cases, POSIX_SEPS);
    }
}

#[cfg(test)]
mod xattr_filename_tests {
    /// `OSErrorTests.test_oserror_filename`: `os.getxattr(<missing path>,
    /// "user.test")` must report the path as `OSError.filename` (`assertIs`).
    /// Wrapping each `FsEncodedPath` in `with_roots!` closed the path's
    /// bracket before the attribute conversion reused that slot, so the
    /// filename became the attribute. The builtins pass the original path
    /// argument to `wrap_oserror2`.
    #[test]
    fn getxattr_oserror_filename_is_the_path() {
        crate::typedef::init_typeobjects();
        let mut w_path = pyre_object::w_str_new("/missing/getxattr-path");
        let mut w_attribute = pyre_object::w_str_new("user.test");
        let orig_path = w_path;
        let hold = pyre_object::gc_roots::push_roots();
        let hold_base = hold.pin_roots(&[w_path, w_attribute]);
        let (_roots, path, _attribute) =
            super::fsencode_path_then_attribute(&mut w_path, &mut w_attribute, "getxattr")
                .expect("str path and attribute convert");
        let wrapped = hold.get(hold_base);
        assert!(
            std::ptr::eq(wrapped, orig_path),
            "wrap_oserror2 must keep the original path argument"
        );
        let filename = crate::baseobjspace::str_utf8_w(path.w_path()).expect("filename is a str");
        assert_eq!(filename, "/missing/getxattr-path");
        let filename = crate::baseobjspace::str_utf8_w(wrapped).expect("held path is a str");
        assert_eq!(filename, "/missing/getxattr-path");
    }

    /// `OSErrorTests.test_oserror_filename`: `os.setxattr(<missing path>,
    /// "user.test", b'user')` must report the path as `OSError.filename`.
    #[test]
    fn setxattr_oserror_filename_is_the_path() {
        crate::typedef::init_typeobjects();
        let mut w_path = pyre_object::w_str_new("/missing/getxattr-path");
        let mut w_attribute = pyre_object::w_str_new("user.test");
        let (_roots, path, _attribute) =
            super::fsencode_path_then_attribute(&mut w_path, &mut w_attribute, "setxattr")
                .expect("str path and attribute convert");
        let filename = crate::baseobjspace::str_utf8_w(path.w_path()).expect("filename is a str");
        assert_eq!(filename, "/missing/getxattr-path");
    }
}
