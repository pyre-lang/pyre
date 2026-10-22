use indexmap::IndexMap;
use majit_rlib::rmarshal::StatResult;
use std::cell::Cell;
use std::fs;
use std::io::{self, Cursor, Read, Seek};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

// vfs.py mirrors stat-module bitmasks that are POSIX-standard regardless of
// host OS, so the constants are hard-coded here rather than pulled from
// libc (which omits these symbols on Windows targets).
pub type Mode = u32;
pub type FsNode = Rc<dyn FSObject>;
pub type VfsResult<T> = Result<T, VfsError>;

/// A readable + seekable handle — the Rust shape of vfs.py's file-like objects
/// (`cStringIO.StringIO` for in-memory `File`, an `open(...)` handle for
/// `RealFile`). The controller needs `Seek` to service `ll_os_lseek`.
pub trait ReadSeek: Read + Seek {}
impl<T: Read + Seek> ReadSeek for T {}

// vfs.py:4
pub const UID: u32 = 1000;
// vfs.py:5
pub const GID: u32 = 1000;
// vfs.py:6
pub const ATIME: i64 = 0;
// vfs.py:6
pub const MTIME: i64 = 0;
// vfs.py:6
pub const CTIME: i64 = 0;
// vfs.py:7
static INO_COUNTER: AtomicU64 = AtomicU64::new(0);

// Error sentinels for VfsError.errno.  `libc` omits these symbols on
// wasm32-unknown-unknown, so the standard POSIX numeric values are
// hard-coded for that target.
#[cfg(not(target_arch = "wasm32"))]
use libc::{EACCES, EIO, ENOENT, ENOTDIR};
#[cfg(target_arch = "wasm32")]
const ENOTDIR: i32 = 20;
#[cfg(target_arch = "wasm32")]
const EACCES: i32 = 13;
#[cfg(target_arch = "wasm32")]
const ENOENT: i32 = 2;
#[cfg(target_arch = "wasm32")]
const EIO: i32 = 5;

bitflags::bitflags! {
    /// POSIX `st_mode` bits, hard-coded so Windows/wasm hosts still have them.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct ModeBits: Mode {
        const IXOTH = 0o000001;
        const IWOTH = 0o000002;
        const IROTH = 0o000004;
        const IRWXO = 0o000007;
        const IXGRP = 0o000010;
        const IRGRP = 0o000040;
        const IRWXG = 0o000070;
        const IXUSR = 0o000100;
        const IWUSR = 0o000200;
        const IRUSR = 0o000400;
        const IRWXU = 0o000700;
        const IFDIR = 0o040000;
        const IFREG = 0o100000;
        const IFMT = 0o170000;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VfsError {
    pub errno: i32,
    pub object: String,
}

#[derive(Debug, Default)]
pub struct FSObjectState {
    st_ino: Cell<Option<u64>>,
}

// vfs.py FSObject
pub trait FSObject {
    // Rust support for vfs.py:15
    fn state(&self) -> &FSObjectState;

    // Rust support for vfs.py:23
    fn kind(&self) -> Mode;

    // vfs.py:11
    fn read_only(&self) -> bool {
        true
    }

    // vfs.py stat
    fn stat(&self) -> VfsResult<StatResult> {
        let st_ino = match self.state().st_ino.get() {
            Some(st_ino) => st_ino,
            None => {
                let st_ino = INO_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
                self.state().st_ino.set(Some(st_ino));
                st_ino
            }
        };
        let st_dev = 1;
        let st_nlink = 1;
        let st_size = self.getsize()?;
        let mut st_mode = self.kind();
        st_mode |= ModeBits::IWUSR.bits()
            | ModeBits::IRUSR.bits()
            | ModeBits::IRGRP.bits()
            | ModeBits::IROTH.bits();
        if is_dir(self.kind()) {
            st_mode |= ModeBits::IXUSR.bits() | ModeBits::IXGRP.bits() | ModeBits::IXOTH.bits();
        }
        let (st_uid, st_gid) = if self.read_only() { (0, 0) } else { (UID, GID) };
        Ok(StatResult {
            st_mode,
            st_ino,
            st_dev,
            st_nlink,
            st_uid,
            st_gid,
            st_size,
            st_atime: ATIME,
            st_mtime: MTIME,
            st_ctime: CTIME,
        })
    }

    // vfs.py access
    fn access(&self, mode: Mode) -> VfsResult<bool> {
        let s = self.stat()?;
        let mut e_mode = s.st_mode & ModeBits::IRWXO.bits();
        if UID == s.st_uid {
            e_mode |= (s.st_mode & ModeBits::IRWXU.bits()) >> 6;
        }
        if GID == s.st_gid {
            e_mode |= (s.st_mode & ModeBits::IRWXG.bits()) >> 3;
        }
        Ok((e_mode & mode) == mode)
    }

    // vfs.py keys
    fn keys(&self) -> VfsResult<Vec<String>> {
        Err(VfsError {
            errno: ENOTDIR,
            object: "self".to_owned(),
        })
    }

    // vfs.py only defines `join` on `Dir`/`RealDir`; a non-directory raising on
    // `join` is the Rust equivalent of the `AttributeError`/`ENOTDIR` a file
    // would produce when `translate_path` walks into it.
    fn join(&self, name: &str) -> VfsResult<FsNode> {
        Err(VfsError {
            errno: ENOTDIR,
            object: name.to_owned(),
        })
    }

    // vfs.py open
    fn open(&self) -> VfsResult<Box<dyn ReadSeek>> {
        Err(VfsError {
            errno: EACCES,
            object: "self".to_owned(),
        })
    }

    // vfs.py getsize
    fn getsize(&self) -> VfsResult<u64> {
        Ok(0)
    }
}

// vfs.py Dir
#[derive(Default)]
pub struct Dir {
    state: FSObjectState,
    pub entries: IndexMap<String, FsNode>,
}

impl Dir {
    // vfs.py __init__
    pub fn new(entries: IndexMap<String, FsNode>) -> Self {
        Self {
            state: FSObjectState::default(),
            entries,
        }
    }
}

impl FSObject for Dir {
    // Rust support for vfs.py:15
    fn state(&self) -> &FSObjectState {
        &self.state
    }

    // vfs.py:60
    fn kind(&self) -> Mode {
        ModeBits::IFDIR.bits()
    }

    // vfs.py keys
    fn keys(&self) -> VfsResult<Vec<String>> {
        Ok(self.entries.keys().cloned().collect())
    }

    // vfs.py join
    fn join(&self, name: &str) -> VfsResult<FsNode> {
        self.entries.get(name).cloned().ok_or_else(|| VfsError {
            errno: ENOENT,
            object: name.to_owned(),
        })
    }
}

// vfs.py RealDir
pub struct RealDir {
    state: FSObjectState,
    pub path: PathBuf,
    pub show_dotfiles: bool,
    pub follow_links: bool,
    pub exclude: Vec<String>,
}

impl RealDir {
    // vfs.py __init__
    pub fn new(
        path: impl Into<PathBuf>,
        show_dotfiles: bool,
        follow_links: bool,
        exclude: Vec<String>,
    ) -> Self {
        Self {
            state: FSObjectState::default(),
            path: path.into(),
            show_dotfiles,
            follow_links,
            exclude: exclude
                .into_iter()
                .map(|excl| excl.to_lowercase())
                .collect(),
        }
    }

    // vfs.py __repr__
    pub fn repr(&self) -> String {
        format!("<RealDir {}>", self.path.display())
    }
}

impl FSObject for RealDir {
    // Rust support for vfs.py:15
    fn state(&self) -> &FSObjectState {
        &self.state
    }

    // vfs.py:60
    fn kind(&self) -> Mode {
        ModeBits::IFDIR.bits()
    }

    // vfs.py keys
    fn keys(&self) -> VfsResult<Vec<String>> {
        let mut names = Vec::new();
        for entry in fs::read_dir(&self.path)
            .map_err(|err| io_error(err, self.path.display().to_string()))?
        {
            let entry = entry.map_err(|err| io_error(err, self.path.display().to_string()))?;
            // The VFS is keyed by `String`, so a non-UTF-8 host filename is
            // decoded lossily (U+FFFD) instead of preserved as raw bytes like
            // PyPy's os.listdir. A lossy name only fails to resolve through the
            // single-`Normal`-component jail in `join`; it cannot escape.
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        if !self.show_dotfiles {
            names.retain(|name| !name.starts_with('.'));
        }
        for excl in &self.exclude {
            names.retain(|name| !name.to_lowercase().ends_with(excl));
        }
        Ok(names)
    }

    // vfs.py join
    //
    // Sandbox-hardening deviation from RPython upstream
    // `rpython/translator/sandbox/vfs.py join`, which does a bare
    // `os.path.join(self.path, name)`.  RPython's sandbox is a separate
    // trusted parent process that intercepts every syscall; pyre's VFS
    // runs in-process so we must reject any `name` that would escape the
    // base directory.  Forbidden inputs: absolute paths (`PathBuf::join`
    // silently swaps the base when joined with an absolute), parent
    // traversal (`..`) and embedded path separators (single-component
    // child names only).  This keeps the join inside `self.path`
    // unconditionally — the rest of the sandbox depends on that
    // invariant.
    fn join(&self, name: &str) -> VfsResult<FsNode> {
        // A child name must be exactly one ordinary path component. Reject both
        // separators explicitly (the VFS is platform-neutral, so `\` is barred on
        // unix too) and require a single `Component::Normal`, which also turns
        // away `..`, the empty string, absolute paths, and Windows drive-prefixed
        // names like `C:` / `C:foo` (neither absolute nor separator-bearing, yet
        // they would escape the base on join).
        let mut components = std::path::Path::new(name).components();
        let single_normal_child =
            matches!(components.next(), Some(std::path::Component::Normal(_)))
                && components.next().is_none();
        if name.contains(['/', '\\']) || !single_normal_child {
            return Err(VfsError {
                errno: ENOENT,
                object: name.to_owned(),
            });
        }
        if name.starts_with('.') && !self.show_dotfiles {
            return Err(VfsError {
                errno: ENOENT,
                object: name.to_owned(),
            });
        }
        let lower_name = name.to_lowercase();
        for excl in &self.exclude {
            if lower_name.ends_with(excl) {
                return Err(VfsError {
                    errno: ENOENT,
                    object: name.to_owned(),
                });
            }
        }

        let path = self.path.join(name);
        let st = if self.follow_links {
            fs::metadata(&path)
        } else {
            fs::symlink_metadata(&path)
        }
        .map_err(|err| io_error(err, path.display().to_string()))?;

        if st.file_type().is_dir() {
            Ok(Rc::new(RealDir::new(
                path,
                self.show_dotfiles,
                self.follow_links,
                self.exclude.clone(),
            )))
        } else if st.file_type().is_file() {
            Ok(Rc::new(RealFile::new(path, 0)))
        } else {
            Err(VfsError {
                errno: EACCES,
                object: path.display().to_string(),
            })
        }
    }
}

// vfs.py File
pub struct File {
    state: FSObjectState,
    pub data: Vec<u8>,
}

impl File {
    // vfs.py __init__
    pub fn new(data: impl AsRef<[u8]>) -> Self {
        Self {
            state: FSObjectState::default(),
            data: data.as_ref().to_vec(),
        }
    }
}

impl FSObject for File {
    // Rust support for vfs.py:15
    fn state(&self) -> &FSObjectState {
        &self.state
    }

    // vfs.py:116
    fn kind(&self) -> Mode {
        ModeBits::IFREG.bits()
    }

    // vfs.py getsize
    fn getsize(&self) -> VfsResult<u64> {
        Ok(self.data.len() as u64)
    }

    // vfs.py open
    fn open(&self) -> VfsResult<Box<dyn ReadSeek>> {
        Ok(Box::new(Cursor::new(self.data.clone())))
    }
}

// vfs.py RealFile
pub struct RealFile {
    state: FSObjectState,
    pub path: PathBuf,
    pub kind: Mode,
}

impl RealFile {
    // vfs.py __init__
    pub fn new(path: impl Into<PathBuf>, mode: Mode) -> Self {
        Self {
            state: FSObjectState::default(),
            path: path.into(),
            kind: ModeBits::IFREG.bits() | mode,
        }
    }

    // vfs.py __repr__
    pub fn repr(&self) -> String {
        format!("<RealFile {}>", self.path.display())
    }
}

impl FSObject for RealFile {
    // Rust support for vfs.py:15
    fn state(&self) -> &FSObjectState {
        &self.state
    }

    // vfs.py:128
    fn kind(&self) -> Mode {
        self.kind
    }

    // vfs.py getsize
    fn getsize(&self) -> VfsResult<u64> {
        fs::metadata(&self.path)
            .map(|st| st.len())
            .map_err(|err| io_error(err, self.path.display().to_string()))
    }

    // vfs.py open
    fn open(&self) -> VfsResult<Box<dyn ReadSeek>> {
        fs::File::open(&self.path)
            .map(|file| Box::new(file) as Box<dyn ReadSeek>)
            .map_err(|err| io_error(err, self.path.display().to_string()))
    }
}

// vfs.py:25
pub fn is_dir(mode: Mode) -> bool {
    (mode & ModeBits::IFMT.bits()) == ModeBits::IFDIR.bits()
}

// vfs.py:134
fn io_error(err: io::Error, object: String) -> VfsError {
    VfsError {
        errno: err.raw_os_error().unwrap_or(EIO),
        object,
    }
}

// Port of rpython/translator/sandbox/test/test_vfs.py.
#[cfg(test)]
mod tests {
    use super::*;

    // POSIX access() mode bits.
    const R_OK: Mode = 4;
    const W_OK: Mode = 2;
    const X_OK: Mode = 1;

    fn read_all(node: &FsNode) -> Vec<u8> {
        let mut data = Vec::new();
        node.open().unwrap().read_to_end(&mut data).unwrap();
        data
    }

    // `FsNode` is not `Debug`, so `Result::unwrap_err` is unavailable on a
    // `VfsResult<FsNode>`; extract the errno of an expected failure by hand.
    fn errno_of(r: VfsResult<FsNode>) -> i32 {
        match r {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.errno,
        }
    }

    fn sorted_keys(node: &FsNode) -> Vec<String> {
        let mut names = node.keys().unwrap();
        names.sort();
        names
    }

    // test_vfs.py test_dir
    #[test]
    fn test_dir() {
        let mut entries: IndexMap<String, FsNode> = IndexMap::new();
        entries.insert("foo".to_owned(), Rc::new(Dir::default()));
        let d = Dir::new(entries);

        assert_eq!(d.keys().unwrap(), vec!["foo".to_owned()]);
        assert!(d.open().is_err());
        assert!(d.getsize().is_ok());

        let d1 = d.join("foo").unwrap();
        assert!(is_dir(d1.kind()));
        assert_eq!(d1.keys().unwrap(), Vec::<String>::new());

        // join('bar') raises
        assert!(d.join("bar").is_err());

        let st = d.stat().unwrap();
        assert!(is_dir(st.st_mode));
        assert!(d.access(R_OK | X_OK).unwrap());
        assert!(!d.access(W_OK).unwrap());
    }

    // test_vfs.py test_file
    #[test]
    fn test_file() {
        let f = File::new("hello world");
        assert_eq!(f.kind() & ModeBits::IFMT.bits(), ModeBits::IFREG.bits());
        assert!(f.keys().is_err());
        assert_eq!(f.getsize().unwrap(), 11);
        assert_eq!(
            read_all(&(Rc::new(File::new("hello world")) as FsNode)),
            b"hello world"
        );

        let st = f.stat().unwrap();
        assert_eq!(st.st_mode & ModeBits::IFMT.bits(), ModeBits::IFREG.bits());
        assert_eq!(st.st_size, 11);
        assert!(f.access(R_OK).unwrap());
        assert!(!f.access(W_OK).unwrap());
    }

    // test_vfs.py test_realdir_realfile — RealDir/RealFile keys + join + read.
    #[test]
    fn test_realdir_realfile() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("file1"), b"somedata1").unwrap();
        std::fs::write(tmp.path().join(".hidden"), b"secret").unwrap();
        std::fs::create_dir(tmp.path().join("subdir1")).unwrap();
        std::fs::write(tmp.path().join("subdir1/subfile1"), b"spam").unwrap();

        let v = RealDir::new(tmp.path(), false, false, Vec::new());
        // dotfiles hidden by default
        assert_eq!(
            {
                let mut k = v.keys().unwrap();
                k.sort();
                k
            },
            vec!["file1".to_owned(), "subdir1".to_owned()]
        );

        assert_eq!(read_all(&v.join("file1").unwrap()), b"somedata1");

        let sub = v.join("subdir1").unwrap();
        assert!(is_dir(sub.kind()));
        assert_eq!(sorted_keys(&sub), vec!["subfile1".to_owned()]);

        // missing + hidden + traversal are all ENOENT
        assert!(v.join("does_not_exist").is_err());
        assert!(v.join(".hidden").is_err());
        assert_eq!(errno_of(v.join("..")), ENOENT);
        assert_eq!(errno_of(v.join("subdir1/subfile1")), ENOENT);
    }

    // test_vfs.py test_realdir_exclude — case-insensitive suffix exclusion.
    #[test]
    fn test_realdir_exclude() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("thing.yes"), b"").unwrap();
        std::fs::write(tmp.path().join("thing.no"), b"").unwrap();

        let v = RealDir::new(tmp.path(), false, false, vec![".no".to_owned()]);
        let keys = v.keys().unwrap();
        assert!(keys.contains(&"thing.yes".to_owned()));
        assert!(!keys.contains(&"thing.no".to_owned()));

        assert!(v.join("thing.yes").is_ok());
        assert!(v.join("thing.no").is_err());
        // case variants are excluded too
        assert!(v.join("thing.No").is_err());
        assert!(v.join("thing.NO").is_err());
    }
}
