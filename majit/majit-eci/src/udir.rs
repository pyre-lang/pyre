//! Port of `rpython/tool/udir.py`.
//!
//! Upstream creates a process-wide "usession" directory named
//! `$PYPY_USESSION_DIR/usession-$PYPY_USESSION_BASENAME-N` and exposes it
//! as module global `udir` (`udir.py`).  This Rust port mirrors
//! upstream's environment-variable contract, the basename defaulting that
//! consults `rpython.tool.version.get_repo_version_info`, and
//! `py.path.local.make_numbered_dir` (nested `parse_num`, `create_lockfile`,
//! `atexit_remove_lockfile`).
//!
//! The optional `usession-<basename>-$USER` symlink `udir.py` documents is
//! created the way `LocalPath.make_numbered_dir` does, from `$USER` /
//! `$USERNAME` / `"current"`.

use std::env;
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Once, OnceLock};
use std::time::{Duration, SystemTime};

/// Upstream `LocalPath.make_numbered_dir` default `lock_timeout` (two days).
const LOCK_TIMEOUT: u64 = 172800;
/// Upstream `LocalPath.make_numbered_dir` default `min_timeout` (five minutes).
const MIN_TIMEOUT: u64 = 300;

static LOCKFILES: parking_lot::Mutex<Vec<(u32, PathBuf)>> = parking_lot::Mutex::new(Vec::new());
static REGISTER_ATEXIT: Once = Once::new();

/// Upstream module global `PYPY_KEEP = int(os.environ.get(
/// 'PYPY_USESSION_KEEP', '3'))` (`udir.py` `PYPY_KEEP`).
fn pypy_keep() -> usize {
    env::var("PYPY_USESSION_KEEP")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3)
}

/// Process-wide equivalent of upstream module global `udir` (`udir.py`).
pub fn udir() -> &'static Path {
    static UDIR: OnceLock<PathBuf> = OnceLock::new();
    UDIR.get_or_init(|| {
        make_udir(
            env::var_os("PYPY_USESSION_DIR").map(PathBuf::from),
            env::var("PYPY_USESSION_BASENAME").ok(),
        )
        .expect("failed to create RPython usession directory")
    })
    .as_path()
}

/// Port of upstream `make_udir(dir=None, basename=None)` (`udir.py`).
pub fn make_udir(dir: Option<PathBuf>, basename: Option<String>) -> std::io::Result<PathBuf> {
    let root = dir.unwrap_or_else(env::temp_dir);
    // Upstream `make_udir`: if `basename is None`, look up the repo's
    // `(hgtag, hgid)` via `get_repo_version_info`.  The first element
    // (tag/branch) is used as basename; '?' is rewritten to 'unknown'.
    let basename = match basename {
        Some(b) => b,
        None => match get_repo_version_info() {
            Some((hgtag, _hgid)) => {
                if hgtag == "?" {
                    "unknown".to_string()
                } else {
                    hgtag
                }
            }
            None => String::new(),
        },
    };
    let mut basename = basename.replace('/', "--");
    // Upstream `make_udir`: force leading and trailing '-'.
    if !basename.starts_with('-') {
        basename.insert(0, '-');
    }
    if !basename.ends_with('-') {
        basename.push('-');
    }
    let prefix = format!("usession{basename}");
    make_numbered_dir(&root, &prefix, pypy_keep(), LOCK_TIMEOUT, MIN_TIMEOUT)
}

/// Port of `py.path.local.make_numbered_dir`.
///
/// `udir.py` `make_udir` calls it as `make_numbered_dir(rootdir=dir, prefix=...,
/// keep=PYPY_KEEP)`, taking the upstream defaults for `lock_timeout` and
/// `min_timeout`.
fn make_numbered_dir(
    root: &Path,
    prefix: &str,
    keep: usize,
    lock_timeout: u64,
    min_timeout: u64,
) -> std::io::Result<PathBuf> {
    fs::create_dir_all(root)?;
    let mut lastmax = None;
    let (udir, lockfile, maxnum) = loop {
        let mut maxnum: i64 = -1;
        for entry in fs::read_dir(root)? {
            let entry = entry?;
            if let Some(num) = parse_num(&entry.path(), prefix) {
                maxnum = maxnum.max(num);
            }
        }
        let created = (|| -> io::Result<(PathBuf, Option<PathBuf>)> {
            let udir = root.join(format!("{prefix}{}", maxnum + 1));
            fs::create_dir(&udir)?;
            let lockfile = if lock_timeout != 0 {
                let lockfile = create_lockfile(&udir)?;
                atexit_remove_lockfile(lockfile.clone());
                Some(lockfile)
            } else {
                None
            };
            Ok((udir, lockfile))
        })();
        match created {
            Ok((udir, lockfile)) => break (udir, lockfile, maxnum),
            Err(e) if is_creation_race(&e) => {
                if lastmax == Some(maxnum) {
                    return Err(e);
                }
                lastmax = Some(maxnum);
            }
            Err(e) => return Err(e),
        }
    };

    if keep != 0 {
        prune_older_numbered_dirs(
            root,
            prefix,
            keep,
            maxnum,
            lockfile.as_deref(),
            lock_timeout,
            min_timeout,
        )?;
    }
    refresh_user_symlink(&udir);
    Ok(udir)
}

fn is_creation_race(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        ErrorKind::AlreadyExists | ErrorKind::NotFound | ErrorKind::ResourceBusy
    ) || e.raw_os_error() == Some(libc::EBUSY)
}

/// Nested `parse_num` of `LocalPath.make_numbered_dir`.
fn parse_num(path: &Path, prefix: &str) -> Option<i64> {
    let nbasename = path.file_name()?.to_str()?.to_lowercase();
    if nbasename.starts_with(&prefix.to_lowercase()) {
        nbasename.get(prefix.len()..)?.parse::<i64>().ok()
    } else {
        None
    }
}

/// Nested `create_lockfile` of `LocalPath.make_numbered_dir`.
fn create_lockfile(path: &Path) -> io::Result<PathBuf> {
    let mypid = std::process::id();
    let lockfile = path.join(".lock");
    let pid_string = mypid.to_string();
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&pid_string, &lockfile)?;
    }
    #[cfg(not(unix))]
    {
        use std::io::Write;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lockfile)?;
        file.write_all(pid_string.as_bytes())?;
    }
    Ok(lockfile)
}

extern "C" fn remove_registered_lockfiles() {
    let pid = std::process::id();
    let lockfiles = LOCKFILES.lock();
    for (owner, path) in lockfiles.iter() {
        if *owner == pid {
            let _ = fs::remove_file(path);
        }
    }
}

/// Nested `atexit_remove_lockfile` of `LocalPath.make_numbered_dir`.
fn atexit_remove_lockfile(lockfile: PathBuf) {
    let mypid = std::process::id();
    REGISTER_ATEXIT.call_once(|| unsafe {
        let _ = libc::atexit(remove_registered_lockfiles);
    });
    LOCKFILES.lock().push((mypid, lockfile));
}

fn modified_time(path: &Path) -> io::Result<SystemTime> {
    fs::symlink_metadata(path)?.modified()
}

fn abs_duration(a: SystemTime, b: SystemTime) -> Duration {
    a.duration_since(b).unwrap_or_else(|err| err.duration())
}

/// Pruning loop of `LocalPath.make_numbered_dir`.
fn prune_older_numbered_dirs(
    root: &Path,
    prefix: &str,
    keep: usize,
    maxnum: i64,
    lockfile: Option<&Path>,
    lock_timeout: u64,
    min_timeout: u64,
) -> io::Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let Some(num) = parse_num(&path, prefix) else {
            continue;
        };
        if num > maxnum - keep as i64 {
            continue;
        }
        if min_timeout != 0 {
            let recent_or_unreadable = (|| {
                let t1 = modified_time(&path).ok()?;
                let t2 = modified_time(lockfile?).ok()?;
                Some(abs_duration(t2, t1) < Duration::from_secs(min_timeout))
            })();
            match recent_or_unreadable {
                Some(true) | None => continue,
                Some(false) => {}
            }
        }
        let lf = path.join(".lock");
        let still_locked = (|| {
            let t1 = modified_time(&lf).ok()?;
            let t2 = modified_time(lockfile?).ok()?;
            Some(lock_timeout == 0 || abs_duration(t2, t1) < Duration::from_secs(lock_timeout))
        })();
        if still_locked == Some(true) {
            continue;
        }
        let _ = fs::remove_dir_all(&path);
    }
    Ok(())
}

/// User symlink at the end of `LocalPath.make_numbered_dir`.
fn refresh_user_symlink(udir: &Path) {
    let username = env::var("USER")
        .or_else(|_| env::var("USERNAME"))
        .unwrap_or_else(|_| "current".to_string());
    let src = udir.to_string_lossy();
    let Some(dash) = src.rfind('-') else {
        return;
    };
    let dest = format!("{}-{username}", &src[..dash]);
    let _ = fs::remove_file(&dest);
    #[cfg(unix)]
    {
        let _ = std::os::unix::fs::symlink(src.as_ref(), &dest);
    }
}

/// Port of `rpython.tool.version.get_repo_version_info` (`version.py`)
/// for the git path only.  Returns `(branch_or_tag, revision_id)` when the
/// surrounding source tree is a git checkout, `None` otherwise.  Mercurial
/// is upstream's other branch (the hg branch of `get_repo_version_info`); this port targets the
/// majit/pypy git mirror layout that pyre-stdlib actually ships in.
fn get_repo_version_info() -> Option<(String, String)> {
    let git_root = locate_git_root()?;
    let revision = run_git(&git_root, &["rev-parse", "HEAD"])
        .map(|s| s.chars().take(12).collect::<String>())?;

    if let Some(tag) = run_git(&git_root, &["describe", "--tags", "--exact-match"]) {
        return Some((tag, revision));
    }

    if let Some(branch_text) = run_git(&git_root, &["branch"]) {
        for line in branch_text.lines() {
            if let Some(rest) = line.strip_prefix("* ") {
                let rest = rest.trim();
                let branch = if rest == "(no branch)" || rest.starts_with("(HEAD detached") {
                    "pypy-HEAD"
                } else {
                    rest
                };
                return Some((branch.to_string(), revision));
            }
        }
        return Some(("pypy-HEAD".to_string(), revision));
    }
    Some(("pypy-HEAD".to_string(), revision))
}

fn locate_git_root() -> Option<PathBuf> {
    let mut cur: PathBuf = env::current_dir().ok()?;
    loop {
        if cur.join(".git").exists() {
            return Some(cur);
        }
        if !cur.pop() {
            return None;
        }
    }
}

fn run_git(cwd: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8(output.stdout).ok()?.trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    fn numbered_children(root: &Path, prefix: &str) -> Vec<i64> {
        let mut numbered = Vec::new();
        for entry in fs::read_dir(root).unwrap() {
            let entry = entry.unwrap();
            if let Some(n) = parse_num(&entry.path(), prefix) {
                numbered.push(n);
            }
        }
        numbered.sort();
        numbered
    }

    #[test]
    fn make_udir_matches_upstream_prefix_rules() {
        let root = tempfile::tempdir().expect("tempdir");
        let first = make_udir(Some(root.path().to_path_buf()), Some("fun/bar".to_string()))
            .expect("first udir");
        let second = make_udir(Some(root.path().to_path_buf()), Some("fun/bar".to_string()))
            .expect("second udir");

        assert_eq!(first.file_name().unwrap(), "usession-fun--bar-0");
        assert_eq!(second.file_name().unwrap(), "usession-fun--bar-1");
        assert!(first.join(".lock").symlink_metadata().is_ok());
        assert!(second.join(".lock").symlink_metadata().is_ok());
    }

    #[test]
    fn make_udir_normalizes_empty_basename() {
        let root = tempfile::tempdir().expect("tempdir");
        let path = make_udir(Some(root.path().to_path_buf()), Some(String::new())).expect("udir");

        assert_eq!(path.file_name().unwrap(), "usession-0");
        assert!(path.join(".lock").symlink_metadata().is_ok());
    }

    #[test]
    fn prune_unlocked_older_than_keep_leaves_locked() {
        // `LocalPath.make_numbered_dir` pruning: with `keep = 2`,
        // `min_timeout = 0`, and the default `lock_timeout`, each finished
        // process drops its `.lock` so older unlocked dirs can go. The 5th
        // call sees `maxnum = 3` and removes `num <= 1`.
        let root = tempfile::tempdir().expect("tempdir");
        let prefix = "usession-x-";
        for _ in 0..5 {
            let dir = make_numbered_dir(root.path(), prefix, 2, LOCK_TIMEOUT, 0).expect("mkdir");
            let _ = fs::remove_file(dir.join(".lock"));
        }
        assert_eq!(numbered_children(root.path(), prefix), vec![2, 3, 4]);

        create_lockfile(&root.path().join(format!("{prefix}2"))).expect("relock 2");
        make_numbered_dir(root.path(), prefix, 2, LOCK_TIMEOUT, 0).expect("sixth");
        let numbered = numbered_children(root.path(), prefix);
        assert!(
            numbered.contains(&2),
            "locked dir 2 must survive: {numbered:?}"
        );
        assert!(
            numbered.iter().all(|&n| n > 1),
            "nothing <= 1 must remain: {numbered:?}"
        );
    }

    #[test]
    fn make_udir_with_unknown_repo_uses_repo_default_or_empty() {
        // Upstream falls back to `''` when get_repo_version_info returns
        // None, then prepends '-' (`make_udir`). The Rust port must
        // produce a directory whose name still starts with `usession-`
        // and ends in a number.
        let root = tempfile::tempdir().expect("tempdir");
        let path = make_udir(Some(root.path().to_path_buf()), None).expect("udir");
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.starts_with("usession-"));
        assert!(
            name.chars().last().is_some_and(|c| c.is_ascii_digit()),
            "expected trailing digit, got {name}"
        );
        assert!(path.join(".lock").symlink_metadata().is_ok());
    }

    #[test]
    fn make_numbered_dir_tolerates_concurrent_creators() {
        let root = tempfile::tempdir().expect("tempdir");
        let n = 16;
        let barrier = Barrier::new(n);
        std::thread::scope(|scope| {
            let mut joins = Vec::new();
            for _ in 0..n {
                joins.push(scope.spawn(|| {
                    barrier.wait();
                    make_numbered_dir(root.path(), "usession-c-", 3, LOCK_TIMEOUT, MIN_TIMEOUT)
                }));
            }
            let mut paths = Vec::new();
            for join in joins {
                let path = join.join().expect("thread").expect("mkdir");
                paths.push(path);
            }
            let mut unique = paths.clone();
            unique.sort();
            unique.dedup();
            assert_eq!(unique.len(), n, "paths must be distinct: {paths:?}");
            for path in &paths {
                assert!(path.is_dir(), "fresh dir vanished: {}", path.display());
            }
        });
    }
}
