//! Per-thread critical-section suspension around external blocking calls.
//!
//! This thread-local stack is the `tstate->critical_section` analog used by
//! `_PyCriticalSection_SuspendAll`: each entry records one recursion level of
//! an address-striped reentrant mutex held by the current thread.

use std::cell::RefCell;

#[derive(Default)]
struct CriticalSections {
    held: Vec<usize>,
    suspension_marks: Vec<usize>,
}

thread_local! {
    static CRITICAL_SECTIONS: RefCell<CriticalSections> = RefCell::new(CriticalSections::default());
}

/// Record one recursion level acquired by the current thread.
pub fn note_acquired(word: usize) {
    if word == 0 {
        return;
    }
    CRITICAL_SECTIONS.with(|sections| sections.borrow_mut().held.push(word));
}

/// Retire the most recently acquired matching recursion level.
///
/// Returns whether the word was recorded. A destroyed thread-local returns
/// true so thread teardown can still discharge its native lock acquisition.
pub fn note_released(word: usize) -> bool {
    if word == 0 {
        return false;
    }
    CRITICAL_SECTIONS
        .try_with(|sections| {
            let mut sections = sections.borrow_mut();
            if let Some(index) = sections.held.iter().rposition(|&held| held == word) {
                sections.held.remove(index);
                for mark in sections.suspension_marks.iter_mut() {
                    if index < *mark {
                        *mark -= 1;
                    }
                }
                true
            } else {
                false
            }
        })
        .unwrap_or(true)
}

/// Discard critical-section ownership inherited across `fork()`.
pub fn after_fork_child() {
    CRITICAL_SECTIONS.with(|sections| {
        let mut sections = sections.borrow_mut();
        sections.held.clear();
        sections.suspension_marks.clear();
    });
}

unsafe fn force_unlock(word: usize) {
    unsafe { (&*(word as *const parking_lot::ReentrantMutex<()>)).force_unlock() };
}

unsafe fn try_lock(word: usize) -> bool {
    let lock = unsafe { &*(word as *const parking_lot::ReentrantMutex<()>) };
    let Some(guard) = lock.try_lock() else {
        return false;
    };
    std::mem::forget(guard);
    true
}

/// Wait for a stripe to become available without retaining an acquisition.
/// Call while detached so the wait holds neither the running-mutator census
/// nor any suspended critical-section levels.
pub fn wait_until_unlocked(lock: &parking_lot::ReentrantMutex<()>) {
    let guard = lock.lock();
    std::mem::forget(guard);
    unsafe { lock.force_unlock() };
}

/// Suspend every critical-section level acquired since the enclosing detach.
pub(crate) fn suspend_all() -> bool {
    CRITICAL_SECTIONS.with(|sections| {
        let mut sections = sections.borrow_mut();
        let start = sections.suspension_marks.last().copied().unwrap_or(0);
        let end = sections.held.len();
        if start == end {
            return false;
        }
        sections.suspension_marks.push(end);
        for &word in sections.held[start..end].iter().rev() {
            unsafe { force_unlock(word) };
        }
        true
    })
}

/// Reacquire the levels suspended by [`suspend_all`] in acquisition order.
pub(crate) fn resume_all() {
    let words = CRITICAL_SECTIONS.with(|sections| {
        let sections = sections.borrow();
        // `os.fork` runs inside a `before_external_block` guard, and
        // [`after_fork_child`] discards that guard's mark in the child along
        // with the stripes it named: nothing is left to resume.
        let Some(&end) = sections.suspension_marks.last() else {
            return Vec::new();
        };
        let start = sections
            .suspension_marks
            .iter()
            .rev()
            .nth(1)
            .copied()
            .unwrap_or(0);
        sections.held[start..end].to_vec()
    });

    if !words.is_empty() {
        loop {
            let mut acquired = 0;
            while acquired < words.len() && unsafe { try_lock(words[acquired]) } {
                acquired += 1;
            }
            if acquired == words.len() {
                break;
            }
            for &word in words[..acquired].iter().rev() {
                unsafe { force_unlock(word) };
            }
            wait_until_unlocked(unsafe {
                &*(words[acquired] as *const parking_lot::ReentrantMutex<()>)
            });
        }
    }

    CRITICAL_SECTIONS.with(|sections| {
        sections.borrow_mut().suspension_marks.pop();
    });
}
