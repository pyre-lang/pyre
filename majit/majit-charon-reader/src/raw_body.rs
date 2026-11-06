//! Raw JSON text of a `FunDecl` body, owned or as a range into the
//! artefact's read-only file mapping.

use serde::Deserialize;
use serde_json::value::RawValue;
use std::cell::RefCell;
use std::sync::Arc;

pub(crate) enum Backing {
    Mapped(memmap2::Mmap),
}

impl std::ops::Deref for Backing {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Backing::Mapped(map) => map.as_ref(),
        }
    }
}

pub struct RawBody(Repr);

enum Repr {
    Owned(Box<RawValue>),
    Mapped {
        backing: Arc<Backing>,
        start: usize,
        len: usize,
    },
}

impl RawBody {
    /// The body's raw JSON text.
    #[allow(unsafe_code)]
    pub fn get(&self) -> &str {
        match &self.0 {
            Repr::Owned(raw) => raw.get(),
            Repr::Mapped {
                backing,
                start,
                len,
            } => {
                // SAFETY: the range was taken from a `&str` the JSON parser
                // borrowed out of these same bytes, and the mapping is
                // read-only; it relies on the same invariant as the
                // `Mmap::map` call in `Llbc::load` (the artefact is
                // replaced by rename, never rewritten in place).
                unsafe { std::str::from_utf8_unchecked(&backing[*start..*start + *len]) }
            }
        }
    }
}

impl std::fmt::Debug for RawBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("RawBody")
            .field(&format_args!("{}", self.get()))
            .finish()
    }
}

// serde's `Deserialize` has no seed path through
// `LlbcFile -> Translated -> Vec<Option<FunDecl>>`, so the mapping a
// borrowed body range belongs to is passed through this parse-scoped
// slot; it is set only for the duration of one `serde_json::from_slice`
// call on the current thread and holds no semantic state.
thread_local! {
    static PARSE_BACKING: RefCell<Option<Arc<Backing>>> = const { RefCell::new(None) };
}

struct ParseBackingRestore(Option<Arc<Backing>>);

impl Drop for ParseBackingRestore {
    fn drop(&mut self) {
        PARSE_BACKING.with(|slot| {
            *slot.borrow_mut() = self.0.take();
        });
    }
}

pub(crate) fn with_parse_backing<R>(backing: &Arc<Backing>, f: impl FnOnce() -> R) -> R {
    let prev = PARSE_BACKING.with(|slot| slot.borrow_mut().replace(backing.clone()));
    let _restore = ParseBackingRestore(prev);
    f()
}

impl<'de> Deserialize<'de> for RawBody {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let backing = PARSE_BACKING.with(|slot| slot.borrow().clone());
        if let Some(backing) = backing {
            let raw = <&'de RawValue>::deserialize(deserializer)?;
            let s = raw.get();
            // Bind the mapping as `[u8]` so `as_ptr` is the file bytes,
            // not `Arc::as_ptr`.
            let bytes: &[u8] = &backing;
            let base = bytes.as_ptr() as usize;
            let ptr = s.as_ptr() as usize;
            let end = ptr + s.len();
            if ptr >= base && end <= base + bytes.len() {
                let start = ptr - base;
                Ok(RawBody(Repr::Mapped {
                    backing,
                    start,
                    len: s.len(),
                }))
            } else {
                Ok(RawBody(Repr::Owned(
                    RawValue::from_string(s.to_owned()).map_err(serde::de::Error::custom)?,
                )))
            }
        } else {
            Ok(RawBody(Repr::Owned(Box::<RawValue>::deserialize(
                deserializer,
            )?)))
        }
    }
}
