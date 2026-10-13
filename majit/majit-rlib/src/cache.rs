//! `rpython/rlib/cache.py`: finite, annotation-time caches.
//!
//! `CacheBuilder` supplies the subclass's `_build` / `_ready` methods.
//! Values representing Python objects must clone their identity (e.g. `Arc`),
//! not duplicate the object. Ordered maps preserve the host dict's insertion
//! order without imposing a dense integer key space on class/PBC keys.
//!
//! The upstream `getorbuild._annspecialcase_ = "specialize:memo"` still needs
//! source-callable registration in the translator. This module implements the
//! host cache, not that registration or target prebuilt-object serialization.

use std::hash::Hash;

use indexmap::IndexMap;
use parking_lot::{Mutex, ReentrantMutex};

// cache.py's module-global RLock, shared even by distinct cache instances.
// Keep it reentrant: `_build` and `_ready` can consult this or another cache.
// The per-cache mutex below is only a Rust storage-access guard and is never
// held across either callback. A per-cache replacement for LOCK would change
// upstream's cross-cache serialization and allow lock-order deadlocks.
static LOCK: ReentrantMutex<()> = ReentrantMutex::new(());

#[derive(Debug, PartialEq, Eq)]
pub enum CacheError<E> {
    /// cache.py Cache.getorbuild raises RuntimeError on recursive construction.
    RecursiveBuilding,
    Build(E),
}

pub trait CacheBuilder<K, V> {
    type Error;

    fn _build(&self, key: &K) -> Result<V, CacheError<Self::Error>>;

    fn _ready(&self, _result: &V) -> Result<(), CacheError<Self::Error>> {
        Ok(())
    }
}

#[derive(Debug)]
struct CacheState<K, V> {
    content: IndexMap<K, V>,
    _building: IndexMap<K, bool>,
}

#[derive(Debug)]
pub struct Cache<K, V> {
    state: Mutex<Option<CacheState<K, V>>>,
}

fn occupied<K, V>(state: &mut Option<CacheState<K, V>>) -> &mut CacheState<K, V> {
    state.get_or_insert_with(|| CacheState {
        content: IndexMap::new(),
        _building: IndexMap::new(),
    })
}

impl<K, V> Cache<K, V> {
    /// Empty cache for a prebuilt owner (`cache.py Cache.__init__`).
    pub const EMPTY: Self = Self {
        state: Mutex::new(None),
    };
}

impl<K, V> Default for Cache<K, V> {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl<K, V> Cache<K, V> {
    /// Visit the references in Cache.content for an owning GC root walker.
    /// RPython's GC transform traces this dict automatically; native caches
    /// must expose the same stored values, never copies in a side table.
    /// The visitor must not allocate or re-enter cache construction.
    pub fn visit_values_mut(&self, mut visit: impl FnMut(&mut V)) {
        let mut guard = self.state.lock();
        let Some(state) = guard.as_mut() else {
            return;
        };
        for value in state.content.values_mut() {
            visit(value);
        }
    }
}

// Cache.getorbuild's inner finally, including Rust panic unwinding.
struct Building<'a, K: Eq + Hash, V> {
    state: &'a Mutex<Option<CacheState<K, V>>>,
    key: K,
}

impl<K: Eq + Hash, V> Drop for Building<'_, K, V> {
    fn drop(&mut self) {
        if let Some(state) = self.state.lock().as_mut() {
            state._building.shift_remove(&self.key);
        }
    }
}

impl<K: Clone + Eq + Hash, V: Clone> Cache<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    /// `Cache.getorbuild`: publish before `_ready`, but clear `_building`
    /// whether `_build` succeeds, fails, or unwinds.
    pub fn getorbuild<B: CacheBuilder<K, V>>(
        &self,
        key: K,
        builder: &B,
    ) -> Result<V, CacheError<B::Error>> {
        // cache.py getorbuild takes the module RLock while the host thread
        // already holds the GIL. Host `threading.RLock.acquire` keeps the GIL
        // on the uncontended path and drops it only while waiting. Always
        // releasing first opened a window where a libtest waiter could take
        // the GIL and then block on `TYPEOBJECT_CACHE.get_or_init` while this
        // thread needed the GIL back after taking LOCK.
        // Unregistered translator threads naturally need no runtime
        // transition. Keys and builder references must be nonmoving
        // host/prebuilt identities, or explicitly rooted and reloaded across
        // this allocation boundary.
        let _lock = if let Some(guard) = LOCK.try_lock() {
            guard
        } else {
            let blocked = majit_gc::gc_sync::before_external_block();
            let guard = LOCK.lock();
            drop(blocked);
            guard
        };
        {
            let mut guard = self.state.lock();
            let state = occupied(&mut guard);
            if let Some(result) = state.content.get(&key) {
                return Ok(result.clone());
            }
            if state._building.contains_key(&key) {
                return Err(CacheError::RecursiveBuilding);
            }
            state._building.insert(key.clone(), true);
        }
        let building = Building {
            state: &self.state,
            key: key.clone(),
        };
        let result = builder._build(&key)?;
        occupied(&mut self.state.lock())
            .content
            .insert(key, result.clone());
        drop(building);
        builder._ready(&result)?;
        Ok(result)
    }

    /// `Cache._freeze_`: admit a prebuilt cache as a PBC, without sealing it.
    /// Annotation may discover more getorbuild calls afterwards.
    pub fn _freeze_(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn external_wait_ends_before_build_and_ready() {
        struct Owner;
        impl CacheBuilder<usize, usize> for Owner {
            type Error = ();
            fn _build(&self, key: &usize) -> Result<usize, CacheError<()>> {
                assert!(majit_gc::rgil::am_i_holding_the_gil());
                Ok(*key)
            }
            fn _ready(&self, _: &usize) -> Result<(), CacheError<()>> {
                assert!(majit_gc::rgil::am_i_holding_the_gil());
                Ok(())
            }
        }
        let _gil = majit_gc::rgil::GilGuard::acquire();
        let owner = Owner;
        let cache = Cache::new();
        assert_eq!(cache.getorbuild(1, &owner), Ok(1));
        assert_eq!(cache.getorbuild(1, &owner), Ok(1));
        assert!(majit_gc::rgil::am_i_holding_the_gil());
    }

    #[derive(Default)]
    struct MyCache {
        cache: Cache<usize, Arc<usize>>,
        counter: AtomicUsize,
    }

    impl CacheBuilder<usize, Arc<usize>> for MyCache {
        type Error = ();

        fn _build(&self, key: &usize) -> Result<Arc<usize>, CacheError<()>> {
            self.counter.fetch_add(1, Ordering::Relaxed);
            Ok(Arc::new(key * 7))
        }
    }

    #[test]
    fn getorbuild() {
        // rpython/rlib/test/test_cache.py TestCache.test_getorbuild.
        let owner = MyCache::default();
        let one = owner.cache.getorbuild(1, &owner).unwrap();
        assert_eq!(*one, 7);
        assert_eq!(owner.counter.load(Ordering::Relaxed), 1);
        assert!(Arc::ptr_eq(
            &one,
            &owner.cache.getorbuild(1, &owner).unwrap()
        ));
        assert_eq!(owner.counter.load(Ordering::Relaxed), 1);
        assert!(owner.cache._freeze_());
        let three = owner.cache.getorbuild(3, &owner).unwrap();
        assert_eq!(*three, 21);
        assert!(Arc::ptr_eq(
            &one,
            &owner.cache.getorbuild(1, &owner).unwrap()
        ));
        assert!(Arc::ptr_eq(
            &three,
            &owner.cache.getorbuild(3, &owner).unwrap()
        ));
        assert_eq!(owner.counter.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn concurrent_getorbuild_publishes_one_identity() {
        let owner = MyCache::default();
        std::thread::scope(|scope| {
            let threads: Vec<_> = (0..4)
                .map(|_| scope.spawn(|| owner.cache.getorbuild(1, &owner).unwrap()))
                .collect();
            let values: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
            assert!(values.iter().all(|v| Arc::ptr_eq(v, &values[0])));
        });
        assert_eq!(owner.counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn build_can_enter_another_cache_and_another_key() {
        struct Owner {
            first: Cache<usize, usize>,
            second: Cache<usize, usize>,
        }
        impl CacheBuilder<usize, usize> for Owner {
            type Error = ();

            fn _build(&self, key: &usize) -> Result<usize, CacheError<()>> {
                match key {
                    0 => self.second.getorbuild(1, self),
                    1 => self.first.getorbuild(2, self),
                    _ => Ok(42),
                }
            }
        }
        let owner = Owner {
            first: Cache::new(),
            second: Cache::new(),
        };
        assert_eq!(owner.first.getorbuild(0, &owner), Ok(42));
        assert!(
            owner
                .first
                .state
                .lock()
                .as_ref()
                .is_none_or(|s| s._building.is_empty())
        );
        assert!(
            owner
                .second
                .state
                .lock()
                .as_ref()
                .is_none_or(|s| s._building.is_empty())
        );
    }

    struct RecursiveCache {
        cache: Cache<usize, usize>,
        attempts: AtomicUsize,
        fail_ready: bool,
    }

    impl CacheBuilder<usize, usize> for RecursiveCache {
        type Error = &'static str;

        fn _build(&self, key: &usize) -> Result<usize, CacheError<Self::Error>> {
            match self.attempts.fetch_add(1, Ordering::Relaxed) {
                0 => self.cache.getorbuild(*key, self),
                1 => Err(CacheError::Build("build failed")),
                2 => panic!("build unwound"),
                _ => Ok(*key),
            }
        }

        fn _ready(&self, result: &usize) -> Result<(), CacheError<Self::Error>> {
            // Publication precedes ready: this is a hit, not recursive build.
            assert_eq!(self.cache.getorbuild(*result, self)?, *result);
            if self.fail_ready {
                Err(CacheError::Build("ready failed"))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn failed_build_cleans_up_and_ready_sees_published_result() {
        let owner = RecursiveCache {
            cache: Cache::new(),
            attempts: AtomicUsize::new(0),
            fail_ready: true,
        };
        assert_eq!(
            owner.cache.getorbuild(1, &owner),
            Err(CacheError::RecursiveBuilding)
        );
        assert_eq!(
            owner.cache.getorbuild(1, &owner),
            Err(CacheError::Build("build failed"))
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                owner.cache.getorbuild(1, &owner)
            }))
            .is_err()
        );
        assert_eq!(
            owner.cache.getorbuild(1, &owner),
            Err(CacheError::Build("ready failed"))
        );
        // Like upstream, a ready failure leaves the value cached.
        assert_eq!(owner.cache.getorbuild(1, &owner), Ok(1));
        assert_eq!(owner.attempts.load(Ordering::Relaxed), 4);
    }
}
