//! _random module — PyPy `pypy/module/_random/`.
//!
//! `W_Random` wraps a Mersenne Twister generator (`rrandom.Random`) and
//! exposes `random`/`seed`/`getstate`/`setstate`/`getrandbits`, matching
//! `pypy/module/_random/interp_random.py W_Random`.

use majit_rlib::rbigint::{RBigInt as BigInt, RBigIntSign as Sign};
use pyre_object::*;

/// Mersenne Twister state — `rpython/rlib/rrandom.py Random`.
const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7fffffff;
const TEMPERING_MASK_A: u32 = 0x9d2c5680;
const TEMPERING_MASK_B: u32 = 0xefc60000;
const MAGIC_CONSTANT_A: u32 = 1812433253;
const MAGIC_CONSTANT_B: u32 = 19650218;
const MAGIC_CONSTANT_C: u32 = 1664525;
const MAGIC_CONSTANT_D: u32 = 1566083941;

/// `rrandom.py class Random` is an ordinary instance that
/// `interp_random.py` allocates on its own — `self._rnd = rrandom.Random()`
/// — so the twister lives *behind* a reference rather than inside its holder.
/// Keeping that shape is what makes the holder's `self.rnd` field readable:
/// borrowing an inline aggregate field has no IR spelling, and the front-end's
/// `&x ≡ x` model would hand `genrand32` the first word of `state` where its
/// address was meant.  It owns no object references, so the collector forwards
/// nothing but its header.
#[pyre_interpreter::pyre_class("_random._mersenne_twister", static_name = "MERSENNE_TWISTER")]
pub struct Random {
    state: [u32; N],
    index: usize,
}

impl Random {
    /// `rrandom.Random.__init__` — a fresh state run through `init_genrand(0)`.
    fn new_instance() -> PyObjectRef {
        let obj = Random::allocate_stable(Random {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            state: [0; N],
            index: 0,
        });
        Random::from_obj(obj)
            .expect("a freshly allocated twister has the Random layout")
            .init_genrand(0);
        obj
    }

    fn init_genrand(&mut self, s: u32) {
        let mt = &mut self.state;
        mt[0] = s;
        for mti in 1..N {
            mt[mti] = MAGIC_CONSTANT_A
                .wrapping_mul(mt[mti - 1] ^ (mt[mti - 1] >> 30))
                .wrapping_add(mti as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, init_key: &[u32]) {
        let key_length = init_key.len();
        self.init_genrand(MAGIC_CONSTANT_B);
        let mut i = 1usize;
        let mut j = 0usize;
        let max_k = if N > key_length { N } else { key_length };
        for _ in 0..max_k {
            let mt = &mut self.state;
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(MAGIC_CONSTANT_C))
                .wrapping_add(init_key[j])
                .wrapping_add(j as u32); // non linear
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= key_length {
                j = 0;
            }
        }
        for _ in (1..N).rev() {
            let mt = &mut self.state;
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(MAGIC_CONSTANT_D))
                .wrapping_sub(i as u32); // non linear
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
        }
        self.state[0] = UPPER_MASK;
    }

    fn conditionally_apply(val: u32, y: u32) -> u32 {
        if y & 1 != 0 { val ^ MATRIX_A } else { val }
    }

    pub(crate) fn genrand32(&mut self) -> u32 {
        if self.index >= N {
            let mt = &mut self.state;
            for kk in 0..(N - M) {
                let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = Random::conditionally_apply(mt[kk + M] ^ (y >> 1), y);
            }
            for kk in (N - M)..(N - 1) {
                let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = Random::conditionally_apply(mt[kk + M - N] ^ (y >> 1), y);
            }
            let y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
            mt[N - 1] = Random::conditionally_apply(mt[M - 1] ^ (y >> 1), y);
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & TEMPERING_MASK_A;
        y ^= (y << 15) & TEMPERING_MASK_B;
        y ^= y >> 18;
        y
    }

    fn random(&mut self) -> f64 {
        let a = (self.genrand32() >> 5) as f64;
        let b = (self.genrand32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }
}

// CPython 3.14 Modules/_randommodule.c:_random_exec uses
// PyType_FromModuleAndSpec and does not request IMMUTABLETYPE.
/// `interp_random.py` `W_Random`. A user subclass is `typedef.py`
/// `_getusercls` (`MapdictStorageMixin`), allocated by `objspace.py`
/// `allocate_instance`. The builtin keeps `self._rnd` only.
#[pyre_interpreter::pyre_class("_random.Random", cpython_mutable, user_layout, weakrefable)]
#[derive(Default)]
pub struct W_Random {
    /// `interp_random.py self._rnd = rrandom.Random()` — the reference to a
    /// separately allocated generator.
    pub rnd: PyObjectRef,
}

impl W_Random {
    /// The generator behind `self._rnd`.
    fn rnd(&self) -> &'static mut Random {
        Random::from_obj(self.rnd).expect("_random.Random holds a Mersenne Twister")
    }
}

#[pyre_interpreter::pyre_methods(
    doc = "Random() -> create a random number generator.\n\nNot for security or cryptographic use.",
    weakrefable
)]
impl W_Random {
    /// PyPy `interp_random.py` `descr_new__`: allocate the requested subtype and
    /// initialise it from the first constructor argument (or `None`).
    ///
    /// `__args__.firstarg()` deliberately leaves surplus-argument reporting to
    /// the later `__init__` dispatch, which is essential for subclasses whose
    /// initializer accepts its own positional or keyword arguments.
    #[staticmethod]
    fn __new__(
        cls: PyObjectRef,
        args: &[PyObjectRef],
    ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let user_args = args.get(1..).unwrap_or(&[]);
        let w_anything = user_args.first().copied().unwrap_or_else(w_none);
        // `objspace.py` `allocate_instance` validates the requested subtype.
        // The builtin is the base layout; a subclass is `typedef.py`
        // `_getusercls`.
        let random_type = type_object();
        pyre_interpreter::typedef::check_user_subclass(random_type, cls)?;
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(cls);
        let _ = pyre_object::gc_roots::pin_root(w_anything);
        let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 2;
        let seed_slot = cls_slot + 1;
        // `interp_random.py` builds the generator before anything can reach
        // it through the wrapper, so each allocation has to hold the previous
        // one down: the twister across the wrapper's allocation, the wrapper
        // across `seed`. Exact `allocate_stable` does not reload a payload
        // `PyObjectRef`, so `rnd` is stored after the wrapper exists.
        let _ = pyre_object::gc_roots::pin_root(Random::new_instance());
        let rnd_slot = seed_slot + 1;
        let obj = W_Random::allocate_instance(
            W_Random::default(),
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        );
        let _ = pyre_object::gc_roots::pin_root(obj);
        let obj_slot = rnd_slot + 1;
        let random = W_Random::from_obj(pyre_object::gc_roots::shadow_stack_get(obj_slot))
            .expect("a freshly allocated _random.Random has the Random layout");
        random.rnd = pyre_object::gc_roots::shadow_stack_get(rnd_slot);
        random.seed(pyre_object::gc_roots::shadow_stack_get(seed_slot))?;
        Ok(pyre_object::gc_roots::shadow_stack_get(obj_slot))
    }

    fn random(&mut self) -> f64 {
        self.rnd().random()
    }

    fn seed(
        &mut self,
        #[default(pyre_object::w_none())] w_n: PyObjectRef,
    ) -> Result<(), pyre_interpreter::PyError> {
        // None: seed from `interp_posix.urandom(space, 8)`; fall back to a
        // time-based int only when that raises OSError (interp_random.py).
        let w_n = if unsafe { is_none(w_n) } {
            match posix_urandom(8) {
                Ok(w_bytes) => w_bytes,
                Err(mut error) => {
                    let w_oserror = pyre_interpreter::builtins::lookup_exc_class("OSError")
                        .expect("OSError is installed");
                    if !error.match_(PY_NULL, w_oserror) {
                        return Err(error);
                    }
                    w_int_new(seed_from_time() as i64)
                }
            }
        } else {
            w_n
        };
        let n = unsafe {
            if is_int_or_long(w_n) {
                // space.abs(w_n)
                let v = pyre_interpreter::builtins::obj_to_bigint(w_n);
                if v.get_sign() < 0 { -v } else { v }
            } else {
                // n = space.hash_w(w_n); w_n = space.newint(r_uint(n))
                BigInt::from(pyre_interpreter::baseobjspace::hash_w_strict(w_n)? as u64)
            }
        };
        // Split into little-endian 32-bit chunks.
        let (_sign, key) = n.to_u32_digits();
        let key = if key.is_empty() { vec![0u32] } else { key };
        self.rnd().init_by_array(&key);
        Ok(())
    }

    fn getstate(&self) -> PyObjectRef {
        let rnd = self.rnd();
        // 625 successive allocations, each over the words already produced,
        // and a plain `Vec` is not a root area (`build_list_storage`).
        let mut state = pyre_object::gc_roots::RootedItems::new();
        for i in 0..N {
            state.push(w_int_new(rnd.state[i] as i64));
        }
        state.push(w_int_new(rnd.index as i64));
        w_tuple_new(state.take())
    }

    fn setstate(&mut self, w_state: PyTuple) -> Result<(), pyre_interpreter::PyError> {
        unsafe {
            if w_tuple_len(w_state) != N + 1 {
                pyre_interpreter::bail_value_error!("state vector is the wrong size");
            }
            let mut new_state = [0u32; N];
            // `obj_to_bigint` of a machine int collects; the state tuple
            // comes back off the shadow stack.
            let _roots = pyre_object::gc_roots::push_roots();
            let state_slot = pyre_object::gc_roots::pin_roots(&[w_state]);
            for i in 0..N {
                let Some(item) = w_tuple_getitem(
                    pyre_object::gc_roots::shadow_stack_get(state_slot),
                    i as i64,
                ) else {
                    pyre_interpreter::bail_value_error!("state vector is the wrong size");
                };
                if !is_int_or_long(item) {
                    pyre_interpreter::bail_type_error!("state vector must contain ints");
                }
                let mut v = pyre_interpreter::builtins::obj_to_bigint(item);
                if v.get_sign() < 0 {
                    v = v.int_add(1i64 << 32);
                }
                // space.uint_w: every word must fit an unsigned 32-bit int.
                if v.get_sign() < 0 {
                    pyre_interpreter::bail_overflow_error!(
                        "cannot convert negative integer to unsigned int"
                    );
                }
                if v.int_ge(1i64 << 32) {
                    pyre_interpreter::bail_overflow_error!(
                        "int too large to convert to unsigned int"
                    );
                }
                let (_s, digits) = v.to_u32_digits();
                new_state[i] = digits.first().copied().unwrap_or(0);
            }
            let Some(item) = w_tuple_getitem(
                pyre_object::gc_roots::shadow_stack_get(state_slot),
                N as i64,
            ) else {
                pyre_interpreter::bail_value_error!("state vector is the wrong size");
            };
            // space.int_w: handles long overflow / non-int (TypeError).
            let index = pyre_interpreter::baseobjspace::int_w(item)?;
            if index < 0 || index > N as i64 {
                pyre_interpreter::bail_value_error!("invalid state");
            }
            let rnd = self.rnd();
            rnd.state = new_state;
            rnd.index = index as usize;
        }
        Ok(())
    }

    fn getrandbits(&mut self, k: PyIndexInt) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let mut k = k;
        if k < 0 {
            pyre_interpreter::bail_value_error!("number of bits must be non-negative");
        }
        if k == 0 {
            return Ok(w_int_new(0));
        }
        if k < 32 {
            // fits an int, skip the bytes-to-long dance
            let r = self.rnd().genrand32() >> (32 - k);
            return Ok(w_int_new(r as i64));
        }
        let nbytes = usize::try_from(((k - 1) / 32 + 1) * 4)
            .map_err(|_| pyre_interpreter::PyError::memory_error(""))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(nbytes)
            .map_err(|_| pyre_interpreter::PyError::memory_error(""))?;
        while k > 0 {
            let mut r = self.rnd().genrand32();
            if k < 32 {
                r >>= 32 - k;
            }
            bytes.push((r & 0xff) as u8);
            bytes.push(((r >> 8) & 0xff) as u8);
            bytes.push(((r >> 16) & 0xff) as u8);
            bytes.push(((r >> 24) & 0xff) as u8);
            k -= 32;
        }
        // little endian order to match the byte append order
        let result = BigInt::from_bytes_le(Sign::Plus, &bytes);
        Ok(w_long_new(result))
    }
}

/// `interp_posix.urandom(space, size)`, called directly as `W_Random.seed`
/// does: it owns the entropy source on every host (`rurandom.urandom`, the
/// Windows CSPRNG, or the sandbox controller), whatever `posix.urandom` has
/// been rebound to.
fn posix_urandom(size: i64) -> pyre_interpreter::PyResult {
    pyre_interpreter::module::posix::urandom(&[w_int_new(size)])
}

/// Time-based fallback seed — `int(time.time() * 256)`.
///
/// Read through the same accessor `time.time()` uses, which is what routes
/// the reading to the sandbox controller and to the wasm embedder rather than
/// to a `SystemTime` neither of them has.
fn seed_from_time() -> u64 {
    let secs = pyre_interpreter::module::time::interp_time::duration_since_epoch().as_secs_f64();
    (secs * 256.0) as u64
}

pyre_interpreter::py_module! {
    "_random",
    interpleveldefs: {
        "Random" => type_object(),
    },
}

#[cfg(test)]
mod macro_smoke;

/// The GC types this module owns, in `build_gc` registration order.
pub(crate) fn gc_types(types: &mut Vec<pyre_interpreter::importing::ModuleGcType>) {
    use pyre_interpreter::importing::{ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    // `objspace.py` `allocate_instance`: the builtin traces `w_class` and
    // `rnd`. A subclass is `typedef.py` `_getusercls`, whose mapdict storage
    // the user layout walks on top of those edges.
    let pyre_class = ModuleGcLayout::PyreClass {
        memory_pressure_offset: None,
    };
    types.push(ModuleGcType {
        descriptor: <W_Random as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: pyre_class,
        destructor: None,
    });
    types.push(ModuleGcType {
        descriptor: &W_RANDOM_USER_PYRE_CLASS_DESCRIPTOR,
        layout: pyre_class,
        destructor: None,
    });
    // `rrandom.Random` — the Mersenne Twister `interp_random.py` allocates
    // beside its holder. Like `W_DequeBlock` it is GC-managed without being an
    // rclass.OBJECT subclass and has no Python-visible vtable, so it takes a
    // bare `with_gc_ptrs` id.
    types.push(ModuleGcType {
        descriptor: <Random as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::WithGcPtrs,
        destructor: None,
    });
}
