//! `_scproxy` — the macOS SystemConfiguration proxy probe that
//! `urllib.request.getproxies_macosx_sysconf` / `proxy_bypass_macosx_sysconf`
//! import.  Report "no system proxy configured" so the import succeeds and
//! proxy resolution yields an empty mapping.

use pyre_object::gc_roots;
use pyre_object::*;

pub fn init(ns: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "_get_proxies",
        pyre_interpreter::make_builtin_function("_get_proxies", |_| Ok(w_dict_new()))
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "_get_proxy_settings",
        pyre_interpreter::make_builtin_function("_get_proxy_settings", |_| {
            // The `dict` moves across the allocations each store makes.
            let roots = gc_roots::push_roots();
            let d_slot = roots.base();
            let _ = roots.pin_root(w_dict_new());
            unsafe {
                let key_slot = gc_roots::shadow_stack_len();
                let _ = roots.pin_root(w_str_new("exclude_simple"));
                let val_slot = gc_roots::shadow_stack_len();
                let _ = roots.pin_root(w_bool_from(false));
                w_dict_store(roots.get(d_slot), roots.get(key_slot), roots.get(val_slot));
                let key_slot = gc_roots::shadow_stack_len();
                let _ = roots.pin_root(w_str_new("exceptions"));
                let val_slot = gc_roots::shadow_stack_len();
                let _ = roots.pin_root(w_list_new(Vec::new()));
                w_dict_store(roots.get(d_slot), roots.get(key_slot), roots.get(val_slot));
            }
            Ok(roots.get(d_slot))
        })
    );
    Ok(())
}
