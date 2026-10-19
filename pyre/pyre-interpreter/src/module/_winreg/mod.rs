//! winreg module — PyPy: pypy/module/_winreg/ (`applevel_name = 'winreg'`).
//!
//! This file is `moduledef.py`: the module entry and its integer constants.
//! The handle object and the functions are `interp_winreg.rs`.
//!
//! `importlib._bootstrap_external` does an eager `import winreg` on every
//! `sys.platform == "win32"` build, so the module must exist for the import
//! machinery — and therefore `import site` — to come up.  The integer
//! constants are always present; the `PyHKEY` handle object and the
//! Open/Query/Enum/Set/Delete key & value functions are registered when the
//! `host_env` feature is on (they delegate every Win32 `Reg*` call to
//! `rustpython_host_env::winreg`).
#[cfg(feature = "host_env")]
pub mod interp_winreg;

#[cfg(feature = "host_env")]
macro_rules! reg_int {
    (HKEY_CLASSES_ROOT) => {
        rustpython_host_env::winreg::HKEY_CLASSES_ROOT as u32 as i64
    };
    (HKEY_CURRENT_USER) => {
        rustpython_host_env::winreg::HKEY_CURRENT_USER as u32 as i64
    };
    (HKEY_LOCAL_MACHINE) => {
        rustpython_host_env::winreg::HKEY_LOCAL_MACHINE as u32 as i64
    };
    (HKEY_USERS) => {
        rustpython_host_env::winreg::HKEY_USERS as u32 as i64
    };
    (HKEY_PERFORMANCE_DATA) => {
        rustpython_host_env::winreg::HKEY_PERFORMANCE_DATA as u32 as i64
    };
    (HKEY_CURRENT_CONFIG) => {
        rustpython_host_env::winreg::HKEY_CURRENT_CONFIG as u32 as i64
    };
    (HKEY_DYN_DATA) => {
        rustpython_host_env::winreg::HKEY_DYN_DATA as u32 as i64
    };
    (ERROR_MORE_DATA) => {
        windows_sys::Win32::Foundation::ERROR_MORE_DATA as i64
    };
    ($name:ident) => {
        rustpython_host_env::winreg::$name as i64
    };
}

#[cfg(not(feature = "host_env"))]
macro_rules! reg_int {
    (HKEY_CLASSES_ROOT) => {
        0x8000_0000u32 as i64
    };
    (HKEY_CURRENT_USER) => {
        0x8000_0001u32 as i64
    };
    (HKEY_LOCAL_MACHINE) => {
        0x8000_0002u32 as i64
    };
    (HKEY_USERS) => {
        0x8000_0003u32 as i64
    };
    (HKEY_PERFORMANCE_DATA) => {
        0x8000_0004u32 as i64
    };
    (HKEY_CURRENT_CONFIG) => {
        0x8000_0005u32 as i64
    };
    (HKEY_DYN_DATA) => {
        0x8000_0006u32 as i64
    };
    (KEY_QUERY_VALUE) => {
        0x0001 as i64
    };
    (KEY_SET_VALUE) => {
        0x0002 as i64
    };
    (KEY_CREATE_SUB_KEY) => {
        0x0004 as i64
    };
    (KEY_ENUMERATE_SUB_KEYS) => {
        0x0008 as i64
    };
    (KEY_NOTIFY) => {
        0x0010 as i64
    };
    (KEY_CREATE_LINK) => {
        0x0020 as i64
    };
    (KEY_WOW64_64KEY) => {
        0x0100 as i64
    };
    (KEY_WOW64_32KEY) => {
        0x0200 as i64
    };
    (KEY_WRITE) => {
        0x0002_0006 as i64
    };
    (KEY_READ) => {
        0x0002_0019 as i64
    };
    (KEY_EXECUTE) => {
        0x0002_0019 as i64
    };
    (KEY_ALL_ACCESS) => {
        0x000F_003F as i64
    };
    (REG_NONE) => {
        0 as i64
    };
    (REG_SZ) => {
        1 as i64
    };
    (REG_EXPAND_SZ) => {
        2 as i64
    };
    (REG_BINARY) => {
        3 as i64
    };
    (REG_DWORD) => {
        4 as i64
    };
    (REG_DWORD_LITTLE_ENDIAN) => {
        4 as i64
    };
    (REG_DWORD_BIG_ENDIAN) => {
        5 as i64
    };
    (REG_LINK) => {
        6 as i64
    };
    (REG_MULTI_SZ) => {
        7 as i64
    };
    (REG_RESOURCE_LIST) => {
        8 as i64
    };
    (REG_FULL_RESOURCE_DESCRIPTOR) => {
        9 as i64
    };
    (REG_RESOURCE_REQUIREMENTS_LIST) => {
        10 as i64
    };
    (REG_QWORD) => {
        11 as i64
    };
    (REG_QWORD_LITTLE_ENDIAN) => {
        11 as i64
    };
    (REG_OPTION_RESERVED) => {
        0 as i64
    };
    (REG_OPTION_NON_VOLATILE) => {
        0 as i64
    };
    (REG_OPTION_VOLATILE) => {
        1 as i64
    };
    (REG_OPTION_CREATE_LINK) => {
        2 as i64
    };
    (REG_OPTION_BACKUP_RESTORE) => {
        4 as i64
    };
    (REG_OPTION_OPEN_LINK) => {
        8 as i64
    };
    (REG_CREATED_NEW_KEY) => {
        1 as i64
    };
    (REG_OPENED_EXISTING_KEY) => {
        2 as i64
    };
    (REG_WHOLE_HIVE_VOLATILE) => {
        1 as i64
    };
    (REG_REFRESH_HIVE) => {
        2 as i64
    };
    (REG_NO_LAZY_FLUSH) => {
        4 as i64
    };
    (REG_NOTIFY_CHANGE_NAME) => {
        1 as i64
    };
    (REG_NOTIFY_CHANGE_ATTRIBUTES) => {
        2 as i64
    };
    (REG_NOTIFY_CHANGE_LAST_SET) => {
        4 as i64
    };
    (REG_NOTIFY_CHANGE_SECURITY) => {
        8 as i64
    };
    (REG_LEGAL_CHANGE_FILTER) => {
        0x1000_000F as i64
    };
    (REG_LEGAL_OPTION) => {
        0x001F as i64
    };
    (ERROR_MORE_DATA) => {
        234 as i64
    };
}

pyre_interpreter::py_module! {
    "winreg",
    int_constants: {
        // Predefined root handles (winreg.h). Python surfaces these as the
        // unsigned handle value, so cast through u32.
        "HKEY_CLASSES_ROOT" => reg_int!(HKEY_CLASSES_ROOT),
        "HKEY_CURRENT_USER" => reg_int!(HKEY_CURRENT_USER),
        "HKEY_LOCAL_MACHINE" => reg_int!(HKEY_LOCAL_MACHINE),
        "HKEY_USERS" => reg_int!(HKEY_USERS),
        "HKEY_PERFORMANCE_DATA" => reg_int!(HKEY_PERFORMANCE_DATA),
        "HKEY_CURRENT_CONFIG" => reg_int!(HKEY_CURRENT_CONFIG),
        "HKEY_DYN_DATA" => reg_int!(HKEY_DYN_DATA),

        // Access-right masks (winnt.h).
        "KEY_QUERY_VALUE" => reg_int!(KEY_QUERY_VALUE),
        "KEY_SET_VALUE" => reg_int!(KEY_SET_VALUE),
        "KEY_CREATE_SUB_KEY" => reg_int!(KEY_CREATE_SUB_KEY),
        "KEY_ENUMERATE_SUB_KEYS" => reg_int!(KEY_ENUMERATE_SUB_KEYS),
        "KEY_NOTIFY" => reg_int!(KEY_NOTIFY),
        "KEY_CREATE_LINK" => reg_int!(KEY_CREATE_LINK),
        "KEY_WOW64_64KEY" => reg_int!(KEY_WOW64_64KEY),
        "KEY_WOW64_32KEY" => reg_int!(KEY_WOW64_32KEY),
        "KEY_WRITE" => reg_int!(KEY_WRITE),
        "KEY_READ" => reg_int!(KEY_READ),
        "KEY_EXECUTE" => reg_int!(KEY_EXECUTE),
        "KEY_ALL_ACCESS" => reg_int!(KEY_ALL_ACCESS),

        // Value types (winnt.h REG_*).
        "REG_NONE" => reg_int!(REG_NONE),
        "REG_SZ" => reg_int!(REG_SZ),
        "REG_EXPAND_SZ" => reg_int!(REG_EXPAND_SZ),
        "REG_BINARY" => reg_int!(REG_BINARY),
        "REG_DWORD" => reg_int!(REG_DWORD),
        "REG_DWORD_LITTLE_ENDIAN" => reg_int!(REG_DWORD_LITTLE_ENDIAN),
        "REG_DWORD_BIG_ENDIAN" => reg_int!(REG_DWORD_BIG_ENDIAN),
        "REG_LINK" => reg_int!(REG_LINK),
        "REG_MULTI_SZ" => reg_int!(REG_MULTI_SZ),
        "REG_RESOURCE_LIST" => reg_int!(REG_RESOURCE_LIST),
        "REG_FULL_RESOURCE_DESCRIPTOR" => reg_int!(REG_FULL_RESOURCE_DESCRIPTOR),
        "REG_RESOURCE_REQUIREMENTS_LIST" => reg_int!(REG_RESOURCE_REQUIREMENTS_LIST),
        "REG_QWORD" => reg_int!(REG_QWORD),
        "REG_QWORD_LITTLE_ENDIAN" => reg_int!(REG_QWORD_LITTLE_ENDIAN),

        // Key-creation options and disposition (winnt.h).
        "REG_OPTION_RESERVED" => reg_int!(REG_OPTION_RESERVED),
        "REG_OPTION_NON_VOLATILE" => reg_int!(REG_OPTION_NON_VOLATILE),
        "REG_OPTION_VOLATILE" => reg_int!(REG_OPTION_VOLATILE),
        "REG_OPTION_CREATE_LINK" => reg_int!(REG_OPTION_CREATE_LINK),
        "REG_OPTION_BACKUP_RESTORE" => reg_int!(REG_OPTION_BACKUP_RESTORE),
        "REG_OPTION_OPEN_LINK" => reg_int!(REG_OPTION_OPEN_LINK),
        "REG_CREATED_NEW_KEY" => reg_int!(REG_CREATED_NEW_KEY),
        "REG_OPENED_EXISTING_KEY" => reg_int!(REG_OPENED_EXISTING_KEY),
        "REG_WHOLE_HIVE_VOLATILE" => reg_int!(REG_WHOLE_HIVE_VOLATILE),
        "REG_REFRESH_HIVE" => reg_int!(REG_REFRESH_HIVE),
        "REG_NO_LAZY_FLUSH" => reg_int!(REG_NO_LAZY_FLUSH),

        // RegNotifyChangeKeyValue filters (winnt.h).
        "REG_NOTIFY_CHANGE_NAME" => reg_int!(REG_NOTIFY_CHANGE_NAME),
        "REG_NOTIFY_CHANGE_ATTRIBUTES" => reg_int!(REG_NOTIFY_CHANGE_ATTRIBUTES),
        "REG_NOTIFY_CHANGE_LAST_SET" => reg_int!(REG_NOTIFY_CHANGE_LAST_SET),
        "REG_NOTIFY_CHANGE_SECURITY" => reg_int!(REG_NOTIFY_CHANGE_SECURITY),
        // Both masks carry a bit whose own name `winreg` does not publish:
        // `REG_NOTIFY_THREAD_AGNOSTIC` in the filter, `REG_OPTION_DONT_VIRTUALIZE`
        // in the option word.
        "REG_LEGAL_CHANGE_FILTER" => reg_int!(REG_LEGAL_CHANGE_FILTER),
        "REG_LEGAL_OPTION" => reg_int!(REG_LEGAL_OPTION),

        // `rwinreg.constant_names` ends with the one Win32 error code the
        // module itself acts on, and `moduledef.py` publishes every name
        // there.
        "ERROR_MORE_DATA" => reg_int!(ERROR_MORE_DATA),
    },
    extra_init: |ns| {
        #[cfg(feature = "host_env")]
        interp_winreg::install(ns);
    }
}
