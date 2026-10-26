//! The `key=value` line format shared by the LLBC fingerprint stamp and the
//! extraction driver's `--fingerprint` output.
//!
//! One format, two producers, two readers.  `scripts/llbc_extract.py` writes
//! the stamp beside each `.ullbc` and prints the same fields on stdout for
//! `--fingerprint`; `build.rs` reads both.  Keeping the parse here — rather
//! than open-coding it per reader — is deliberate: the fields are extended
//! from the Python side, and a reader that models the format loosely goes
//! quiet instead of failing when a field is added.

/// The host spelling `platform_info` in `scripts/llbc_extract.py` writes into
/// `platform=`. It is the only stamp field that separates hosts -- the rest
/// hash the tracked tree -- and it has to, because `cfg` gates which sources
/// charon walks, so artefacts from another host describe different types at
/// different offsets.
///
/// Windows collapses to one key on the producing side regardless of
/// architecture, and must collapse the same way here. `None` is a host the
/// engine refuses to extract on.
pub fn platform_key(target_os: &str, target_arch: &str) -> Option<&'static str> {
    Some(match (target_os, target_arch) {
        ("windows", _) => "windows",
        ("macos", "aarch64") => "darwin-arm64",
        ("macos", "x86_64") => "darwin-x86_64",
        ("linux", "aarch64") => "linux-aarch64",
        ("linux", "x86_64") => "linux-x86_64",
        _ => return None,
    })
}

/// Read one `key=value` line out of a `key=value`-per-line block.
///
/// `key` includes the `=`.  A prefix match is the whole parse.
pub fn stamp_field(text: &str, key: &str) -> Option<String> {
    text.lines()
        .find_map(|line| line.strip_prefix(key).map(str::to_string))
}

/// What the freshness check should do, given its two switches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreshnessMode {
    /// Do not check at all.
    Skip,
    /// Check; report a stale artefact as a warning.
    Warn,
    /// Check; report a stale artefact as an error and fail the build.
    Strict,
    /// The switches contradict each other; fail without checking.
    Refuse,
}

/// The outcome of reading both switches: what to do, and what to say.
#[derive(Debug, Clone)]
pub struct FreshnessPolicy {
    pub mode: FreshnessMode,
    /// Lines the caller must emit. Never silently empty when a switch was set
    /// to something unrecognised.
    pub diagnostics: Vec<String>,
}

/// One switch's raw value, classified. An unrecognised value must not be
/// silently treated as either enabled or disabled.
enum Switch {
    Unset,
    On,
    Off,
    Unrecognised,
}

fn classify_skip(value: Option<&str>) -> Switch {
    match value {
        None => Switch::Unset,
        Some("1") => Switch::On,
        Some(_) => Switch::Unrecognised,
    }
}

fn classify_strict(value: Option<&str>) -> Switch {
    match value {
        None => Switch::Unset,
        Some("1") => Switch::On,
        Some("0") => Switch::Off,
        Some(_) => Switch::Unrecognised,
    }
}

/// Decide the mode from the raw values of `PYRE_LLBC_STRICT` and
/// `PYRE_LLBC_SKIP_FINGERPRINT_CHECK`.
///
/// Strict checking is the default; `PYRE_LLBC_STRICT=0` deliberately demotes
/// stale artefacts to warnings. The skip switch takes exactly `1`. Unknown
/// values are diagnosed and retain the safe strict default.
///
/// Setting both is refused rather than resolved by precedence.  Honouring the
/// opt-out silently would tell a caller who asked for a hard gate that their
/// build passed; honouring strict would override an opt-out that exists so
/// offline, hermetic and vendored builds keep working.  Neither is a defensible
/// reading of "strict, and also do not check".
pub fn freshness_policy(strict: Option<&str>, skip: Option<&str>) -> FreshnessPolicy {
    const STRICT: &str = "PYRE_LLBC_STRICT";
    const SKIP: &str = "PYRE_LLBC_SKIP_FINGERPRINT_CHECK";

    let mut diagnostics = Vec::new();
    let mut note_unrecognised = |name: &str, value: Option<&str>, valid: &str| {
        if let Some(raw) = value {
            diagnostics.push(format!(
                "{name}={raw:?} is not a value this build understands, so it is \
                 being ignored; {valid}"
            ));
        }
    };

    let strict_switch = classify_strict(strict);
    if matches!(strict_switch, Switch::Unrecognised) {
        note_unrecognised(
            STRICT,
            strict,
            "use `0` to demote stale artefacts to warnings or `1` for strict mode",
        );
    }
    let skip_switch = classify_skip(skip);
    if matches!(skip_switch, Switch::Unrecognised) {
        note_unrecognised(SKIP, skip, "the only value that skips the check is `1`");
    }

    let mode = match (strict_switch, skip_switch) {
        (Switch::On, Switch::On) => FreshnessMode::Refuse,
        (_, Switch::On) => FreshnessMode::Skip,
        (Switch::Off, _) => FreshnessMode::Warn,
        (Switch::Unset | Switch::On | Switch::Unrecognised, _) => FreshnessMode::Strict,
    };

    match mode {
        FreshnessMode::Refuse => diagnostics.push(format!(
            "{STRICT}=1 and {SKIP}=1 are contradictory: the first asks for a \
             hard failure on a stale artefact, the second asks for no check at \
             all. Refusing rather than silently honouring one of them — unset \
             whichever you did not mean."
        )),
        // Announced, not silent: an opt-out that says nothing makes "did not
        // check" and "checked and fresh" the same absence of output, which is
        // the conflation the FRESHNESS UNKNOWN channel exists to prevent.
        FreshnessMode::Skip => diagnostics.push(format!(
            "{SKIP}=1 — the LLBC freshness check was skipped; field offsets \
             read out of target/llbc may not match the current sources"
        )),
        FreshnessMode::Warn | FreshnessMode::Strict => {}
    }

    FreshnessPolicy { mode, diagnostics }
}

/// Parse the source digest out of `--fingerprint`'s stdout.
///
/// The driver prints the same `key=value` lines it writes into the stamp, so
/// the field is parsed by name and the hex test is an assertion on the parsed
/// value — not the parser.  Reading the whole of stdout as the digest is what
/// broke when `external=` was added: a second line made the bare-sha256 test
/// fail, every crate collapsed to "the oracle did not answer", and the build
/// went quiet.
///
/// Deliberately no bare-digest fallback.  Accepting the pre-`external=` shape
/// would let this keep silently honouring a format the driver no longer emits,
/// which is the mechanism that produced that defect; the two versions travel
/// together in one repository, so refusing an unmodelled shape is the safe
/// direction.
///
/// One half of the answer.  A consumer deciding whether an artefact is current
/// wants [`parse_fingerprint_fields`], which reads the other half too.
pub fn parse_fingerprint_stdout(stdout: &str) -> Option<String> {
    let source = stamp_field(stdout, "source=")?;
    let is_hash = source.len() == 64 && source.bytes().all(|b| b.is_ascii_hexdigit());
    is_hash.then_some(source)
}

/// The fingerprint fields returned by the extraction oracle.
///
/// `source=` is the artefact-declared read set plus proc macros, build scripts
/// and manifests. `closure=` preserves the former conservative whole-closure
/// hash as a residual warning. `external=` covers closure inputs outside this
/// repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FingerprintFields {
    /// `source=`, validated as a bare sha256.
    pub source: String,
    /// `closure=`, validated as a bare sha256.
    pub closure: String,
    /// `external=` verbatim: one `<root label>=<digest>` per out-of-repo
    /// group, space-separated, and empty when the whole closure lives in this
    /// repository — the common case.  Kept opaque; this side does not model
    /// the encoding, only whether the value moved.
    pub external: String,
    /// `artefacts=` verbatim: one `<file name>=<size>:<sha256>` per artefact
    /// the crate writes, space-separated and `shlex`-quoted.  The only field
    /// here that describes an OUTPUT rather than an input: the other three
    /// answer "would a fresh extraction write different bytes", this one
    /// answers "are these still the bytes it wrote".  Kept opaque for the same
    /// reason as `external=`.
    pub artefacts: String,
}

/// Parse all required fields out of `--fingerprint`'s stdout.
///
/// `external=` and `artefacts=` are required rather than defaulted to empty.
/// The driver prints both on every invocation, so an output without one is a
/// shape this reader does not model, and defaulting would read "nothing
/// outside the repo" — or "the bytes are the ones extraction wrote" — off an
/// answer that never made that claim.  Their absence collapses to `None` — the
/// oracle did not answer — which is the same disposition an unparseable digest
/// gets, and which a caller reports rather than passes over in silence.
pub fn parse_fingerprint_fields(stdout: &str) -> Option<FingerprintFields> {
    let closure = stamp_field(stdout, "closure=")?;
    let closure_is_hash =
        closure.len() == 64 && closure.bytes().all(|byte| byte.is_ascii_hexdigit());
    Some(FingerprintFields {
        source: parse_fingerprint_stdout(stdout)?,
        closure: closure_is_hash.then_some(closure)?,
        external: stamp_field(stdout, "external=")?,
        artefacts: stamp_field(stdout, "artefacts=")?,
    })
}
