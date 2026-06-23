//! Byte-identity guard for the rand_philox extraction (the Phase 2 source swap).
//!
//! engine-core's RNG primitive — Philox4x32-10 plus the splitmix64 /
//! u32_to_unit_f32 leaf helpers — now comes from the `rand_philox` crate instead
//! of a local copy. Its output is the reproducibility contract: these pinned
//! values are the exact bytes the local copies produced, and `golden_rng.rs`
//! rides on them. A failure here means `rand_philox` diverged from the frozen
//! primitive — STOP and reconcile; do NOT regenerate goldens to match.
//!
//! The Philox vectors are the canonical Random123 known-answer vectors (re-homed
//! from the deleted src/philox.rs); the splitmix64 / u32_to_unit_f32 values are
//! pinned from the extracted crate.

use rand_philox::{philox4x32_10, splitmix64, u32_to_unit_f32};

#[test]
fn philox_known_answer_vectors() {
    assert_eq!(
        philox4x32_10([0, 0, 0, 0], [0, 0]),
        [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
    );
    assert_eq!(
        philox4x32_10([0xffff_ffff; 4], [0xffff_ffff, 0xffff_ffff]),
        [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
    );
    assert_eq!(
        philox4x32_10(
            [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
            [0xa409_3822, 0x299f_31d0]
        ),
        [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
    );
}

#[test]
fn splitmix64_pinned() {
    // 0 is the Mix13 finalizer's fixed point (0 → 0) — pinned deliberately.
    assert_eq!(splitmix64(0), 0x0000_0000_0000_0000);
    assert_eq!(splitmix64(1), 0x5692_161d_100b_05e5);
    assert_eq!(splitmix64(0x0123_4567_89ab_cdef), 0xb2c0_58e4_ebb5_112c);
    assert_eq!(splitmix64(u64::MAX), 0xb4d0_55fc_f2cb_bd7b);
}

#[test]
fn u32_to_unit_f32_pinned() {
    // Bit patterns (f32::to_bits) — the open-interval construction, exact.
    assert_eq!(u32_to_unit_f32(0).to_bits(), 0x3380_0000);
    assert_eq!(u32_to_unit_f32(0xffff_ffff).to_bits(), 0x3f7f_ffff);
    assert_eq!(u32_to_unit_f32(0x9e37_79b9).to_bits(), 0x3f1e_3779);
}
