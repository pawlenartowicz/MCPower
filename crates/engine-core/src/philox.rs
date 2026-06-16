//! Philox4x32-10 counter-based PRNG core (Salmon et al., SC'11; Random123).
//!
//! Deliberately self-contained: imports nothing from the rest of `engine-core`
//! and pulls in no `rand`/`rand_pcg`. This is the extraction-insurance
//! discipline for an eventual standalone synthetic-data library (see the RNG
//! overhaul spec, Architecture A). The output stream is part of the
//! reproducibility contract — the KAT below pins the algorithm; `golden_rng.rs`
//! pins the derived draws.

/// Round multiplier constants (Random123 PHILOX_M4x32_{0,1}).
const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
/// Per-round key bumps: golden ratio and √3−1 (Random123 PHILOX_W32_{0,1}).
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;

#[inline]
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let p = (a as u64) * (b as u64);
    ((p >> 32) as u32, p as u32) // (hi, lo)
}

#[inline]
fn round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo(PHILOX_M0, ctr[0]);
    let (hi1, lo1) = mulhilo(PHILOX_M1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

/// 10-round Philox4x32. Round 0 uses the original key; rounds 1..9 bump first
/// (canonical Random123 order — verified by the KAT below).
#[inline]
pub fn philox4x32_10(ctr: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    let mut c = round(ctr, key);
    for _ in 1..10 {
        key[0] = key[0].wrapping_add(PHILOX_W0);
        key[1] = key[1].wrapping_add(PHILOX_W1);
        c = round(c, key);
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    // Random123 v1.x published known-answer vectors for philox4x32_10. A wrong
    // round/bump order or constant fails the first line.
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
}
