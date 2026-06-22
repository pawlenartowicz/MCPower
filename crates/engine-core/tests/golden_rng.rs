//! Golden bit-pattern tests for the reproducibility contract.
//!
//! The RNG streams (`SimRng`, `scenario_rng`) and the deterministic transform
//! surface (`phi`, `marginal_uniform`, `sample_t`) are part of the
//! reproducibility contract: equal versions must mean equal numbers in every
//! port. These tests compare checked-in bit patterns directly, so any change
//! to the RNG (Philox4x32-10, the f32 inverse-CDF draw structure) or the
//! transform algorithms fails loudly at the first diverging draw.
//!
//! A failure here means a result-moving change: it must land in all ports
//! simultaneously and carry a version bump — never as a silent upgrade. If
//! the change is deliberate, regenerate the constants from the new stream and
//! bump the version.
//!
//! Word widths track the draw precision: the f32 data-plane draws
//! (`next_normal`, `next_uniform`, `scenario_rng`) pin `f32::to_bits()` (`u32`);
//! the f64 transform plane (`phi`, `marginal_uniform`, `sample_t`) pins
//! `f64::to_bits()` (`u64`). The `phi`/`marginal_uniform` goldens move only
//! with a sanctioned reproducibility-contract change to the transform internals
//! (precedent: the 2026-06-11 owned-exp swap inside `erfc` — which left these
//! pinned values bit-identical); a movement outside such a change means
//! something widened or reordered that shouldn't have.
//!
//! The marginal/residual transforms that live inside `data_gen.rs`
//! (`apply_marginal`, `draw_residual`) are private; their golden coverage is
//! in `data_gen.rs`'s unit tests (same contract, same regeneration rule).

use engine_core::distributions::{marginal_uniform, phi, sample_t};
use engine_core::rng::SimRng;
use engine_core::scenarios::scenario_rng;

/// Fixed z-grid for the deterministic transforms. Includes the two critical
/// quantiles (±1.96, 1.645) used in CI/power calculations plus tail values.
const Z_GRID: [f64; 11] = [
    -8.0,
    -3.0,
    -1.959963985,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.644853627,
    3.0,
    8.0,
];

// f32 data-plane goldens (`f32::to_bits`).
#[rustfmt::skip]
const GOLDEN_NORMALS: [u32; 64] = [
    0x3ee0aaf7, 0xbf9bd3af, 0xbe8fe872, 0x3c52259d,
    0x3f5a4c8f, 0xbe42acbf, 0x3f94a0a1, 0x3f919220,
    0x3f21db97, 0xbebae28c, 0xbf666e70, 0xbf212633,
    0xbf07afd0, 0x3e93ab00, 0x3f497b6d, 0x3f580533,
    0x3f2a1545, 0x3f77ccd4, 0x3e5df031, 0xbf233e74,
    0xbf4e2c49, 0x3f2d70eb, 0xbf0ebcf1, 0xbc8fbb00,
    0x4017456d, 0xbdfc10cc, 0xbe91cf05, 0xbf89a347,
    0xbf8f61ed, 0xbf37fe3e, 0x3fdd5f54, 0x3fc74378,
    0xbe947755, 0xbf956272, 0x3e8845d7, 0xbf7d21ec,
    0x3eef82e3, 0xbf16200d, 0xbf2c026a, 0x3db8b3e7,
    0xbfaf58dd, 0x3f36bd95, 0x3f561f09, 0x3fbe3494,
    0x3e10a215, 0x3c6eec21, 0x3c52d79a, 0x3f696aa8,
    0xbeffbdfa, 0xbe94f7c8, 0x3fee0a89, 0x3e8b6932,
    0xbec15c27, 0xbc9d6965, 0x3f8ef6cd, 0x3f65ee6d,
    0x400c48e5, 0x3e0b88d3, 0xbfa03858, 0x3ee852ff,
    0x3eb49fdf, 0x3fe3ea1a, 0xbfaa9505, 0x3fa1f7f6,
];

#[rustfmt::skip]
const GOLDEN_UNIFORMS: [u32; 64] = [
    0x3f3ee253, 0x3f15ca37, 0x3e17fdc4, 0x3f0d2643,
    0x3f4d0d41, 0x3d99b4a8, 0x3f336785, 0x3f138a51,
    0x3e9e8e6a, 0x3ee283ca, 0x3f7947b5, 0x3f7ef17b,
    0x3f1885f7, 0x3e4be864, 0x3edca16a, 0x3ec560c2,
    0x3f0b676f, 0x3f79f4f3, 0x3ce1f9e0, 0x3eba69f6,
    0x3f0e5c8d, 0x3f30bde9, 0x3edb4a5e, 0x3f3ea2dd,
    0x3e562f44, 0x3f0ddfd9, 0x3f525dc3, 0x3f3d350b,
    0x3f3d414f, 0x3cb72a20, 0x3ec188fe, 0x3f672959,
    0x3c8c6fa0, 0x3f560b53, 0x3f37418f, 0x3e88f72e,
    0x3c47f540, 0x3ec9ff46, 0x3f4e2c15, 0x3f79f9e5,
    0x3d0d5eb0, 0x3f47b611, 0x3f0ee93b, 0x3f40cd83,
    0x3e5ba574, 0x3e2f44d4, 0x3f1bf5fb, 0x3f508325,
    0x3eeb9c82, 0x3f463a1b, 0x3f560599, 0x3e53f754,
    0x3ee6d1b6, 0x3efc0bc2, 0x3f39571f, 0x3db35df8,
    0x3f2c3f7f, 0x3d46f2f0, 0x3f1e0547, 0x3e2bf994,
    0x3eebfb42, 0x3e7e4a44, 0x3efa23ca, 0x3e290c44,
];

#[rustfmt::skip]
const GOLDEN_CATEGORICALS: [usize; 64] = [
    1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2,
    2, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2, 0, 2, 0,
    2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1,
    2, 2, 0, 2, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 2, 0,
];

#[rustfmt::skip]
const GOLDEN_SCENARIO_U32S: [u32; 64] = [
    0x3f30bc7f, 0x3e925de6, 0x3f31405b, 0x3e3f8784,
    0x3e0b7514, 0x3ef8610e, 0x3f77ab6d, 0x3f35d941,
    0x3ed7c87e, 0x3f50e9a5, 0x3f713381, 0x3ef6faa2,
    0x3dc76db8, 0x3e9b87ce, 0x3f04a03b, 0x3f0467c5,
    0x3e798564, 0x3e51ff8c, 0x3f6a6911, 0x3eaf5f0e,
    0x3d025710, 0x3e3b1fac, 0x3f70292d, 0x3eeef116,
    0x3f6f9f33, 0x3f43e9f1, 0x3e76a27c, 0x3f797ac5,
    0x3f5e8e1f, 0x3f7a111d, 0x3e888f8a, 0x3f20a225,
    0x3ed437ea, 0x3ed0efba, 0x3e4bc66c, 0x3f404253,
    0x3e8f310e, 0x3e820e86, 0x3eb99a72, 0x3f6fa6c3,
    0x3f41cb8f, 0x3f368633, 0x3bb09080, 0x3e9e5b7a,
    0x3f23efe7, 0x3e91a262, 0x3f3b8df1, 0x3df35f68,
    0x3ccc2720, 0x3e3172b4, 0x3e894346, 0x3ef95e4a,
    0x3ea1017a, 0x3f2f8649, 0x3ee1f8a6, 0x3f374ec9,
    0x3f4e2d9f, 0x3f50426f, 0x3f3e7549, 0x3eb5c822,
    0x3eac48ae, 0x3f15eccd, 0x3ee046a6, 0x3f4a07d5,
];

#[rustfmt::skip]
const GOLDEN_PHI: [u64; 11] = [
    0x3cc6000000000000, 0x3f561e2c54bb0e00, 0x3f999994f70d9320, 0x3fc44ed0d0bfc6c6,
    0x3fd3bf143a0c15bb, 0x3fdfffffff768fa1, 0x3fe62075e2f9f522, 0x3feaec4bcbd00e4e,
    0x3fee66666ddc3614, 0x3feff4f0e9d5a279, 0x3feffffffffffffa,
];

#[rustfmt::skip]
const GOLDEN_MARGINAL_UNIFORM: [u64; 11] = [
    0xbffbb67ae8584ca0, 0xbffba35352610c22, 0xbffa53c1d026c2fb, 0xbff2eb53ae7dfbdc,
    0xbfe5394e78ee11be, 0xbe1dc1a3c0000000, 0x3fe5394e78ee11bc, 0x3ff2eb53ae7dfbdc,
    0x3ff8f108446e8f44, 0x3ffba35352610c22, 0x3ffbb67ae8584ca0,
];

// sample_t returns f64 (draws widen), so its goldens stay `u64`; values moved
// because the draw stream changed.
#[rustfmt::skip]
const GOLDEN_SAMPLE_T_DF5: [u64; 16] = [
    0x3ff08753d4939991, 0xbffaa5dda54b0add, 0x3ffb47502099b951, 0x3fed39a4e44fcf8c,
    0xbf9830a9292f2cf1, 0xbfd211a4998b76fe, 0x3ff21d1e503bb16c, 0x3fbf36f7bc545ff6,
    0xbfe3499c7f01533f, 0xbfeaa69902d0c3ee, 0xbfe89a0c9cec9e2f, 0x4002930715335445,
    0x3fb6fcd6403005c1, 0xbfedf57796924cb3, 0xbfc2016bd67d9ef1, 0xbff2feafe0aa20f7,
];

#[rustfmt::skip]
const GOLDEN_SAMPLE_T_DF30: [u64; 16] = [
    0x3ff0be9cb272e81b, 0x3ff9f9d25966991b, 0xbfd990ba818253bd, 0x3fe26a8f167cee0f,
    0xbfea1bd7bf8403d5, 0x3fd39b8722a0c906, 0x3fa746cb333686d9, 0xbff99a20fd29d2e7,
    0xbf7d9d5ad17280de, 0xbfb37d12c5445632, 0x3fd762c17b04d45a, 0x3ff6a1b88d8816d9,
    0xbfd015f111f707c6, 0xbfd9b4e287bf21d8, 0xbff9c555558fb6b1, 0xbfec2565d20634f0,
];

/// Compare f64-bit draws, failing at the first divergence with the decoded f64.
fn assert_bits(name: &str, got: &[u64], want: &[u64]) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g,
            w,
            "{name}: first divergence at draw {i}: got {g:#018x} ({}), want {w:#018x} ({})",
            f64::from_bits(*g),
            f64::from_bits(*w),
        );
    }
}

/// f32 sibling of `assert_bits` for the data-plane draws (decoded as f32).
fn assert_bits_f32(name: &str, got: &[u32], want: &[u32]) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g,
            w,
            "{name}: first divergence at draw {i}: got {g:#010x} ({}), want {w:#010x} ({})",
            f32::from_bits(*g),
            f32::from_bits(*w),
        );
    }
}

#[test]
fn golden_next_normal() {
    let mut rng = SimRng::new(42, 0);
    let got: Vec<u32> = (0..64).map(|_| rng.next_normal().to_bits()).collect();
    assert_bits_f32("next_normal", &got, &GOLDEN_NORMALS);
}

#[test]
fn golden_next_uniform() {
    let mut rng = SimRng::new(42, 1);
    let got: Vec<u32> = (0..64).map(|_| rng.next_uniform().to_bits()).collect();
    assert_bits_f32("next_uniform", &got, &GOLDEN_UNIFORMS);
}

#[test]
fn golden_next_categorical() {
    let mut rng = SimRng::new(42, 2);
    let probs = [0.2, 0.3, 0.5];
    let got: Vec<usize> = (0..64).map(|_| rng.next_categorical(&probs)).collect();
    assert_eq!(got, GOLDEN_CATEGORICALS, "next_categorical stream moved");
}

#[test]
fn golden_scenario_rng() {
    // Pin the scenario stream through the public f32 uniform draw (raw Philox
    // words are already pinned by the Phase A KAT, so a raw-word pin is redundant).
    let mut rng = scenario_rng(42, 0);
    let got: Vec<u32> = (0..64).map(|_| rng.next_uniform().to_bits()).collect();
    assert_bits_f32("scenario_rng", &got, &GOLDEN_SCENARIO_U32S);
}

#[test]
fn golden_phi() {
    let got: Vec<u64> = Z_GRID.iter().map(|&z| phi(z).to_bits()).collect();
    assert_bits("phi", &got, &GOLDEN_PHI);
}

#[test]
fn golden_marginal_uniform() {
    // (a, b) = (-√3, √3) — the Uniform marginal's unit-variance bounds.
    let got: Vec<u64> = Z_GRID
        .iter()
        .map(|&z| marginal_uniform(z, -1.732_050_807_568_877_2, 1.732_050_807_568_877_2).to_bits())
        .collect();
    assert_bits("marginal_uniform", &got, &GOLDEN_MARGINAL_UNIFORM);
}

#[test]
fn golden_sample_t() {
    let mut rng = SimRng::new(42, 3);
    let got: Vec<u64> = (0..16).map(|_| sample_t(&mut rng, 5.0).to_bits()).collect();
    assert_bits("sample_t df=5", &got, &GOLDEN_SAMPLE_T_DF5);

    let mut rng = SimRng::new(42, 4);
    let got: Vec<u64> = (0..16)
        .map(|_| sample_t(&mut rng, 30.0).to_bits())
        .collect();
    assert_bits("sample_t df=30", &got, &GOLDEN_SAMPLE_T_DF30);
}
