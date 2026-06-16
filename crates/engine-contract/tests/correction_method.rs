use engine_contract::CorrectionMethod;

#[test]
fn correction_method_is_alias_of_correction_with_snake_case_tags() {
    let cases = [
        (CorrectionMethod::None, "none"),
        (CorrectionMethod::Bonferroni, "bonferroni"),
        (CorrectionMethod::Holm, "holm"),
        (CorrectionMethod::BenjaminiHochberg, "benjamini_hochberg"),
        (CorrectionMethod::TukeyHsd, "tukey_hsd"),
    ];
    for (c, tag) in cases {
        let bytes = rmp_serde::to_vec_named(&c).unwrap();
        let back: CorrectionMethod = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(c, back);
        let found = bytes.windows(tag.len()).any(|w| w == tag.as_bytes());
        assert!(found, "msgpack must contain {tag:?}; bytes={bytes:?}");
    }
}
