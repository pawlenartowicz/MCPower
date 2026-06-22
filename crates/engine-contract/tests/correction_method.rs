use engine_contract::CorrectionMethod;

// T4 — unknown tag → clean Err; CorrectionMethod is rename_all="snake_case" plain string.
#[test]
fn correction_method_unknown_tag_returns_err() {
    let result = serde_json::from_str::<CorrectionMethod>(r#""sidak""#);
    assert!(result.is_err());
}

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
