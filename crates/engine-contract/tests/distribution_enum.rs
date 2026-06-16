use engine_contract::Distribution;

#[test]
fn distribution_serializes_snake_case() {
    let cases = [
        (Distribution::Normal, "normal"),
        (Distribution::Binary, "binary"),
        (Distribution::RightSkewed, "right_skewed"),
        (Distribution::LeftSkewed, "left_skewed"),
        (Distribution::HighKurtosis, "high_kurtosis"),
        (Distribution::Uniform, "uniform"),
        (Distribution::UploadedFactor, "uploaded_factor"),
        (Distribution::UploadedBinary, "uploaded_binary"),
        (Distribution::UploadedData, "uploaded_data"),
    ];
    for (d, expected_tag) in cases {
        let bytes = rmp_serde::to_vec_named(&d).unwrap();
        let back: Distribution = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(d, back);
        let found = bytes
            .windows(expected_tag.len())
            .any(|w| w == expected_tag.as_bytes());
        assert!(
            found,
            "msgpack must contain UTF-8 tag {expected_tag:?} for {d:?}; bytes={bytes:?}"
        );
    }
}
