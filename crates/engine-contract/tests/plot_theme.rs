use engine_contract::plot_theme::validate_plot_theme;
use std::{fs, path::Path};

#[test]
fn all_themes_validate() {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/plot-themes.json");
    let bytes = fs::read(&p).unwrap_or_else(|e| panic!("read {p:?}: {e}"));
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes).expect("plot-themes.json must be a JSON object");
    assert!(
        map.keys()
            .any(|k| ["light", "dark", "wild", "print"].contains(&k.as_str())),
        "expected the four bundled theme keys",
    );
    for (name, theme) in &map {
        let theme_bytes = serde_json::to_vec(theme).unwrap();
        validate_plot_theme(&theme_bytes).unwrap_or_else(|e| panic!("theme {name:?} failed: {e}"));
    }
}

#[test]
fn rejects_non_object_root() {
    let err = validate_plot_theme(b"42").unwrap_err();
    assert!(err.to_string().contains("plot theme"));
}

// Non-UTF-8 bytes and non-JSON UTF-8 text must both fail as
// InvalidPlotTheme (error-path; not a parse panic).
#[test]
fn rejects_non_json_and_non_utf8() {
    // Invalid UTF-8 byte sequence.
    let err = validate_plot_theme(&[0xff, 0xfe, 0x00]).unwrap_err();
    assert!(
        err.to_string().contains("plot theme"),
        "non-UTF-8 must be rejected as plot theme error"
    );
    // Well-formed UTF-8 that is not JSON.
    let err = validate_plot_theme(b"not json at all").unwrap_err();
    assert!(
        err.to_string().contains("plot theme"),
        "non-JSON text must be rejected as plot theme error"
    );
}
