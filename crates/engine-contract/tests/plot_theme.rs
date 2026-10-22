use std::{fs, path::Path};

#[test]
fn bundled_themes_are_the_four_expected_objects() {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/plot-themes.json");
    let bytes = fs::read(&p).unwrap_or_else(|e| panic!("read {p:?}: {e}"));
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes).expect("plot-themes.json must be a JSON object");
    let mut keys: Vec<&str> = map.keys().map(|k| k.as_str()).collect();
    keys.sort_unstable();
    let mut expected = ["light-print", "dark-print", "light-app", "dark-app"];
    expected.sort_unstable();
    assert_eq!(
        keys, expected,
        "expected exactly the four bundled theme keys"
    );
    for (name, theme) in &map {
        assert!(theme.is_object(), "theme {name:?} must be a JSON object");
    }
}
