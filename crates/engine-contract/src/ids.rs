//! `ColumnId` — stable, label-free integer identity for a column in `GenerationSpec`.

use serde::{Deserialize, Serialize};

/// Index into `GenerationSpec.columns` — a newtype so the coordinate cannot
/// be confused with design-term positions or uploaded-frame columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ColumnId(pub u32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_id_serializes_transparently_as_u32() {
        let id = ColumnId(7);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "7");
        let back: ColumnId = serde_json::from_str("7").unwrap();
        assert_eq!(back, id);
    }
}
