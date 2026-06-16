use engine_app_spec::progress::{EmitterSink, ProgressEmitter};
use engine_orchestrator::{ProgressEvent, ProgressSink};
use std::sync::Mutex;

struct RecordingEmitter(Mutex<Vec<serde_json::Value>>);
impl ProgressEmitter for RecordingEmitter {
    fn emit(&self, event: serde_json::Value) {
        self.0.lock().unwrap().push(event);
    }
}

fn record_one(event: ProgressEvent) -> serde_json::Value {
    let rec = RecordingEmitter(Mutex::new(vec![]));
    let mut sink = EmitterSink::new(&rec);
    sink.on_event(event);
    let mut events = rec.0.lock().unwrap();
    assert_eq!(events.len(), 1);
    events.pop().unwrap()
}

#[test]
fn started_forwards_with_totals() {
    let v = record_one(ProgressEvent::Started {
        total_sims: 1600,
        total_scenarios: 2,
        total_grid_points: 5,
    });
    assert_eq!(v["kind"], "started");
    assert_eq!(v["total_sims"], 1600);
    assert_eq!(v["total_scenarios"], 2);
    assert_eq!(v["total_grid_points"], 5);
}

#[test]
fn scenario_started_forwards_fields() {
    let v = record_one(ProgressEvent::ScenarioStarted {
        label: "my_scenario".into(),
        idx: 1,
        total: 3,
    });
    assert_eq!(v["kind"], "scenario_started");
    assert_eq!(v["label"], "my_scenario");
    assert_eq!(v["idx"], 1);
    assert_eq!(v["total"], 3);
}

#[test]
fn sims_completed_forwards_fields() {
    let v = record_one(ProgressEvent::SimsCompleted {
        n: 100,
        completed: 400,
        total: 1600,
    });
    assert_eq!(v["kind"], "sims_completed");
    assert_eq!(v["n"], 100);
    assert_eq!(v["completed"], 400);
    assert_eq!(v["total"], 1600);
}

#[test]
fn n_point_completed_forwards_fields() {
    let v = record_one(ProgressEvent::NPointCompleted {
        n: 50,
        power_uncorrected: vec![0.8, 0.75],
        power_corrected: vec![0.7, 0.65],
    });
    assert_eq!(v["kind"], "n_point_completed");
    assert_eq!(v["n"], 50);
    assert_eq!(v["power_uncorrected"][0], 0.8);
    assert_eq!(v["power_uncorrected"][1], 0.75);
    assert_eq!(v["power_corrected"][0], 0.7);
    assert_eq!(v["power_corrected"][1], 0.65);
}

#[test]
fn scenario_completed_forwards_fields() {
    let v = record_one(ProgressEvent::ScenarioCompleted {
        label: "done_scenario".into(),
        idx: 2,
    });
    assert_eq!(v["kind"], "scenario_completed");
    assert_eq!(v["label"], "done_scenario");
    assert_eq!(v["idx"], 2);
}

#[test]
fn cancelled_has_only_kind() {
    let v = record_one(ProgressEvent::Cancelled);
    assert_eq!(v["kind"], "cancelled");
    let obj = v.as_object().unwrap();
    assert_eq!(obj.len(), 1, "cancelled should have exactly one field");
}

#[test]
fn completed_has_only_kind() {
    let v = record_one(ProgressEvent::Completed);
    assert_eq!(v["kind"], "completed");
    let obj = v.as_object().unwrap();
    assert_eq!(obj.len(), 1, "completed should have exactly one field");
}

#[test]
fn every_variant_serializes_with_a_kind_field() {
    use engine_app_spec::progress::serialize_event;
    let samples = vec![
        ProgressEvent::Started {
            total_sims: 1,
            total_scenarios: 1,
            total_grid_points: 0,
        },
        ProgressEvent::ScenarioStarted {
            label: "s".into(),
            idx: 0,
            total: 1,
        },
        ProgressEvent::SimsCompleted {
            n: 10,
            completed: 5,
            total: 10,
        },
        ProgressEvent::NPointCompleted {
            n: 10,
            power_uncorrected: vec![0.5],
            power_corrected: vec![0.5],
        },
        ProgressEvent::ScenarioCompleted {
            label: "s".into(),
            idx: 0,
        },
        ProgressEvent::Cancelled,
        ProgressEvent::Completed,
    ];
    for ev in samples {
        let v = serialize_event(ev);
        assert!(
            v.get("kind").and_then(|k| k.as_str()).is_some(),
            "missing kind in {v:?}"
        );
    }
}
