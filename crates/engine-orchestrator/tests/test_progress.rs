use engine_orchestrator::{NoOpSink, ProgressEvent, ProgressSink};

#[derive(Default)]
struct Recorder {
    events: Vec<String>,
}

impl ProgressSink for Recorder {
    fn on_event(&mut self, e: ProgressEvent) {
        let tag = match e {
            ProgressEvent::Started { .. } => "Started",
            ProgressEvent::ScenarioStarted { .. } => "ScenarioStarted",
            ProgressEvent::SimsCompleted { .. } => "SimsCompleted",
            ProgressEvent::NPointCompleted { .. } => "NPointCompleted",
            ProgressEvent::ScenarioCompleted { .. } => "ScenarioCompleted",
            ProgressEvent::Cancelled => "Cancelled",
            ProgressEvent::Completed => "Completed",
        };
        self.events.push(tag.to_string());
    }
}

#[test]
fn noop_sink_accepts_all_variants() {
    let mut sink = NoOpSink;
    sink.on_event(ProgressEvent::Started {
        total_sims: 0,
        total_scenarios: 0,
        total_grid_points: 0,
    });
    sink.on_event(ProgressEvent::ScenarioStarted {
        label: "a".into(),
        idx: 0,
        total: 1,
    });
    sink.on_event(ProgressEvent::SimsCompleted {
        n: 100,
        completed: 1,
        total: 2,
    });
    sink.on_event(ProgressEvent::NPointCompleted {
        n: 100,
        power_uncorrected: vec![],
        power_corrected: vec![],
    });
    sink.on_event(ProgressEvent::ScenarioCompleted {
        label: "a".into(),
        idx: 0,
    });
    sink.on_event(ProgressEvent::Cancelled);
    sink.on_event(ProgressEvent::Completed);
}

#[test]
fn recorder_captures_event_ordering() {
    let mut r = Recorder::default();
    r.on_event(ProgressEvent::Started {
        total_sims: 10,
        total_scenarios: 1,
        total_grid_points: 1,
    });
    r.on_event(ProgressEvent::ScenarioStarted {
        label: "x".into(),
        idx: 0,
        total: 1,
    });
    r.on_event(ProgressEvent::Completed);
    assert_eq!(r.events, vec!["Started", "ScenarioStarted", "Completed"]);
}
