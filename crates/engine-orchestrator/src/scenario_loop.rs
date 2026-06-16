//! Shared per-scenario iteration scaffolding for `find_power`,
//! `find_sample_size`, and the `single_core_*` variants.
//!
//! Hides the cancellation check, lifecycle events, and the borrow-checker
//! dance around moving the progress sink into `EngineSinkAdapter` and back.

use crate::cancel::CancellationToken;
use crate::progress::{ProgressEvent, ProgressSink};
use crate::result::{OrchestratorError, Scenario};

/// Run `op` per scenario, threading the progress sink through. `op` receives
/// `(idx, &Scenario, &mut Option<&mut dyn ProgressSink>)` and returns the
/// per-scenario payload `T`. Cancellation is checked before each call;
/// `ScenarioStarted`/`ScenarioCompleted` are emitted around the call.
///
/// `progress` is taken by `&mut` (not by value) so the caller can still emit
/// envelope events (`Started`, `Completed`) before and after this call.
pub(crate) fn for_each_scenario<T, F>(
    scenarios: &[Scenario],
    progress: &mut Option<&mut dyn ProgressSink>,
    cancel: &CancellationToken,
    fallback_n: Option<usize>,
    mut op: F,
) -> Result<Vec<(String, T)>, OrchestratorError>
where
    F: FnMut(usize, &Scenario, &mut Option<&mut dyn ProgressSink>) -> Result<T, OrchestratorError>,
{
    let total = scenarios.len();
    let mut out = Vec::with_capacity(total);
    for (idx, scenario) in scenarios.iter().enumerate() {
        if cancel.is_cancelled() {
            if let Some(p) = progress.as_deref_mut() {
                p.on_event(ProgressEvent::Cancelled);
            }
            return Err(OrchestratorError::Cancelled {
                scenario_idx: idx,
                n: fallback_n,
            });
        }
        if let Some(p) = progress.as_deref_mut() {
            p.on_event(ProgressEvent::ScenarioStarted {
                label: scenario.label.clone(),
                idx,
                total,
            });
        }
        let payload = op(idx, scenario, progress)?;
        if let Some(p) = progress.as_deref_mut() {
            p.on_event(ProgressEvent::ScenarioCompleted {
                label: scenario.label.clone(),
                idx,
            });
        }
        out.push((scenario.label.clone(), payload));
    }
    Ok(out)
}
