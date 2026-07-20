import { useMutation } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import {
  ApiError,
  api,
  type CalibrationBatch,
  type CalibrationCurve,
  type CalibrationFlow,
  type CalibrationLabel,
  type CalibrationRun,
} from "../api";

const FEATURE_LABELS: Record<string, string> = {
  flowduration: "Flow duration (µs)",
  totfwdpkt: "Forward packets",
  totbwdpkt: "Backward packets",
  flowbyts: "Bytes / second",
  flowpkts: "Packets / second",
  flowiatmean: "Mean inter-arrival (µs)",
  fwdpktlenmean: "Mean forward packet bytes",
  bwdpktlenmean: "Mean backward packet bytes",
};

const FEATURE_PRIORITY = Object.keys(FEATURE_LABELS);

function displayName(name: string) {
  if (FEATURE_LABELS[name]) return FEATURE_LABELS[name];
  return name.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function displayValue(value: number) {
  if (Math.abs(value) >= 10_000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
    return value.toExponential(2);
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: 3 });
}

function replaceFlow(batch: CalibrationBatch, flow: CalibrationFlow): CalibrationBatch {
  const flows = batch.flows.map((item) => (item.id === flow.id ? flow : item));
  return {
    ...batch,
    flows,
    n_labelled: flows.filter((item) => item.operator_label !== undefined).length,
  };
}

function message(error: unknown) {
  if (error instanceof ApiError && error.status === 404) {
    return "The calibration lab is disabled on this server. Set SENTINEL_API_ENABLE_CALIBRATION=true after building the local pack.";
  }
  return error instanceof Error ? error.message : "Calibration request failed.";
}

export function Calibrate() {
  const [batch, setBatch] = useState<CalibrationBatch | null>(null);
  const [revealed, setRevealed] = useState<CalibrationFlow | null>(null);
  const [run, setRun] = useState<CalibrationRun | null>(null);
  const [curve, setCurve] = useState<CalibrationCurve | null>(null);
  const [error, setError] = useState<string | null>(null);
  const startRequest = useMutation({ mutationFn: () => api.createCalibrationBatch(50) });
  const labelRequest = useMutation({
    mutationFn: ({ id, label }: { id: number; label: CalibrationLabel }) =>
      api.labelCalibrationFlow(id, label),
  });
  const simulateRequest = useMutation({
    mutationFn: (id: number) => api.simulateCalibrationLabel(id),
  });
  const retrainRequest = useMutation({
    mutationFn: (id: number) =>
      Promise.all([api.retrainCalibrationBatch(id), api.calibrationCurve()]),
  });
  const busy =
    startRequest.isPending ||
    labelRequest.isPending ||
    simulateRequest.isPending ||
    retrainRequest.isPending;

  const current = batch?.flows.find((flow) => flow.operator_label === undefined) ?? null;
  const featureRows = useMemo(
    () => {
      const features = current?.features ?? {};
      const selected = FEATURE_PRIORITY.filter((name) => name in features).map(
        (name) => [name, features[name]] as const,
      );
      return selected.length >= 6 ? selected.slice(0, 8) : Object.entries(features).slice(0, 8);
    },
    [current],
  );
  const accuracy = useMemo(() => {
    const labelled = batch?.flows.filter((flow) => flow.true_label !== undefined) ?? [];
    if (!labelled.length) return null;
    const correct = labelled.filter((flow) => flow.operator_label === flow.true_label).length;
    return correct / labelled.length;
  }, [batch]);

  async function start() {
    setError(null);
    try {
      const created = await startRequest.mutateAsync();
      setBatch(created);
      setRevealed(null);
      setRun(null);
      setCurve(null);
    } catch (cause) {
      setError(message(cause));
    }
  }

  async function label(labelValue: CalibrationLabel) {
    if (!current || !batch || busy) return;
    setError(null);
    try {
      const labelled = await labelRequest.mutateAsync({ id: current.id, label: labelValue });
      setBatch(replaceFlow(batch, labelled));
      setRevealed(labelled);
    } catch (cause) {
      setError(message(cause));
    }
  }

  async function simulateRemaining() {
    if (!batch || busy) return;
    setError(null);
    let updated = batch;
    try {
      for (const flow of batch.flows.filter((item) => item.operator_label === undefined)) {
        const labelled = await simulateRequest.mutateAsync(flow.id);
        updated = replaceFlow(updated, labelled);
        setBatch(updated);
      }
      setRevealed(null);
    } catch (cause) {
      setError(message(cause));
    }
  }

  async function retrain() {
    if (!batch || busy) return;
    setError(null);
    try {
      const [result, reference] = await retrainRequest.mutateAsync(batch.id);
      setRun(result);
      setCurve(reference);
      setBatch({ ...batch, status: "retrained", runs: [result, ...batch.runs] });
    } catch (cause) {
      setError(message(cause));
    }
  }

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (busy || revealed || !current) return;
      if (event.key.toLowerCase() === "b") void label("benign");
      if (event.key.toLowerCase() === "a") void label("attack");
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  if (!batch) {
    return (
      <main className="calibrate-intro">
        <section className="panel cal-hero">
          <span className="cal-kicker">CROSS-NETWORK ADAPTATION LAB</span>
          <h2>Teach the detector what traffic looks like here.</h2>
          <p>
            A model trained on one network can fail silently on another. Label 50 held-out
            CIC-IDS2018 flows, retrain with those examples, then measure the change on a test set
            the model never saw.
          </p>
          <div className="cal-guardrails">
            <span><b>01</b> Ground truth stays hidden while you decide.</span>
            <span><b>02</b> Selection never inspects target labels.</span>
            <span><b>03</b> Final recall is graded on a disjoint test split.</span>
          </div>
          <button className="cal-primary" disabled={busy} onClick={() => void start()}>
            {busy ? "Preparing frozen flow pack…" : "Start 50-flow calibration"}
          </button>
          {error && <p className="error cal-error">{error}</p>}
        </section>
      </main>
    );
  }

  if (run) {
    const familyRecall = run.metrics.per_family_recall ?? {};
    return (
      <main>
        <section className="panel cal-result">
          <span className="cal-kicker">HELD-OUT RESULT · {run.n_labels_used} LABELS</span>
          <h2>Calibration changed target-network recall</h2>
          <div className="cal-scoreboard">
            <div><span>Blind transfer</span><strong>{(run.recall_before * 100).toFixed(1)}%</strong></div>
            <i>→</i>
            <div className="after"><span>After calibration</span><strong>{(run.recall_after * 100).toFixed(1)}%</strong></div>
          </div>
          <div className="metric-row">
            <div className="metric"><span className="ml">Operator accuracy</span><div className="mv">{(run.operator_accuracy * 100).toFixed(0)}%</div><span className="mh">labels matched hidden truth</span></div>
            <div className="metric"><span className="ml">False-positive rate</span><div className="mv">{(run.fpr_after * 100).toFixed(2)}%</div><span className="mh">target-benign calibrated</span></div>
            <div className="metric"><span className="ml">ROC AUC</span><div className="mv">{run.auc_after.toFixed(3)}</div><span className="mh">held-out target traffic</span></div>
          </div>
          <div className="cal-family-grid">
            {Object.entries(familyRecall).map(([family, recall]) => (
              <div key={family}><span>{family}</span><b>{(recall * 100).toFixed(1)}%</b></div>
            ))}
          </div>
        </section>
        {curve && (
          <section className="panel">
            <h2>Recorded label-efficiency curve <small>{curve.strategy} selection · 5 seeds</small></h2>
            <div className="cal-curve">
              {curve.points.map((point) => (
                <div className="cal-curve-row" key={point.n}>
                  <span>{point.n} labels</span>
                  <div><i style={{ width: `${point.mean_recall * 100}%` }} /></div>
                  <b>{(point.mean_recall * 100).toFixed(1)}%</b>
                </div>
              ))}
            </div>
            <p className="hint">Reference results are the frozen multi-family experiment, not a re-grade of this operator session.</p>
          </section>
        )}
        <button onClick={() => void start()} disabled={busy}>Start another batch</button>
      </main>
    );
  }

  const complete = batch.n_labelled === batch.n_flows;
  return (
    <main>
      <div className="cal-progress-head">
        <div><span className="cal-kicker">BATCH {batch.id} · SCORE-STRATIFIED</span><h2>{complete ? "Labels ready" : "Classify the flow"}</h2></div>
        <b>{batch.n_labelled}<small> / {batch.n_flows}</small></b>
      </div>
      <div className="cal-progress"><i style={{ width: `${(batch.n_labelled / batch.n_flows) * 100}%` }} /></div>

      {complete ? (
        <section className="panel cal-ready">
          <span className="cal-kicker">FROZEN TEST SPLIT IS STILL UNTOUCHED</span>
          <h2>Ready to retrain with {batch.n_labelled} operator labels.</h2>
          <p>Your label accuracy is {accuracy === null ? "—" : `${(accuracy * 100).toFixed(0)}%`}. The next step fits a fresh source-plus-target model and grades it once.</p>
          <button className="cal-primary" onClick={() => void retrain()} disabled={busy}>
            {busy ? "Retraining and grading…" : "Retrain and reveal held-out result"}
          </button>
        </section>
      ) : revealed ? (
        <section className={`panel cal-reveal ${revealed.operator_label === revealed.true_label ? "correct" : "wrong"}`}>
          <span className="cal-kicker">GROUND TRUTH REVEALED</span>
          <h2>{revealed.operator_label === revealed.true_label ? "Correct classification" : "This one was misclassified"}</h2>
          <p>You chose <b>{revealed.operator_label}</b>. The frozen label is <b>{revealed.true_label}</b>.</p>
          <button className="cal-primary" onClick={() => setRevealed(null)}>Next flow</button>
        </section>
      ) : current ? (
        <section className="cal-flow-layout">
          <div className="panel cal-flow-card">
            <div className="cal-flow-title"><span>FLOW #{current.pool_row}</span><small>blind score {current.model_score.toFixed(3)}</small></div>
            <div className="cal-features">
              {featureRows.map(([name, value]) => (
                <div key={name}><span>{displayName(name)}</span><b>{displayValue(value)}</b></div>
              ))}
            </div>
            {Object.keys(current.features).length > featureRows.length && <p className="hint">Showing {featureRows.length} of {Object.keys(current.features).length} aligned flow features.</p>}
          </div>
          <aside className="panel cal-decision">
            <span className="cal-kicker">YOUR DECISION</span>
            <h2>Benign or attack?</h2>
            <p>Use only the flow evidence shown. Its hidden benchmark label is not sent to this screen.</p>
            <button disabled={busy} onClick={() => void label("benign")}><kbd>B</kbd> Benign</button>
            <button className="attack" disabled={busy} onClick={() => void label("attack")}><kbd>A</kbd> Attack</button>
            <small>Keyboard shortcuts B / A</small>
          </aside>
        </section>
      ) : null}

      {!complete && !revealed && (
        <div className="cal-skip">
          <button onClick={() => void simulateRemaining()} disabled={busy}>
            {busy ? "Applying simulation labels…" : "Simulate remaining labels"}
          </button>
          <span>Demo shortcut; the server applies hidden truth so a reviewer can preview the payoff.</span>
        </div>
      )}
      {error && <p className="error">{error}</p>}
    </main>
  );
}
