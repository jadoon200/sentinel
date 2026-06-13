const detectors = [
  ["Supervised LightGBM", "known attack families", "seen families ≈ 1.0", "ti-target"],
  ["Benign-only autoencoder", "unseen-family anomalies", "Infiltration 0.84 · DDoS 0.71", "ti-wave-sine"],
  ["Per-host sequence model", "slow web attacks", "XSS 1.00 · brute-force 0.94", "ti-timeline"],
  ["Host-profile fan-out", "scans & sweeps", "PortScan 0.998", "ti-affiliate"],
];

const fixes: [string, string, string, "bad" | "mid" | "good"][] = [
  ["baseline (train 2017)", "0.000", "0.93", "bad"],
  ["CORAL covariance alignment", "0.000", "0.56", "bad"],
  ["transfer-stable features", "0.000", "0.01", "bad"],
  ["target-trained autoencoder", "0.000", "0.81", "mid"],
  ["few-shot: +50 labelled flows", "1.000", "1.00", "good"],
];

export function ReportCard() {
  return (
    <>
      <section className="panel callout">
        <h2>
          <i className="ti ti-alert-triangle" aria-hidden="true" style={{ color: "var(--warn)" }} />
          &nbsp;The honest result
        </h2>
        <p>
          Trained on one network the IDS scores a perfect <b>1.0000</b> AUC. On a{" "}
          <i>different</i> network it still ranks attacks (0.940) but at any usable threshold it
          either floods you (100% recall at 23% false alarms) or goes blind (
          <b style={{ color: "var(--bad)" }}>0%</b> recall). The platform's value is that it knows
          exactly where its models break — because it measured it.
        </p>
      </section>

      <section className="panel">
        <h2>Can we beat the transfer failure? Four fixes, measured</h2>
        <p className="muted" style={{ marginTop: 0 }}>
          recall at a target-calibrated 1% false-positive rate, 2017 → 2018
        </p>
        {fixes.map(([name, recall, auc, kind]) => (
          <div key={name} className="fix-row">
            <span className="fix-name">{name}</span>
            <div className="fix-bar">
              <div className={`fix-fill fix-${kind}`} style={{ width: `${Number(recall) * 100}%` }} />
            </div>
            <span className={`fix-val fix-${kind}`}>recall {recall}</span>
            <span className="hint">AUC {auc}</span>
          </div>
        ))}
        <p style={{ marginBottom: 8 }}>
          The clever label-free tricks <b>failed</b> — alignment collapsed the model to chance,
          feature pruning made it worse, and a target-trained autoencoder couldn't clear a usable
          threshold either. What works: <b style={{ color: "var(--good)" }}>50 labelled flows</b>{" "}
          from the target network, verified on held-out data across three different attack families.
        </p>
        <div className="xfam">
          {[
            ["DoS", "0.05", "0.96"],
            ["Bot", "0.00", "0.99"],
            ["Brute-force", "0.00", "1.00"],
          ].map(([fam, before, after]) => (
            <div key={fam} className="xfam-row">
              <span className="xfam-name">{fam}</span>
              <span className="fix-bad">recall {before}</span>
              <i className="ti ti-arrow-right" aria-hidden="true" />
              <span className="fix-good">recall {after}</span>
              <span className="hint">with 50 labels</span>
            </div>
          ))}
        </div>
        <p style={{ marginBottom: 0 }}>
          Even Bot — where the blind 2017 model ranks <i>worse than a coin flip</i> (AUC 0.40) — is
          recovered to AUC 0.997. Cross-network transfer is a <i>few-shot</i> problem: the
          unsupervised detectors surface candidates, an analyst confirms ~50, the model adapts.
        </p>
      </section>

      <section className="panel">
        <h2>Four detectors, complementary by design</h2>
        {detectors.map(([name, role, score, icon]) => (
          <div key={name} className="roster-row">
            <i className={`ti ${icon}`} aria-hidden="true" />
            <div className="roster-main">
              <div>{name}</div>
              <div className="hint">catches {role}</div>
            </div>
            <div className="roster-score">{score}</div>
          </div>
        ))}
        <p className="muted" style={{ marginBottom: 0 }}>
          Technique mapper: zero-shot over 697 ATT&CK techniques, parent hit@5 0.690 on 10,411 TRAM
          sentences. Autoencoder backend: MLX, 3.3× faster than torch at recall parity (10 seeds).
        </p>
      </section>
    </>
  );
}
