import { Link } from "../router";

const METRICS = [
  "ROC-AUC",
  "Macro-F1",
  "Accuracy",
  "False positive rate",
  "False negative rate",
];

const GENERATORS = ["DALL·E", "Midjourney", "SDXL", "Stable Diffusion"];

function Pending({ note }: { note?: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-faint">Awaiting final evaluation</span>
      {note && <span className="label text-faint">{note}</span>}
    </div>
  );
}

export function Insights() {
  return (
    <div className="mx-auto max-w-6xl px-5 py-14 sm:px-8">
      <span className="label">Model evidence</span>
      <h1 className="display mt-5 max-w-3xl text-4xl text-fg sm:text-5xl">
        Does it <em>generalize?</em>
      </h1>
      <p className="mt-5 max-w-2xl text-muted">
        Insights is reserved for measured project evidence. This interface
        intentionally does not invent benchmark values while evaluation is
        still being finalized.
      </p>

      <div className="mt-8 max-w-2xl border border-line bg-panel p-5">
        <p className="label text-accent">Evaluation data is configuration-ready</p>
        <p className="mt-2 text-sm text-muted">
          Connect verified measurements before presenting them in a demo. No
          placeholder statistics are shown as facts.
        </p>
      </div>

      {/* Overall performance */}
      <section className="mt-16">
        <span className="label">Overall performance</span>
        <h2 className="display mt-4 text-3xl text-fg">Evaluation metrics</h2>
        <p className="label mt-2 text-faint">Validation set / pending</p>

        <div className="mt-6 overflow-hidden rounded-lg border border-line">
          {METRICS.map((m, i) => (
            <div
              key={m}
              className={`flex items-center justify-between px-5 py-4 ${
                i % 2 === 0 ? "bg-paper" : "bg-panel"
              }`}
            >
              <span className="font-serif text-lg text-fg">{m}</span>
              <Pending />
            </div>
          ))}
        </div>

        <p className="mt-4 max-w-2xl text-xs leading-relaxed text-faint">
          A small real-photo adaptation set has been measured during
          development (see the project&rsquo;s real_photo_adaptation report),
          but final public-benchmark numbers are not available here and are not
          fabricated.
        </p>
      </section>

      {/* Unseen generators */}
      <section className="mt-16 grid gap-10 md:grid-cols-2">
        <div>
          <span className="label">Unseen generator evaluation</span>
          <h2 className="display mt-4 text-3xl text-fg">
            Beyond the training <em>distribution.</em>
          </h2>
          <p className="mt-4 text-sm leading-relaxed text-muted">
            A detector can perform well on familiar synthetic images while
            struggling with images produced by different generators. SignalScope
            evaluates generalisation on a separate generator distribution.
          </p>
        </div>
        <div className="overflow-hidden rounded-lg border border-line">
          {GENERATORS.map((g, i) => (
            <div
              key={g}
              className={`flex items-center justify-between px-5 py-4 ${
                i % 2 === 0 ? "bg-paper" : "bg-panel"
              }`}
            >
              <span className="text-sm text-fg">{g}</span>
              <Pending />
            </div>
          ))}
        </div>
      </section>

      {/* Confusion matrix */}
      <section className="mt-16">
        <span className="label">Classification view</span>
        <h2 className="display mt-4 text-3xl text-fg">Confusion matrix</h2>

        <div className="mt-6 grid w-full max-w-md grid-cols-[auto_1fr_1fr] gap-px overflow-hidden rounded-lg border border-line bg-line text-center">
          <div className="bg-panel p-3" />
          <div className="bg-panel p-3">
            <span className="label">Predicted real</span>
          </div>
          <div className="bg-panel p-3">
            <span className="label">Predicted AI</span>
          </div>

          <div className="flex items-center bg-panel px-4">
            <span className="label">Actual real</span>
          </div>
          <div className="bg-paper p-5 font-mono text-sm text-faint">—</div>
          <div className="bg-panel p-5 font-mono text-sm text-faint">—</div>

          <div className="flex items-center bg-panel px-4">
            <span className="label">Actual AI</span>
          </div>
          <div className="bg-panel p-5 font-mono text-sm text-faint">—</div>
          <div className="bg-paper p-5 font-mono text-sm text-faint">—</div>
        </div>

        <p className="mt-4 max-w-2xl text-xs leading-relaxed text-faint">
          A confusion matrix shows how often a model correctly and incorrectly
          classifies real and AI-generated images. Values are filled in once a
          verified held-out evaluation is available.
        </p>

        <Link to="/analyze" className="btn-ghost mt-8">
          Test an image
        </Link>
      </section>
    </div>
  );
}
