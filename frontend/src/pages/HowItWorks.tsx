import { Link } from "../router";

const PIPELINE = [
  { n: "01", title: "Image input", desc: "One image enters the analysis flow." },
  { n: "02", title: "Preprocessing", desc: "The image is prepared for the classifier." },
  { n: "03", title: "Vision model", desc: "EfficientNet-B0 performs binary classification." },
  { n: "04", title: "Probability", desc: "The model returns a confidence-based likelihood." },
  { n: "05", title: "Grad-CAM", desc: "An explanation aid shows areas of model attention." },
  { n: "06", title: "Responsible result", desc: "A clear assessment with limitations in view." },
];

export function HowItWorks() {
  return (
    <div className="mx-auto max-w-6xl px-5 py-14 sm:px-8">
      <span className="label">The SignalScope pipeline</span>
      <h1 className="display mt-5 max-w-3xl text-4xl text-fg sm:text-5xl">
        From pixels to a <em>responsible assessment.</em>
      </h1>
      <p className="mt-5 max-w-2xl text-muted">
        SignalScope combines a transfer-learning classifier with a visual
        explanation layer so the result is easier to inspect and harder to
        overread.
      </p>

      {/* Pipeline */}
      <div className="mt-12 grid gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-2 lg:grid-cols-3">
        {PIPELINE.map((p) => (
          <div key={p.n} className="bg-paper p-6">
            <span className="font-mono text-xs text-accent">{p.n}</span>
            <p className="mt-3 font-serif text-lg text-fg">{p.title}</p>
            <p className="mt-1.5 text-sm text-muted">{p.desc}</p>
          </div>
        ))}
      </div>

      {/* Architecture */}
      <section className="mt-20">
        <span className="label">Model architecture</span>
        <h2 className="display mt-4 text-3xl text-fg">EfficientNet-B0</h2>
        <p className="mt-3 max-w-2xl text-muted">
          SignalScope uses a transfer-learning image classification model
          adapted for binary real-vs-AI image classification.
        </p>

        <div className="mt-6 flex flex-wrap gap-2">
          {["224 × 224 preprocessing", "Binary classification", "Grad-CAM explanation", "Soft-voting ensemble"].map(
            (chip) => (
              <span key={chip} className="rounded-full border border-line px-4 py-1.5 text-xs text-muted">
                {chip}
              </span>
            ),
          )}
        </div>

        <div className="mt-8 flex flex-col items-stretch gap-2 md:flex-row md:items-center">
          <div className="flex-1 border border-line bg-panel p-5 text-center">
            <p className="label text-faint">Image</p>
            <p className="mt-2 font-mono text-fg">224 × 224</p>
          </div>
          <div className="flex-1 border border-line bg-panel p-5 text-center">
            <p className="label text-faint">Backbone</p>
            <p className="mt-2 font-mono text-fg">EfficientNet-B0</p>
          </div>
          <div className="flex-1 border border-line bg-panel p-5 text-center">
            <p className="label text-faint">Output</p>
            <p className="mt-2 font-mono text-fg">Probability</p>
          </div>
          <div className="flex-1 border border-line bg-panel p-5 text-center">
            <p className="label text-faint">Explain</p>
            <p className="mt-2 font-mono text-fg">Result + Grad-CAM</p>
          </div>
        </div>
      </section>

      {/* Unseen generators */}
      <section className="mt-20 grid gap-10 border-t border-line pt-12 md:grid-cols-2">
        <div>
          <span className="label">Why unseen generators matter</span>
          <h2 className="display mt-4 text-3xl text-fg">
            Beyond the training <em>distribution.</em>
          </h2>
        </div>
        <div className="text-muted">
          <p className="leading-relaxed">
            A detector can perform well on familiar synthetic images while
            struggling with images produced by different generators. SignalScope
            therefore treats separate-generator evaluation as a key part of
            technical credibility.
          </p>
          <Link to="/insights" className="mt-4 inline-block text-sm text-fg underline underline-offset-4 hover:text-accent">
            Explore insights
          </Link>
        </div>
      </section>

      {/* Responsible output */}
      <section className="mt-16 rounded-lg border border-line bg-panel p-8">
        <span className="label">Responsible output</span>
        <p className="mt-4 max-w-3xl text-sm leading-relaxed text-muted">
          Grad-CAM highlights regions that contributed more strongly to the
          model&rsquo;s prediction. It is an explanation aid, not proof of
          manipulation or provenance.
        </p>
        <Link to="/analyze" className="btn-accent mt-6">
          Run the analysis flow
        </Link>
      </section>
    </div>
  );
}
