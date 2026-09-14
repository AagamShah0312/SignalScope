import { Link } from "../router";

const CAPABILITIES = [
  { tag: "CORE", title: "Real vs AI classification" },
  { tag: "BONUS A", title: "Faithful visual explanation" },
  { tag: "BONUS C", title: "Robustness analysis" },
  { tag: "BONUS F", title: "Deployable real-time interface" },
];

const LIMITATIONS = [
  "Performance can vary across image generators, image transformations, compression levels, and image domains.",
  "Grad-CAM highlights regions that contributed more strongly to the model's prediction. It is an explanation aid — not a forensic chain of custody.",
  "SignalScope should be treated as an analytical aid rather than definitive proof of image origin.",
];

export function About() {
  return (
    <div className="mx-auto max-w-6xl px-5 py-14 sm:px-8">
      <span className="label">About SignalScope</span>
      <h1 className="display mt-5 max-w-3xl text-4xl text-fg sm:text-5xl">
        Detection with <em>the brakes on.</em>
      </h1>
      <p className="mt-5 max-w-2xl text-muted">
        Built for responsible synthetic-media detection — with enough context to
        know what a model can and cannot tell you.
      </p>

      <div className="mt-10 max-w-3xl space-y-5 leading-relaxed text-muted">
        <p>
          SignalScope helps people understand whether an image shows signals
          associated with synthetic generation. It is designed to make a
          model&rsquo;s output clearer, not to turn uncertainty into certainty.
        </p>
        <p>
          The interface pairs a confidence-based verdict with a visual
          explanation. That pairing matters: a label without context is easy to
          overread, while an explanation without a clear limitation can create
          false confidence.
        </p>
        <p>
          Use SignalScope when you need a fast analytical read, then bring your
          own judgment, provenance, and source context to the decision.
        </p>
      </div>

      {/* Capabilities */}
      <section className="mt-20">
        <span className="label">Built for the SignalScope challenge</span>
        <h2 className="display mt-4 text-3xl text-fg">
          One product story, <em>four capabilities.</em>
        </h2>
        <div className="mt-8 grid gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-2">
          {CAPABILITIES.map((c) => (
            <div key={c.tag} className="bg-paper p-7">
              <p className="label text-accent">{c.tag}</p>
              <p className="mt-3 font-serif text-xl text-fg">{c.title}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Limitations */}
      <section className="mt-20">
        <span className="label">Important limitations</span>
        <h2 className="display mt-4 text-3xl text-fg">
          Read the result in <em>context.</em>
        </h2>
        <div className="mt-8 space-y-3">
          {LIMITATIONS.map((lim, i) => (
            <div key={i} className="flex gap-4 border border-line bg-panel p-5">
              <span className="font-mono text-xs text-faint">{String(i + 1).padStart(2, "0")}</span>
              <p className="text-sm leading-relaxed text-muted">{lim}</p>
            </div>
          ))}
        </div>
        <Link to="/analyze" className="btn-primary mt-8">
          Inspect an image
        </Link>
      </section>
    </div>
  );
}
