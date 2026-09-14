import { Link } from "../router";

const CAPABILITIES = [
  { title: "Transfer learning", desc: "ImageNet-pretrained EfficientNet-B0 backbone." },
  { title: "Grad-CAM explanations", desc: "Visual evidence of model attention." },
  { title: "Robustness testing", desc: "JPEG · resize · blur · noise · crop." },
  { title: "Real-time analysis", desc: "CPU-friendly, sub-second inference." },
];

const STEPS = [
  { n: "01", title: "Upload", desc: "Provide one image in a familiar format. Keep the source in view from the first step." },
  { n: "02", title: "Analyze", desc: "A trained classifier evaluates visual patterns associated with synthetic generation." },
  { n: "03", title: "Inspect", desc: "Review likelihood, confidence, and a Grad-CAM explanation of model attention." },
];

function SignalMap() {
  // Decorative "instrument" panel — a faux waveform + coordinate readout.
  const bars = Array.from({ length: 64 }, (_, i) => {
    const v = 0.18 + 0.62 * Math.abs(Math.sin(i * 0.42) * Math.cos(i * 0.13));
    return (18 + v * 62).toFixed(0);
  });

  return (
    <section className="border-y border-line">
      <div className="mx-auto max-w-6xl px-5 py-16 sm:px-8">
        <div className="flex items-center justify-between">
          <span className="label">Signal map / 01</span>
          <span className="label text-faint">Visual features ready</span>
        </div>

        <div className="mt-6 grid-bg relative overflow-hidden rounded-lg border border-line">
          <div className="scanline absolute inset-0" />
          <div className="relative flex h-40 items-end gap-[3px] px-6 pb-6 pt-10">
            {bars.map((h, i) => (
              <div
                key={i}
                className="flex-1 rounded-t-sm bg-fg/25"
                style={{ height: `${h}px`, opacity: i % 7 === 0 ? 0.5 : 0.22 }}
              />
            ))}
          </div>
          <div className="relative flex items-center justify-between border-t border-line px-6 py-3">
            <span className="label">X 048.120 &nbsp; Y 19.774</span>
            <span className="label text-accent">Signal acquired</span>
          </div>
        </div>
      </div>
    </section>
  );
}

export function Home() {
  return (
    <>
      {/* Hero */}
      <section className="grid-bg border-b border-line">
        <div className="mx-auto max-w-6xl px-5 pb-20 pt-16 sm:px-8 sm:pt-24">
          <p className="label">Image authenticity analysis · 01 / 04</p>

          <h1 className="display mt-6 max-w-4xl text-5xl leading-[1.05] text-fg sm:text-6xl md:text-7xl">
            See the signals <em className="text-accent">behind an image.</em>
          </h1>

          <p className="mt-7 max-w-2xl text-lg leading-relaxed text-muted">
            SignalScope analyzes visual patterns associated with synthetic media
            and provides a confidence-based assessment with visual evidence.
          </p>

          <div className="mt-9 flex flex-wrap items-center gap-4">
            <Link to="/analyze" className="btn-primary">Analyze an image</Link>
            <Link to="/how-it-works" className="btn-ghost">How it works</Link>
          </div>

          <div className="mt-16 flex flex-wrap gap-x-10 gap-y-3 border-t border-line pt-6">
            {["Confidence-based", "Visual evidence", "Responsible by design"].map((chip) => (
              <span key={chip} className="label flex items-center gap-2">
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent" />
                {chip}
              </span>
            ))}
          </div>
        </div>
      </section>

      <SignalMap />

      {/* Capabilities */}
      <section className="border-b border-line">
        <div className="mx-auto max-w-6xl px-5 py-16 sm:px-8">
          <div className="flex items-center justify-between">
            <span className="label">System capabilities</span>
            <span className="label text-faint">02 / 04</span>
          </div>
          <div className="mt-8 grid gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-2 lg:grid-cols-4">
            {CAPABILITIES.map((c) => (
              <div key={c.title} className="bg-paper p-6">
                <p className="font-serif text-lg text-fg">{c.title}</p>
                <p className="mt-2 text-sm text-muted">{c.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Context */}
      <section className="border-b border-line">
        <div className="mx-auto max-w-6xl px-5 py-16 sm:px-8">
          <div className="grid gap-10 md:grid-cols-2">
            <div>
              <span className="label">02 / Context</span>
              <h2 className="display mt-4 text-3xl text-fg sm:text-4xl">
                Better tools start with <em>better questions.</em>
              </h2>
            </div>
            <div className="text-muted">
              <p className="text-base leading-relaxed">
                Synthetic media is getting harder to recognize. SignalScope is
                built to make an image model&rsquo;s assessment more legible —
                pairing a clear likelihood with the visual areas that shaped it.
              </p>
              <p className="mt-4 text-base leading-relaxed">
                A label without context is easy to overread. An explanation
                without limitations can create false confidence. SignalScope
                keeps both in view.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section>
        <div className="mx-auto max-w-6xl px-5 py-16 sm:px-8">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <span className="label">A clearer way to inspect</span>
              <h2 className="display mt-4 text-3xl text-fg sm:text-4xl">
                From upload to <em>understanding.</em>
              </h2>
            </div>
            <Link to="/about" className="btn-ghost">Our approach</Link>
          </div>

          <div className="mt-10 grid gap-px overflow-hidden rounded-lg border border-line bg-line md:grid-cols-3">
            {STEPS.map((s) => (
              <div key={s.n} className="bg-paper p-7">
                <span className="font-mono text-xs text-accent">{s.n}</span>
                <p className="mt-3 font-serif text-xl text-fg">{s.title}</p>
                <p className="mt-2 text-sm leading-relaxed text-muted">{s.desc}</p>
              </div>
            ))}
          </div>

          <div className="mt-8 flex flex-col gap-3 rounded-lg border border-line bg-panel p-6 sm:flex-row sm:items-start sm:justify-between">
            <p className="text-sm text-muted">
              <span className="font-medium text-fg">Not a provenance oracle.</span>{" "}
              SignalScope offers an analytical aid, not definitive proof of where
              an image came from.
            </p>
            <Link to="/about" className="whitespace-nowrap text-sm text-fg underline underline-offset-4 hover:text-accent">
              Read the limitations
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}
