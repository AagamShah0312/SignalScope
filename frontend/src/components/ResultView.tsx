import { useState } from "react";
import type { PredictResponse } from "../types";
import { ConfidenceBar } from "./ConfidenceBar";

type Tab = "original" | "heatmap" | "overlay";

function verdictTone(verdict: string): string {
  if (verdict === "likely_ai_generated") return "text-ai";
  if (verdict === "likely_real") return "text-real";
  return "text-muted";
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-line bg-panel px-4 py-3">
      <p className="label text-faint">{label}</p>
      <p className="mt-1 font-mono text-xl text-fg">{value}</p>
    </div>
  );
}

export function ResultView({
  result,
  preview,
  fileName,
}: {
  result: PredictResponse;
  preview: string | null;
  fileName: string;
}) {
  const [tab, setTab] = useState<Tab>("overlay");
  const vis = result.visualization;

  const tabs: { id: Tab; label: string; src?: string | null }[] = [
    { id: "original", label: "Original", src: vis?.original ?? preview },
    { id: "heatmap", label: "Heatmap", src: vis?.heatmap },
    { id: "overlay", label: "Evidence overlay", src: vis?.overlay },
  ];
  const active = tabs.find((t) => t.id === tab);

  const pAi = (result.probability_ai * 100).toFixed(1);
  const pReal = (result.probability_real * 100).toFixed(1);
  const latency = result.latency_ms?.total_ms;

  const robustnessRows = result.robustness?.included
    ? Object.entries(result.robustness.conditions)
        .filter(([, v]) => typeof v.probability_ai === "number")
        .slice(0, 8)
    : [];

  return (
    <div className="space-y-6">
      {/* Verdict band */}
      <div className="grid gap-px overflow-hidden rounded-lg border border-line bg-line lg:grid-cols-[1fr_auto]">
        <div className="bg-panel p-7">
          <p className="label">Visual model verdict</p>
          <p className={`mt-3 font-serif text-4xl sm:text-5xl ${verdictTone(result.verdict)}`}>
            {result.label}
          </p>
          <p className="mt-2 font-mono text-xs text-faint">
            status: {result.status} · threshold: {result.threshold.toFixed(2)} ·{" "}
            {result.model.architecture}
          </p>
          <div className="mt-6 max-w-md">
            <ConfidenceBar value={result.confidence} label="Confidence" />
          </div>
        </div>
        <div className="grid grid-cols-2 bg-panel lg:grid-cols-1">
          <Stat label="P(AI-generated)" value={`${pAi}%`} />
          <Stat label="P(real)" value={`${pReal}%`} />
        </div>
      </div>

      {/* Evidence grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Left: image + tabs */}
        <div className="border border-line bg-panel">
          <div className="flex items-center gap-1 border-b border-line px-3 py-2">
            {tabs.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`px-3 py-1.5 font-mono text-xs uppercase tracking-wider transition-colors ${
                  tab === t.id
                    ? "bg-fg text-paper"
                    : "text-muted hover:bg-panel-2 hover:text-fg"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
          <div className="p-3">
            {active?.src ? (
              <img
                src={active.src}
                alt={`${active.label} visualization`}
                className="max-h-[460px] w-full rounded-sm border border-line object-contain"
              />
            ) : (
              <div className="flex h-64 items-center justify-center text-sm text-faint">
                No visualization available
              </div>
            )}
          </div>
          <div className="flex items-center justify-between border-t border-line px-4 py-2">
            <span className="label text-faint">Evidence map / Grad-CAM</span>
            <span className="label text-faint">{fileName}</span>
          </div>
        </div>

        {/* Right: explanation + robustness + provenance */}
        <div className="space-y-4">
          {result.explanation && (
            <div className="border border-line bg-panel p-6">
              <p className="label">Why this verdict</p>
              <p className="mt-3 text-sm leading-relaxed text-fg">
                {result.explanation.summary}
              </p>
              <ul className="mt-4 space-y-2">
                {result.explanation.evidence.map((e, i) => (
                  <li key={i} className="flex gap-3 text-sm text-muted">
                    <span className="font-mono text-xs text-accent">{String(i + 1).padStart(2, "0")}</span>
                    {e}
                  </li>
                ))}
              </ul>
              <p className="mt-4 border-t border-line pt-4 text-xs italic leading-relaxed text-faint">
                {result.explanation.uncertainty}
              </p>
            </div>
          )}

          {robustnessRows.length > 0 && (
            <div className="border border-line bg-panel p-6">
              <p className="label">Robustness check</p>
              <p className="mt-1 text-xs text-faint">
                P(AI) after realistic degradations of this image.
              </p>
              <div className="mt-4 grid grid-cols-2 gap-2">
                {robustnessRows.map(([name, v]) => (
                  <div
                    key={name}
                    className="flex items-center justify-between border border-line bg-paper px-3 py-2"
                  >
                    <span className="font-mono text-xs text-muted">{name}</span>
                    <span className="font-mono text-sm text-fg">
                      {((v.probability_ai ?? 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.provenance && (
            <div className="border border-line bg-panel p-6">
              <p className="label">Provenance signals</p>
              <div className="mt-4 space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-muted">EXIF metadata</span>
                  <span className="font-mono text-xs text-fg">
                    {result.provenance.exif.present ? "Present" : "Not detected"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted">Content credentials</span>
                  <span className="font-mono text-xs text-fg">
                    {result.provenance.c2pa.present ? "Present (unverified)" : "Not detected"}
                  </span>
                </div>
              </div>
              <p className="mt-4 border-t border-line pt-3 text-xs text-faint">
                Missing metadata is not evidence of AI generation.
              </p>
            </div>
          )}

          {typeof latency === "number" && (
            <p className="label text-faint">
              Inference {latency.toFixed(0)} ms · device CPU
            </p>
          )}

          <p className="text-xs leading-relaxed text-faint">{result.disclaimer}</p>
        </div>
      </div>
    </div>
  );
}
