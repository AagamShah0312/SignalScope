import { useState } from "react";
import type { PredictResponse } from "../types";
import { ConfidenceBar } from "./ConfidenceBar";

type Tab = "original" | "heatmap" | "overlay";

function verdictColor(verdict: string): string {
  if (verdict === "likely_ai_generated") return "text-amber-400";
  if (verdict === "likely_real") return "text-emerald-400";
  return "text-slate-300";
}

export function ResultView({ result }: { result: PredictResponse }) {
  const [tab, setTab] = useState<Tab>("original");
  const vis = result.visualization;
  const tabs: { id: Tab; label: string; src?: string }[] = [
    { id: "original", label: "Original", src: vis?.original },
    { id: "heatmap", label: "Heatmap", src: vis?.heatmap },
    { id: "overlay", label: "Evidence Overlay", src: vis?.overlay },
  ];
  const active = tabs.find((t) => t.id === tab);

  return (
    <div className="grid gap-6 md:grid-cols-2">
      {/* Left: image + tabs */}
      <div className="rounded-xl bg-panel p-4">
        <div className="flex gap-2 mb-3">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                tab === t.id ? "bg-accent text-ink font-medium" : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        {active?.src ? (
          <img src={active.src} alt={`${active.label} visualization`} className="w-full rounded-lg border border-slate-700" />
        ) : (
          <div className="flex items-center justify-center h-64 text-slate-500">No visualization available</div>
        )}
      </div>

      {/* Right: verdict + explanation + provenance + robustness */}
      <div className="space-y-4">
        <div className="rounded-xl bg-panel p-4">
          <p className="text-xs uppercase tracking-wider text-slate-400 mb-1">Visual model verdict</p>
          <p className={`text-3xl font-bold ${verdictColor(result.verdict)}`}>{result.label}</p>
          <div className="mt-3">
            <ConfidenceBar probability={result.confidence} label="Confidence" />
          </div>
          <p className="mt-2 text-xs text-slate-500">
            P(AI-generated) = {(result.probability_ai * 100).toFixed(1)}% · threshold = {result.threshold}
          </p>
        </div>

        {result.explanation && (
          <div className="rounded-xl bg-panel p-4">
            <p className="text-xs uppercase tracking-wider text-slate-400 mb-2">Why?</p>
            <ul className="list-disc list-inside space-y-1 text-sm text-slate-200">
              {result.explanation.evidence.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>
            <p className="mt-3 text-sm text-slate-300">{result.explanation.summary}</p>
            <p className="mt-2 text-xs text-slate-500 italic">{result.explanation.uncertainty}</p>
          </div>
        )}

        {result.robustness?.included && (
          <div className="rounded-xl bg-panel p-4">
            <p className="text-xs uppercase tracking-wider text-slate-400 mb-2">Robustness check</p>
            <div className="grid grid-cols-2 gap-2 text-sm">
              {Object.entries(result.robustness.conditions)
                .filter(([, v]) => typeof v.probability_ai === "number")
                .slice(0, 6)
                .map(([name, v]) => (
                  <div key={name} className="flex justify-between bg-slate-800 rounded px-2 py-1">
                    <span className="text-slate-400">{name}</span>
                    <span className="text-slate-200">{((v.probability_ai ?? 0) * 100).toFixed(0)}%</span>
                  </div>
                ))}
            </div>
          </div>
        )}

        {result.provenance && (
          <div className="rounded-xl bg-panel p-4">
            <p className="text-xs uppercase tracking-wider text-slate-400 mb-2">Provenance signals</p>
            <div className="text-sm space-y-1">
              <p>
                <span className="text-slate-400">EXIF:</span>{" "}
                {result.provenance.exif.present ? "Present" : "Not detected"}
              </p>
              <p>
                <span className="text-slate-400">Content Credentials:</span>{" "}
                {result.provenance.c2pa.present ? "Present (unverified)" : "Not detected"}
              </p>
              <p className="text-xs text-slate-500">
                Missing metadata is not evidence of AI generation.
              </p>
            </div>
          </div>
        )}

        <p className="text-xs text-slate-500">{result.disclaimer}</p>
      </div>
    </div>
  );
}
