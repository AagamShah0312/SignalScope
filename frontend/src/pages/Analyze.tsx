import { useCallback, useRef, useState } from "react";
import { analyzeImage } from "../api";
import type { PredictResponse } from "../types";
import { ResultView } from "../components/ResultView";
import { useBackendStatus } from "../useBackend";

type Stage = "idle" | "dragging" | "processing" | "done" | "error";

const ACCEPTED = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
const MAX_BYTES = 15 * 1024 * 1024;

function StatusPill() {
  const ok = useBackendStatus();
  if (ok === null) return <span className="label">Connecting…</span>;
  return (
    <span className="flex items-center gap-2 rounded-full border border-line px-3 py-1">
      <span className={`status-dot ${ok ? "" : "off"}`} />
      <span className="label !text-fg">{ok ? "Model online" : "Model offline"}</span>
    </span>
  );
}

export function Analyze() {
  const [stage, setStage] = useState<Stage>("idle");
  const [fileName, setFileName] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const runAnalysis = useCallback(async (file: File) => {
    if (!ACCEPTED.includes(file.type)) {
      setError("Unsupported file type. Use JPG, PNG, WebP or BMP.");
      setStage("error");
      return;
    }
    if (file.size > MAX_BYTES) {
      setError("File is too large (maximum 15 MB).");
      setStage("error");
      return;
    }

    setFileName(file.name);
    setPreview(URL.createObjectURL(file));
    setStage("processing");
    setError("");

    try {
      const res = await analyzeImage(file);
      setResult(res);
      setStage("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed.");
      setStage("error");
    }
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files?.[0];
      if (file) runAnalysis(file);
    },
    [runAnalysis],
  );

  const reset = () => {
    setStage("idle");
    setResult(null);
    setPreview(null);
    setFileName("");
    setError("");
  };

  return (
    <div className="mx-auto max-w-6xl px-5 py-14 sm:px-8">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <span className="label">Analysis workspace · live</span>
        <StatusPill />
      </div>
      <h1 className="display mt-5 text-4xl text-fg sm:text-5xl">Analyze an image</h1>
      <p className="mt-4 max-w-2xl text-muted">
        Inspect the likelihood of synthetic generation with visual evidence
        from the model.
      </p>

      {/* Step 01 — select source */}
      {stage !== "done" && (
        <section className="mt-12">
          <div className="flex items-center gap-3">
            <span className="font-mono text-xs text-accent">01</span>
            <span className="label">Select source</span>
          </div>

          <div
            onDragOver={(e) => {
              e.preventDefault();
              setStage((s) => (s === "processing" ? s : "dragging"));
            }}
            onDragLeave={() => setStage((s) => (s === "dragging" ? "idle" : s))}
            onDrop={onDrop}
            onClick={() => stage !== "processing" && inputRef.current?.click()}
            className={`mt-5 cursor-pointer border transition-colors ${
              stage === "dragging"
                ? "border-accent bg-panel"
                : "border-line bg-panel hover:border-line-strong"
            }`}
          >
            <div className="flex items-center justify-between border-b border-line px-5 py-3">
              <span className="label">Input / Image</span>
              <span className="label text-faint">Max 15 MB</span>
            </div>

            <div className="flex flex-col items-center px-6 py-16 text-center">
              <input
                ref={inputRef}
                type="file"
                accept={ACCEPTED.join(",")}
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) runAnalysis(file);
                  e.target.value = "";
                }}
              />

              {stage === "processing" ? (
                <div className="flex flex-col items-center gap-4">
                  <span className="status-dot" />
                  <p className="font-serif text-xl text-fg">Analyzing {fileName}…</p>
                  <p className="label text-faint">Running the visual model</p>
                </div>
              ) : (
                <>
                  <p className="font-serif text-2xl text-fg">Drop an image to inspect</p>
                  <p className="mt-2 text-sm text-muted">or browse from your device</p>
                  <div className="mt-8 flex flex-wrap items-center justify-center gap-x-6 gap-y-2">
                    <span className="label">JPG · PNG · WEBP · BMP</span>
                    <span className="label text-faint">Local upload</span>
                  </div>
                </>
              )}
            </div>
          </div>

          {stage !== "processing" && (
            <div className="mt-5 flex flex-wrap items-center justify-between gap-4">
              <p className="text-sm text-muted">Start with one image</p>
              <button
                className="btn-accent"
                onClick={(e) => {
                  e.stopPropagation();
                  inputRef.current?.click();
                }}
              >
                Analyze image
              </button>
            </div>
          )}

          {stage === "error" && (
            <div className="mt-5 border border-red-900/60 bg-red-950/20 p-5">
              <p className="label !text-red-400">Analysis failed</p>
              <p className="mt-2 text-sm text-red-200">{error}</p>
              <button className="mt-3 text-sm text-fg underline underline-offset-4" onClick={reset}>
                Try again
              </button>
            </div>
          )}
        </section>
      )}

      {/* Step 02 — result */}
      {stage === "done" && result && (
        <section className="mt-12">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <span className="font-mono text-xs text-accent">02</span>
              <span className="label">Result</span>
            </div>
            <button className="btn-ghost !px-4 !py-2 text-xs" onClick={reset}>
              Analyze another image
            </button>
          </div>
          <div className="mt-5">
            <ResultView result={result} preview={preview} fileName={fileName} />
          </div>
        </section>
      )}

      {/* Note */}
      <section className="mt-16 border-t border-line pt-8">
        <span className="label">03 / Note</span>
        <p className="mt-4 max-w-3xl text-sm leading-relaxed text-muted">
          SignalScope provides a likelihood assessment, not definitive proof of
          image origin. Performance can vary across image domains and
          transformations.
        </p>
        <a href="#/about" className="mt-3 inline-block text-sm text-fg underline underline-offset-4 hover:text-accent">
          Learn about limitations
        </a>
      </section>
    </div>
  );
}
