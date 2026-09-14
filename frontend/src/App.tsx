import { useCallback, useEffect, useRef, useState } from "react";
import { analyzeImage, checkHealth } from "./api";
import type { PredictResponse } from "./types";
import { ResultView } from "./components/ResultView";

type Stage = "idle" | "dragging" | "processing" | "done" | "error";

const ACCEPTED = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

export default function App() {
  const [stage, setStage] = useState<Stage>("idle");
  const [error, setError] = useState<string>("");
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    checkHealth()
      .then((h) => setBackendOk(h.model_loaded))
      .catch(() => setBackendOk(false));
  }, []);

  const runAnalysis = useCallback(async (file: File) => {
    if (!ACCEPTED.includes(file.type)) {
      setError("Unsupported file type. Use JPEG, PNG, WebP or BMP.");
      setStage("error");
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setError("File is too large (maximum 15 MB).");
      setStage("error");
      return;
    }

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
      setStage("idle");
      const file = e.dataTransfer.files?.[0];
      if (file) runAnalysis(file);
    },
    [runAnalysis],
  );

  return (
    <div className="min-h-screen px-4 py-10">
      <div className="max-w-5xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold tracking-tight text-white">
            SIGNAL<span className="text-accent">SCOPE</span>
          </h1>
          <p className="mt-2 text-slate-400">Telling Real From Synthetic</p>
          {backendOk === false && (
            <p className="mt-3 text-sm text-amber-400">
              Backend unavailable — start the API (see README) and refresh.
            </p>
          )}
        </header>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setStage("dragging");
          }}
          onDragLeave={() => setStage((s) => (s === "dragging" ? "idle" : s))}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          className={`cursor-pointer rounded-2xl border-2 border-dashed p-10 text-center transition-colors ${
            stage === "dragging"
              ? "border-accent bg-slate-800/40"
              : "border-slate-700 bg-panel hover:border-slate-500"
          }`}
        >
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
          <p className="text-lg text-slate-200">Drag and drop an image</p>
          <p className="text-sm text-slate-500 mt-1">or click to upload</p>
          {stage === "processing" && (
            <p className="mt-4 text-accent">Analyzing…</p>
          )}
        </div>

        {stage === "error" && (
          <div className="mt-6 rounded-xl border border-rose-700 bg-rose-950/40 p-4 text-rose-200">
            <p className="font-medium">Something went wrong</p>
            <p className="text-sm mt-1">{error}</p>
            <button
              className="mt-3 text-sm underline"
              onClick={() => {
                setStage("idle");
                setError("");
              }}
            >
              Try again
            </button>
          </div>
        )}

        {stage === "done" && result && (
          <div className="mt-8 space-y-6">
            {preview && (
              <div className="flex justify-center">
                <img src={preview} alt="Uploaded" className="max-h-48 rounded-lg border border-slate-700" />
              </div>
            )}
            <ResultView result={result} />
            <div className="text-center">
              <button
                className="rounded-lg border border-slate-600 px-5 py-2 text-sm text-slate-300 hover:bg-slate-800"
                onClick={() => {
                  setStage("idle");
                  setResult(null);
                  setPreview(null);
                }}
              >
                Analyze another image
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
