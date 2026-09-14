import type { PredictResponse } from "./types";

export async function analyzeImage(
  file: File,
  options: { explanation?: boolean; metadata?: boolean; robustness?: boolean } = {},
): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);

  const params = new URLSearchParams({
    include_explanation: String(options.explanation ?? true),
    include_metadata: String(options.metadata ?? true),
    include_robustness: String(options.robustness ?? true),
  });

  const response = await fetch(`/predict?${params.toString()}`, {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }

  return (await response.json()) as PredictResponse;
}

export async function checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
  const response = await fetch("/health");
  if (!response.ok) throw new Error("Backend unavailable");
  return response.json();
}
