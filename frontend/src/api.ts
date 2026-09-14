import type { PredictResponse } from "./types";

/**
 * API base URL.
 *
 * - Local dev: leave `VITE_API_BASE_URL` unset — the Vite dev server proxies
 *   `/predict`, `/health` and `/files` to the backend (see vite.config.ts).
 * - Production (Vercel): set `VITE_API_BASE_URL=https://<render-service>.onrender.com`
 *   so the browser calls the deployed backend directly (CORS is enabled there).
 */
const API_BASE =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/+$/, "") ?? "";

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

/**
 * Resolve a backend-relative asset URL (e.g. `/files/.../gradcam_overlay.jpg`)
 * into a fetchable URL, prefixing the API base in production. Absolute URLs
 * and blob: object URLs pass through untouched.
 */
export function resolveAssetUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  if (/^https?:\/\//i.test(path) || path.startsWith("blob:")) return path;
  return apiUrl(path);
}

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

  const response = await fetch(apiUrl(`/predict?${params.toString()}`), {
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
  const response = await fetch(apiUrl("/health"));
  if (!response.ok) throw new Error("Backend unavailable");
  return response.json();
}
