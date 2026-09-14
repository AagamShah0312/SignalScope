import { useEffect, useState } from "react";
import { checkHealth } from "./api";

let cached: Promise<boolean> | null = null;

/** Whether the backend reports a loaded model. Cached across mounts. */
export function useBackendStatus(): boolean | null {
  const [ok, setOk] = useState<boolean | null>(null);

  useEffect(() => {
    if (!cached) {
      cached = checkHealth()
        .then((h) => h.model_loaded === true)
        .catch(() => false);
    }
    let alive = true;
    cached.then((v) => {
      if (alive) setOk(v);
    });
    return () => {
      alive = false;
    };
  }, []);

  return ok;
}
