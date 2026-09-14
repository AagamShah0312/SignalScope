export function ConfidenceBar({ probability, label }: { probability: number; label: string }) {
  const pct = Math.round(probability * 100);
  return (
    <div className="w-full">
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-2xl font-semibold text-white">{label}</span>
        <span className="text-2xl font-semibold text-accent">{pct}%</span>
      </div>
      <div className="h-3 w-full rounded-full bg-slate-800 overflow-hidden">
        <div
          className="h-full rounded-full bg-gradient-to-r from-accent-dim to-accent transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="mt-2 text-xs text-slate-400">
        Likelihood estimate based on visual evidence — not a definitive determination.
      </p>
    </div>
  );
}
