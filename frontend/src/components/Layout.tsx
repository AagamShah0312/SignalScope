import type { ReactNode } from "react";
import { Link, type Route } from "../router";
import { useBackendStatus } from "../useBackend";

const NAV: { to: Route; label: string }[] = [
  { to: "/analyze", label: "Analyze" },
  { to: "/how-it-works", label: "How it works" },
  { to: "/insights", label: "Insights" },
  { to: "/about", label: "About" },
];

function ModelStatus({ showLabel = false }: { showLabel?: boolean }) {
  const ok = useBackendStatus();
  if (ok === null) {
    return <span className="label">Connecting…</span>;
  }
  return (
    <span className="flex items-center gap-2">
      <span className={`status-dot ${ok ? "" : "off"}`} />
      <span className="label">
        {ok ? "Model online" : "Model offline"}
        {showLabel ? "" : ""}
      </span>
    </span>
  );
}

export function Layout({ route, children }: { route: Route; children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 border-b border-line bg-paper/85 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-5 py-4 sm:px-8">
          <Link to="/" className="group flex items-baseline gap-2">
            <span className="font-serif text-lg tracking-tight text-fg">
              Signal<span className="text-muted">Scope</span>
            </span>
            <span className="label hidden text-faint sm:inline">/ Image authenticity analysis</span>
          </Link>

          <nav className="hidden items-center gap-6 md:flex">
            {NAV.map((item) => (
              <Link
                key={item.to}
                to={item.to}
                className={`text-sm transition-colors ${
                  route === item.to
                    ? "text-fg"
                    : "text-muted hover:text-fg"
                }`}
              >
                {item.label}
              </Link>
            ))}
          </nav>

          <div className="flex items-center gap-3">
            <ModelStatus />
            <Link to="/analyze" className="btn-primary hidden !px-4 !py-2 text-xs sm:inline-flex">
              Analyze
            </Link>
          </div>
        </div>

        {/* Mobile nav */}
        <nav className="flex items-center gap-4 overflow-x-auto border-t border-line px-5 py-2 md:hidden">
          {NAV.map((item) => (
            <Link
              key={item.to}
              to={item.to}
              className={`whitespace-nowrap text-xs ${
                route === item.to ? "text-fg" : "text-muted"
              }`}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </header>

      <main className="flex-1">{children}</main>

      <footer className="border-t border-line">
        <div className="mx-auto max-w-6xl px-5 py-12 sm:px-8">
          <div className="flex flex-col gap-8 md:flex-row md:items-start md:justify-between">
            <div className="max-w-sm">
              <p className="font-serif text-lg text-fg">
                Signal<span className="text-muted">Scope</span>
              </p>
              <p className="mt-3 text-sm text-muted">
                Telling real from synthetic in the age of generative media. A
                confidence-based assessment with visual evidence — never a
                definitive accusation.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-10">
              <div>
                <p className="label mb-3">Product</p>
                <ul className="space-y-2 text-sm text-muted">
                  <li><Link to="/analyze" className="hover:text-fg">Analyze an image</Link></li>
                  <li><Link to="/how-it-works" className="hover:text-fg">How it works</Link></li>
                  <li><Link to="/insights" className="hover:text-fg">Model evidence</Link></li>
                </ul>
              </div>
              <div>
                <p className="label mb-3">Context</p>
                <ul className="space-y-2 text-sm text-muted">
                  <li><Link to="/about" className="hover:text-fg">About & limitations</Link></li>
                  <li><Link to="/" className="hover:text-fg">Overview</Link></li>
                </ul>
              </div>
            </div>
          </div>

          <div className="mt-12 flex flex-col gap-2 border-t border-line pt-6 sm:flex-row sm:items-center sm:justify-between">
            <p className="label">
              SignalScope · SIH 2026 · Problem statement 02
            </p>
            <p className="label text-faint">
              Not a provenance oracle — likelihood assessment only
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
