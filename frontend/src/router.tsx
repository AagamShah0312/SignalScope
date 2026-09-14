import { useEffect, useState, type ReactNode, type MouseEvent } from "react";

export type Route = "/" | "/analyze" | "/how-it-works" | "/about" | "/insights";

const ROUTES: Route[] = ["/", "/analyze", "/how-it-works", "/about", "/insights"];

export function getRoute(): Route {
  const raw = window.location.hash.replace(/^#/, "").split("?")[0];
  const path = (raw || "/") as Route;
  return ROUTES.includes(path) ? path : "/";
}

export function navigate(path: Route) {
  if (getRoute() === path) {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  window.location.hash = path === "/" ? "/" : path;
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(getRoute);

  useEffect(() => {
    const onChange = () => {
      setRoute(getRoute());
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  return route;
}

interface LinkProps {
  to: Route;
  children: ReactNode;
  className?: string;
  onClick?: (e: MouseEvent<HTMLAnchorElement>) => void;
  title?: string;
}

export function Link({ to, children, className, onClick, title }: LinkProps) {
  return (
    <a
      href={`#${to}`}
      className={className}
      title={title}
      onClick={onClick}
    >
      {children}
    </a>
  );
}
