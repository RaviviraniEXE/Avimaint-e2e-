import { useEffect, useState, type ComponentType, type ReactNode } from "react";
import {
  Activity,
  BarChart3,
  BookOpen,
  ClipboardList,
  Gauge,
  GitBranch,
  Menu,
  Search,
  ShieldCheck,
  X,
} from "lucide-react";
import type { HealthResponse } from "../api/types";

export type PageKey =
  | "overview"
  | "diagnose"
  | "insights"
  | "knowledge-graph"
  | "planning"
  | "evaluation"
  | "guide";

const nav: Array<{ key: PageKey; label: string; short: string; icon: ComponentType<{ size?: number }> }> = [
  { key: "overview", label: "Overview", short: "Overview", icon: Gauge },
  { key: "diagnose", label: "Diagnose", short: "Diagnose", icon: Search },
  { key: "insights", label: "Insights", short: "Insights", icon: BarChart3 },
  { key: "knowledge-graph", label: "Knowledge Graph", short: "Graph", icon: GitBranch },
  { key: "planning", label: "Planning", short: "Planning", icon: ClipboardList },
  { key: "evaluation", label: "Evaluation", short: "Evaluation", icon: Activity },
  { key: "guide", label: "Guide", short: "Guide", icon: BookOpen },
];

export function AppShell({
  page,
  onNavigate,
  health,
  children,
}: {
  page: PageKey;
  onNavigate: (page: PageKey) => void;
  health: HealthResponse | null;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const [healthOpen, setHealthOpen] = useState(false);
  useEffect(() => setOpen(false), [page]);

  const ready = health?.status === "ready" && health.frontend?.ready;
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="nav-frame">
          <button className="mobile-menu" type="button" onClick={() => setOpen(!open)} aria-label="Open navigation">
            {open ? <X size={22} /> : <Menu size={22} />}
          </button>
          <button className="brand" type="button" onClick={() => onNavigate("overview")}>
            <span className="brand-mark"><ShieldCheck size={22} /></span>
            <span>
              <b>AviMaint</b><em>DSS</em>
              <small>Evidence-grounded planning support</small>
            </span>
          </button>
          <nav className={`desktop-nav ${open ? "open" : ""}`} aria-label="Primary navigation">
            {nav.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  type="button"
                  key={item.key}
                  className={page === item.key ? "active" : ""}
                  onClick={() => onNavigate(item.key)}
                >
                  <Icon size={17} />
                  <span className="full-label">{item.label}</span>
                  <span className="short-label">{item.short}</span>
                </button>
              );
            })}
          </nav>
          <button
            className={`system-status ${ready ? "ready" : "degraded"}`}
            type="button"
            onClick={() => setHealthOpen(!healthOpen)}
            aria-expanded={healthOpen}
          >
            <span className="status-dot" />
            <span>{ready ? "System ready" : health ? "Degraded" : "Connecting"}</span>
          </button>
          {healthOpen && (
            <div className="health-popover">
              <div className="health-heading">
                <strong>Runtime services</strong>
                <span>API {health?.api_version || "…"}</span>
              </div>
              <HealthRow label="Validated raw SpERT" ready={health?.raw_spert.ready} />
              <HealthRow label="Guarded ByT5" ready={health?.normalization.ready} optional />
              <HealthRow label="Semantic SpERT" ready={health?.semantic_spert.ready} optional />
              <HealthRow label="RQ5 calibrator" ready={health?.rq5_calibrator.ready} />
              <HealthRow label="Phase 5 frontend" ready={health?.frontend.ready} />
              <p>Optional semantic services may fall back safely to the validated raw branch.</p>
            </div>
          )}
        </div>
      </header>

      <main className="main-content">{children}</main>

      <footer className="footer">
        <div>
          <strong>AviMaint-DSS v7.2 · Phase 5</strong>
          <span>Decision support only · Historical evidence is not approved maintenance data.</span>
        </div>
        <span>Verify all actions with qualified personnel and current authorised documentation.</span>
      </footer>
    </div>
  );
}

function HealthRow({ label, ready, optional = false }: { label: string; ready?: boolean; optional?: boolean }) {
  return (
    <div className="health-row">
      <span className={ready ? "ok" : optional ? "optional" : "bad"} />
      <b>{label}</b>
      <small>{ready ? "ready" : optional ? "fallback available" : "required"}</small>
    </div>
  );
}
