import { lazy, Suspense, useEffect, useState } from "react";
import { AlertTriangle } from "lucide-react";
import { api } from "./api/client";
import type { HealthResponse } from "./api/types";
import { AppShell, type PageKey } from "./components/AppShell";
import { LoadingState } from "./components/ui";

const DiagnosePage = lazy(() => import("./pages/DiagnosePage").then((module) => ({ default: module.DiagnosePage })));
const EvaluationPage = lazy(() => import("./pages/EvaluationPage").then((module) => ({ default: module.EvaluationPage })));
const GuidePage = lazy(() => import("./pages/GuidePage").then((module) => ({ default: module.GuidePage })));
const InsightsPage = lazy(() => import("./pages/InsightsPage").then((module) => ({ default: module.InsightsPage })));
const KnowledgeGraphPage = lazy(() => import("./pages/KnowledgeGraphPage").then((module) => ({ default: module.KnowledgeGraphPage })));
const OverviewPage = lazy(() => import("./pages/OverviewPage").then((module) => ({ default: module.OverviewPage })));
const PlanningPage = lazy(() => import("./pages/PlanningPage").then((module) => ({ default: module.PlanningPage })));

const validPages = new Set<PageKey>([
  "overview", "diagnose", "insights", "knowledge-graph", "planning", "evaluation", "guide",
]);

function pageFromHash(): PageKey {
  const value = window.location.hash.replace(/^#\/?/, "") as PageKey;
  return validPages.has(value) ? value : "overview";
}

export default function App() {
  const [page, setPage] = useState<PageKey>(pageFromHash);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    if (!window.location.hash) window.history.replaceState(null, "", "#/overview");
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => {
    let alive = true;
    const refresh = () => api.health().then((value) => alive && setHealth(value)).catch(() => alive && setHealth(null));
    refresh();
    const timer = window.setInterval(refresh, 15000);
    return () => { alive = false; window.clearInterval(timer); };
  }, []);

  function navigate(next: PageKey) {
    if (next === page) window.scrollTo({ top: 0, behavior: "smooth" });
    window.location.hash = `/${next}`;
  }

  return (
    <AppShell page={page} onNavigate={navigate} health={health}>
      {health && health.status !== "ready" && (
        <div className="global-degraded" role="alert"><AlertTriangle size={18} /><span><b>Backend degraded.</b> A required model or frozen calibrator is unavailable. Inspect “Degraded” in the top bar before diagnosing.</span></div>
      )}
      <Suspense fallback={<LoadingState label="Loading interface…" />}>
        {page === "overview" && <OverviewPage onDiagnose={() => navigate("diagnose")} />}
        {page === "diagnose" && <DiagnosePage />}
        {page === "insights" && <InsightsPage />}
        {page === "knowledge-graph" && <KnowledgeGraphPage />}
        {page === "planning" && <PlanningPage onDiagnose={() => navigate("diagnose")} />}
        {page === "evaluation" && <EvaluationPage />}
        {page === "guide" && <GuidePage />}
      </Suspense>
    </AppShell>
  );
}
