import { useEffect, useState } from "react";
import { ClipboardList, Download, FileCheck2, Trash2 } from "lucide-react";
import { api } from "../api/client";
import type { JobCard, RecurringItem, WatchlistEntry } from "../api/types";
import { EmptyState, ErrorState, LoadingState, Notice, PageHeader, Section, Tag } from "../components/ui";
import { loadWatchlist, percent, saveWatchlist } from "../utils";

type PlanningTab = "watchlist" | "recurring";

export function PlanningPage({ onDiagnose }: { onDiagnose: () => void }) {
  const [tab, setTab] = useState<PlanningTab>("watchlist");
  const [watchlist, setWatchlist] = useState<WatchlistEntry[]>(loadWatchlist);
  const [recurring, setRecurring] = useState<RecurringItem[] | null>(null);
  const [error, setError] = useState("");
  const [card, setCard] = useState<{ data: JobCard; warning: string } | null>(null);
  const [cardBusy, setCardBusy] = useState("");

  useEffect(() => {
    const update = () => setWatchlist(loadWatchlist());
    window.addEventListener("avimaint-watchlist", update);
    return () => window.removeEventListener("avimaint-watchlist", update);
  }, []);
  useEffect(() => {
    let active = true;
    api.recurring().then((value) => active && setRecurring(value.items)).catch((reason) => active && setError(reason.message));
    return () => { active = false; };
  }, []);

  function remove(id: string) {
    const next = watchlist.filter((item) => item.id !== id);
    setWatchlist(next);
    saveWatchlist(next);
  }
  function exportWatchlist() {
    const blob = new Blob([JSON.stringify({ exported_at: new Date().toISOString(), warning: "Historical planning evidence only; verify against current approved maintenance data.", items: watchlist }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = "avimaint_planning_watchlist.json"; anchor.click();
    URL.revokeObjectURL(url);
  }
  async function buildCard(clusterId: string) {
    setCardBusy(clusterId); setError("");
    try {
      const value = await api.jobCard(clusterId);
      setCard({ data: value.card, warning: value.warning });
      window.setTimeout(() => document.getElementById("job-card")?.scrollIntoView({ behavior: "smooth" }), 30);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not build the historical job card.");
    } finally { setCardBusy(""); }
  }

  return (
    <>
      <PageHeader eyebrow="Traceable planning handoff" title="Planning support" description="Organise evidence promoted from Diagnose and inspect deterministic job-card drafts assembled only from recorded actions. Nothing here constitutes maintenance authorisation." actions={watchlist.length ? <button type="button" className="button secondary" onClick={exportWatchlist}><Download size={17} /> Export watchlist</button> : undefined} />
      <Notice kind="warning" title="Approval remains external">Every candidate action and job-card step must be checked against current approved maintenance data, aircraft applicability, and qualified maintenance authority.</Notice>
      <div className="tab-strip" role="tablist">
        <button type="button" role="tab" className={tab === "watchlist" ? "active" : ""} onClick={() => setTab("watchlist")}>Diagnosis watchlist <span>{watchlist.length}</span></button>
        <button type="button" role="tab" className={tab === "recurring" ? "active" : ""} onClick={() => setTab("recurring")}>Recurring clusters</button>
      </div>
      {error && <ErrorState message={error} />}
      {tab === "watchlist" && (
        <Section title="Promoted diagnosis evidence" description="Stored locally in this browser; each item keeps its evidence grade and support metadata.">
          {watchlist.length ? <div className="watchlist-grid">{watchlist.map((item) => (
            <article className="watchlist-card" key={item.id}>
              <div className="watchlist-top"><Tag>{item.evidenceGrade}</Tag><button type="button" title="Remove item" onClick={() => remove(item.id)}><Trash2 size={17} /></button></div>
              <small>{item.component || "Unspecified component"}</small><h3>{item.query}</h3><p>{item.recommendation}</p>
              <dl><div><dt>Independent support</dt><dd>{item.supportClusters} clusters</dd></div><div><dt>Anchor coverage</dt><dd>{percent(item.anchorCoverage)}</dd></div><div><dt>Historical agreement</dt><dd>{percent(item.agreement, 1)}</dd></div></dl>
            </article>
          ))}</div> : <EmptyState title="No diagnosis has been promoted" detail="Run Diagnose, review the supporting cases, then add an eligible result to this planning watchlist." />}
          {!watchlist.length && <div className="center-action"><button className="button primary" type="button" onClick={onDiagnose}><ClipboardList size={17} /> Open Diagnose</button></div>}
        </Section>
      )}
      {tab === "recurring" && (
        <Section title="Recurring historical problem clusters" description="Build a traceable draft from recorded non-negative action text for one cluster.">
          {!recurring && !error && <LoadingState label="Loading recurring clusters…" />}
          {recurring && <div className="table-scroll"><table className="data-table"><thead><tr><th>Problem</th><th>Component</th><th>Fault</th><th>Dominant action</th><th className="number">Orders</th><th /></tr></thead><tbody>
            {recurring.map((item) => <tr key={item.cluster_id}><td><strong>{item.problem}</strong><small className="mono-cell">{item.cluster_id}</small></td><td>{item.component || "—"}</td><td>{item.fault || "—"}</td><td>{item.dominant_action || "—"}</td><td className="number">{item.work_orders}</td><td className="number"><button type="button" className="button tiny" disabled={Boolean(cardBusy)} onClick={() => void buildCard(item.cluster_id)}>{cardBusy === item.cluster_id ? "Building…" : "Draft card"}</button></td></tr>)}
          </tbody></table></div>}
        </Section>
      )}
      {card && <JobCardView card={card.data} warning={card.warning} />}
    </>
  );
}

function JobCardView({ card, warning }: { card: JobCard; warning: string }) {
  return (
    <Section className="job-card" title="Source-grounded job-card draft" description="A planning handoff assembled from recorded historical actions; no new step is generated.">
      <article id="job-card">
        <div className="job-card-head"><FileCheck2 size={26} /><div><small>HISTORICAL DRAFT · NOT APPROVED DATA</small><h2>{card.title}</h2><p>{[card.component, card.fault].filter(Boolean).join(" · ")}</p></div></div>
        <div className="job-metrics"><span><b>{card.work_orders}</b> source work orders</span><span><b>{card.problem_groups}</b> problem groups</span><span><b>{card.dominant_action || "—"}</b> dominant family</span></div>
        <h3>Recorded action steps</h3>
        {card.steps.length ? <ol className="job-steps">{card.steps.map((step, index) => <li key={`${step.text}-${index}`}><span>{index + 1}</span><div><b>{step.text}</b><small>Sources: {step.source_idents.join(", ")}</small></div></li>)}</ol> : <p>No eligible non-negative recorded action steps were available.</p>}
        <div className="source-block"><b>Source record IDs</b><p>{card.source_idents.join(" · ") || "None"}</p></div>
        <Notice kind="warning">{warning}</Notice>
      </article>
    </Section>
  );
}
