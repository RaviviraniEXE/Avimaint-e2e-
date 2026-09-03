import { useEffect, useState } from "react";
import { Boxes, ClipboardCheck, Database, Layers3, Repeat2, Sparkles } from "lucide-react";
import { api } from "../api/client";
import type { OverviewResponse } from "../api/types";
import { HorizontalBars, OutcomeDonut } from "../components/Charts";
import { ErrorState, LoadingState, MetricCard, Notice, PageHeader, Section } from "../components/ui";
import { integer } from "../utils";

export function OverviewPage({ onDiagnose }: { onDiagnose: () => void }) {
  const [data, setData] = useState<OverviewResponse | null>(null);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);
  useEffect(() => {
    let active = true;
    setError("");
    api.overview().then((value) => active && setData(value)).catch((reason) => active && setError(reason.message));
    return () => { active = false; };
  }, [reload]);

  return (
    <>
      <PageHeader
        eyebrow="Fleet history, made inspectable"
        title="Maintenance evidence overview"
        description="A count-based view of the historical work-order library used for planning support. Values describe recorded events, not component reliability or failure rates."
        actions={<button className="button primary" type="button" onClick={onDiagnose}><Sparkles size={17} /> Diagnose a problem</button>}
      />
      {error && <ErrorState message={error} retry={() => setReload((value) => value + 1)} />}
      {!data && !error && <LoadingState label="Loading corpus overview…" />}
      {data && (
        <>
          <div className="metric-grid six">
            <MetricCard label="Historical work orders" value={integer.format(data.kpis.work_orders)} detail="Observed records" icon={<Database size={19} />} />
            <MetricCard label="Problem clusters" value={integer.format(data.kpis.problem_clusters)} detail="Independent evidence groups" icon={<Layers3 size={19} />} />
            <MetricCard label="Unique problems" value={integer.format(data.kpis.unique_problems)} detail="Normalized descriptions" icon={<ClipboardCheck size={19} />} />
            <MetricCard label="Components tracked" value={integer.format(data.kpis.components_tracked)} detail="Structured anchors" icon={<Boxes size={19} />} />
            <MetricCard label="Recurring clusters" value={integer.format(data.kpis.recurring_faults)} detail="At configured support gate" icon={<Repeat2 size={19} />} />
            <MetricCard label="Recorded outcomes" value={`${data.kpis.recorded_outcomes_pct.toFixed(1)}%`} detail="Known outcome labels" icon={<ClipboardCheck size={19} />} />
          </div>
          <Notice>
            <b>Interpretation boundary:</b> counts are descriptive. They do not establish reliability, causality, severity, airworthiness, or approval of a historical action.
          </Notice>
          <div className="two-column wide-left">
            <Section title="Most recurring problem clusters" description="Repeated normalized problem groups, ordered by observed work orders.">
              <div className="table-scroll">
                <table className="data-table">
                  <thead><tr><th>Historical problem</th><th>Component</th><th>Fault</th><th>Top action</th><th className="number">Orders</th></tr></thead>
                  <tbody>{data.top_recurring.map((item) => (
                    <tr key={item.cluster_id}>
                      <td><strong>{item.problem}</strong><small className="mono-cell">{item.cluster_id}</small></td>
                      <td>{item.component || "—"}</td><td>{item.fault || "—"}</td><td>{item.top_action || "—"}</td>
                      <td className="number"><b>{item.work_orders}</b></td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            </Section>
            <Section title="Recorded outcome mix" description="Positive, unknown, negative and mixed labels.">
              <OutcomeDonut data={data.outcome_mix} />
            </Section>
          </div>
          <div className="two-column">
            <Section title="Component occurrence" description="Components most often mentioned in the library.">
              <HorizontalBars data={data.component_frequency.slice(0, 10)} dataKey="component" color="#1769c2" height={360} />
            </Section>
            <Section title="Fault-mode occurrence" description="Most frequently extracted fault anchors.">
              <HorizontalBars data={data.fault_frequency.slice(0, 10)} dataKey="fault" color="#d65e68" height={360} />
            </Section>
          </div>
        </>
      )}
    </>
  );
}
