import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { InsightsResponse } from "../api/types";
import { HorizontalBars, Matrix, OutcomeDonut } from "../components/Charts";
import { ErrorState, LoadingState, PageHeader, Section } from "../components/ui";

type Tab = "recurring" | "components" | "faults" | "actions" | "matrix" | "mapping" | "outcomes";

const tabs: Array<{ key: Tab; label: string }> = [
  { key: "recurring", label: "Recurring faults" },
  { key: "components", label: "Components" },
  { key: "faults", label: "Fault modes" },
  { key: "actions", label: "Actions" },
  { key: "matrix", label: "Component × Fault" },
  { key: "mapping", label: "Problem → Action" },
  { key: "outcomes", label: "Outcomes" },
];

export function InsightsPage() {
  const [tab, setTab] = useState<Tab>("recurring");
  const [component, setComponent] = useState("");
  const [data, setData] = useState<InsightsResponse | null>(null);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);
  useEffect(() => {
    let active = true;
    setError("");
    api.insights(component).then((value) => active && setData(value)).catch((reason) => active && setError(reason.message));
    return () => { active = false; };
  }, [component, reload]);

  return (
    <>
      <PageHeader eyebrow="Count-based fleet history" title="Recorded maintenance insights" description="Explore repeated problems, components, faults, action families, and outcomes. These are work-order occurrence counts—not failure or reliability rates." />
      <div className="tab-strip" role="tablist">
        {tabs.map((item) => <button type="button" role="tab" aria-selected={tab === item.key} className={tab === item.key ? "active" : ""} onClick={() => setTab(item.key)} key={item.key}>{item.label}</button>)}
      </div>
      {error && <ErrorState message={error} retry={() => setReload((value) => value + 1)} />}
      {!data && !error && <LoadingState label="Computing count-based insights…" />}
      {data && tab === "recurring" && (
        <Section title="Chronic-defect register" description="Historical problem clusters appearing at least the configured number of times.">
          <div className="table-scroll"><table className="data-table"><thead><tr><th>Problem</th><th>Component</th><th>Fault</th><th>Dominant action</th><th className="number">Orders</th><th className="number">Positive</th><th className="number">Negative / mixed</th></tr></thead><tbody>
            {data.recurring.map((item) => <tr key={item.cluster_id}><td><strong>{item.problem}</strong><small className="mono-cell">{item.cluster_id}</small></td><td>{item.component || "—"}</td><td>{item.fault || "—"}</td><td>{item.dominant_action || "—"}</td><td className="number">{item.work_orders}</td><td className="number positive-text">{item.positive_outcomes ?? 0}</td><td className="number negative-text">{item.negative_outcomes ?? 0}</td></tr>)}
          </tbody></table></div>
        </Section>
      )}
      {data && tab === "components" && <Section title="Component occurrence Pareto" description="The components represented most often in recorded work orders."><HorizontalBars data={data.components} dataKey="component" height={550} /></Section>}
      {data && tab === "faults" && <Section title="Fault-mode occurrence" description="Structured fault anchors observed most often."><HorizontalBars data={data.faults} dataKey="fault" color="#d65e68" height={550} /></Section>}
      {data && tab === "actions" && <Section title="Historical action-family occurrence" description="How often each grouped action family is recorded."><HorizontalBars data={data.actions} dataKey="action_family" color="#15936c" height={520} /></Section>}
      {data && tab === "matrix" && <Section title="Component × fault co-occurrence" description="Cells count work orders where the component and fault were both extracted."><Matrix {...data.matrix} /></Section>}
      {data && tab === "mapping" && (
        <Section title="Problem anchor → recorded action family" description="Filter the historical action distribution for one component.">
          <label className="field-label" htmlFor="component-insight">Component</label>
          <select id="component-insight" className="select" value={component} onChange={(event) => setComponent(event.target.value)}>
            <option value="">All components</option>{data.component_options.map((item) => <option key={item}>{item}</option>)}
          </select>
          <HorizontalBars data={data.component_actions} dataKey="action_family" color="#15936c" height={420} />
        </Section>
      )}
      {data && tab === "outcomes" && <Section title="Recorded outcome labels" description="Outcome labels preserved from historical evidence."><OutcomeDonut data={data.outcomes} /></Section>}
      {data && <p className="page-note">{data.note}</p>}
    </>
  );
}
