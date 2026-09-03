import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { GraphResponse } from "../api/types";
import { KnowledgeNetwork } from "../components/KnowledgeNetwork";
import { EmptyState, ErrorState, LoadingState, Notice, PageHeader, Section } from "../components/ui";

export function KnowledgeGraphPage() {
  const [focus, setFocus] = useState("");
  const [topComponents, setTopComponents] = useState(10);
  const [topFaults, setTopFaults] = useState(8);
  const [minEdge, setMinEdge] = useState(3);
  const [data, setData] = useState<GraphResponse | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    const timer = window.setTimeout(() => {
      setError("");
      api.graph({ topComponents, topFaults, minEdge: focus ? 1 : minEdge, focusComponent: focus })
        .then((value) => active && setData(value)).catch((reason) => active && setError(reason.message));
    }, 180);
    return () => { active = false; window.clearTimeout(timer); };
  }, [focus, topComponents, topFaults, minEdge]);

  return (
    <>
      <PageHeader eyebrow="Structured corpus view" title="Maintenance knowledge graph" description="Explore observed component–fault and fault–action co-occurrences. Select a node to isolate its direct historical connections." />
      <div className="filter-bar">
        <label><span>Focus component</span><select value={focus} onChange={(event) => setFocus(event.target.value)}><option value="">Whole corpus</option>{data?.component_options.map((item) => <option key={item}>{item}</option>)}</select></label>
        <label className={focus ? "disabled" : ""}><span>Components · {topComponents}</span><input type="range" min="5" max="20" value={topComponents} disabled={Boolean(focus)} onChange={(event) => setTopComponents(Number(event.target.value))} /></label>
        <label><span>Faults · {topFaults}</span><input type="range" min="4" max="15" value={topFaults} onChange={(event) => setTopFaults(Number(event.target.value))} /></label>
        <label className={focus ? "disabled" : ""}><span>Min. co-occurrence · {focus ? 1 : minEdge}</span><input type="range" min="1" max="20" value={minEdge} disabled={Boolean(focus)} onChange={(event) => setMinEdge(Number(event.target.value))} /></label>
      </div>
      <Notice><b>No causal claim:</b> an edge means two structured labels co-occurred in historical work orders. It does not prove that one caused or resolved the other.</Notice>
      {error && <ErrorState message={error} />}
      {!data && !error && <LoadingState label="Building the graph view…" />}
      {data && (
        <Section className="network-panel">
          {data.nodes.length ? <KnowledgeNetwork nodes={data.nodes} edges={data.edges} /> : <EmptyState title="No graph at this threshold" detail="Lower the minimum co-occurrence or remove the component filter." />}
        </Section>
      )}
    </>
  );
}
