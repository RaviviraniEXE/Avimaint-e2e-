import { useMemo, useState } from "react";
import type { GraphEdge, GraphNode } from "../api/types";

const kindX = { component: 155, fault: 560, action: 965 } as const;
const color = { component: "#1769c2", fault: "#d65e68", action: "#15936c" } as const;

export function KnowledgeNetwork({ nodes, edges }: { nodes: GraphNode[]; edges: GraphEdge[] }) {
  const [selected, setSelected] = useState<string>("");
  const grouped = useMemo(() => ({
    component: nodes.filter((node) => node.kind === "component"),
    fault: nodes.filter((node) => node.kind === "fault"),
    action: nodes.filter((node) => node.kind === "action"),
  }), [nodes]);
  const maxRows = Math.max(1, grouped.component.length, grouped.fault.length, grouped.action.length);
  const height = Math.max(470, 90 + maxRows * 58);
  const positions = useMemo(() => {
    const out: Record<string, { x: number; y: number }> = {};
    (Object.keys(grouped) as Array<keyof typeof grouped>).forEach((kind) => {
      const list = grouped[kind];
      const spacing = Math.min(62, (height - 90) / Math.max(1, list.length));
      const start = (height - spacing * Math.max(0, list.length - 1)) / 2;
      list.forEach((node, index) => { out[node.id] = { x: kindX[kind], y: start + index * spacing }; });
    });
    return out;
  }, [grouped, height]);
  const neighbours = useMemo(() => new Set(
    edges.flatMap((edge) => edge.source === selected ? [edge.target] : edge.target === selected ? [edge.source] : []),
  ), [edges, selected]);
  const maxEdge = Math.max(1, ...edges.map((edge) => edge.count));

  const connected = (id: string) => !selected || id === selected || neighbours.has(id);
  return (
    <div className="network-wrap">
      <div className="network-legend">
        {(["component", "fault", "action"] as const).map((kind) => (
          <span key={kind}><i style={{ background: color[kind] }} />{kind === "action" ? "Action family" : kind}</span>
        ))}
        <small>Click a node to isolate its observed connections</small>
      </div>
      <svg viewBox={`0 0 1120 ${height}`} role="img" aria-label="Component, fault and action co-occurrence network">
        <text className="network-column-label" x={kindX.component} y={28}>COMPONENT</text>
        <text className="network-column-label" x={kindX.fault} y={28}>FAULT</text>
        <text className="network-column-label" x={kindX.action} y={28}>ACTION FAMILY</text>
        {edges.map((edge) => {
          const source = positions[edge.source];
          const target = positions[edge.target];
          if (!source || !target) return null;
          const active = !selected || edge.source === selected || edge.target === selected;
          return (
            <path
              d={`M ${source.x + 116} ${source.y} C ${source.x + 215} ${source.y}, ${target.x - 215} ${target.y}, ${target.x - 116} ${target.y}`}
              className={active ? "network-edge active" : "network-edge"}
              style={{ strokeWidth: 1 + (edge.count / maxEdge) * 7 }}
              key={`${edge.source}-${edge.target}`}
            />
          );
        })}
        {nodes.map((node) => {
          const point = positions[node.id];
          if (!point) return null;
          const visible = connected(node.id);
          const label = node.label.length > 25 ? `${node.label.slice(0, 23)}…` : node.label;
          return (
            <g
              key={node.id}
              className={`network-node ${visible ? "visible" : "muted"}`}
              onClick={() => setSelected(selected === node.id ? "" : node.id)}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => { if (event.key === "Enter") setSelected(selected === node.id ? "" : node.id); }}
            >
              <title>{`${node.label} · ${node.count} observed work orders`}</title>
              <rect x={point.x - 116} y={point.y - 19} width={232} height={38} rx={10} fill={color[node.kind]} />
              <text x={point.x - 101} y={point.y + 4}>{label}</text>
              <text className="network-count" x={point.x + 100} y={point.y + 4} textAnchor="end">{node.count}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
