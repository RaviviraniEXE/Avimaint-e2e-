import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { FrequencyItem } from "../api/types";

const colors = ["#1769c2", "#32a4df", "#16a076", "#f2a341", "#d65e68", "#7967c9"];

function labelFor(item: FrequencyItem, key: "component" | "fault" | "action_family" | "outcome") {
  return item[key] || "Unknown";
}

export function HorizontalBars({
  data,
  dataKey,
  color = "#1769c2",
  height = 360,
}: {
  data: FrequencyItem[];
  dataKey: "component" | "fault" | "action_family";
  color?: string;
  height?: number;
}) {
  const values = data.map((item) => ({ ...item, label: labelFor(item, dataKey) })).reverse();
  return (
    <div style={{ height }} className="chart-frame">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={values} layout="vertical" margin={{ left: 8, right: 22, top: 8, bottom: 8 }}>
          <CartesianGrid stroke="#e9eef5" horizontal={false} />
          <XAxis type="number" axisLine={false} tickLine={false} tick={{ fill: "#617086", fontSize: 11 }} />
          <YAxis
            type="category"
            dataKey="label"
            width={132}
            axisLine={false}
            tickLine={false}
            tick={{ fill: "#34445a", fontSize: 11 }}
            tickFormatter={(value) => (String(value).length > 20 ? `${String(value).slice(0, 18)}…` : String(value))}
          />
          <Tooltip cursor={{ fill: "#f3f7fb" }} formatter={(value) => [Number(value).toLocaleString(), "Work orders"]} />
          <Bar dataKey="work_orders" fill={color} radius={[0, 5, 5, 0]} maxBarSize={24} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function OutcomeDonut({ data }: { data: FrequencyItem[] }) {
  const values = data.map((item) => ({ ...item, label: labelFor(item, "outcome") }));
  return (
    <div style={{ height: 310 }} className="chart-frame">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie data={values} dataKey="work_orders" nameKey="label" innerRadius={65} outerRadius={100} paddingAngle={2}>
            {values.map((_, index) => <Cell fill={colors[index % colors.length]} key={index} />)}
          </Pie>
          <Tooltip formatter={(value) => [Number(value).toLocaleString(), "Work orders"]} />
          <Legend iconType="circle" />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

export function GroupedHits({ problem, action }: { problem: number[]; action: number[] }) {
  const values = [1, 3, 5].map((depth, index) => ({
    depth: `@${depth}`,
    problem: problem[index] * 100,
    action: action[index] * 100,
  }));
  return (
    <div style={{ height: 330 }} className="chart-frame">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={values} margin={{ left: 0, right: 8, top: 18, bottom: 4 }}>
          <CartesianGrid stroke="#e9eef5" vertical={false} />
          <XAxis dataKey="depth" axisLine={false} tickLine={false} />
          <YAxis domain={[0, 100]} unit="%" axisLine={false} tickLine={false} width={42} />
          <Tooltip formatter={(value) => [`${Number(value).toFixed(1)}%`, ""]} />
          <Legend />
          <Bar name="Relevant problem found" dataKey="problem" fill="#1769c2" radius={[5, 5, 0, 0]} />
          <Bar name="Useful action evidence" dataKey="action" fill="#16a076" radius={[5, 5, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Matrix({ components, faults, values }: { components: string[]; faults: string[]; values: number[][] }) {
  const max = Math.max(1, ...values.flat());
  return (
    <div className="heatmap-scroll">
      <table className="heatmap">
        <thead>
          <tr><th>Component</th>{faults.map((fault) => <th key={fault}>{fault}</th>)}</tr>
        </thead>
        <tbody>
          {components.map((component, row) => (
            <tr key={component}>
              <th>{component}</th>
              {faults.map((fault, column) => {
                const value = values[row]?.[column] || 0;
                const opacity = value ? 0.12 + (value / max) * 0.78 : 0.03;
                return <td key={fault} style={{ backgroundColor: `rgba(23,105,194,${opacity})` }} title={`${component} · ${fault}: ${value}`}>{value}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
