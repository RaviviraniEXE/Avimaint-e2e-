import type { ReactNode } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Database,
  LoaderCircle,
  ShieldAlert,
} from "lucide-react";
import { badgeLabel } from "../utils";

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  );
}

export function Section({
  title,
  description,
  action,
  children,
  className = "",
}: {
  title?: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || description || action) && (
        <div className="panel-heading">
          <div>
            {title && <h2>{title}</h2>}
            {description && <p>{description}</p>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

export function MetricCard({
  label,
  value,
  detail,
  icon,
}: {
  label: string;
  value: ReactNode;
  detail?: string;
  icon?: ReactNode;
}) {
  return (
    <article className="metric-card">
      <div className="metric-top">
        <span>{label}</span>
        {icon}
      </div>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </article>
  );
}

export function LoadingState({ label = "Loading evidence…" }: { label?: string }) {
  return (
    <div className="state-card" role="status">
      <LoaderCircle className="spin" size={25} />
      <div>
        <strong>{label}</strong>
        <p>The local service is preparing this view.</p>
      </div>
    </div>
  );
}

export function ErrorState({ message, retry }: { message: string; retry?: () => void }) {
  return (
    <div className="state-card error" role="alert">
      <AlertCircle size={25} />
      <div className="grow">
        <strong>This view could not be loaded</strong>
        <p>{message}</p>
      </div>
      {retry && (
        <button className="button secondary" onClick={retry} type="button">
          Try again
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="empty-state">
      <Database size={24} />
      <strong>{title}</strong>
      <p>{detail}</p>
    </div>
  );
}

export function Notice({
  kind = "info",
  title,
  children,
}: {
  kind?: "info" | "warning" | "success";
  title?: string;
  children: ReactNode;
}) {
  const Icon = kind === "warning" ? ShieldAlert : kind === "success" ? CheckCircle2 : AlertCircle;
  return (
    <div className={`notice ${kind}`}>
      <Icon size={20} />
      <div>
        {title && <strong>{title}</strong>}
        <div>{children}</div>
      </div>
    </div>
  );
}

export function EvidenceBadge({ badge }: { badge: string }) {
  return <span className={`evidence-badge ${badge}`}>{badgeLabel[badge] || badge}</span>;
}

export function Tag({ children, tone = "blue" }: { children: ReactNode; tone?: string }) {
  return <span className={`tag ${tone}`}>{children}</span>;
}

export function Definition({ term, children }: { term: string; children: ReactNode }) {
  return (
    <div className="definition">
      <dt>{term}</dt>
      <dd>{children}</dd>
    </div>
  );
}

export function SkeletonGrid() {
  return (
    <div className="metric-grid" aria-hidden="true">
      {[0, 1, 2, 3].map((item) => (
        <div className="metric-card skeleton" key={item} />
      ))}
    </div>
  );
}
