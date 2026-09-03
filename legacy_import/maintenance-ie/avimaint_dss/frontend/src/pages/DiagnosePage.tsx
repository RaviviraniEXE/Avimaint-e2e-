import { useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  Braces,
  Check,
  ClipboardPlus,
  FileSearch,
  History,
  Plane,
  Search,
  ShieldCheck,
} from "lucide-react";
import { api } from "../api/client";
import type {
  CaseEvidence,
  DiagnoseResponse,
  Entity,
  Recommendation,
  Relation,
  Strategy,
} from "../api/types";
import {
  Definition,
  EmptyState,
  ErrorState,
  EvidenceBadge,
  Notice,
  PageHeader,
  Section,
  Tag,
} from "../components/ui";
import {
  badgeLabel,
  getEntityText,
  getEntityType,
  loadWatchlist,
  percent,
  recommendationToWatchlist,
  saveWatchlist,
  score,
} from "../utils";

const examples = [
  { label: "Strong evidence", query: "#2 & 4 INTAKES LEAKING." },
  { label: "Diagnostic evidence", query: "L/H ENG #2 CYL HAS LOW COMPRESSION." },
  { label: "Mixed outcome", query: "ENGINE OVER RIDE SPEED WAS 1225 RPM." },
  { label: "Compound problem", query: "#2 & 4 INTAKES LEAKING. ALL ROCKER COVER SCREWS LOOSE." },
  { label: "Weak / unseen", query: "NOISE AND INTERMITTENT VIBRATION FROM AN UNIDENTIFIED ENGINE AREA." },
];

export function DiagnosePage() {
  const [query, setQuery] = useState("#2 & 4 INTAKES LEAKING.");
  const [response, setResponse] = useState<DiagnoseResponse | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event?: React.FormEvent) {
    event?.preventDefault();
    if (!query.trim() || busy) return;
    setBusy(true);
    setError("");
    try {
      setResponse(await api.diagnose(query.trim()));
    } catch (reason) {
      setResponse(null);
      setError(reason instanceof Error ? reason.message : "Diagnosis failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader eyebrow="Problem-only retrieval" title="Diagnose with historical evidence" description="Describe the maintenance problem. AviMaint-DSS retrieves cluster-distinct historical cases, preserves alternatives and negative outcomes, and abstains when the evidence gate is not met." />
      <form className="diagnose-form" onSubmit={submit}>
        <div className="query-box">
          <Plane size={23} />
          <textarea aria-label="Maintenance problem" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="e.g. LEFT MAGNETO RPM DROP ON RUN UP" rows={3} maxLength={5000} />
          <button className="button primary diagnose-button" type="submit" disabled={!query.trim() || busy}>
            {busy ? <><span className="button-spinner" /> Analysing…</> : <><Search size={18} /> Diagnose</>}
          </button>
        </div>
        <div className="example-row"><span>Representative tests</span>{examples.map((item) => <button key={item.label} type="button" onClick={() => setQuery(item.query)}>{item.label}</button>)}</div>
      </form>
      <Notice kind="warning" title="Operational boundary">
        Results are historical planning evidence—not a maintenance instruction, airworthiness finding, regulatory approval, or probability of technical correctness.
      </Notice>
      {busy && <AnalysisProgress />}
      {error && <ErrorState message={error} retry={() => void submit()} />}
      {!busy && !error && !response && <EmptyState title="Ready for a problem description" detail="Run one of the representative tests or enter a problem from a current planning scenario." />}
      {!busy && response && <DiagnoseResult response={response} />}
    </>
  );
}

function AnalysisProgress() {
  return (
    <div className="analysis-progress" role="status">
      <div className="radar"><span /><span /><i /></div>
      <div><strong>Evaluating structured historical evidence</strong><p>Normalizing safely, extracting anchors, retrieving TRAIN cases and applying evidence gates. Local model inference can take a moment.</p></div>
    </div>
  );
}

function DiagnoseResult({ response }: { response: DiagnoseResponse }) {
  const result = response.result;
  return (
    <div className="diagnose-results">
      <Interpretation result={result} />
      {!result.rq4_live_validated && (
        <Notice kind="warning" title="Validated retrieval branch unavailable">
          The required TRUE-RAW SpERT representation was not validated for this query. No primary RQ4/RQ5 action should be treated as available.
        </Notice>
      )}
      {result.compound_detected && result.subproblems.length > 0 ? (
        <>
          <Notice title={`${result.subproblems.length} independently grounded issues detected`}>
            Each subproblem below received a separate retrieval and evidence decision. No combined procedure was invented. Decomposition source: <b>{result.decomposition_source || "structured relations"}</b>.
          </Notice>
          <div className="subproblem-stack">
            {result.subproblems.map((item) => (
              <section className="subproblem" key={`${item.index}-${item.query}`}>
                <div className="subproblem-head">
                  <span>Issue {item.index}</span><h2>{item.title}</h2><p>{item.query}</p>
                  <div className="tag-row"><Tag>{item.component || "Unspecified component"}</Tag>{item.location && <Tag tone="slate">{item.location}</Tag>}<Tag tone="rose">{item.issue || item.issue_type}</Tag></div>
                </div>
                <RecommendationView result={item.recommendation} compact />
              </section>
            ))}
          </div>
          <CasesSection title="Combined-query nearest cases" description="Shown only for research transparency; per-issue evidence above remains authoritative." cases={result.nearest_cases} />
        </>
      ) : (
        <RecommendationView result={result} />
      )}
      <Section title="Model and retrieval provenance" description="Inspect what each branch received and which safeguards were active.">
        <div className="provenance-grid">
          <Definition term="User input"><code>{result.query}</code></Definition>
          <Definition term="Guarded normalized interpretation"><code>{result.normalized_interpretation || result.query}</code></Definition>
          <Definition term="Validated RQ4 input"><code>{result.model_input || "Not reported"}</code></Definition>
          <Definition term="Primary semantic branch">{result.semantic_branch_used ? "Verified normalized semantic SpERT" : "Validated raw-SpERT fallback"}</Definition>
          <Definition term="Normalization method">{result.normalization_method || "none"}</Definition>
          <Definition term="Evidence library">{response.meta.candidate_split.toUpperCase()} only</Definition>
          <Definition term="Retrieval base">{response.meta.rq4_base}</Definition>
          <Definition term="Reranker role">{response.meta.reranker_role.replaceAll("_", " ")}</Definition>
        </div>
        {(result.normalization_warning || result.semantic_warning || result.partial_structure_warning) && (
          <div className="warning-list">
            {[result.normalization_warning, result.semantic_warning, result.partial_structure_warning].filter(Boolean).map((warning) => <p key={warning}><AlertTriangle size={16} />{warning}</p>)}
          </div>
        )}
      </Section>
    </div>
  );
}

function Interpretation({ result }: { result: Recommendation }) {
  const anchors = [...result.model_components, ...result.model_faults];
  return (
    <Section className="interpretation-card">
      <div className="interpretation-title"><div><span className="eyebrow">Structured interpretation</span><h2>{result.normalized_interpretation || result.query}</h2></div><Tag tone={result.semantic_branch_used ? "green" : "amber"}>{result.semantic_branch_used ? "Verified semantic branch" : "Safe raw fallback"}</Tag></div>
      <div className="anchor-row"><b>Primary anchors</b>{anchors.length ? anchors.map((item, index) => <Tag key={`${item}-${index}`}>{item}</Tag>) : <span>No learned anchor predicted</span>}</div>
      <details className="details-panel"><summary><Braces size={17} /> Inspect extracted entities and relations</summary><StructurePayload entities={result.entities} relations={result.relations} /></details>
      <details className="details-panel"><summary><FileSearch size={17} /> Validated RQ4 representation</summary><StructurePayload entities={result.rq4_entities} relations={result.rq4_relations} /></details>
    </Section>
  );
}

function RecommendationView({ result, compact = false }: { result: Recommendation; compact?: boolean }) {
  const [added, setAdded] = useState(false);
  const canPromote = Boolean(result.structured_sentence && !result.abstain);
  function addToPlanning() {
    const current = loadWatchlist();
    saveWatchlist([recommendationToWatchlist(result), ...current].slice(0, 100));
    window.dispatchEvent(new Event("avimaint-watchlist"));
    setAdded(true);
  }
  return (
    <div className={compact ? "recommendation compact" : "recommendation"}>
      <Section className={`decision-card ${result.badge}`}>
        <div className="decision-top">
          <EvidenceBadge badge={result.badge} />
          <div className="decision-metrics">
            <span><b>{result.support_clusters}</b> independent clusters</span>
            <span><b>{percent(result.anchor_coverage)}</b> anchor coverage</span>
            <span><b>{percent(result.historical_agreement_probability, 1)}</b> historical agreement</span>
          </div>
        </div>
        {result.structured_sentence ? (
          <div className="primary-action"><ArrowRight size={25} /><div><small>Source-grounded planning statement</small><h2>{result.structured_sentence}</h2><p>{result.headline_reason}</p></div></div>
        ) : (
          <div className="abstention"><ShieldCheck size={27} /><div><h2>No primary action passed the evidence gate</h2><p>{result.headline_reason || `Closest historical action family: ${result.evidence_family || "none"}. Review the nearest cases without treating them as a recommendation.`}</p></div></div>
        )}
        {result.badge === "limited" && <Notice kind="warning"><b>One-cluster evidence.</b> This is a traceable historical example, not a recurring strategy.</Notice>}
        <div className="technical-strip">
          <span>Evidence tier <b>{result.evidence_tier || badgeLabel[result.badge]}</b></span>
          <span>Top retrieval <b>{score(result.base_top_score)}</b></span>
          <span>Family margin <b>{score(result.family_evidence_margin)}</b></span>
          <span>Library <b>{result.candidate_split.toUpperCase()}</b></span>
        </div>
        {canPromote && <button className="button secondary planning-add" type="button" onClick={addToPlanning} disabled={added}>{added ? <><Check size={17} /> Added to planning</> : <><ClipboardPlus size={17} /> Add to planning watchlist</>}</button>}
      </Section>
      {result.strategies.length > 0 && (
        <Section title="Recorded strategy alternatives" description="Grouped from eligible historical actions after problem-only retrieval; single-case options stay explicitly weak.">
          <div className="strategy-grid">{result.strategies.map((strategy) => <StrategyCard strategy={strategy} key={`${strategy.family}-${strategy.tier}`} />)}</div>
        </Section>
      )}
      <CasesSection title="Supporting historical work orders" description="Cases supporting the displayed primary action family, diversified by independent evidence cluster." cases={result.recommended_cases} />
      {result.negative_evidence.length > 0 && <CasesSection negative title="Negative or unresolved evidence" description="Preserved for transparency and excluded from positive action-family support." cases={result.negative_evidence} />}
      <CasesSection title="Nearest historical cases" description="All nearest problem-side cases, including evidence that did not pass the primary gate." cases={result.nearest_cases} />
    </div>
  );
}

function StrategyCard({ strategy }: { strategy: Strategy }) {
  return (
    <article className={`strategy-card ${strategy.tier === "single_case" ? "weak" : ""}`}>
      <div className="strategy-heading"><div><span>{strategy.tier === "single_case" ? "Single-case option" : "Corroborated strategy"}</span><h3>{strategy.family}</h3></div>{strategy.is_primary && <Tag tone="green">Primary</Tag>}</div>
      <p className="strategy-sentence">{strategy.sentence}</p><p>{strategy.meaning}</p>
      <div className="strategy-stats"><span><b>{strategy.support_clusters}</b> clusters</span><span><b>{strategy.case_count}</b> cases</span><span className="positive-text"><b>{strategy.outcome_positive}</b> positive</span><span className="negative-text"><b>{strategy.outcome_negative}</b> neg./mixed</span></div>
      {strategy.examples.length > 0 && <details><summary>Recorded examples</summary>{strategy.examples.map((example) => <div className="strategy-example" key={`${example.ident}-${example.action}`}><b>{example.ident}</b><span>{example.action}</span><em>{example.outcome}</em></div>)}</details>}
    </article>
  );
}

function CasesSection({ title, description, cases, negative = false }: { title: string; description: string; cases: CaseEvidence[]; negative?: boolean }) {
  if (!cases.length) return null;
  return (
    <Section title={title} description={description} className={negative ? "negative-section" : ""}>
      <div className="case-list">{cases.map((item, index) => <CaseCard item={item} rank={index + 1} key={`${item.ident}-${item.cluster_id}-${index}`} />)}</div>
    </Section>
  );
}

function CaseCard({ item, rank }: { item: CaseEvidence; rank: number }) {
  return (
    <details className="case-card">
      <summary>
        <span className="case-rank">{rank}</span>
        <div className="case-summary"><b>{item.problem}</b><span>{item.action_family} · {item.outcome || "unknown outcome"}</span></div>
        <span className="case-score">{score(item.score)}</span>
      </summary>
      <div className="case-detail">
        <div><span>Recorded action</span><p>{item.action || "No action text recorded"}</p></div>
        <dl><Definition term="Source record">{item.ident}</Definition><Definition term="Evidence cluster">{item.cluster_id}</Definition><Definition term="Text score">{score(item.text_score)}</Definition><Definition term="Structure score">{score(item.structure_score)}</Definition></dl>
      </div>
    </details>
  );
}

function StructurePayload({ entities, relations }: { entities: Entity[]; relations: Relation[] }) {
  return (
    <div className="structure-grid">
      <div><h4>Entities · {entities.length}</h4>{entities.length ? <div className="entity-list">{entities.map((entity, index) => <span key={index}><b>{getEntityType(entity)}</b>{getEntityText(entity)}</span>)}</div> : <p>No model entities returned.</p>}</div>
      <div><h4>Relations · {relations.length}</h4>{relations.length ? <div className="relation-list">{relations.map((relation, index) => <span key={index}>{String(relation.type ?? relation.label ?? "RELATION")}</span>)}</div> : <p>No model relations returned.</p>}</div>
    </div>
  );
}
