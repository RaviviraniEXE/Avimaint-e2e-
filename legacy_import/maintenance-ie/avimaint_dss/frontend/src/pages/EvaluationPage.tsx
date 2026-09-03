import { useEffect, useState } from "react";
import { CheckCircle2, LockKeyhole, ShieldCheck } from "lucide-react";
import { api } from "../api/client";
import type { EvaluationResponse } from "../api/types";
import { GroupedHits } from "../components/Charts";
import { ErrorState, LoadingState, MetricCard, Notice, PageHeader, Section } from "../components/ui";
import { percent, score } from "../utils";

function numberAt(value: unknown, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function EvaluationPage() {
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    api.evaluation().then((value) => active && setData(value)).catch((reason) => active && setError(reason.message));
    return () => { active = false; };
  }, []);
  if (error) return <><PageHeader title="Frozen research evaluation" description="Locked RQ4/RQ5 evidence." /><ErrorState message={error} /></>;
  if (!data) return <><PageHeader title="Frozen research evaluation" description="Locked RQ4/RQ5 evidence." /><LoadingState label="Reading frozen evaluation summaries…" /></>;

  const { rq4, rq5, manual, lock } = data.frozen;
  const metrics = rq4?.selected_mode_metrics || {};
  const phaseA = manual?.phase_a || {};
  const phaseB = manual?.phase_b || {};
  const complete = Boolean(rq4 && rq5 && manual && lock?.locked);
  const problemHits = [numberAt(phaseA.problem_relevance_hit_at_1), numberAt(phaseA.problem_relevance_hit_at_3), numberAt(phaseA.problem_relevance_hit_at_5)];
  const actionHits = [numberAt(phaseB.usable_historical_action_evidence_hit_at_1), numberAt(phaseB.usable_historical_action_evidence_hit_at_3), numberAt(phaseB.usable_historical_action_evidence_hit_at_5)];
  return (
    <>
      <PageHeader eyebrow="Immutable thesis evidence" title="Frozen RQ4 and RQ5 evaluation" description="Results from the leakage-safe DEV selection, one-time locked TEST evaluation, and blinded semantic review. Operational Phase 2/3 extensions do not rewrite these values." actions={<span className={`lock-state ${complete ? "locked" : "missing"}`}><LockKeyhole size={17} />{complete ? "Freeze locked" : "Freeze incomplete"}</span>} />
      {!complete && <Notice kind="warning">One or more frozen summaries are missing. Run `FINAL_11_VERIFY_ALL.bat` and verify the frozen output directory before citing results.</Notice>}
      <div className="metric-grid six">
        <MetricCard label="RQ4 Hit@1" value={percent(numberAt(metrics.hit_at_1), 1)} detail="Locked TEST" />
        <MetricCard label="RQ4 Hit@3" value={percent(numberAt(metrics.hit_at_3), 1)} detail="Locked TEST" />
        <MetricCard label="RQ4 Hit@5" value={percent(numberAt(metrics.hit_at_5), 1)} detail="Locked TEST" />
        <MetricCard label="nDCG@5" value={score(numberAt(metrics.ndcg_at_5))} detail="Rank-sensitive relevance" />
        <MetricCard label="RQ5 ECE" value={score(numberAt(rq5?.ece_10_bins))} detail="10 bins" />
        <MetricCard label="RQ5 Brier" value={score(numberAt(rq5?.brier_score))} detail="Historical agreement" />
      </div>
      <Notice><b>Calibration boundary:</b> RQ5 estimates agreement with recorded historical action families. It is not a probability of safety, technical correctness, applicability, approval, or aircraft release.</Notice>
      <div className="two-column">
        <Section title="Blinded semantic review" description="Problem relevance and historical action usefulness at retrieval depths 1, 3 and 5.">
          <GroupedHits problem={problemHits} action={actionHits} />
        </Section>
        <Section title="Manual-review evidence" description="100 locked-TEST queries, 500 problem pairs and 380 Phase-B action pairs.">
          <div className="review-summary">
            <div><CheckCircle2 size={20} /><span><b>{percent(numberAt(phaseA.problem_relevance_hit_at_5), 1)}</b> of queries found relevant or partially relevant problem evidence by rank 5.</span></div>
            <div><CheckCircle2 size={20} /><span><b>{percent(numberAt(phaseB.usable_historical_action_evidence_hit_at_5), 1)}</b> found usable historical action evidence by rank 5.</span></div>
            <div><ShieldCheck size={20} /><span><b>{percent(numberAt(phaseB.applicable_or_partial_rate_among_problem_relevant_pairs), 1)}</b> of problem-relevant pairs contained at least partially useful action evidence.</span></div>
            <div><ShieldCheck size={20} /><span><b>{percent(numberAt(phaseB.clearly_applicable_rate_among_problem_relevant_pairs), 1)}</b> were clearly useful historical actions.</span></div>
          </div>
          <p className="limitation">Limitation: single-reviewer/AI-assisted judgments do not provide inter-rater reliability.</p>
        </Section>
      </div>
      <Section title="Frozen protocol boundary" description="The retrieval mode was selected on DEV before TEST was opened once.">
        <div className="protocol-grid"><div><span>Selected mode</span><b>{rq4?.selected_mode || "Unavailable"}</b></div><div><span>TEST queries</span><b>{numberAt(metrics.queries).toLocaleString()}</b></div><div><span>Candidate split</span><b>TRAIN only</b></div><div><span>Manual review</span><b>Two blinded phases</b></div></div>
        <p className="page-note">{data.warning}</p>
      </Section>
    </>
  );
}
