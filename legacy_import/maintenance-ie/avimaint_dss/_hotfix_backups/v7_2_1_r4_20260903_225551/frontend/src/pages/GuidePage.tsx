import { BookOpenCheck, CircleHelp, GitBranch, Search, ShieldCheck } from "lucide-react";
import { Notice, PageHeader, Section } from "../components/ui";

export function GuidePage() {
  return (
    <>
      <PageHeader eyebrow="How to interpret the system" title="AviMaint-DSS guide" description="A concise reference for evidence tiers, the dual extraction pipeline, provenance, and safe use of historical maintenance evidence." />
      <div className="guide-grid">
        <Section title="Diagnose workflow">
          <ol className="flow-steps">
            <li><span>1</span><div><Search size={19} /><b>Problem text only</b><p>The query action is never used to retrieve supporting cases.</p></div></li>
            <li><span>2</span><div><GitBranch size={19} /><b>Dual, fail-closed structure</b><p>Verified normalized semantics can improve interpretation; the frozen RQ4/RQ5 decision remains on validated TRUE-RAW SpERT.</p></div></li>
            <li><span>3</span><div><BookOpenCheck size={19} /><b>Cluster-distinct history</b><p>TRAIN cases are retrieved by problem evidence, then actions are grouped from independent clusters.</p></div></li>
            <li><span>4</span><div><ShieldCheck size={19} /><b>Evidence gate or abstention</b><p>Weak evidence remains exploratory or nearest-case-only, with alternatives and negative outcomes visible.</p></div></li>
          </ol>
        </Section>
        <Section title="Evidence tiers">
          <div className="tier-list">
            <div className="strong"><b>Strong historical evidence</b><p>Multiple independent clusters and a clear action-family margin.</p></div>
            <div className="moderate"><b>Moderate historical evidence</b><p>Corroborated historical support with a smaller margin.</p></div>
            <div className="limited"><b>Limited historical evidence</b><p>One grounded cluster meeting the explicit Phase 3 safeguards; review as a traceable example.</p></div>
            <div className="exploratory"><b>Exploratory evidence</b><p>Related historical evidence exists but does not justify a primary action.</p></div>
            <div className="abstain"><b>Nearest cases only</b><p>The system abstains and exposes cases for manual inspection.</p></div>
          </div>
        </Section>
      </div>
      <Section title="What every Diagnose result exposes" description="Use these fields together; no single score replaces engineering review.">
        <div className="guide-features">
          <div><b>Raw and normalized text</b><p>Lets you verify whether normalization preserved the maintenance meaning and protected values.</p></div>
          <div><b>Entities and relations</b><p>Shows structured components, faults, locations, observations, and their indexed connections.</p></div>
          <div><b>Independent support</b><p>Counts distinct evidence clusters so near-duplicate rows cannot inflate support.</p></div>
          <div><b>Historical agreement</b><p>RQ5 calibration estimates action-family agreement in recorded data, never technical correctness.</p></div>
          <div><b>Source work orders</b><p>Every displayed action stays linked to record and cluster identifiers.</p></div>
          <div><b>Negative evidence</b><p>Failed, mixed, diagnostic-only, or unresolved historical actions remain visible and separated.</p></div>
        </div>
      </Section>
      <Notice kind="warning" title="The system does not authorise work">
        AviMaint-DSS does not select approved maintenance data, determine airworthiness, certify applicability, generate a maintenance procedure, approve parts, satisfy regulatory requirements, or release an aircraft to service.
      </Notice>
      <Section title="Recommended review sequence">
        <div className="review-sequence"><span>1. Confirm the interpreted problem</span><span>2. Check each compound issue separately</span><span>3. Read evidence tier and support</span><span>4. Inspect alternatives and negative cases</span><span>5. Verify source records</span><span>6. Consult current approved data</span></div>
      </Section>
      <div className="help-callout"><CircleHelp size={24} /><div><b>When evidence feels too confident</b><p>Do not act on it. Capture the query and visible provenance, then include it in the end-to-end diagnostic review.</p></div></div>
    </>
  );
}
