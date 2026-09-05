export type ServiceHealth = {
  ready: boolean;
  error?: string;
  metadata?: Record<string, unknown>;
};

export type HealthResponse = {
  status: "ready" | "degraded" | string;
  api_version: string;
  rq4_base: string;
  candidate_split: string;
  critical_ready: boolean;
  phase2_compound_decomposition: boolean;
  phase3_limited_evidence: boolean;
  raw_spert: ServiceHealth;
  normalization: ServiceHealth;
  semantic_spert: ServiceHealth;
  rq5_calibrator: ServiceHealth & { status?: string };
  frontend: {
    ready: boolean;
    version: string;
    expected_version: string;
    mode: string;
    url: string;
  };
  runtime_model_lock?: Record<string, unknown>;
};

export type Kpis = {
  work_orders: number;
  unique_problems: number;
  problem_clusters: number;
  components_tracked: number;
  action_families: number;
  recurring_faults: number;
  recorded_outcomes_pct: number;
  spert_backed: boolean;
};

export type RecurringItem = {
  cluster_id: string;
  problem: string;
  component: string;
  fault: string;
  work_orders: number;
  top_action?: string;
  dominant_action?: string;
  positive_outcomes?: number;
  negative_outcomes?: number;
  outcome_unknown?: number;
};

export type FrequencyItem = {
  component?: string;
  fault?: string;
  action_family?: string;
  outcome?: string;
  work_orders: number;
  share_pct?: number;
  cumulative_pct?: number;
};

export type OverviewResponse = {
  kpis: Kpis;
  top_recurring: RecurringItem[];
  component_frequency: FrequencyItem[];
  fault_frequency: FrequencyItem[];
  action_frequency: FrequencyItem[];
  outcome_mix: FrequencyItem[];
  note: string;
};

export type Entity = {
  type?: string;
  label?: string;
  text?: string;
  phrase?: string;
  start?: number;
  end?: number;
  score?: number;
  [key: string]: unknown;
};

export type Relation = {
  type?: string;
  label?: string;
  head?: number | Entity;
  tail?: number | Entity;
  score?: number;
  [key: string]: unknown;
};

export type CaseEvidence = {
  ident: string;
  problem: string;
  action: string;
  action_family: string;
  outcome: string;
  cluster_id: string;
  score: number;
  text_score: number;
  structure_score: number;
  channels?: Record<string, number>;
};

export type Strategy = {
  family: string;
  sentence: string;
  meaning: string;
  support_clusters: number;
  case_count: number;
  outcome_positive: number;
  outcome_negative: number;
  outcome_unknown: number;
  examples: Array<{ ident: string; action: string; outcome?: string }>;
  is_primary: boolean;
  tier: "corroborated" | "single_case" | string;
};

export type Subproblem = {
  index: number;
  title: string;
  query: string;
  component: string;
  location: string;
  issue: string;
  issue_type: string;
  relation_score: number;
  recommendation: Recommendation;
};

export type Recommendation = {
  query: string;
  components: string[];
  faults: string[];
  badge: "strong" | "moderate" | "limited" | "exploratory" | "abstain" | string;
  lens: string;
  headline_action: string;
  headline_reason: string;
  support_clusters: number;
  structured_sentence: string;
  strategies: Strategy[];
  recommended_cases: CaseEvidence[];
  alternatives: unknown[];
  nearest_cases: CaseEvidence[];
  negative_evidence: CaseEvidence[];
  structure_source: string;
  entities: Entity[];
  relations: Relation[];
  family_evidence_margin: number;
  retrieval_margin: number;
  anchor_coverage: number;
  abstain: boolean;
  evidence_family: string;
  base_top_score: number;
  evidence_tier: string;
  evidence_note: string;
  historical_agreement_probability: number | null;
  calibration_source: string;
  reranker_used: boolean;
  candidate_split: string;
  model_components: string[];
  model_faults: string[];
  derived_components: string[];
  derived_faults: string[];
  target_location: string;
  partial_structure_warning: string;
  model_input: string;
  input_adapter: string;
  input_adapted: boolean;
  normalized_interpretation: string;
  normalization_model_input: string;
  normalization_method: string;
  normalization_warning: string;
  normalization_model: string;
  semantic_branch_used: boolean;
  semantic_status: string;
  semantic_warning: string;
  semantic_entities: Entity[];
  semantic_relations: Relation[];
  semantic_components: string[];
  semantic_faults: string[];
  semantic_locations: string[];
  rq4_entities: Entity[];
  rq4_relations: Relation[];
  rq4_components: string[];
  rq4_faults: string[];
  rq4_structure_source: string;
  rq4_live_validated: boolean;
  compound_detected: boolean;
  decomposition_source: string;
  subproblems: Subproblem[];
};

export type DiagnoseResponse = {
  result: Recommendation;
  meta: {
    rq4_base: string;
    candidate_split: string;
    reranker_role: string;
    rq5_meaning: string;
    technical_correctness_probability: boolean;
  };
};

export type InsightsResponse = {
  recurring: RecurringItem[];
  components: FrequencyItem[];
  faults: FrequencyItem[];
  actions: FrequencyItem[];
  outcomes: FrequencyItem[];
  matrix: { components: string[]; faults: string[]; values: number[][] };
  component_options: string[];
  selected_component: string;
  component_actions: FrequencyItem[];
  note: string;
};

export type GraphNode = {
  id: string;
  label: string;
  kind: "component" | "fault" | "action";
  count: number;
  focused: boolean;
};

export type GraphEdge = {
  source: string;
  target: string;
  kind: "component_fault" | "fault_action";
  count: number;
};

export type GraphResponse = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  focus_component: string;
  component_options: string[];
  parameters: { top_components: number; top_faults: number; min_edge: number };
  note: string;
};

export type JobCard = {
  title: string;
  component: string;
  fault: string;
  work_orders: number;
  problem_groups: number;
  dominant_action: string;
  steps: Array<{ text: string; source_idents: string[] }>;
  references: string[];
  outcome_positive: number;
  outcome_negative: number;
  outcome_unknown: number;
  source_idents: string[];
};

export type EvaluationResponse = {
  frozen: {
    rq4: Record<string, any> | null;
    rq5: Record<string, any> | null;
    manual: Record<string, any> | null;
    lock: Record<string, any> | null;
  };
  warning: string;
};

export type WatchlistEntry = {
  id: string;
  addedAt: string;
  query: string;
  recommendation: string;
  actionFamily: string;
  component: string;
  evidenceGrade: string;
  agreement: number | null;
  supportClusters: number;
  anchorCoverage: number;
};
