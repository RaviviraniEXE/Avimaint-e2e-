import type {
  DiagnoseResponse,
  EvaluationResponse,
  GraphResponse,
  HealthResponse,
  InsightsResponse,
  JobCard,
  OverviewResponse,
  RecurringItem,
} from "./types";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = 60000): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, {
      ...init,
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = payload?.detail;
      throw new ApiError(
        typeof detail === "string" ? detail : `Request failed (${response.status})`,
        response.status,
      );
    }
    return payload as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError("The request timed out. The local models may still be loading.");
    }
    throw new ApiError("Cannot reach the AviMaint-DSS backend at port 8780.");
  } finally {
    window.clearTimeout(timer);
  }
}

function params(values: Record<string, string | number | undefined>) {
  const query = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => {
    if (value !== undefined && value !== "") query.set(key, String(value));
  });
  const rendered = query.toString();
  return rendered ? `?${rendered}` : "";
}

export const api = {
  health: () => request<HealthResponse>("/api/v1/health", undefined, 8000),
  overview: () => request<OverviewResponse>("/api/v1/overview"),
  diagnose: (query: string, topK = 25) =>
    request<DiagnoseResponse>(
      "/api/v1/diagnose",
      { method: "POST", body: JSON.stringify({ query, top_k: topK }) },
      120000,
    ),
  insights: (component = "") =>
    request<InsightsResponse>(`/api/v1/insights${params({ component })}`),
  graph: (options: {
    topComponents: number;
    topFaults: number;
    minEdge: number;
    focusComponent: string;
  }) =>
    request<GraphResponse>(
      `/api/v1/knowledge-graph${params({
        top_components: options.topComponents,
        top_faults: options.topFaults,
        min_edge: options.minEdge,
        focus_component: options.focusComponent,
      })}`,
    ),
  recurring: (minSupport = 5) =>
    request<{ items: RecurringItem[]; min_support: number; note: string }>(
      `/api/v1/planning/recurring${params({ min_support: minSupport })}`,
    ),
  jobCard: (clusterId: string) =>
    request<{ card: JobCard; cluster_id: string; warning: string }>(
      "/api/v1/planning/job-card",
      { method: "POST", body: JSON.stringify({ cluster_id: clusterId }) },
    ),
  evaluation: () => request<EvaluationResponse>("/api/v1/evaluation"),
};
