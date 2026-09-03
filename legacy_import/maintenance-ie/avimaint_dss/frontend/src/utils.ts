import type { Recommendation, WatchlistEntry } from "./api/types";

export const integer = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export function percent(value: number | null | undefined, digits = 0) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "n/a";
  return `${(value * 100).toFixed(digits)}%`;
}

export function score(value: number | null | undefined, digits = 3) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "n/a";
  return value.toFixed(digits);
}

export const badgeLabel: Record<string, string> = {
  strong: "Strong historical evidence",
  moderate: "Moderate historical evidence",
  limited: "Limited historical evidence",
  exploratory: "Exploratory evidence",
  abstain: "Nearest cases only",
};

const WATCHLIST_KEY = "avimaint-dss-phase5-watchlist";

export function loadWatchlist(): WatchlistEntry[] {
  try {
    const value = JSON.parse(localStorage.getItem(WATCHLIST_KEY) || "[]");
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

export function saveWatchlist(items: WatchlistEntry[]) {
  localStorage.setItem(WATCHLIST_KEY, JSON.stringify(items));
}

export function recommendationToWatchlist(result: Recommendation): WatchlistEntry {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    addedAt: new Date().toISOString(),
    query: result.query,
    recommendation: result.structured_sentence,
    actionFamily: result.headline_action,
    component: result.components[0] || "",
    evidenceGrade: badgeLabel[result.badge] || result.badge,
    agreement: result.historical_agreement_probability,
    supportClusters: result.support_clusters,
    anchorCoverage: result.anchor_coverage,
  };
}

export function getEntityText(entity: Record<string, unknown>) {
  return String(entity.text ?? entity.phrase ?? entity.label ?? "(unnamed)");
}

export function getEntityType(entity: Record<string, unknown>) {
  return String(entity.type ?? entity.label ?? "ENTITY");
}
