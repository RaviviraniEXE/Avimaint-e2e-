import { describe, expect, it } from "vitest";
import { badgeLabel, percent, recommendationToWatchlist, score } from "../utils";
import type { Recommendation } from "../api/types";

describe("evidence presentation", () => {
  it("uses evidence wording rather than safety-confidence wording", () => {
    expect(badgeLabel.strong).toBe("Strong historical evidence");
    expect(badgeLabel.limited).toBe("Limited historical evidence");
    expect(badgeLabel.abstain).toBe("Nearest cases only");
  });

  it("formats missing and finite values safely", () => {
    expect(percent(0.88684, 1)).toBe("88.7%");
    expect(percent(null)).toBe("n/a");
    expect(score(Number.NaN)).toBe("n/a");
  });

  it("preserves evidence metadata in a planning handoff", () => {
    const recommendation = {
      query: "#2 INTAKE LEAKING",
      structured_sentence: "Review the recorded replacement strategy.",
      headline_action: "Replace",
      components: ["INTAKE"],
      badge: "limited",
      historical_agreement_probability: 0.62,
      support_clusters: 1,
      anchor_coverage: 0.75,
    } as Recommendation;
    const item = recommendationToWatchlist(recommendation);
    expect(item.evidenceGrade).toBe("Limited historical evidence");
    expect(item.supportClusters).toBe(1);
    expect(item.agreement).toBe(0.62);
  });
});
