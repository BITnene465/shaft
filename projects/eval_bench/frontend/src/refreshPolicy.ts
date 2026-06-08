import { isConstrainedNetworkMode } from "./networkHints";

const CONSTRAINED_REFRESH_MULTIPLIER = 3;
const CONSTRAINED_REFRESH_MIN_MS = 15_000;

export function adaptiveRefreshInterval(baseIntervalMs: number | false) {
  if (baseIntervalMs === false) {
    return false;
  }
  if (!isConstrainedNetworkMode()) {
    return baseIntervalMs;
  }
  return Math.max(baseIntervalMs * CONSTRAINED_REFRESH_MULTIPLIER, CONSTRAINED_REFRESH_MIN_MS);
}
