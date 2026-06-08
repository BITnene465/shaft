type NetworkHintNavigator = Navigator & {
  connection?: {
    effectiveType?: string;
    saveData?: boolean;
  };
};

const CONSTRAINED_EFFECTIVE_TYPES = new Set(["slow-2g", "2g"]);

export function isConstrainedNetworkMode() {
  if (typeof navigator === "undefined") {
    return false;
  }
  const connection = (navigator as NetworkHintNavigator).connection;
  return Boolean(
    connection?.saveData ||
      (connection?.effectiveType &&
        CONSTRAINED_EFFECTIVE_TYPES.has(connection.effectiveType.toLowerCase()))
  );
}

export function shouldAvoidSpeculativeNetworkWork() {
  return isConstrainedNetworkMode();
}
