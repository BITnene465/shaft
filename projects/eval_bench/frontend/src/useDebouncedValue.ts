import { useEffect, useState } from "react";

export function useDebouncedValue<T>(value: T, delayMs = 220): T {
  return useDebouncedValueState(value, delayMs).value;
}

export function useDebouncedValueState<T>(value: T, delayMs = 220): {
  value: T;
  pending: boolean;
} {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timeout = window.setTimeout(() => setDebouncedValue(value), delayMs);
    return () => window.clearTimeout(timeout);
  }, [delayMs, value]);

  return {
    value: debouncedValue,
    pending: debouncedValue !== value
  };
}
