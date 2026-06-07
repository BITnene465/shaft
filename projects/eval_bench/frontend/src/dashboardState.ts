import { useQuery } from "@tanstack/react-query";

import { fetchState } from "./api";

export function useDashboardState({
  refetchInterval = false,
  staleTime = 15_000
}: {
  refetchInterval?: number | false;
  staleTime?: number;
} = {}) {
  return useQuery({
    queryKey: ["dashboard-state"],
    queryFn: ({ signal }) => fetchState({ signal }),
    refetchInterval,
    staleTime
  });
}
