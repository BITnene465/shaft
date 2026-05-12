import { useQuery } from "@tanstack/react-query";

import { fetchState } from "./api";

export function useDashboardState() {
  return useQuery({ queryKey: ["dashboard-state"], queryFn: fetchState });
}
