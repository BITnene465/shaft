import type { QueryClient } from "@tanstack/react-query";

import {
  fetchBenchmarks,
  fetchComparisons,
  fetchJobs,
  fetchRankBoard,
  fetchRuns,
  fetchSchedulerStatus,
  fetchServices,
  fetchSettingsPreviewSample
} from "./api";
import { shouldAvoidSpeculativeNetworkWork } from "./networkHints";
import { RANK_PAGE_SIZE } from "./rankBoardModel";

const DEFAULT_LIST_PAGE_SIZE = 80;
const DEFAULT_COMPARISON_HISTORY_PAGE_SIZE = 50;
const ROUTE_PREFETCH_STALE_MS = 15_000;
const ROUTE_PREFETCH_INTENT_DELAY_MS = 80;

const prefetchInFlightPathnames = new Set<string>();
const pendingPrefetchTimers = new Map<string, number>();

export function prefetchEvalBenchRouteData(queryClient: QueryClient, pathname: string) {
  if (shouldSkipRoutePrefetch()) {
    return;
  }
  const normalizedPathname = routeRootPathname(pathname);
  if (
    prefetchInFlightPathnames.has(normalizedPathname) ||
    pendingPrefetchTimers.has(normalizedPathname)
  ) {
    return;
  }
  clearPendingRoutePrefetchesExcept(normalizedPathname);
  const timeoutHandle = window.setTimeout(() => {
    pendingPrefetchTimers.delete(normalizedPathname);
    runRouteDataPrefetch(queryClient, normalizedPathname);
  }, ROUTE_PREFETCH_INTENT_DELAY_MS);
  pendingPrefetchTimers.set(normalizedPathname, timeoutHandle);
}

function runRouteDataPrefetch(queryClient: QueryClient, normalizedPathname: string) {
  if (prefetchInFlightPathnames.has(normalizedPathname)) {
    return;
  }
  const prefetchQueries = prefetchQueriesForPathname(queryClient, normalizedPathname);
  if (prefetchQueries.length === 0) {
    return;
  }
  prefetchInFlightPathnames.add(normalizedPathname);
  void Promise.allSettled(prefetchQueries).finally(() => {
    prefetchInFlightPathnames.delete(normalizedPathname);
  });
}

function clearPendingRoutePrefetchesExcept(pathname: string) {
  for (const [pendingPathname, timeoutHandle] of pendingPrefetchTimers) {
    if (pendingPathname === pathname) {
      continue;
    }
    window.clearTimeout(timeoutHandle);
    pendingPrefetchTimers.delete(pendingPathname);
  }
}

function routeRootPathname(pathname: string) {
  if (pathname.startsWith("/benchmarks")) {
    return "/benchmarks";
  }
  if (pathname.startsWith("/services")) {
    return "/services";
  }
  if (pathname.startsWith("/jobs")) {
    return "/jobs";
  }
  if (pathname.startsWith("/runs")) {
    return "/runs";
  }
  if (pathname.startsWith("/rank-board")) {
    return "/rank-board";
  }
  if (pathname.startsWith("/suite-report")) {
    return "/suite-report";
  }
  if (pathname.startsWith("/compare")) {
    return "/compare";
  }
  if (pathname.startsWith("/settings")) {
    return "/settings";
  }
  return "/";
}

function prefetchQueriesForPathname(queryClient: QueryClient, pathname: string) {
  switch (pathname) {
    case "/benchmarks": {
      const filters = { offset: 0, limit: DEFAULT_LIST_PAGE_SIZE };
      return [
        queryClient.prefetchQuery({
          queryKey: ["benchmarks", filters],
          queryFn: ({ signal }) => fetchBenchmarks(filters, { signal, silent: true }),
          staleTime: ROUTE_PREFETCH_STALE_MS
        })
      ];
    }
    case "/services": {
      const filters = { offset: 0, limit: DEFAULT_LIST_PAGE_SIZE };
      return [
        queryClient.prefetchQuery({
          queryKey: ["services", filters],
          queryFn: ({ signal }) => fetchServices(filters, { signal, silent: true }),
          staleTime: ROUTE_PREFETCH_STALE_MS
        })
      ];
    }
    case "/jobs": {
      const filters = { offset: 0, limit: DEFAULT_LIST_PAGE_SIZE };
      return [
        queryClient.prefetchQuery({
          queryKey: ["jobs", filters],
          queryFn: ({ signal }) => fetchJobs(filters, { signal, silent: true }),
          staleTime: ROUTE_PREFETCH_STALE_MS
        }),
        queryClient.prefetchQuery({
          queryKey: ["scheduler-status"],
          queryFn: ({ signal }) => fetchSchedulerStatus({ signal, silent: true }),
          staleTime: ROUTE_PREFETCH_STALE_MS
        })
      ];
    }
    case "/runs": {
      const filters = { offset: 0, limit: DEFAULT_LIST_PAGE_SIZE };
      return [
        queryClient.prefetchQuery({
          queryKey: ["runs", filters],
          queryFn: ({ signal }) => fetchRuns(filters, { signal, silent: true }),
          staleTime: ROUTE_PREFETCH_STALE_MS
        })
      ];
    }
    case "/rank-board": {
      return [
        queryClient.prefetchQuery({
          queryKey: [
            "rank-board",
            "",
            "all",
            "all",
            "all",
            "all",
            "all",
            "all",
            "all",
            "all",
            "",
            "f1_iou50",
            "desc",
            0
          ],
          queryFn: ({ signal }) =>
            fetchRankBoard(
              {
                offset: 0,
                limit: RANK_PAGE_SIZE,
                sortBy: "f1_iou50",
                sortOrder: "desc"
              },
              { signal, silent: true }
            ),
          staleTime: ROUTE_PREFETCH_STALE_MS
        })
      ];
    }
    case "/suite-report": {
      return [];
    }
    case "/compare": {
      const runFilters = { offset: 0, limit: DEFAULT_LIST_PAGE_SIZE };
      const comparisonFilters = { offset: 0, limit: DEFAULT_COMPARISON_HISTORY_PAGE_SIZE };
      return [
        queryClient.prefetchQuery({
          queryKey: ["runs", "compare", runFilters],
          queryFn: ({ signal }) => fetchRuns(runFilters, { signal, silent: true }),
          staleTime: ROUTE_PREFETCH_STALE_MS
        }),
        queryClient.prefetchQuery({
          queryKey: ["comparisons", comparisonFilters],
          queryFn: ({ signal }) => fetchComparisons(comparisonFilters, { signal, silent: true }),
          staleTime: ROUTE_PREFETCH_STALE_MS
        })
      ];
    }
    case "/settings": {
      return [
        queryClient.prefetchQuery({
          queryKey: ["settings-preview-sample"],
          queryFn: ({ signal }) => fetchSettingsPreviewSample({ signal, silent: true }),
          retry: false,
          staleTime: 60_000
        })
      ];
    }
    default: {
      return [];
    }
  }
}

function shouldSkipRoutePrefetch() {
  return shouldAvoidSpeculativeNetworkWork();
}
