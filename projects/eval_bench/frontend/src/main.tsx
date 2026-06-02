import React from "react";
import ReactDOM from "react-dom/client";
import {
  QueryClient,
  QueryClientProvider
} from "@tanstack/react-query";
import {
  RouterProvider,
  createRootRoute,
  createRoute,
  createRouter,
  lazyRouteComponent
} from "@tanstack/react-router";

import { AppErrorBoundary, AppShell } from "./appShell";
import { JobsPage } from "./jobsPage";
import { OverviewPage } from "./overviewPage";
import { ServicesPage } from "./servicesPage";
import { SettingsPage } from "./settingsPage";
import "./appBase.css";
import "./appChrome.css";
import "./sharedControlsTheme.css";
import "./sharedControls.css";
import "./sharedButtons.css";
import "./sharedIndicators.css";
import "./sharedMetrics.css";
import "./sharedPager.css";
import "./sharedSplit.css";
import "./labelColorControls.css";
import "./appTheme.css";
import "./appChromeVisual.css";
import "./appChromeCollapsed.css";
import "./design.css";
import "./interactionFeedback.css";
import "./appTypography.css";
import "./viewerTheme.css";
import "./compositeTheme.css";
import "./adaptiveContent.css";
import "./workspaceTheme.css";
import "./workspaceShell.css";
import "./workspaceDialog.css";
import "./pageCommand.css";
import "./dataTable.css";
import "./runTables.css";
import "./formControls.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchInterval: 10_000,
      staleTime: 5_000,
      retry: 1
    }
  }
});

const rootRoute = createRootRoute({ component: AppShell });
const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: OverviewPage
});
const benchmarksRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/benchmarks",
  component: lazyRouteComponent(() => import("./benchmarksPage"), "BenchmarksPage")
});
const benchmarkDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/benchmarks/$benchmarkId",
  component: lazyRouteComponent(() => import("./benchmarksPage"), "BenchmarkDetailPage")
});
const jobsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/jobs",
  component: JobsPage
});
const servicesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/services",
  component: ServicesPage
});
const runsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/runs",
  component: lazyRouteComponent(() => import("./runsPage"), "RunsPage")
});
const runDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/runs/$runId",
  component: lazyRouteComponent(() => import("./runsPage"), "RunDetailPage")
});
const rankBoardRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/rank-board",
  component: lazyRouteComponent(() => import("./rankBoardPage"), "RankBoardPage")
});
const suiteReportRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/suite-report",
  component: lazyRouteComponent(() => import("./suiteReportPage"), "SuiteReportPage")
});
const compareRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/compare",
  component: lazyRouteComponent(() => import("./comparePage"), "ComparePage")
});
const comparisonSampleRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/compare/$baselineRunId/$candidateRunId/$sampleIndex",
  component: lazyRouteComponent(() => import("./comparisonSamplePage"), "ComparisonSamplePage")
});
const settingsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/settings",
  component: SettingsPage
});

const routeTree = rootRoute.addChildren([
  indexRoute,
  benchmarksRoute,
  benchmarkDetailRoute,
  servicesRoute,
  jobsRoute,
  runsRoute,
  runDetailRoute,
  rankBoardRoute,
  suiteReportRoute,
  compareRoute,
  comparisonSampleRoute,
  settingsRoute
]);
const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <AppErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </AppErrorBoundary>
  </React.StrictMode>
);
