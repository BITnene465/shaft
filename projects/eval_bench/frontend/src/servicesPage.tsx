import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { fetchServices } from "./api";
import { PagerControl, clampListPageOffset, updatePagedFilterValue } from "./samplePager";
import { errorMessage, facetValues } from "./formatters";
import { AdvancedFilterBar } from "./filterControls";
import { AppIcon } from "./iconLibrary";
import { ServiceCreatePanel } from "./servicesCreatePanel";
import { ServiceGrid } from "./servicesGrid";
import {
  CommandButton,
  EmptyState,
  WorkspaceDialog
} from "./ui";
import { useDebouncedValueState } from "./useDebouncedValue";

import "./servicesPage.css";

const SERVICE_PAGE_SIZE = 80;

export function ServicesPage() {
  const [registerOpen, setRegisterOpen] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [kindFilter, setKindFilter] = useState("all");
  const [pageOffset, setPageOffset] = useState(0);
  const debouncedSearch = useDebouncedValueState(searchText);
  const serviceFilters = useMemo(
    () => ({
      offset: pageOffset,
      limit: SERVICE_PAGE_SIZE,
      status: statusFilter !== "all" ? statusFilter : undefined,
      kind: kindFilter !== "all" ? kindFilter : undefined,
      query: debouncedSearch.value.trim() || undefined
    }),
    [debouncedSearch.value, kindFilter, pageOffset, statusFilter]
  );
  const servicesQuery = useQuery({
    queryKey: ["services", serviceFilters],
    queryFn: ({ signal }) => fetchServices(serviceFilters, { signal }),
    placeholderData: (previousData) => previousData
  });
  const services = servicesQuery.data?.services ?? [];
  const totalServices = servicesQuery.data?.total ?? services.length;
  const facets = servicesQuery.data?.facets;
  const statuses = facetValues(facets, "statuses", [
    "registered",
    "starting",
    "running",
    "stopped",
    "failed",
    ...services.map((service) => service.status)
  ]);
  const kinds = facetValues(facets, "kinds", [
    "local_vllm",
    "external_vllm",
    ...services.map((service) => service.kind)
  ]);
  useEffect(() => {
    const nextOffset = clampListPageOffset(pageOffset, totalServices, SERVICE_PAGE_SIZE);
    if (nextOffset !== pageOffset) {
      setPageOffset(nextOffset);
    }
  }, [pageOffset, totalServices]);
  return (
    <section className="page-stack density-page">
      <div className="page-command-row">
        <div>
          <h2>模型服务</h2>
          <span>长期 endpoint、健康检查和 runtime 日志</span>
        </div>
        <CommandButton
          icon={<AppIcon name="registerService" size={17} />}
          onClick={() => setRegisterOpen(true)}
        >
          登记服务
        </CommandButton>
      </div>
      {servicesQuery.isLoading ? (
        <EmptyState title="正在加载服务" />
      ) : servicesQuery.error || !servicesQuery.data ? (
        <EmptyState title={`服务加载失败：${errorMessage(servicesQuery.error)}`} tone="danger" />
      ) : (
        <>
          <AdvancedFilterBar
            title="服务高级检索"
            meta={`${services.length.toLocaleString()} / ${totalServices.toLocaleString()} 个服务`}
            controls={[
              {
                type: "search",
                id: "service-query",
                label: "全文检索",
                value: searchText,
                onChange: (value) =>
                  updatePagedFilterValue(searchText, value, setSearchText, setPageOffset),
                placeholder: "搜索服务、模型、endpoint、CUDA、健康状态"
              },
              {
                type: "select",
                id: "service-status",
                label: "状态",
                value: statusFilter,
                values: ["all", ...statuses],
                labels: { all: "全部" },
                onChange: (value) =>
                  updatePagedFilterValue(statusFilter, value, setStatusFilter, setPageOffset)
              },
              {
                type: "select",
                id: "service-kind",
                label: "类型",
                value: kindFilter,
                values: ["all", ...kinds],
                labels: { all: "全部" },
                onChange: (value) =>
                  updatePagedFilterValue(kindFilter, value, setKindFilter, setPageOffset)
              }
            ]}
          />
          <ServiceGrid
            services={services}
            refreshing={servicesQuery.isPlaceholderData || debouncedSearch.pending}
          />
          <PagerControl
            className="rank-board-pager service-list-pager"
            offset={servicesQuery.data.offset ?? pageOffset}
            limit={servicesQuery.data.limit ?? SERVICE_PAGE_SIZE}
            total={totalServices}
            onPageChange={setPageOffset}
          />
        </>
      )}
      <WorkspaceDialog
        open={registerOpen}
        title="登记模型服务"
        meta="保存本地或外部 vLLM 服务参数"
        onClose={() => setRegisterOpen(false)}
      >
        <ServiceCreatePanel bare />
      </WorkspaceDialog>
    </section>
  );
}
