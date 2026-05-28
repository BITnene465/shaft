import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import {
  useMutation,
  useQuery,
  useQueryClient
} from "@tanstack/react-query";
import type { UseQueryResult } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";

import {
  ServiceLog,
  ServiceSummary,
  checkServiceHealth,
  createService,
  deleteService,
  fetchServiceLogs,
  fetchServices,
  startService,
  stopService
} from "./api";
import { FormSelectControl, NumberInputControl, TextInputControl } from "./controlPrimitives";
import {
  canDeleteService,
  canStartService,
  canStopService
} from "./statusModel";
import { PagerControl, clampListPageOffset } from "./samplePager";
import {
  errorMessage,
  facetValues,
  formatDate,
  runtimeValue,
  serviceConfigValue,
  serviceEndpointValue,
  serviceHealth
} from "./formatters";
import { AdvancedFilterBar } from "./filterControls";
import { AppIcon } from "./iconLibrary";
import {
  ActionButton,
  Badge,
  CommandButton,
  ConfigItem,
  DangerConfirmDialog,
  EmptyState,
  IconActionButton,
  WorkspaceDialog
} from "./ui";

const SERVICE_PAGE_SIZE = 80;

export function ServicesPage() {
  const [registerOpen, setRegisterOpen] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [kindFilter, setKindFilter] = useState("all");
  const [pageOffset, setPageOffset] = useState(0);
  const serviceFilters = useMemo(
    () => ({
      offset: pageOffset,
      limit: SERVICE_PAGE_SIZE,
      status: statusFilter !== "all" ? statusFilter : undefined,
      kind: kindFilter !== "all" ? kindFilter : undefined,
      query: searchText.trim() || undefined
    }),
    [kindFilter, pageOffset, searchText, statusFilter]
  );
  const servicesQuery = useQuery({
    queryKey: ["services", serviceFilters],
    queryFn: () => fetchServices(serviceFilters),
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
    setPageOffset(0);
  }, [searchText, statusFilter, kindFilter]);
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
                onChange: setSearchText,
                placeholder: "搜索服务、模型、endpoint、CUDA、健康状态"
              },
              {
                type: "select",
                id: "service-status",
                label: "状态",
                value: statusFilter,
                values: ["all", ...statuses],
                labels: { all: "全部" },
                onChange: setStatusFilter
              },
              {
                type: "select",
                id: "service-kind",
                label: "类型",
                value: kindFilter,
                values: ["all", ...kinds],
                labels: { all: "全部" },
                onChange: setKindFilter
              }
            ]}
          />
          <ServiceGrid services={services} refreshing={servicesQuery.isPlaceholderData} />
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

function ServiceCreatePanel({ bare }: { bare?: boolean }) {
  const queryClient = useQueryClient();
  const [kind, setKind] = useState("local_vllm");
  const [serviceId, setServiceId] = useState("local-vllm-0");
  const [modelPath, setModelPath] = useState("");
  const [servedModelName, setServedModelName] = useState("");
  const [endpoint, setEndpoint] = useState("");
  const [cudaVisibleDevices, setCudaVisibleDevices] = useState("0");
  const [tensorParallelSize, setTensorParallelSize] = useState(1);
  const [port, setPort] = useState(8000);
  const [maxModelLen, setMaxModelLen] = useState(32768);
  const [gpuMemoryUtilization, setGpuMemoryUtilization] = useState(0.9);
  const [maxNumSeqs, setMaxNumSeqs] = useState(8);
  const mutation = useMutation({
    mutationFn: createService,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["services"] });
    }
  });

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    mutation.mutate({
      kind,
      service_id: serviceId.trim() || undefined,
      model_path: modelPath.trim() || undefined,
      served_model_name: servedModelName.trim() || undefined,
      endpoint: endpoint.trim() || undefined,
      cuda_visible_devices: cudaVisibleDevices.trim() || undefined,
      tensor_parallel_size: tensorParallelSize,
      port,
      max_model_len: maxModelLen,
      gpu_memory_utilization: gpuMemoryUtilization,
      max_num_seqs: maxNumSeqs
    });
  }

  const content = (
    <form className="job-form service-form" onSubmit={submit}>
      <FormSelectControl
        label="类型"
        value={kind}
        options={[
          { value: "local_vllm", label: "本地 vLLM" },
          { value: "external_vllm", label: "外部 vLLM" }
        ]}
        onChange={setKind}
      />
      <TextInputControl label="服务 ID" value={serviceId} onChange={setServiceId} />
      <TextInputControl
        className="wide-field"
        label="模型路径"
        value={modelPath}
        onChange={setModelPath}
        placeholder="outputs/qwen3vl-sft/run/best"
      />
      <TextInputControl
        label="服务模型名"
        value={servedModelName}
        onChange={setServedModelName}
        placeholder="qwen3vl-best"
      />
      <TextInputControl
        className="wide-field"
        label="端点"
        value={endpoint}
        onChange={setEndpoint}
        placeholder="http://127.0.0.1:8000"
      />
      <TextInputControl
        label="CUDA"
        value={cudaVisibleDevices}
        onChange={setCudaVisibleDevices}
        placeholder="0"
      />
      <NumberInputControl
        label="TP 大小"
        min={1}
        value={tensorParallelSize}
        onChange={setTensorParallelSize}
      />
      <NumberInputControl label="端口" min={1} value={port} onChange={setPort} />
      <NumberInputControl
        label="最大上下文"
        min={1}
        value={maxModelLen}
        onChange={setMaxModelLen}
      />
      <NumberInputControl
        label="显存占比"
        min={0}
        max={1}
        step={0.01}
        value={gpuMemoryUtilization}
        onChange={setGpuMemoryUtilization}
      />
      <NumberInputControl
        label="最大并发序列"
        min={1}
        value={maxNumSeqs}
        onChange={setMaxNumSeqs}
      />
      <ActionButton
        variant="primary"
        type="submit"
        icon={<AppIcon name="saveService" size={16} />}
        disabled={mutation.isPending}
      >
        {mutation.isPending ? "保存中" : "保存服务"}
      </ActionButton>
      {mutation.error ? (
        <div className="form-error full-field">服务保存失败：{errorMessage(mutation.error)}</div>
      ) : null}
    </form>
  );
  return bare ? content : <div className="workspace-card compact-form-card">{content}</div>;
}

function ServiceGrid({
  services,
  refreshing = false
}: {
  services: ServiceSummary[];
  refreshing?: boolean;
}) {
  if (services.length === 0) {
    return <EmptyState title="没有符合高级检索条件的模型服务。" />;
  }
  return (
    <div className={refreshing ? "service-grid refreshing" : "service-grid"}>
      {refreshing ? (
        <span className="table-refresh-indicator" aria-live="polite">
          服务列表更新中
        </span>
      ) : null}
      {services.map((service) => (
        <ServiceCard key={service.service_id} service={service} />
      ))}
    </div>
  );
}

function ServiceCard({ service }: { service: ServiceSummary }) {
  const queryClient = useQueryClient();
  const [showLog, setShowLog] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const startMutation = useMutation({
    mutationFn: () => startService(service.service_id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["services"] });
    }
  });
  const healthMutation = useMutation({
    mutationFn: () => checkServiceHealth(service.service_id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["services"] });
    }
  });
  const stopMutation = useMutation({
    mutationFn: () => stopService(service.service_id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["services"] });
    }
  });
  const deleteMutation = useMutation({
    mutationFn: () => deleteService(service.service_id),
    onSuccess: () => {
      setDeleteConfirmOpen(false);
      void queryClient.invalidateQueries({ queryKey: ["services"] });
    }
  });
  const logQuery = useQuery({
    queryKey: ["service-log", service.service_id],
    queryFn: () => fetchServiceLogs(service.service_id),
    enabled: showLog
  });
  const command = Array.isArray(service.runtime.command)
    ? service.runtime.command.map(String).join(" ")
    : "";
  const health = serviceHealth(service);
  return (
    <div className="service-card">
      <div className="service-card-heading">
        <div>
          <h2>{service.service_id}</h2>
          <p>{serviceConfigValue(service, "model_path") || serviceConfigValue(service, "endpoint")}</p>
        </div>
        <Badge value={service.status} domain="service" />
      </div>
      <div className="service-config-grid">
        <ConfigItem label="类型" value={service.kind} />
        <ConfigItem label="服务模型" value={serviceConfigValue(service, "served_model_name")} />
        <ConfigItem label="端点" value={serviceEndpointValue(service)} />
        <ConfigItem label="CUDA" value={serviceConfigValue(service, "cuda_visible_devices")} />
        <ConfigItem label="TP" value={serviceConfigValue(service, "tensor_parallel_size")} />
        <ConfigItem label="端口" value={serviceConfigValue(service, "port")} />
        <ConfigItem label="上下文" value={serviceConfigValue(service, "max_model_len")} />
        <ConfigItem label="显存占比" value={serviceConfigValue(service, "gpu_memory_utilization")} />
        <ConfigItem label="并发序列" value={serviceConfigValue(service, "max_num_seqs")} />
        <ConfigItem label="PID" value={runtimeValue(service, "pid")} />
        <ConfigItem label="健康状态" value={health.status} />
        <ConfigItem label="探测时间" value={health.checkedAt} />
        <ConfigItem label="更新时间" value={formatDate(service.updated_at)} />
      </div>
      <div className={health.ok ? "service-health ok" : "service-health"}>
        <span>{health.ok ? "就绪" : health.status}</span>
        <strong title={health.message}>{health.message}</strong>
      </div>
      {command ? <pre className="service-command">{command}</pre> : null}
      {service.error ? <div className="form-error">{service.error}</div> : null}
      <div className="row-actions">
        <ActionButton
          variant="secondary"
          disabled={!canStartService(service) || startMutation.isPending}
          onClick={() => startMutation.mutate()}
        >
          {startMutation.isPending ? "启动中" : "启动"}
        </ActionButton>
        <ActionButton
          variant="mini"
          disabled={healthMutation.isPending}
          onClick={() => healthMutation.mutate()}
        >
          {healthMutation.isPending ? "探测中" : "探测"}
        </ActionButton>
        <ActionButton
          variant="mini"
          disabled={!canStopService(service) || stopMutation.isPending}
          onClick={() => stopMutation.mutate()}
        >
          {stopMutation.isPending ? "停止中" : "停止"}
        </ActionButton>
        <ActionButton variant="mini" onClick={() => setShowLog((value) => !value)}>
          {showLog ? "隐藏日志" : "日志"}
        </ActionButton>
        <IconActionButton
          icon={<Trash2 size={14} />}
          danger
          disabled={!canDeleteService(service) || deleteMutation.isPending}
          title="删除服务记录"
          onClick={() => setDeleteConfirmOpen(true)}
        />
      </div>
      {showLog ? <ServiceLogPanel query={logQuery} /> : null}
      <DangerConfirmDialog
        open={deleteConfirmOpen}
        title="删除服务记录"
        subject={service.service_id}
        description="服务登记会移入回收站，健康检查、启动命令和 runtime 日志入口会从模型服务页移除。"
        confirmLabel="删除服务"
        pending={deleteMutation.isPending}
        onCancel={() => setDeleteConfirmOpen(false)}
        onConfirm={() => deleteMutation.mutate()}
      />
    </div>
  );
}

function ServiceLogPanel({
  query
}: {
  query: UseQueryResult<ServiceLog, Error>;
}) {
  if (query.isLoading) {
    return <div className="service-log-panel muted-line">正在加载日志</div>;
  }
  if (query.isError || !query.data) {
    return <div className="service-log-panel form-error">日志加载失败：{errorMessage(query.error)}</div>;
  }
  return (
    <div className="service-log-panel">
      <div className="service-log-heading">
        <span>日志尾部</span>
        <strong title={query.data.log_path ?? ""}>{query.data.log_path ?? "没有日志文件"}</strong>
      </div>
      <pre>{query.data.text || "没有日志内容。"}</pre>
    </div>
  );
}
