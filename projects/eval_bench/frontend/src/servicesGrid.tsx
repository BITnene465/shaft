import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { UseQueryResult } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";

import type { ServiceLog, ServiceSummary } from "./api";
import {
  checkServiceHealth,
  deleteService,
  fetchServiceLogs,
  startService,
  stopService
} from "./api";
import {
  errorMessage,
  formatDate,
  runtimeValue,
  serviceConfigValue,
  serviceEndpointValue,
  serviceHealth
} from "./formatters";
import {
  canDeleteService,
  canStartService,
  canStopService
} from "./statusModel";
import {
  ActionButton,
  Badge,
  ConfigItem,
  DangerConfirmDialog,
  IconActionButton,
  TableEmptyState
} from "./ui";

export function ServiceGrid({
  services,
  refreshing = false
}: {
  services: ServiceSummary[];
  refreshing?: boolean;
}) {
  if (services.length === 0) {
    return (
      <TableEmptyState
        emptyText="没有符合高级检索条件的模型服务。"
        refreshing={refreshing}
        refreshLabel="服务列表更新中"
      />
    );
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
    queryFn: ({ signal }) => fetchServiceLogs(service.service_id, { signal }),
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
