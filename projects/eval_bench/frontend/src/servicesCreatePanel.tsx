import { useState } from "react";
import type { FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { createService } from "./api";
import { FormSelectControl, NumberInputControl, TextInputControl } from "./controlPrimitives";
import { errorMessage } from "./formatters";
import { AppIcon } from "./iconLibrary";
import { ActionButton } from "./ui";
import "./formControls.css";

export function ServiceCreatePanel({ bare }: { bare?: boolean }) {
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
