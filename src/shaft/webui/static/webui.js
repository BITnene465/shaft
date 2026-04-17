(function () {
  const initialState = window.__SHAFT_INITIAL_STATE__ || {};

  const elements = {
    configPath: document.getElementById("config-path"),
    yamlEditor: document.getElementById("yaml-editor"),
    resolvedYaml: document.getElementById("resolved-yaml"),
    logViewer: document.getElementById("log-viewer"),
    statusHtml: document.getElementById("status-html"),
    runsTableBody: document.getElementById("runs-table-body"),
    runSelector: document.getElementById("run-selector"),
    themeToggle: document.getElementById("theme-toggle"),
    themeValue: document.querySelector("[data-shaft-theme-value]"),
    loadConfigBtn: document.getElementById("load-config-btn"),
    validateBtn: document.getElementById("validate-btn"),
    startBtn: document.getElementById("start-btn"),
    stopBtn: document.getElementById("stop-btn"),
    refreshBtn: document.getElementById("refresh-btn"),
    openRunBtn: document.getElementById("open-run-btn"),
    runId: document.getElementById("run-id"),
    seed: document.getElementById("seed"),
    finetuneMode: document.getElementById("finetune-mode"),
    epochs: document.getElementById("epochs"),
    learningRate: document.getElementById("learning-rate"),
    mixStrategy: document.getElementById("mix-strategy"),
    trainBatchSize: document.getElementById("train-batch-size"),
    evalBatchSize: document.getElementById("eval-batch-size"),
  };

  let currentRunId = "";
  let refreshTimer = null;

  function setTheme(theme) {
    const resolvedTheme = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-shaft-theme", resolvedTheme);
    localStorage.setItem("shaft-webui-theme", resolvedTheme);
    if (elements.themeValue) {
      elements.themeValue.textContent = resolvedTheme === "dark" ? "Dark" : "Light";
    }
  }

  function toggleTheme() {
    const nextTheme = document.documentElement.getAttribute("data-shaft-theme") === "dark" ? "light" : "dark";
    setTheme(nextTheme);
  }

  function hydrateTheme() {
    const savedTheme = localStorage.getItem("shaft-webui-theme");
    setTheme(savedTheme || "light");
  }

  function collectFormPayload() {
    return {
      run_id: elements.runId ? elements.runId.value : "",
      seed: elements.seed ? elements.seed.value : "",
      finetune_mode: elements.finetuneMode ? elements.finetuneMode.value : "",
      epochs: elements.epochs ? elements.epochs.value : "",
      learning_rate: elements.learningRate ? elements.learningRate.value : "",
      mix_strategy: elements.mixStrategy ? elements.mixStrategy.value : "",
      train_batch_size: elements.trainBatchSize ? elements.trainBatchSize.value : "",
      eval_batch_size: elements.evalBatchSize ? elements.evalBatchSize.value : "",
    };
  }

  function renderRuns(rows) {
    if (!elements.runsTableBody) {
      return;
    }
    const html = (rows || [])
      .map((row) => {
        const values = [
          row.run_id,
          row.status,
          row.pid,
          row.return_code,
          row.output_dir,
          row.started_at,
        ];
        const canDelete = row.is_terminal === "true";
        const deleteButton = canDelete
          ? `<button type="button" class="shaft-inline-delete" data-shaft-delete-run="${escapeAttribute(row.run_id)}">Delete</button>`
          : `<span class="shaft-inline-delete shaft-inline-delete-disabled">Running</span>`;
        return `<tr>${values.map((value) => `<td>${escapeHtml(String(value ?? "-"))}</td>`).join("")}<td>${deleteButton}</td></tr>`;
      })
      .join("");
    elements.runsTableBody.innerHTML = html || '<tr><td colspan="7">No runs tracked yet.</td></tr>';
  }

  function renderRunChoices(choices, selectedRun) {
    if (!elements.runSelector) {
      return;
    }
    const html = (choices || [])
      .map((runId) => {
        const selected = runId === selectedRun ? " selected" : "";
        return `<option value="${escapeAttribute(runId)}"${selected}>${escapeHtml(runId)}</option>`;
      })
      .join("");
    elements.runSelector.innerHTML = html;
    if (selectedRun && !choices.includes(selectedRun)) {
      elements.runSelector.value = "";
    }
  }

  function applyState(state) {
    if (elements.configPath && typeof state.config_path === "string") {
      elements.configPath.value = state.config_path;
    }
    if (elements.yamlEditor && typeof state.yaml_text === "string") {
      elements.yamlEditor.value = state.yaml_text;
    }
    if (elements.statusHtml && typeof state.status_html === "string") {
      elements.statusHtml.innerHTML = state.status_html;
    }
    if (elements.resolvedYaml && typeof state.resolved_yaml === "string") {
      elements.resolvedYaml.value = state.resolved_yaml;
    }
    if (elements.logViewer && typeof state.log_text === "string") {
      elements.logViewer.value = state.log_text;
      elements.logViewer.scrollTop = elements.logViewer.scrollHeight;
    }
    if (Array.isArray(state.runs)) {
      renderRuns(state.runs);
    }
    if (Array.isArray(state.run_choices)) {
      renderRunChoices(state.run_choices, state.selected_run || "");
    }
    if (typeof state.current_run_id === "string") {
      currentRunId = state.current_run_id;
    }
    syncAutoRefresh();
  }

  function setBusy(isBusy) {
    document.body.classList.toggle("shaft-is-busy", isBusy);
    [
      elements.loadConfigBtn,
      elements.validateBtn,
      elements.startBtn,
      elements.stopBtn,
      elements.refreshBtn,
      elements.openRunBtn,
    ].forEach((button) => {
      if (button) {
        button.disabled = isBusy;
      }
    });
  }

  async function postJson(url, payload) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
  }

  async function performAction(url, payload, options = {}) {
    const useBusyState = options.busy !== false;
    if (useBusyState) {
      setBusy(true);
    }
    try {
      const state = await postJson(url, payload);
      applyState(state);
    } catch (error) {
      if (elements.statusHtml) {
        elements.statusHtml.innerHTML = `
          <div class="shaft-card shaft-status-card">
            <div class="shaft-note shaft-note-error">Web UI request failed: ${escapeHtml(String(error))}</div>
          </div>
        `;
      }
    } finally {
      if (useBusyState) {
        setBusy(false);
      }
    }
  }

  function escapeHtml(value) {
    return value
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function escapeAttribute(value) {
    return escapeHtml(value);
  }

  function syncAutoRefresh() {
    if (refreshTimer !== null) {
      window.clearInterval(refreshTimer);
      refreshTimer = null;
    }
    if (!currentRunId) {
      return;
    }
    refreshTimer = window.setInterval(() => {
      if (document.hidden) {
        return;
      }
      performAction("/api/refresh", { current_run_id: currentRunId }, { busy: false });
    }, 4000);
  }

  function bindActions() {
    if (elements.themeToggle) {
      elements.themeToggle.addEventListener("click", toggleTheme);
    }

    if (!elements.configPath) {
      return;
    }

    elements.loadConfigBtn.addEventListener("click", () => {
      performAction("/api/load-config", { config_path: elements.configPath.value });
    });

    elements.validateBtn.addEventListener("click", () => {
      performAction("/api/validate", {
        config_path: elements.configPath.value,
        yaml_text: elements.yamlEditor.value,
        form: collectFormPayload(),
      });
    });

    elements.startBtn.addEventListener("click", () => {
      performAction("/api/start", {
        config_path: elements.configPath.value,
        yaml_text: elements.yamlEditor.value,
        form: collectFormPayload(),
      });
    });

    elements.stopBtn.addEventListener("click", () => {
      performAction("/api/stop", { current_run_id: currentRunId });
    });

    elements.refreshBtn.addEventListener("click", () => {
      performAction("/api/refresh", { current_run_id: currentRunId });
    });

    elements.openRunBtn.addEventListener("click", () => {
      performAction("/api/load-run", { run_id: elements.runSelector.value });
    });

    if (elements.runsTableBody) {
      elements.runsTableBody.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof HTMLElement)) {
          return;
        }
        const runId = target.getAttribute("data-shaft-delete-run");
        if (!runId) {
          return;
        }
        performAction("/api/delete-run", {
          run_id: runId,
          current_run_id: currentRunId,
        });
      });
    }
  }

  hydrateTheme();
  applyState(initialState);
  bindActions();
})();
