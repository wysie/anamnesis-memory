from __future__ import annotations

from anamnesis import Anamnesis
from anamnesis.dashboard import DashboardAPI


def test_dashboard_serves_static_shell_and_assets(tmp_path):
    api = DashboardAPI(Anamnesis(tmp_path / "anamnesis.db"))

    status, headers, body = api.handle("GET", "/")
    html = body.decode()

    assert status == 200
    assert headers["content-type"] == "text/html; charset=utf-8"
    assert "Anamnesis" in html
    assert "data-view=\"overview\"" in html
    assert "data-view=\"memories\"" in html
    assert "data-view=\"inbox\"" in html
    assert "data-view=\"preview\"" in html
    assert "data-view=\"simulator\"" in html
    assert "data-view=\"maintenance\"" in html
    assert "data-view=\"runtime\"" in html
    assert "data-view=\"settings\"" in html
    assert "data-view=\"benchmarks\"" not in html
    assert "Smoke benchmark plan" not in html
    assert "Embedding model" in html
    assert "Appearance" in html
    assert "Theme" in html
    assert "Backfill now" in html
    assert "Enable embedding recall" in html
    assert "Enable LLM synthesis" in html
    assert "Set or change password" in html
    assert "Disable password" in html
    assert "clears the current browser session" in html
    assert "Recommended defaults: temperature 0.0, max tokens 512" in html
    assert ">Audit</button>" not in html
    assert "hamburger" in html
    assert "/static/dashboard.css" in html
    assert "/static/dashboard.js" in html

    status, headers, css = api.handle("GET", "/static/dashboard.css")
    css_text = css.decode()
    assert status == 200
    assert headers["content-type"] == "text/css; charset=utf-8"
    assert "oklch(" in css_text
    assert "#000" not in css_text
    assert "#fff" not in css_text
    assert "linear-gradient(90deg" not in css_text
    assert "background-clip: text" not in css_text

    assert "@media (max-width: 820px)" in css_text
    assert ".panel-head { align-items: stretch; flex-direction: column; }" in css_text
    assert ".metric-grid, .card-grid { grid-template-columns: minmax(0, 1fr); }" in css_text

    status, headers, js = api.handle("GET", "/static/dashboard.js")
    js_text = js.decode()
    assert status == 200
    assert headers["content-type"] == "application/javascript; charset=utf-8"
    assert "'/api/overview'" in js_text
    assert "'/api/preview-memory-write'" in js_text
    assert "'/api/auth/status'" in js_text
    assert "'/api/auth/login'" in js_text
    assert "showModal" in js_text
    assert "window.prompt" not in js_text
    assert "window.confirm" not in js_text
    assert "window.alert" not in js_text


def test_dashboard_shell_uses_semantic_action_labels(tmp_path):
    api = DashboardAPI(Anamnesis(tmp_path / "anamnesis.db"))

    _status, _headers, body = api.handle("GET", "/")
    html = body.decode()
    _status, _headers, js = api.handle("GET", "/static/dashboard.js")
    js_text = js.decode()

    assert "Preview Tools" in html
    assert "Run preview" in html
    assert "Run legacy preview check" not in html
    assert "General memory" in html
    assert "Infrastructure" in html
    assert "Open memory" in js_text
    assert "Replace memory" in js_text
    assert "appModal" in html
    assert "modal-card" in html
    assert "Replace this memory?" in js_text
    assert "will be invalidated" in js_text
    assert "Audit trail" in js_text
    assert "Advanced metadata" in js_text
    assert "All platforms" in js_text
    assert "User memory import" in js_text
    assert "data-open-audit" not in js_text
    assert "/api/settings" in js_text
    assert "settingsEmbeddingEnabled" in js_text
    assert "settingsLlmEnabled" in js_text
    assert "Search" in html
    assert "Recall Simulator" in html
    assert "Autopilot hygiene" in html
    assert "default 30-day threshold" in html
    assert "Integration status" in html
    assert "Test recall now" in html
    assert "runtimeTestQuery" in html
    assert "runRuntimeTestRecall" in js_text
    assert "/api/runtime/test-recall" in js_text
    assert "Smoke benchmark plan" not in html
    assert "savePassword" in js_text
    assert "clearPassword" in js_text
    assert "applyTheme" in js_text
    assert "backfillEmbeddings" in js_text
    assert "is missing vectors for" in js_text
    assert "Backfill embeddings now?" in js_text
    assert "Later" in js_text
    assert "runSimulator" in js_text
    assert "runMaintenance" in js_text
    assert "loadRuntime" in js_text
    assert "loadBenchmarks" not in js_text
    assert "history.pushState" in js_text
    assert "popstate" in js_text
    assert "saveEmbedding" in js_text
    assert "saveLlm" in js_text
    assert "Invalidate selected" in html
    assert "Superseded" in html
    assert "Invalidated" in html
    assert "Accept memory" in js_text
    assert "Reject proposal" in js_text
    assert "Confidence" in js_text
    assert "Confidence" in html
    assert "inboxConfidence" in html
    assert "min_confidence" in js_text
    assert "max_confidence" in js_text
    assert "Superseded" in js_text
    assert "Invalidated" in js_text
    assert ">OK<" not in html
    assert ">Submit<" not in html
    assert ">Yes<" not in html
