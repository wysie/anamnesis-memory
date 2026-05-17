const state = { view: 'overview', memoryOffset: 0, inboxOffset: 0 };

function applyTheme(theme) {
  const chosen = theme || localStorage.getItem('anamnesis-theme') || 'system';
  document.documentElement.dataset.theme = chosen === 'system' ? '' : chosen;
  localStorage.setItem('anamnesis-theme', chosen);
  const select = document.querySelector('#themeSelect');
  if (select) select.value = chosen;
}

applyTheme(localStorage.getItem('anamnesis-theme') || 'system');

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  }[char]));
}

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

async function getJson(path) {
  const response = await fetch(path);
  if (response.status === 401) {
    const ok = await ensureAuthenticated(true);
    if (ok) return getJson(path);
  }
  return response.json();
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (response.status === 401) {
    const ok = await ensureAuthenticated(true);
    if (ok) return postJson(path, payload);
  }
  return response.json();
}

async function ensureAuthenticated(forcePrompt = false) {
  const status = await fetch('/api/auth/status').then((response) => response.json());
  if (!status.password_required || status.authenticated) return true;
  const password = await showModal({
    title: 'Dashboard password',
    body: forcePrompt ? 'Enter the password to continue.' : 'This local dashboard is password protected.',
    confirmText: 'Unlock dashboard',
    cancelText: 'Cancel',
    input: { label: 'Password', type: 'password', autocomplete: 'current-password' },
  });
  if (!password) return false;
  const result = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ password }),
  });
  if (!result.ok) {
    await showModal({
      title: 'Incorrect password',
      body: 'That password did not unlock the dashboard. Try again when ready.',
      confirmText: 'Got it',
      cancelText: '',
    });
    return false;
  }
  return true;
}

function showModal(options) {
  return new Promise((resolve) => {
    const modal = $('#appModal');
    const inputWrap = $('#modalInputWrap');
    const input = $('#modalInput');
    const cancel = $('#modalCancelButton');
    const confirm = $('#modalConfirmButton');
    $('#modalEyebrow').textContent = options.eyebrow || 'Anamnesis';
    $('#modalTitle').textContent = options.title || 'Confirm action';
    $('#modalBody').textContent = options.body || '';
    confirm.textContent = options.confirmText || 'Continue';
    cancel.textContent = options.cancelText || 'Cancel';
    cancel.style.display = options.cancelText === '' ? 'none' : '';
    if (options.input) {
      inputWrap.classList.remove('hidden');
      inputWrap.childNodes[0].textContent = `${options.input.label || 'Value'}\n`;
      input.type = options.input.type || 'text';
      input.autocomplete = options.input.autocomplete || 'off';
      input.value = '';
    } else {
      inputWrap.classList.add('hidden');
      input.value = '';
    }
    modal.classList.add('open');
    modal.setAttribute('aria-hidden', 'false');
    const finish = (value) => {
      modal.classList.remove('open');
      modal.setAttribute('aria-hidden', 'true');
      confirm.removeEventListener('click', onConfirm);
      cancel.removeEventListener('click', onCancel);
      modal.removeEventListener('click', onBackdrop);
      document.removeEventListener('keydown', onKeydown);
      resolve(value);
    };
    const onConfirm = () => finish(options.input ? input.value : true);
    const onCancel = () => finish(false);
    const onBackdrop = (event) => {
      if (event.target === modal) finish(false);
    };
    const onKeydown = (event) => {
      if (event.key === 'Escape') finish(false);
      if (event.key === 'Enter' && options.input) onConfirm();
    };
    confirm.addEventListener('click', onConfirm);
    cancel.addEventListener('click', onCancel);
    modal.addEventListener('click', onBackdrop);
    document.addEventListener('keydown', onKeydown);
    setTimeout(() => (options.input ? input : confirm).focus(), 0);
  });
}

function viewUrl(view) {
  const url = new URL(window.location.href);
  url.searchParams.set('view', view);
  return url;
}

function navigateToView(view) {
  if (view === state.view) return;
  history.pushState({ view }, '', viewUrl(view));
  showView(view);
}

function showView(view) {
  state.view = view;
  $$('.view').forEach((node) => node.classList.toggle('active', node.id === `view-${view}`));
  $$('[data-view]').forEach((node) => node.classList.toggle('active', node.dataset.view === view));
  $('[data-menu]')?.classList.remove('open');
  $('[data-menu-toggle]')?.setAttribute('aria-expanded', 'false');
  if (view === 'overview') loadOverview();
  if (view === 'memories') loadMemories();
  if (view === 'inbox') loadInbox();
  if (view === 'settings') loadSettings();
  if (view === 'runtime') loadRuntime();
}

function badge(text) {
  return `<span class="badge">${escapeHtml(text)}</span>`;
}

function statusLabel(status) {
  return ({ active: 'Active', superseded: 'Superseded', invalidated: 'Invalidated' })[status] || status;
}

function scopeLabel(scope) {
  if (!scope || scope === 'all') return 'All platforms';
  return scope;
}

function sourceLabel(source) {
  return ({
    'yantrikdb:user': 'User memory import',
    'yantrikdb:memory': 'Legacy memory import',
    'yantrikdb:mnemosyne_import': 'External memory provider import',
    'mnemosyne:episodic_memory': 'Episodic memory import',
  })[source] || source || 'Unknown source';
}

function field(label, value) {
  return `<div class="field"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value || '—')}</strong></div>`;
}

function confidenceLabel(confidence) {
  const value = Number(confidence);
  if (!Number.isFinite(value)) return 'Confidence —';
  return `Confidence ${Math.round(value * 100)}%`;
}

function rowCheckbox(kind, id) {
  return `<label class="row-check" aria-label="Select item"><input type="checkbox" data-select-${kind}="${escapeHtml(id)}"></label>`;
}

function memoryCard(item) {
  return `<article class="card selectable" data-rid="${escapeHtml(item.rid)}">
    ${rowCheckbox('memory', item.rid)}
    <div class="title">${escapeHtml(item.text)}</div>
    <div class="meta">${badge(statusLabel(item.status))}${badge(item.owner)}${badge(item.domain || 'general')}${badge(item.platform_scope)}</div>
    <button class="button ghost" type="button" data-open-memory="${escapeHtml(item.rid)}">Open memory</button>
  </article>`;
}

function inboxCard(item) {
  const actions = item.decision === 'pending' ? `<div class="card-actions">
    <button class="button primary" type="button" data-accept-inbox="${escapeHtml(item.cid)}">Accept memory</button>
    <button class="button ghost" type="button" data-reject-inbox="${escapeHtml(item.cid)}">Reject proposal</button>
  </div>` : '';
  return `<article class="card selectable" data-cid="${escapeHtml(item.cid)}">
    ${rowCheckbox('inbox', item.cid)}
    <div class="title">${escapeHtml(item.proposed_text)}</div>
    <p>${escapeHtml(item.why_save || item.review_reason || 'Awaiting review')}</p>
    <div class="meta">${badge(item.decision)}${badge(confidenceLabel(item.confidence))}${badge(item.owner)}${badge(item.domain || 'general')}${badge(sourceLabel(item.source))}${badge(scopeLabel(item.platform_scope))}</div>
    ${actions}
  </article>`;
}

function empty(text) {
  return `<div class="empty">${escapeHtml(text)}</div>`;
}

function formatCount(value) {
  return new Intl.NumberFormat('en-SG').format(Number(value || 0));
}

function selectedValues(selector) {
  return $$(selector).filter((node) => node.checked).map((node) => node.dataset.selectMemory || node.dataset.selectInbox);
}

function pageInfo(payload) {
  if (!payload.total) return '0 results';
  const start = payload.offset + 1;
  const end = Math.min(payload.offset + payload.items.length, payload.total);
  return `${formatCount(start)}–${formatCount(end)} of ${formatCount(payload.total)}`;
}

function setPager(prefix, payload) {
  $(`#${prefix}PageInfo`).textContent = pageInfo(payload);
  $(`#prev${prefix[0].toUpperCase()}${prefix.slice(1)}Button`).disabled = !payload.has_prev;
  $(`#next${prefix[0].toUpperCase()}${prefix.slice(1)}Button`).disabled = !payload.has_next;
}

function populateSelect(select, rows, allLabel) {
  const current = select.value;
  select.innerHTML = `<option value="">${escapeHtml(allLabel)}</option>` + rows.map((row) => (
    `<option value="${escapeHtml(row.value)}">${escapeHtml(row.value)} (${formatCount(row.count)})</option>`
  )).join('');
  if (rows.some((row) => row.value === current)) select.value = current;
}

function mergeFacetRows(...groups) {
  const counts = new Map();
  groups.flat().forEach((row) => counts.set(row.value, (counts.get(row.value) || 0) + Number(row.count || 0)));
  return Array.from(counts, ([value, count]) => ({ value, count })).sort((a, b) => b.count - a.count || a.value.localeCompare(b.value));
}

function populateFixedSelect(select, values) {
  const current = select.value;
  select.innerHTML = values.map(({ value, label }) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`).join('');
  if (values.some((row) => row.value === current)) select.value = current;
}

function populateShadowControls(facets) {
  const owners = mergeFacetRows(facets.memories.owners, facets.inbox.owners);
  populateFixedSelect($('#previewOwner'), owners.length ? owners.map((row) => ({ value: row.value, label: `${row.value} (${formatCount(row.count)})` })) : [{ value: 'primary', label: 'primary' }]);

  const knownPlatforms = [
    { value: 'whatsapp', label: 'WhatsApp' },
    { value: 'telegram', label: 'Telegram' },
    { value: 'cli', label: 'CLI' },
    { value: 'local', label: 'Local' },
    { value: 'all', label: 'All platforms' },
  ];
  const extraPlatforms = mergeFacetRows(facets.memories.platforms || [], facets.inbox.platforms || [])
    .filter((row) => !knownPlatforms.some((platform) => platform.value === row.value))
    .map((row) => ({ value: row.value, label: `${scopeLabel(row.value)} (${formatCount(row.count)})` }));
  populateFixedSelect($('#previewPlatform'), [...knownPlatforms, ...extraPlatforms]);

  const knownTargets = [
    { value: 'memory', label: 'General memory' },
    { value: 'preference', label: 'Preference' },
    { value: 'work', label: 'Work' },
    { value: 'project', label: 'Project' },
    { value: 'infrastructure', label: 'Infrastructure' },
    { value: 'integration', label: 'Integration' },
    { value: 'ops', label: 'Ops' },
    { value: 'personal', label: 'Personal' },
  ];
  const extraTargets = mergeFacetRows(facets.memories.domains || [], facets.inbox.domains || [])
    .filter((row) => row.value && !knownTargets.some((target) => target.value === row.value))
    .map((row) => ({ value: row.value, label: `${row.value} (${formatCount(row.count)})` }));
  populateFixedSelect($('#previewTarget'), [...knownTargets, ...extraTargets]);
}

async function loadFacets() {
  const facets = await getJson('/api/facets');
  populateSelect($('#memoryOwner'), facets.memories.owners, 'All owners');
  populateSelect($('#inboxOwner'), facets.inbox.owners, 'All owners');
  populateSelect($('#inboxSource'), facets.inbox.sources, 'All sources');
  const simOwners = mergeFacetRows(facets.memories.owners, facets.inbox.owners);
  populateFixedSelect($('#simOwner'), simOwners.length ? simOwners.map((row) => ({ value: row.value, label: `${row.value} (${formatCount(row.count)})` })) : [{ value: 'primary', label: 'primary' }]);
  populateSelect($('#maintenanceOwner'), mergeFacetRows(facets.memories.owners, facets.inbox.owners), 'All owners');
  populateFixedSelect($('#runtimeTestOwner'), simOwners.length ? simOwners.map((row) => ({ value: row.value, label: `${row.value} (${formatCount(row.count)})` })) : [{ value: 'primary', label: 'primary' }]);
  populateShadowControls(facets);
}

async function loadOverview() {
  const payload = await getJson('/api/overview');
  $('#overviewMetrics').innerHTML = [
    ['Active', payload.counts.memories.active],
    ['Superseded', payload.counts.memories.superseded],
    ['Invalidated', payload.counts.memories.invalidated],
    ['Pending inbox', payload.counts.inbox.pending],
  ].map(([label, value]) => `<section class="metric"><span class="metric-value">${formatCount(value)}</span><span class="metric-label">${label}</span></section>`).join('');
  $('#recentMemories').innerHTML = payload.recent_memories.length ? payload.recent_memories.map(memoryCard).join('') : empty('No active memories yet.');
  $('#recentInbox').innerHTML = payload.recent_inbox.length ? payload.recent_inbox.map(inboxCard).join('') : empty('No inbox exceptions.');
}

async function loadMemories() {
  const params = new URLSearchParams({
    status: $('#memoryStatus').value || 'active',
    limit: $('#memoryLimit').value || '50',
    offset: String(state.memoryOffset),
  });
  if ($('#memoryOwner').value) params.set('owner', $('#memoryOwner').value);
  if ($('#memorySearch').value.trim()) params.set('q', $('#memorySearch').value.trim());
  const payload = await getJson(`/api/memories?${params.toString()}`);
  $('#selectAllMemories').checked = false;
  $('#memoryGrid').innerHTML = payload.items.length ? payload.items.map(memoryCard).join('') : empty('No memories match these filters. Try changing Owner or Status.');
  setPager('memory', payload);
}

function inboxConfidenceFilter(selectValue) {
  return {
    high: { min_confidence: '0.9' },
    good: { min_confidence: '0.75' },
    medium: { min_confidence: '0.5' },
    low: { max_confidence: '0.5' },
  }[selectValue] || {};
}

async function loadInbox() {
  const params = new URLSearchParams({
    decision: $('#inboxDecision').value || 'pending',
    limit: $('#inboxLimit').value || '50',
    offset: String(state.inboxOffset),
  });
  if ($('#inboxOwner').value) params.set('owner', $('#inboxOwner').value);
  Object.entries(inboxConfidenceFilter($('#inboxConfidence').value)).forEach(([key, value]) => params.set(key, value));
  if ($('#inboxSearch').value.trim()) params.set('q', $('#inboxSearch').value.trim());
  if ($('#inboxSource').value) params.set('source', $('#inboxSource').value);
  const payload = await getJson(`/api/inbox?${params.toString()}`);
  $('#selectAllInbox').checked = false;
  $('#inboxGrid').innerHTML = payload.items.length ? payload.items.map(inboxCard).join('') : empty('No inbox items match these filters.');
  setPager('inbox', payload);
}

async function runPreviewCheck() {
  const payload = await postJson('/api/preview-memory-write', {
    text: $('#previewText').value,
    owner: $('#previewOwner').value || 'primary',
    platform: $('#previewPlatform').value || 'whatsapp',
    target: $('#previewTarget').value || 'memory',
    origin: 'dashboard',
  });
  $('#previewResult').textContent = prettyJson(payload);
}

async function openAudit(rid = $('#auditRid').value) {
  const payload = await getJson(`/api/audit/${encodeURIComponent(rid)}`);
  $('#auditRid').value = rid;
  $('#auditResult').textContent = prettyJson(payload);
  $('#drawerBody').innerHTML = memoryDetail(payload);
  $('#detailDrawer').classList.add('open');
  $('#detailDrawer').setAttribute('aria-hidden', 'false');
}

function memoryDetail(payload) {
  const memory = payload.memory;
  const events = payload.events || [];
  const replacement = payload.correction_chain?.replacement_rid
    ? `<p class="drawer-note">This memory was replaced by <code>${escapeHtml(payload.correction_chain.replacement_rid)}</code>.</p>`
    : '';
  return `<div class="drawer-stack">
    <p class="eyebrow">Memory detail</p>
    <h2>${escapeHtml(statusLabel(memory.status))} memory</h2>
    <p class="drawer-memory-text">${escapeHtml(memory.text)}</p>
    <div class="field-grid">
      ${field('Status', statusLabel(memory.status))}
      ${field('Owner', memory.owner)}
      ${field('Domain', memory.domain || 'General')}
      ${field('Scope', scopeLabel(memory.platform_scope))}
    </div>
    <details class="advanced-meta">
      <summary>Advanced metadata</summary>
      <div class="field-grid">
        ${field('Memory ID', memory.rid)}
        ${field('Source', sourceLabel(memory.source))}
        ${field('Visibility', memory.visibility)}
        ${field('Confidence', memory.confidence)}
      </div>
    </details>
    ${replacement}
    <section class="drawer-section">
      <h3>Replace memory</h3>
      <label>Replacement text<textarea id="drawerCorrectText" rows="5">${escapeHtml(memory.text)}</textarea></label>
      <label>Reason<input id="drawerCorrectReason" value="dashboard replacement" autocomplete="off"></label>
      <button class="button danger" type="button" data-replace-memory="${escapeHtml(memory.rid)}">Replace memory</button>
      <p class="drawer-note">Replacing invalidates this memory and creates a corrected active copy.</p>
    </section>
    <section class="drawer-section">
      <h3>Audit trail</h3>
      ${events.length ? `<div class="audit-list">${events.map(auditEvent).join('')}</div>` : empty('No audit events recorded.')}
    </section>
  </div>`;
}

function auditEvent(event) {
  return `<article class="audit-event">
    <strong>${escapeHtml(event.event_type)}</strong>
    <span>${escapeHtml(event.reason || 'No reason recorded')}</span>
    <code>${escapeHtml(new Date(Number(event.created_at || 0) * 1000).toLocaleString('en-SG'))}</code>
  </article>`;
}

async function replaceMemoryFromDrawer(rid) {
  const text = $('#drawerCorrectText')?.value || '';
  const reason = $('#drawerCorrectReason')?.value || 'dashboard replacement';
  if (!text.trim()) return;
  const confirmed = await showModal({
    title: 'Replace this memory?',
    body: 'The current memory will be invalidated and a corrected active copy will be created. This keeps the old row for audit but removes it from normal recall.',
    confirmText: 'Replace memory',
    cancelText: 'Keep current',
  });
  if (!confirmed) return;
  const payload = await postJson('/api/correct', { rid, text, reason });
  $('#auditResult').textContent = prettyJson(payload);
  await loadFacets();
  if (state.view === 'memories') await loadMemories();
  await loadOverview();
  await openAudit(payload.replacement.rid);
}

async function correctMemory() {
  const payload = await postJson('/api/correct', {
    rid: $('#auditRid').value,
    text: $('#correctText').value,
    reason: $('#correctReason').value || 'dashboard correction',
  });
  $('#auditResult').textContent = prettyJson(payload);
  await loadOverview();
}

async function acceptInbox(cid) {
  const payload = await postJson('/api/inbox/accept', { cid });
  $('#auditResult').textContent = prettyJson(payload);
  await loadFacets();
  await loadInbox();
  await loadOverview();
}

async function rejectInbox(cid) {
  const payload = await postJson('/api/inbox/reject', { cid, reason: 'dashboard rejection' });
  $('#auditResult').textContent = prettyJson(payload);
  await loadFacets();
  await loadInbox();
  await loadOverview();
}

async function batchInbox(action) {
  const cids = selectedValues('[data-select-inbox]');
  if (!cids.length) return;
  const payload = await postJson('/api/inbox/batch', { action, cids, reason: 'dashboard batch action' });
  $('#auditResult').textContent = prettyJson(payload);
  await loadFacets();
  await loadInbox();
  await loadOverview();
}

async function batchMemories(action) {
  const rids = selectedValues('[data-select-memory]');
  if (!rids.length) return;
  const payload = await postJson('/api/memories/batch', { action, rids, reason: 'dashboard batch action' });
  $('#auditResult').textContent = prettyJson(payload);
  await loadFacets();
  await loadMemories();
  await loadOverview();
}

function resetMemoryPage() {
  state.memoryOffset = 0;
  loadMemories();
}

function resetInboxPage() {
  state.inboxOffset = 0;
  loadInbox();
}

async function loadSettings() {
  applyTheme(localStorage.getItem('anamnesis-theme') || 'system');
  const payload = await getJson('/api/settings');
  const models = payload.embedding.available_models;
  const active = payload.embedding.active_model;
  $('#settingsEmbeddingEnabled').checked = Boolean(payload.embedding.enabled);
  $('#settingsEmbeddingModel').innerHTML = models.map((m) => `<option value="${escapeHtml(m.name)}" ${m.name === active ? 'selected' : ''}>${escapeHtml(m.name)} (${m.dimension}d)</option>`).join('');
  if (active) $('#settingsEmbeddingModel').value = active;
  const synth = payload.synthesis;
  $('#settingsLlmEnabled').checked = Boolean(synth.enabled);
  if (synth.base_url) $('#settingsLlmUrl').value = synth.base_url;
  if (synth.model) $('#settingsLlmModel').value = synth.model;
  if (synth.api_key_env) $('#settingsLlmApiKeyEnv').value = synth.api_key_env;
  $('#settingsLlmTemp').value = synth.temperature;
  $('#settingsLlmMaxTokens').value = synth.max_tokens;
  $('#settingsLlmTimeout').value = synth.timeout;
  $('#settingsPasswordStatus').textContent = payload.dashboard_password_set
    ? 'Password is currently enabled.'
    : 'Password is currently disabled. Anyone on this local network URL can open the dashboard.';
  $('#settingsPassword').value = '';
  await loadEmbeddingStatus();
}

async function savePassword() {
  const password = $('#settingsPassword').value;
  if (!password) {
    $('#settingsResult').textContent = 'Enter a new password, or use Disable password.';
    return;
  }
  const confirmed = await showModal({
    title: 'Change dashboard password?',
    body: 'Your current browser session will be cleared. After this save, the next dashboard action will ask for the new password.',
    confirmText: 'Change password',
    cancelText: 'Cancel',
  });
  if (!confirmed) return;
  const payload = await postJson('/api/settings/dashboard-password', { password });
  $('#settingsResult').textContent = prettyJson(payload);
  $('#settingsPassword').value = '';
  $('#settingsPasswordStatus').textContent = 'Password is currently enabled. Session cleared; use the new password on the next prompt.';
}

async function clearPassword() {
  const confirmed = await showModal({
    title: 'Disable dashboard password?',
    body: 'The dashboard will be accessible without a password from this local network URL. Your current browser session cookie will be cleared too.',
    confirmText: 'Disable password',
    cancelText: 'Cancel',
  });
  if (!confirmed) return;
  const payload = await postJson('/api/settings/dashboard-password/clear', {});
  $('#settingsResult').textContent = prettyJson(payload);
  $('#settingsPassword').value = '';
  $('#settingsPasswordStatus').textContent = 'Password is currently disabled. Anyone on this local network URL can open the dashboard.';
}

async function loadEmbeddingStatus() {
  const status = await getJson('/api/embedding/status');
  $('#embeddingStatusGrid').innerHTML = [
    field('Enabled', status.enabled ? 'Yes' : 'No'),
    field('Model', status.active_model || status.model_id),
    field('Active memories', formatCount(status.total_active)),
    field('Embedded', formatCount(status.embedded)),
    field('Missing', formatCount(status.missing)),
    field('Fallback', status.fts_fallback ? 'Keyword recall available' : 'None'),
  ].join('');
  return status;
}

async function backfillEmbeddings() {
  const model = $('#settingsEmbeddingModel').value;
  const confirmed = await showModal({
    title: 'Backfill embeddings now?',
    body: `Generate missing vectors for ${model}. This may take a while on large stores, but keyword recall keeps working meanwhile.`,
    confirmText: 'Backfill now',
    cancelText: 'Cancel',
  });
  if (!confirmed) return;
  const payload = await postJson('/api/embedding/backfill', { model });
  $('#settingsResult').textContent = prettyJson(payload);
  await loadEmbeddingStatus();
}

async function runSimulator() {
  const payload = await postJson('/api/recall/simulate', {
    query: $('#simQuery').value,
    owner: $('#simOwner').value || 'primary',
    platform: $('#simPlatform').value || 'whatsapp',
    domain: $('#simDomain').value,
    limit: Number($('#simLimit').value) || 10,
    visibility: ['private'],
  });
  $('#simResult').textContent = prettyJson(payload);
}

function memorySnippet(text, limit = 160) {
  const clean = String(text || '').replace(/\s+/g, ' ').trim();
  if (clean.length <= limit) return clean;
  return `${clean.slice(0, limit - 1)}…`;
}

function shortId(value) {
  const text = String(value || '');
  return text.length > 8 ? `${text.slice(0, 8)}…` : text || '—';
}

function renderMaintenanceResult(payload) {
  const summary = payload.summary || {};
  const lines = [];
  lines.push(payload.mode === 'dry_run' ? 'Dry-run preview' : 'Applied maintenance');
  lines.push('');
  if (payload.mode === 'dry_run') {
    lines.push(`Inbox items that would expire: ${formatCount(summary.stale_pending_inbox || 0)}`);
    lines.push(`Active memories considered: ${formatCount(summary.active_memories_considered || 0)}`);
    lines.push(`Duplicate pairs compared: ${formatCount(summary.duplicate_pairs_compared || 0)}`);
    lines.push(`Memories that would be marked superseded: ${formatCount(summary.would_supersede_duplicates || 0)}`);
    lines.push('');
    const inbox = payload.would_expire_inbox || [];
    lines.push(inbox.length ? 'Inbox examples to expire:' : 'Inbox examples to expire: none');
    inbox.forEach((item, idx) => lines.push(`${idx + 1}. ${memorySnippet(item.proposed_text || item.source_snippet)}`));
    lines.push('');
    const duplicates = payload.would_supersede_duplicates || [];
    lines.push(duplicates.length ? 'Duplicate examples to supersede:' : 'Duplicate examples to supersede: none');
    duplicates.forEach((item, idx) => {
      lines.push(`${idx + 1}. Overlap ${Math.round(Number(item.overlap || 0) * 100)}%`);
      lines.push(`   Keep: ${memorySnippet(item.canonical_text)}`);
      lines.push(`   Supersede: ${memorySnippet(item.superseded_text)}`);
    });
    if (summary.truncated_duplicate_examples) {
      lines.push(`... plus ${formatCount(summary.truncated_duplicate_examples)} more duplicate example(s)`);
    }
    return lines.join('\n');
  }

  lines.push(`Expired inbox items: ${formatCount(summary.expired_inbox || 0)}`);
  lines.push(`Superseded duplicate memories: ${formatCount(summary.superseded_duplicates || 0)}`);
  lines.push('');
  const expired = payload.expired_inbox || [];
  lines.push(expired.length ? 'Expired inbox examples:' : 'Expired inbox examples: none');
  expired.slice(0, 10).forEach((item, idx) => lines.push(`${idx + 1}. ${memorySnippet(item.proposed_text)}`));
  lines.push('');
  const superseded = payload.superseded_duplicates || [];
  lines.push(superseded.length ? 'Superseded duplicate IDs:' : 'Superseded duplicate IDs: none');
  superseded.slice(0, 10).forEach((item, idx) => {
    lines.push(`${idx + 1}. ${shortId(item.superseded_rid)} superseded by ${shortId(item.canonical_rid)} (${Math.round(Number(item.overlap || 0) * 100)}%)`);
  });
  if (superseded.length > 10) lines.push(`... plus ${formatCount(superseded.length - 10)} more`);
  return lines.join('\n');
}

async function runMaintenance(apply) {
  if (apply) {
    const confirmed = await showModal({
      title: 'Apply autopilot maintenance?',
      body: 'This can expire stale inbox items and mark near-duplicate active memories as superseded. Dry-run first if unsure.',
      confirmText: 'Apply autopilot',
      cancelText: 'Cancel',
    });
    if (!confirmed) return;
  }
  const payload = await postJson('/api/maintenance/autopilot', {
    apply,
    owner: $('#maintenanceOwner').value,
    domain: $('#maintenanceDomain').value,
    max_inbox_age_days: Number($('#maintenanceAge').value) || 30,
    duplicate_threshold: Number($('#maintenanceThreshold').value) || 0.9,
  });
  $('#maintenanceResult').textContent = renderMaintenanceResult(payload);
  await loadFacets();
}

function renderRuntimeStatus(payload) {
  const lines = [
    'Runtime status',
    '',
    `DB: ${payload.db_path}`,
    `Integration: ${payload.runtime_injection}`,
    `Embedding: ${payload.embedding?.enabled ? 'enabled' : 'disabled'} ${payload.embedding?.active_model || ''}`.trim(),
    `LLM synthesis: ${payload.synthesis?.enabled ? 'enabled' : 'disabled'} ${payload.synthesis?.model || ''}`.trim(),
    `Memories: ${formatCount(payload.counts?.memories || 0)}`,
    `Inbox rows: ${formatCount(payload.counts?.inbox || 0)}`,
    `Audit events: ${formatCount(payload.counts?.audit_events || 0)}`,
    '',
  ];
  const last = payload.last_recall || payload.last_recall_event;
  if (last) {
    lines.push('Last recall event');
    lines.push(`Query: ${last.metadata?.query || '—'}`);
    lines.push(`Owner/platform: ${last.metadata?.owner || '—'} / ${last.metadata?.platform || '—'}`);
    lines.push(`Results: ${formatCount(last.metadata?.result_count || 0)}`);
    lines.push(`At: ${new Date(Number(last.created_at || 0) * 1000).toLocaleString('en-SG')}`);
  } else {
    lines.push('Last recall event: none yet');
  }
  return lines.join('\n');
}

function renderRuntimeTest(payload) {
  const lines = [
    'Runtime recall test',
    '',
    `Query: ${payload.query}`,
    `Owner/platform: ${payload.owner} / ${payload.platform}`,
    `Results: ${formatCount(payload.result_count || 0)}`,
    '',
    payload.included?.length ? 'Included memories:' : 'Included memories: none',
  ];
  (payload.included || []).forEach((item, idx) => {
    lines.push(`${idx + 1}. ${memorySnippet(item.text, 220)}`);
    lines.push(`   ${shortId(item.rid)} · score ${item.score} · ${item.reasons.join(', ')}`);
  });
  lines.push('');
  lines.push('Context preview:');
  lines.push(payload.context_preview || '—');
  lines.push('');
  lines.push(`Audit event: ${payload.audit_event?.event_type || 'missing'}`);
  return lines.join('\n');
}

async function loadRuntime() {
  $('#runtimeResult').textContent = renderRuntimeStatus(await getJson('/api/runtime/status'));
}

async function runRuntimeTestRecall() {
  const payload = await postJson('/api/runtime/test-recall', {
    query: $('#runtimeTestQuery').value,
    owner: $('#runtimeTestOwner').value || 'primary',
    platform: $('#runtimeTestPlatform').value || 'whatsapp',
    limit: Number($('#runtimeTestLimit').value) || 10,
    visibility: ['private'],
  });
  $('#runtimeResult').textContent = renderRuntimeTest(payload);
}

async function saveEmbedding() {
  const model = $('#settingsEmbeddingModel').value;
  if (!model) return;
  const enabled = $('#settingsEmbeddingEnabled').checked;
  const confirmed = await showModal({
    title: 'Switch embedding model?',
    body: `Use ${model} for semantic recall? After switching, Anamnesis will check whether this model needs vector backfill.`,
    confirmText: 'Switch model',
    cancelText: 'Cancel',
  });
  if (!confirmed) return;
  const payload = await postJson('/api/settings/embedding', { model, enabled });
  $('#settingsResult').textContent = prettyJson(payload);
  const status = await loadEmbeddingStatus();
  if (enabled && Number(status.missing || 0) > 0) {
    const backfill = await showModal({
      title: 'Backfill embeddings now?',
      body: `${model} is missing vectors for ${formatCount(status.missing)} active memories. Semantic recall for this model is incomplete until backfill runs; keyword recall still works meanwhile.`,
      confirmText: 'Backfill now',
      cancelText: 'Later',
    });
    if (backfill) {
      const backfillPayload = await postJson('/api/embedding/backfill', { model });
      $('#settingsResult').textContent = prettyJson({ settings: payload, backfill: backfillPayload });
      await loadEmbeddingStatus();
    }
  }
}

async function saveLlm() {
  const payload = await postJson('/api/settings/synthesis', {
    enabled: $('#settingsLlmEnabled').checked,
    base_url: $('#settingsLlmUrl').value,
    model: $('#settingsLlmModel').value,
    api_key_env: $('#settingsLlmApiKeyEnv').value,
    temperature: Number($('#settingsLlmTemp').value) || 0.0,
    max_tokens: Number($('#settingsLlmMaxTokens').value) || 512,
    timeout: Number($('#settingsLlmTimeout').value) || 60,
  });
  $('#settingsResult').textContent = prettyJson(payload);
}

function bindEvents() {
  $$('[data-view]').forEach((node) => node.addEventListener('click', () => navigateToView(node.dataset.view)));
  $('[data-menu-toggle]')?.addEventListener('click', () => {
    const menu = $('[data-menu]');
    const open = !menu.classList.contains('open');
    menu.classList.toggle('open', open);
    $('[data-menu-toggle]').setAttribute('aria-expanded', String(open));
  });
  $('#refreshButton')?.addEventListener('click', loadOverview);
  $('#loadMemoriesButton')?.addEventListener('click', loadMemories);
  $('#loadInboxButton')?.addEventListener('click', loadInbox);
  $('#memoryOwner')?.addEventListener('change', resetMemoryPage);
  $('#memorySearch')?.addEventListener('input', resetMemoryPage);
  $('#memoryStatus')?.addEventListener('change', resetMemoryPage);
  $('#memoryLimit')?.addEventListener('change', resetMemoryPage);
  $('#inboxOwner')?.addEventListener('change', resetInboxPage);
  $('#inboxSearch')?.addEventListener('input', resetInboxPage);
  $('#inboxDecision')?.addEventListener('change', resetInboxPage);
  $('#inboxConfidence')?.addEventListener('change', resetInboxPage);
  $('#inboxSource')?.addEventListener('change', resetInboxPage);
  $('#inboxLimit')?.addEventListener('change', resetInboxPage);
  $('#prevMemoriesButton')?.addEventListener('click', () => { state.memoryOffset = Math.max(0, state.memoryOffset - Number($('#memoryLimit').value || 50)); loadMemories(); });
  $('#nextMemoriesButton')?.addEventListener('click', () => { state.memoryOffset += Number($('#memoryLimit').value || 50); loadMemories(); });
  $('#prevInboxButton')?.addEventListener('click', () => { state.inboxOffset = Math.max(0, state.inboxOffset - Number($('#inboxLimit').value || 50)); loadInbox(); });
  $('#nextInboxButton')?.addEventListener('click', () => { state.inboxOffset += Number($('#inboxLimit').value || 50); loadInbox(); });
  $('#selectAllMemories')?.addEventListener('change', (event) => $$('[data-select-memory]').forEach((node) => { node.checked = event.target.checked; }));
  $('#selectAllInbox')?.addEventListener('change', (event) => $$('[data-select-inbox]').forEach((node) => { node.checked = event.target.checked; }));
  $('#batchInvalidateMemoriesButton')?.addEventListener('click', () => batchMemories('invalidate'));
  $('#batchAcceptInboxButton')?.addEventListener('click', () => batchInbox('accept'));
  $('#batchRejectInboxButton')?.addEventListener('click', () => batchInbox('reject'));
  $('#runPreviewButton')?.addEventListener('click', runPreviewCheck);
  $('#openAuditButton')?.addEventListener('click', () => openAudit());
  $('#prepareCorrectionButton')?.addEventListener('click', () => $('#correctText').focus());
  $('#correctMemoryButton')?.addEventListener('click', correctMemory);
  $('#savePasswordButton')?.addEventListener('click', savePassword);
  $('#clearPasswordButton')?.addEventListener('click', clearPassword);
  $('#themeSelect')?.addEventListener('change', (event) => applyTheme(event.target.value));
  $('#saveEmbeddingButton')?.addEventListener('click', saveEmbedding);
  $('#embeddingBackfillButton')?.addEventListener('click', backfillEmbeddings);
  $('#runSimulatorButton')?.addEventListener('click', runSimulator);
  $('#maintenanceDryRunButton')?.addEventListener('click', () => runMaintenance(false));
  $('#maintenanceApplyButton')?.addEventListener('click', () => runMaintenance(true));
  $('#refreshRuntimeButton')?.addEventListener('click', loadRuntime);
  $('#runtimeTestButton')?.addEventListener('click', runRuntimeTestRecall);
  $('#saveLlmButton')?.addEventListener('click', saveLlm);
  $('#closeDrawerButton')?.addEventListener('click', () => {
    $('#detailDrawer').classList.remove('open');
    $('#detailDrawer').setAttribute('aria-hidden', 'true');
  });
  document.addEventListener('click', (event) => {
    const memoryTarget = event.target.closest('[data-open-memory]');
    if (memoryTarget) openAudit(memoryTarget.dataset.openMemory);
    const replaceTarget = event.target.closest('[data-replace-memory]');
    if (replaceTarget) replaceMemoryFromDrawer(replaceTarget.dataset.replaceMemory);
    const acceptTarget = event.target.closest('[data-accept-inbox]');
    if (acceptTarget) acceptInbox(acceptTarget.dataset.acceptInbox);
    const rejectTarget = event.target.closest('[data-reject-inbox]');
    if (rejectTarget) rejectInbox(rejectTarget.dataset.rejectInbox);
  });
}

async function bootstrap() {
  const ok = await ensureAuthenticated();
  if (!ok) {
    document.body.innerHTML = '<main class="main"><section class="panel"><h1>Dashboard locked</h1><p>Reload and enter the dashboard password to continue.</p></section></main>';
    return;
  }
  await loadFacets();
  const initialView = new URLSearchParams(window.location.search).get('view') || 'overview';
  history.replaceState({ view: initialView }, '', viewUrl(initialView));
  showView(initialView);
}

window.addEventListener('popstate', (event) => {
  const view = event.state?.view || new URLSearchParams(window.location.search).get('view') || 'overview';
  showView(view);
});

bindEvents();
bootstrap();
