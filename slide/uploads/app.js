/* ============================================================
   Lec.2 — 序列模型 & 多模態  (最佳化版)
   核心策略：
   - LSTM/Attention 動畫：首次建立 DOM，之後只更新 CSS variables
   - 用 CSS transition 而非 JS 每幀重畫
   - GPU 加速 (translate3d, will-change)
============================================================ */

(function() {
'use strict';

const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => Array.from(ctx.querySelectorAll(sel));
const SVG_NS = 'http://www.w3.org/2000/svg';
function svgEl(name, attrs = {}, text = null) {
  const el = document.createElementNS(SVG_NS, name);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  if (text !== null) el.textContent = text;
  return el;
}
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

// ============================================================
// 1. 頁面切換
// ============================================================
const pageLabels = [
  '序章 — 為什麼需要序列模型？',
  'RNN — 加一條自我循環的線',
  'LSTM — 給模型一條高速公路',
  'ELMo — biLSTM 的最後光輝',
  'Transformer — Attention Is All You Need',
  'Self-Attention 互動可視化',
  'BERT vs GPT — 兩條技術路線',
  '多模態 — CLIP / BLIP / LLaVA',
  '作業 2 — 設計你自己的多模態模型'
];
let currentPage = 0;
const totalPages = 9;
const navTabs = $$('#navTabs .nav-tab');
const pages = $$('.page');
const prevBtn = $('#prevBtn');
const nextBtn = $('#nextBtn');
const centerLabel = $('#centerLabel');
const progBar = $('#progBar');
const stage = $('#stage');

function gotoPage(idx) {
  idx = clamp(idx, 0, totalPages - 1);
  currentPage = idx;
  pages.forEach((p, i) => p.classList.toggle('active', i === idx));
  navTabs.forEach((t, i) => t.classList.toggle('active', i === idx));
  centerLabel.textContent = pageLabels[idx];
  progBar.textContent = `${idx} / ${totalPages - 1}`;
  prevBtn.disabled = idx === 0;
  nextBtn.disabled = idx === totalPages - 1;
  stage.scrollTop = 0;

  if (idx === 1) initGradDemo();
  if (idx === 2) initLstmDemo();
  if (idx === 5) initAttnDemo();
  markPageVisited(idx);
}
navTabs.forEach((tab, i) => tab.addEventListener('click', () => gotoPage(i)));
prevBtn.addEventListener('click', () => gotoPage(currentPage - 1));
nextBtn.addEventListener('click', () => gotoPage(currentPage + 1));
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowLeft') gotoPage(currentPage - 1);
  else if (e.key === 'ArrowRight') gotoPage(currentPage + 1);
});

// ============================================================
// 2. Tooltip (event delegation, 輕量)
// ============================================================
const tipPopup = $('#tipPopup');
let hideTipTimer = null;

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}
function positionTip(mx, my) {
  const rect = tipPopup.getBoundingClientRect();
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  let x = mx + 14;
  let y = my + 14;
  if (x + rect.width > vw - 10) x = mx - rect.width - 14;
  if (y + rect.height > vh - 10) y = my - rect.height - 14;
  if (x < 10) x = 10;
  if (y < 10) y = 10;
  tipPopup.style.left = x + 'px';
  tipPopup.style.top = y + 'px';
}
function showTip(target, mx, my) {
  const title = target.getAttribute('data-title') || '';
  const tip = target.getAttribute('data-tip') || '';
  let html = '';
  if (title) html += `<span class="tip-title">${escapeHtml(title)}</span>`;
  html += `<div>${escapeHtml(tip)}</div>`;
  tipPopup.innerHTML = html;
  tipPopup.classList.add('show');
  positionTip(mx, my);
}
function hideTip() {
  tipPopup.classList.remove('show');
}
document.body.addEventListener('mouseover', (e) => {
  const target = e.target.closest('.hov[data-tip], .node[data-tip]');
  if (!target) return;
  if (hideTipTimer) { clearTimeout(hideTipTimer); hideTipTimer = null; }
  showTip(target, e.clientX, e.clientY);
});
document.body.addEventListener('mousemove', (e) => {
  if (!tipPopup.classList.contains('show')) return;
  const target = e.target.closest('.hov[data-tip], .node[data-tip]');
  if (!target) return;
  positionTip(e.clientX, e.clientY);
});
document.body.addEventListener('mouseout', (e) => {
  const target = e.target.closest('.hov[data-tip], .node[data-tip]');
  if (!target) return;
  hideTipTimer = setTimeout(hideTip, 80);
});

// ============================================================
// 3. DEMO 1 — 梯度消失模擬器 (SVG 重畫，但只在 input 時觸發)
// ============================================================
let gradInited = false;
function initGradDemo() {
  if (gradInited) return;
  gradInited = true;
  const wSlider = $('#gradW');
  const tSlider = $('#gradT');
  const wVal = $('#gradWVal');
  const tVal = $('#gradTVal');
  const svg = $('#gradSvg');
  const msg = $('#gradMsg');

  function render() {
    const W = parseFloat(wSlider.value);
    const T = parseInt(tSlider.value);
    wVal.textContent = W.toFixed(2);
    tVal.textContent = T;

    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const margin = { top: 20, right: 30, bottom: 36, left: 60 };
    const w = 700, h = 240;
    const plotW = w - margin.left - margin.right;
    const plotH = h - margin.top - margin.bottom;

    const data = [];
    for (let t = 0; t <= T; t++) data.push({ t, val: Math.pow(W, t) });
    const allLogs = data.map(d => Math.log10(Math.max(1e-30, d.val)));
    let yMin = Math.min(...allLogs, -2);
    let yMax = Math.max(...allLogs, 2);
    if (yMin > -2) yMin = -2;
    if (yMax < 2) yMax = 2;
    const range = yMax - yMin;
    const xScale = (t) => margin.left + (t / T) * plotW;
    const yScale = (logv) => margin.top + (1 - (logv - yMin) / range) * plotH;

    svg.appendChild(svgEl('rect', { x: margin.left, y: margin.top, width: plotW, height: plotH, fill: '#faf3e3', stroke: '#8b7e68', 'stroke-width': '1' }));

    for (let i = Math.ceil(yMin); i <= Math.floor(yMax); i++) {
      const y = yScale(i);
      svg.appendChild(svgEl('line', { x1: margin.left, x2: margin.left + plotW, y1: y, y2: y, stroke: '#c8b896', 'stroke-width': '0.5', 'stroke-dasharray': '2,2' }));
      svg.appendChild(svgEl('text', { x: margin.left - 8, y: y + 4, 'text-anchor': 'end', 'font-size': '10', fill: '#5a4f3e', 'font-family': 'JetBrains Mono' }, `10^${i}`));
    }

    const safeY = yScale(0);
    if (safeY >= margin.top && safeY <= margin.top + plotH) {
      svg.appendChild(svgEl('line', { x1: margin.left, x2: margin.left + plotW, y1: safeY, y2: safeY, stroke: '#3f5d3a', 'stroke-width': '2', 'stroke-dasharray': '4,3' }));
      svg.appendChild(svgEl('text', { x: margin.left + plotW - 4, y: safeY - 6, 'text-anchor': 'end', 'font-size': '10', fill: '#3f5d3a', 'font-style': 'italic' }, '梯度大小 = 1（安全）'));
    }

    let pathD = '';
    data.forEach((d, i) => {
      const x = xScale(d.t);
      const y = yScale(Math.log10(Math.max(1e-30, d.val)));
      pathD += (i === 0 ? 'M' : 'L') + x + ',' + y + ' ';
    });
    const color = W < 1 ? '#b6432a' : (W > 1 ? '#c98e2b' : '#3f5d3a');
    svg.appendChild(svgEl('path', { d: pathD, stroke: color, 'stroke-width': '2.5', fill: 'none' }));
    const last = data[data.length - 1];
    svg.appendChild(svgEl('circle', { cx: xScale(last.t), cy: yScale(Math.log10(Math.max(1e-30, last.val))), r: '5', fill: color, stroke: '#2b2419', 'stroke-width': '1.5' }));

    for (let t = 0; t <= T; t += Math.max(1, Math.round(T / 5))) {
      svg.appendChild(svgEl('text', { x: xScale(t), y: margin.top + plotH + 16, 'text-anchor': 'middle', 'font-size': '10', fill: '#5a4f3e', 'font-family': 'JetBrains Mono' }, 't=' + t));
    }
    svg.appendChild(svgEl('text', { x: margin.left + plotW / 2, y: h - 4, 'text-anchor': 'middle', 'font-size': '11', fill: '#2b2419', 'font-family': 'Fraunces', 'font-style': 'italic' }, '時間步 t（反向傳播經過的層數）'));
    svg.appendChild(svgEl('text', { x: 16, y: margin.top + plotH / 2, 'text-anchor': 'middle', 'font-size': '11', fill: '#2b2419', 'font-family': 'Fraunces', 'font-style': 'italic', transform: `rotate(-90 16 ${margin.top + plotH / 2})` }, '梯度大小 |W^t|（log10）'));

    const finalVal = Math.pow(W, T);
    let txt;
    if (W < 0.95) {
      txt = `W=${W.toFixed(2)}, t=${T} → |W^t| ≈ ${finalVal.toExponential(2)}　【梯度消失！】梯度幾乎為 0，前面的層學不到東西。`;
    } else if (W > 1.05) {
      txt = `W=${W.toFixed(2)}, t=${T} → |W^t| ≈ ${finalVal.toExponential(2)}　【梯度爆炸！】梯度大到爆，訓練會 NaN。`;
    } else {
      txt = `W=${W.toFixed(2)}, t=${T} → |W^t| ≈ ${finalVal.toFixed(3)}　梯度穩定 — 但這樣的權重很稀有，實務上很難維持。`;
    }
    msg.textContent = txt;
  }
  wSlider.addEventListener('input', render);
  tSlider.addEventListener('input', render);
  render();
}

// ============================================================
// 4. DEMO 2 — LSTM 動畫（重點：CSS variable 更新，不重建 DOM）
// ============================================================
const LSTM_TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat'];
const LSTM_DIMS = ['主詞', '動詞', '位置', '結尾'];
const LSTM_TRACE = [
  { token: 'The', f: [0.0, 0.0, 0.0, 0.0], i: [0.7, 0.1, 0.1, 0.0], cTilde: [0.8, 0.0, 0.0, 0.0],
    note: '「The」是冠詞，預期接下來會有主詞 — input gate 在「主詞」維度打開' },
  { token: 'cat', f: [0.9, 0.5, 0.2, 0.1], i: [0.9, 0.0, 0.0, 0.0], cTilde: [0.95, 0.0, 0.0, 0.0],
    note: '「cat」是主詞，input gate 在「主詞」維度全開，記住主詞' },
  { token: 'sat', f: [0.95, 0.2, 0.2, 0.1], i: [0.0, 0.9, 0.1, 0.0], cTilde: [0.0, 0.85, 0.0, 0.0],
    note: '「sat」是動詞，動詞維度被填入；主詞維度透過 forget gate 保留' },
  { token: 'on', f: [0.95, 0.95, 0.3, 0.1], i: [0.0, 0.0, 0.6, 0.0], cTilde: [0.0, 0.0, 0.7, 0.0],
    note: '「on」預示位置詞，位置維度開始累積；前面的記憶都保留' },
  { token: 'the', f: [0.95, 0.95, 0.95, 0.1], i: [0.0, 0.0, 0.3, 0.0], cTilde: [0.0, 0.0, 0.5, 0.0],
    note: '「the」幾乎只更新位置維度（介系詞片語在進行中）' },
  { token: 'mat', f: [0.95, 0.95, 0.95, 0.0], i: [0.0, 0.0, 0.95, 0.6], cTilde: [0.0, 0.0, 0.9, 0.7],
    note: '「mat」完成介系詞片語；位置 + 結尾維度都被填滿 — 句子訊息完整' }
];

function computeLstmStates() {
  const states = [];
  let C = [0, 0, 0, 0];
  for (const step of LSTM_TRACE) {
    const newC = [0, 0, 0, 0];
    for (let d = 0; d < 4; d++) {
      newC[d] = step.f[d] * C[d] + step.i[d] * step.cTilde[d];
    }
    const o = newC.map(v => Math.min(1, Math.max(0, Math.abs(v) * 0.9 + 0.1)));
    const h = newC.map((v, d) => o[d] * Math.tanh(v));
    states.push({ ...step, C: [...newC], o, h });
    C = [...newC];
  }
  return states;
}
const LSTM_STATES = computeLstmStates();

let lstmInited = false;
let lstmStep = 0;
let lstmPlayTimer = null;
let lstmEls = null; // 儲存所有 cell 元素引用

function initLstmDemo() {
  if (lstmInited) return;
  lstmInited = true;

  // ===== 一次性建立所有 DOM =====
  // 1. Tokens
  const tokensEl = $('#lstmTokens');
  const tokenEls = LSTM_TOKENS.map((t) => {
    const el = document.createElement('span');
    el.className = 'lstm-token';
    el.textContent = t;
    tokensEl.appendChild(el);
    return el;
  });

  // 2. 五個 grid: cell, forget, input, output, hidden
  function buildGrid(parentId, hue) {
    const parent = $(parentId);
    const cells = [];
    for (let d = 0; d < 4; d++) {
      const cell = document.createElement('div');
      cell.className = 'lstm-cell ' + (hue ? 'gate-' + hue : '');
      cell.style.setProperty('--intensity', '0');
      const label = document.createElement('span');
      label.className = 'label';
      label.textContent = LSTM_DIMS[d];
      const val = document.createElement('span');
      val.className = 'val';
      val.textContent = '0.00';
      cell.appendChild(label);
      cell.appendChild(val);
      parent.appendChild(cell);
      cells.push({ cell, val });
    }
    return cells;
  }
  lstmEls = {
    tokens: tokenEls,
    cellState: buildGrid('#lstmCellGrid', 'red'),
    forget:    buildGrid('#lstmForgetGrid', 'mustard'),
    input:     buildGrid('#lstmInputGrid', 'green'),
    output:    buildGrid('#lstmOutputGrid', 'blue'),
    hidden:    buildGrid('#lstmHiddenGrid', null)
  };

  // ===== 控制按鈕 =====
  $('#lstmStep').addEventListener('click', () => {
    if (lstmStep >= LSTM_TOKENS.length) lstmStep = 0;
    else lstmStep++;
    updateLstm();
  });
  $('#lstmReset').addEventListener('click', () => {
    if (lstmPlayTimer) { clearInterval(lstmPlayTimer); lstmPlayTimer = null; $('#lstmPlay').textContent = '▶ 自動播放'; }
    lstmStep = 0;
    updateLstm();
  });
  $('#lstmPlay').addEventListener('click', () => {
    if (lstmPlayTimer) {
      clearInterval(lstmPlayTimer);
      lstmPlayTimer = null;
      $('#lstmPlay').textContent = '▶ 自動播放';
      return;
    }
    if (lstmStep >= LSTM_TOKENS.length) lstmStep = 0;
    $('#lstmPlay').textContent = '⏸ 暫停';
    lstmPlayTimer = setInterval(() => {
      if (lstmStep >= LSTM_TOKENS.length) {
        clearInterval(lstmPlayTimer);
        lstmPlayTimer = null;
        $('#lstmPlay').textContent = '▶ 自動播放';
        return;
      }
      lstmStep++;
      updateLstm();
    }, 1500);
  });

  updateLstm();
}

// 只更新值，不重建 DOM — CSS transition 會自動 smooth 過渡
function updateLstm() {
  if (!lstmEls) return;
  const info = $('#lstmStepInfo');
  const msg = $('#lstmMsg');
  info.textContent = `step ${lstmStep} / ${LSTM_TOKENS.length}`;

  // Tokens 高亮
  lstmEls.tokens.forEach((el, i) => {
    el.classList.remove('past', 'active');
    if (i < lstmStep - 1) el.classList.add('past');
    else if (i === lstmStep - 1) el.classList.add('active');
  });

  function updateRow(cells, values) {
    for (let d = 0; d < 4; d++) {
      const v = clamp(Math.abs(values[d]), 0, 1);
      cells[d].cell.style.setProperty('--intensity', v.toFixed(2));
      cells[d].val.textContent = values[d].toFixed(2);
      cells[d].val.style.color = v > 0.6 ? '#faf3e3' : '#2b2419';
    }
  }

  if (lstmStep === 0) {
    updateRow(lstmEls.cellState, [0, 0, 0, 0]);
    updateRow(lstmEls.forget,    [0, 0, 0, 0]);
    updateRow(lstmEls.input,     [0, 0, 0, 0]);
    updateRow(lstmEls.output,    [0, 0, 0, 0]);
    updateRow(lstmEls.hidden,    [0, 0, 0, 0]);
    msg.textContent = '初始狀態：cell state 全為 0，按「下一步」開始處理第一個 token。';
  } else {
    const s = LSTM_STATES[lstmStep - 1];
    updateRow(lstmEls.cellState, s.C);
    updateRow(lstmEls.forget,    s.f);
    updateRow(lstmEls.input,     s.i);
    updateRow(lstmEls.output,    s.o);
    updateRow(lstmEls.hidden,    s.h);
    msg.textContent = `[t=${lstmStep}] 處理「${s.token}」 — ${s.note}`;
  }
}

// ============================================================
// 5. DEMO 3 — Self-Attention (重點：只更新 cell color，不重建)
// ============================================================
let attnInited = false;
let attnTokens = [];
let attnMatrix = [];
let attnFocusIdx = 0;
let attnEls = null;  // { cells: [[...], ...], rowHeaders: [...], colHeaders: [...], bars: [...] }

function initAttnDemo() {
  if (attnInited) return;
  attnInited = true;
  $('#attnRun').addEventListener('click', runAttn);
  $('#attnHead').addEventListener('change', () => {
    if (attnTokens.length > 0) runAttn();
  });
  $('#attnInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') runAttn();
  });
  runAttn();
}

function runAttn() {
  const input = $('#attnInput').value.trim();
  if (!input) return;
  const newTokens = input.split(/\s+/).filter(t => t.length > 0).slice(0, 14);
  if (newTokens.length < 2) return;

  const headType = $('#attnHead').value;

  // 如果 token 改了，要重建 grid；否則只更新顏色
  const tokensChanged = newTokens.length !== attnTokens.length ||
                        newTokens.some((t, i) => t !== attnTokens[i]);

  attnTokens = newTokens;
  attnMatrix = computeAttnMatrix(attnTokens, headType);
  attnFocusIdx = 0;

  if (tokensChanged) {
    buildAttnDom();
  }
  updateAttnUI();
  updateAttnDesc(headType);
}

function computeAttnMatrix(tokens, type) {
  const N = tokens.length;
  const matrix = [];
  const NOUNS = new Set(['cat', 'dog', 'mat', 'animal', 'street', 'person', 'man', 'woman', 'student', 'teacher', 'tree', 'car', 'book', 'house']);
  const PRONOUNS = new Set(['it', 'he', 'she', 'they', 'i', 'we', 'you', 'this', 'that']);
  const VERBS = new Set(['sat', 'ran', 'jumped', 'is', 'was', 'were', 'are', 'love', 'eat', 'sleep', 'tired', 'happy']);
  const FUNC = new Set(['the', 'a', 'an', 'on', 'in', 'at', 'because', 'and', 'or', 'but']);

  function classify(t) {
    const lt = t.toLowerCase().replace(/[.,!?]/g, '');
    if (PRONOUNS.has(lt)) return 'pronoun';
    if (NOUNS.has(lt)) return 'noun';
    if (VERBS.has(lt)) return 'verb';
    if (FUNC.has(lt)) return 'func';
    return 'other';
  }

  for (let i = 0; i < N; i++) {
    const row = new Array(N).fill(0);
    const ci = classify(tokens[i]);
    for (let j = 0; j < N; j++) {
      const cj = classify(tokens[j]);
      let score = 0;
      if (type === 'position') {
        score = Math.exp(-Math.abs(i - j) * 0.7);
      } else if (type === 'syntax') {
        const dist = Math.abs(i - j);
        if (i === j) score = 0.3;
        else if (ci === 'verb' && cj === 'noun') score = 1.5 * Math.exp(-dist * 0.15);
        else if (ci === 'noun' && cj === 'func' && j === i - 1) score = 1.0;
        else if (cj === 'verb' && ci === 'noun') score = 0.6 * Math.exp(-dist * 0.2);
        else score = 0.15 * Math.exp(-dist * 0.4);
      } else if (type === 'coref') {
        if (ci === 'pronoun' && cj === 'noun' && j < i) score = 2.5 * Math.exp(-(i - j) * 0.04);
        else if (i === j) score = 0.4;
        else if (ci === 'pronoun' && cj === 'pronoun') score = 0.05;
        else score = 0.15 * Math.exp(-Math.abs(i - j) * 0.3);
      } else if (type === 'content') {
        if (i === j) score = 0.5;
        else if (ci === cj && ci !== 'other' && ci !== 'func') score = 1.2;
        else if (ci === 'noun' && cj === 'noun') score = 1.0;
        else score = 0.2 * Math.exp(-Math.abs(i - j) * 0.2);
      }
      row[j] = Math.max(0.0001, score);
    }
    const maxV = Math.max(...row);
    const exps = row.map(v => Math.exp(v - maxV));
    const sum = exps.reduce((a, b) => a + b, 0);
    matrix.push(exps.map(v => v / sum));
  }
  return matrix;
}

// 一次性建立 token chips 和 heatmap grid
function buildAttnDom() {
  const N = attnTokens.length;

  // === Token chips ===
  const tokensContainer = $('#attnTokens');
  tokensContainer.innerHTML = '<div style="font-family:Fraunces; font-weight:700; margin-bottom:8px; color:var(--ink);">點擊任一個 token 看它的注意力分布：</div>';
  const chipEls = [];
  attnTokens.forEach((t, i) => {
    const chip = document.createElement('span');
    chip.className = 'token-chip';
    chip.textContent = t;
    chip.addEventListener('click', () => {
      attnFocusIdx = i;
      updateAttnUI();
    });
    tokensContainer.appendChild(chip);
    chipEls.push(chip);
  });

  // === Heatmap grid ===
  const heatmapContainer = $('#attnHeatmap');
  heatmapContainer.innerHTML = '';
  const cellSize = Math.max(28, Math.min(46, Math.floor(360 / N)));

  const grid = document.createElement('div');
  grid.className = 'attn-grid-wrap';
  grid.style.gridTemplateColumns = `60px repeat(${N}, ${cellSize}px)`;

  // 角落空格
  const corner = document.createElement('div');
  corner.style.background = 'var(--paper)';
  grid.appendChild(corner);

  // 上方 column headers
  const colHeaders = [];
  for (let j = 0; j < N; j++) {
    const head = document.createElement('div');
    head.className = 'attn-header';
    head.style.writingMode = 'vertical-rl';
    head.style.transform = 'rotate(180deg)';
    head.textContent = attnTokens[j];
    grid.appendChild(head);
    colHeaders.push(head);
  }

  // Rows
  const rowHeaders = [];
  const cells = [];  // cells[i][j]
  for (let i = 0; i < N; i++) {
    const rowHead = document.createElement('div');
    rowHead.className = 'attn-header';
    rowHead.style.textAlign = 'right';
    rowHead.style.padding = '4px 6px';
    rowHead.style.display = 'flex';
    rowHead.style.alignItems = 'center';
    rowHead.style.justifyContent = 'flex-end';
    rowHead.textContent = attnTokens[i];
    rowHead.style.cursor = 'pointer';
    rowHead.addEventListener('click', () => {
      attnFocusIdx = i;
      updateAttnUI();
    });
    grid.appendChild(rowHead);
    rowHeaders.push(rowHead);

    cells.push([]);
    for (let j = 0; j < N; j++) {
      const cell = document.createElement('div');
      cell.className = 'attn-cell';
      cell.style.width = cellSize + 'px';
      cell.style.height = cellSize + 'px';
      cell.title = `${attnTokens[i]} → ${attnTokens[j]}`;
      cell.addEventListener('click', () => {
        attnFocusIdx = i;
        updateAttnUI();
      });
      grid.appendChild(cell);
      cells[i].push(cell);
    }
  }
  heatmapContainer.appendChild(grid);

  // === Bars (HTML/CSS, width transition GPU-friendly) ===
  const barsContainer = $('#attnBarsHtml');
  barsContainer.innerHTML = '';
  const barTitle = document.createElement('div');
  barTitle.style.fontFamily = 'Fraunces';
  barTitle.style.fontWeight = '700';
  barTitle.style.color = 'var(--accent-red)';
  barTitle.style.fontSize = '13px';
  barTitle.style.marginBottom = '12px';
  barTitle.id = 'attnBarTitle';
  barsContainer.appendChild(barTitle);

  const bars = [];
  for (let j = 0; j < N; j++) {
    const row = document.createElement('div');
    row.className = 'attn-bar-row';
    const lbl = document.createElement('div');
    lbl.className = 'attn-bar-label';
    lbl.textContent = attnTokens[j];
    const track = document.createElement('div');
    track.className = 'attn-bar-track';
    const fill = document.createElement('div');
    fill.className = 'attn-bar-fill';
    fill.style.width = '0%';
    track.appendChild(fill);
    row.appendChild(lbl);
    row.appendChild(track);
    barsContainer.appendChild(row);
    bars.push({ label: lbl, fill });
  }

  attnEls = { chips: chipEls, cells, rowHeaders, colHeaders, bars, barTitle };
}

// 只更新顏色和 width，不重建 DOM
function updateAttnUI() {
  if (!attnEls) return;
  const N = attnTokens.length;

  // === Chips focus ===
  attnEls.chips.forEach((c, i) => {
    c.classList.toggle('focus', i === attnFocusIdx);
  });

  // === Heatmap colors (只改 backgroundColor — CSS transition 自動 smooth) ===
  for (let i = 0; i < N; i++) {
    attnEls.rowHeaders[i].classList.toggle('row-focus', i === attnFocusIdx);
    for (let j = 0; j < N; j++) {
      const v = attnMatrix[i][j];
      const intensity = Math.pow(v, 0.6);
      const r = Math.round(250 - intensity * 68);
      const g = Math.round(243 - intensity * 200);
      const b = Math.round(227 - intensity * 197);
      const cell = attnEls.cells[i][j];
      cell.style.backgroundColor = `rgb(${r},${g},${b})`;
      cell.style.color = intensity > 0.5 ? '#faf3e3' : '#2b2419';
      cell.classList.toggle('row-focus', i === attnFocusIdx);
      if (v > 0.05) cell.textContent = (v * 100).toFixed(0);
      else cell.textContent = '';
    }
  }

  // === Bars (改 width — GPU-friendly transition) ===
  attnEls.barTitle.textContent = `「${attnTokens[attnFocusIdx]}」的注意力分布`;
  const row = attnMatrix[attnFocusIdx];
  const maxV = Math.max(...row, 0.001);
  attnEls.bars.forEach((b, j) => {
    const v = row[j];
    const w = (v / maxV) * 100;
    b.fill.style.width = w.toFixed(1) + '%';
    b.fill.textContent = (v * 100).toFixed(1) + '%';
    b.label.classList.toggle('focus', j === attnFocusIdx);
    b.fill.classList.toggle('focus', j === attnFocusIdx);
  });
}

function updateAttnDesc(headType) {
  const desc = $('#attnDesc');
  const headDescs = {
    syntax: '這個 head 模擬「句法依賴」— 動詞傾向 attend 主詞，名詞 attend 它前面的冠詞。是 Transformer 中常觀察到的 pattern 之一。',
    position: '這個 head 模擬「位置鄰近」— 每個 token 主要 attend 自己附近的 token。形成對角線 pattern，常出現在淺層 attention head。',
    coref: '這個 head 模擬「共指（coreference）」— 代名詞（it/he/she）會強烈 attend 它指涉的名詞。試試把 "it" 改成 "they" 看看變化。',
    content: '這個 head 模擬「內容相似性」— 同類別的詞（名詞 vs 名詞、動詞 vs 動詞）互相 attend。展現語意層次的關係。'
  };
  desc.textContent = headDescs[headType] || '';
}

// ============================================================
// 6. Accordion (event delegation)
// ============================================================
document.body.addEventListener('click', (e) => {
  const header = e.target.closest('.accordion-header');
  if (!header) return;
  const accordion = header.parentElement;
  accordion.classList.toggle('open');
});

// ============================================================
// 7. Quiz System
// ============================================================
document.body.addEventListener('click', (e) => {
  const option = e.target.closest('.quiz-option');
  if (!option) return;
  const panel = option.closest('.quiz-panel');
  if (!panel) return;

  const alreadyAnswered = panel.querySelector('.quiz-option.correct, .quiz-option.wrong');
  if (alreadyAnswered) return;

  const isCorrect = option.getAttribute('data-correct') === 'true';
  const options = panel.querySelectorAll('.quiz-option');

  options.forEach(opt => {
    if (opt.getAttribute('data-correct') === 'true') {
      opt.classList.add('correct');
    } else if (opt === option && !isCorrect) {
      opt.classList.add('wrong');
    }
  });

  const feedback = panel.querySelector('.quiz-feedback');
  if (feedback) {
    feedback.classList.add('show');
    if (!isCorrect) {
      const originalText = feedback.textContent;
      feedback.textContent = '✗ 再想想 — ' + originalText;
      feedback.style.borderLeftColor = 'var(--accent-red)';
    } else {
      const originalText = feedback.textContent;
      feedback.textContent = '✓ ' + originalText;
    }
  }
});

// ============================================================
// 8. localStorage Progress Tracking
// ============================================================
function markPageVisited(idx) {
  const visited = JSON.parse(localStorage.getItem('lec2_visited') || '[]');
  if (!visited.includes(idx)) {
    visited.push(idx);
    localStorage.setItem('lec2_visited', JSON.stringify(visited));
  }
}

// ============================================================
// Init
// ============================================================
gotoPage(0);

})();
