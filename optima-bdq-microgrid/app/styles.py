"""Industrial design system and CSS tokens for OPTIMA Microgrid Intelligence Platform.
Engineered for operator control rooms: deep navy background, teal/cyan accents,
consistent cards, perfectly aligned tables, and zero unrendered HTML.
"""


def get_custom_css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* =========================================================
   DESIGN TOKENS
   ========================================================= */
:root {
    --bg-main:         #0B1120;
    --bg-sidebar:      #0D1526;
    --bg-panel:        #111827;
    --bg-card:         #141E2E;
    --bg-secondary:    #1A2540;
    --bg-hover:        #1F2E48;
    --border-subtle:   #1E2E46;
    --border-strong:   #253249;
    --border-accent:   rgba(56, 189, 248, 0.25);

    --text-primary:    #F0F6FF;
    --text-secondary:  #94A3B8;
    --text-muted:      #64748B;
    --text-label:      #7A90B0;

    --accent-teal:     #14B8A6;
    --accent-cyan:     #38BDF8;
    --accent-green:    #22C55E;
    --accent-amber:    #F59E0B;
    --accent-red:      #EF4444;
    --accent-indigo:   #818CF8;
    --accent-purple:   #A855F7;

    --shadow-card:     0 2px 12px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.03);
    --shadow-hover:    0 4px 20px rgba(0,0,0,0.55), 0 0 0 1px rgba(56,189,248,0.18);
    --radius-card:     10px;
    --radius-sm:       6px;

    --font-sans:  'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono:  'JetBrains Mono', monospace;

    --transition-fast:  150ms cubic-bezier(0.4,0,0.2,1);
    --transition-med:   220ms cubic-bezier(0.4,0,0.2,1);
}

/* =========================================================
   KEYFRAMES
   ========================================================= */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0);    }
}
@keyframes pulse-dot {
    0%, 100% { box-shadow: 0 0 0 0 rgba(20,184,166,0.6); }
    50%       { box-shadow: 0 0 0 6px rgba(20,184,166,0); }
}
@keyframes countUp {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0);   }
}

.kpi-animate-1 { animation: fadeUp 0.4s ease-out 0.05s both; }
.kpi-animate-2 { animation: fadeUp 0.4s ease-out 0.12s both; }
.kpi-animate-3 { animation: fadeUp 0.4s ease-out 0.19s both; }
.kpi-animate-4 { animation: fadeUp 0.4s ease-out 0.26s both; }

/* =========================================================
   BASE RESET
   ========================================================= */
.stApp {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
}

.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 2.0rem !important;
    max-width: 100% !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    min-height: 0 !important;
}
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* =========================================================
   SIDEBAR
   ========================================================= */
section[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-subtle) !important;
    min-width: 220px !important;
    max-width: 240px !important;
}

section[data-testid="stSidebar"] > div {
    padding: 1.1rem 0.85rem !important;
}

section[data-testid="stSidebar"] hr {
    border-color: var(--border-subtle) !important;
    margin: 0.6rem 0 !important;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding-bottom: 0.5rem;
}
.sidebar-logo-icon {
    width: 32px; height: 32px;
    border-radius: 8px;
    background: linear-gradient(135deg, #14B8A6 0%, #0891B2 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
    box-shadow: 0 2px 10px rgba(20,184,166,0.35);
}
.sidebar-logo-title {
    font-size: 1.05rem; font-weight: 700;
    color: var(--text-primary); letter-spacing: -0.01em; line-height: 1.15;
}
.sidebar-logo-sub {
    font-size: 0.6rem; color: var(--accent-cyan); font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.08em;
}

.sidebar-section-label {
    font-size: 0.62rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.09em;
    color: var(--text-muted);
    margin-bottom: 0.35rem; margin-top: 0.1rem; padding-left: 0.1rem;
}

div[data-testid="stRadio"] > div {
    gap: 0.18rem !important;
    display: flex;
    flex-direction: column;
}

div[data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    padding: 0.38rem 0.6rem !important;
    border-radius: var(--radius-sm) !important;
    cursor: pointer !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    transition: background var(--transition-fast), color var(--transition-fast) !important;
    gap: 0.5rem !important;
    border: 1px solid transparent !important;
}

div[data-testid="stRadio"] label:hover {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-subtle) !important;
}

div[data-testid="stRadio"] [data-baseweb="radio"] div:first-child {
    display: none !important;
}

div[data-baseweb="select"] > div {
    background-color: var(--bg-secondary) !important;
    border-color: var(--border-strong) !important;
    color: var(--text-primary) !important;
    font-size: 0.8rem !important;
    border-radius: var(--radius-sm) !important;
}

div[data-baseweb="input"] > div {
    background-color: var(--bg-secondary) !important;
    border-color: var(--border-strong) !important;
    color: var(--text-primary) !important;
    font-size: 0.8rem !important;
}

/* =========================================================
   STREAMLIT NATIVE BORDERED CONTAINER CARD STYLING
   ========================================================= */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-card) !important;
    box-shadow: var(--shadow-card) !important;
    transition: border-color var(--transition-med), box-shadow var(--transition-med) !important;
    overflow: hidden !important;
    margin-bottom: 0.45rem !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--border-accent) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 0.65rem 0.75rem !important;
    gap: 0.3rem !important;
}

div[data-testid="stHtml"] {
    width: 100% !important;
}

/* =========================================================
   CARD TITLE
   ========================================================= */
.panel-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 0.25rem;
    letter-spacing: 0.01em;
}

/* Section labels for subviews */
.section-title {
    font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-label);
    margin-bottom: 0.5rem; padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border-subtle);
}

/* =========================================================
   CONTROL ROOM HEADER
   ========================================================= */
.ctrl-header {
    background: linear-gradient(135deg, #111827 0%, #0f1f38 100%);
    border: 1px solid var(--border-strong);
    border-top: 2px solid rgba(56,189,248,0.35);
    border-radius: var(--radius-card);
    padding: 0.7rem 1.1rem;
    margin-bottom: 0.75rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.65rem;
    box-shadow: var(--shadow-card);
    animation: fadeUp 0.4s ease-out 0s both;
}

.ctrl-header-logo {
    display: flex; align-items: center; gap: 0.7rem;
}

.ctrl-header-icon {
    width: 36px; height: 36px;
    border-radius: 9px;
    background: linear-gradient(135deg, #14B8A6 0%, #0891B2 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
    box-shadow: 0 2px 10px rgba(20,184,166,0.40);
}

.ctrl-header-title {
    font-size: 1.3rem; font-weight: 800;
    color: var(--text-primary); letter-spacing: -0.02em; line-height: 1;
}

.ctrl-header-subtitle {
    font-size: 0.74rem; color: var(--text-secondary);
    margin-top: 0.1rem; font-weight: 400;
}

.ctrl-header-right {
    display: flex; align-items: center; gap: 0.55rem; flex-wrap: wrap;
}

.badge-crit { background: rgba(239,68,68,0.3); color: #FEE2E2; }
.badge-warn { background: rgba(245,158,11,0.3); color: #FEF3C7; }

/* =========================================================
   OPERATOR POWER BALANCE DISPLAY
   ========================================================= */
.balance-footer {
    border-top: 1px solid var(--border-subtle);
    padding-top: 0.35rem;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.1rem;
}

.balance-caption {
    color: var(--text-muted);
    font-size: 0.62rem;
    line-height: 1.35;
}

.balance-status-ok {
    color: var(--accent-green);
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
}

.balance-status-err {
    color: var(--accent-red);
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
}

/* =========================================================
   NEXT ACTION PREVIEW
   ========================================================= */
.next-action-card {
    background: var(--bg-card);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-card);
    padding: 0.75rem 0.95rem;
}

.next-action-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    font-size: 0.68rem;
    font-family: var(--font-mono);
    font-weight: 600;
}

.next-action-feasible {
    background: rgba(34,197,94,0.14);
    color: var(--accent-green);
    border: 1px solid rgba(34,197,94,0.3);
}

.next-action-complete {
    background: rgba(148,163,184,0.1);
    color: var(--text-secondary);
    border: 1px solid rgba(148,163,184,0.25);
}

.header-chip {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.22rem 0.55rem;
    border-radius: var(--radius-sm);
    font-size: 0.7rem; font-family: var(--font-mono); font-weight: 500;
    background: var(--bg-secondary); border: 1px solid var(--border-strong);
    color: var(--text-secondary); white-space: nowrap;
}

div[data-testid="stSlider"] [data-testid="stThumbValue"] {
    display: none !important;
}

.coverage-summary {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 0.25rem 0.75rem;
    padding: 0.35rem 0.5rem;
    margin-top: 0.1rem;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    font-size: 0.7rem;
}

.coverage-details {
    display: grid;
    gap: 0.2rem;
    margin-top: 0.35rem;
    color: var(--text-secondary);
    font-size: 0.67rem;
    line-height: 1.45;
}

.dispatch-status-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.35rem;
    font-size: 0.72rem;
    color: var(--text-secondary);
}

.dispatch-action-row {
    display: grid;
    grid-template-columns: minmax(55px, 1fr) auto auto minmax(96px, auto);
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0;
    border-bottom: 1px solid rgba(37,50,73,0.3);
    font-size: 0.68rem;
}

.dispatch-action-row:last-child {
    border-bottom: none;
}

.dispatch-summary {
    margin-top: 0.35rem;
    font-size: 0.68rem;
    color: var(--text-secondary);
    line-height: 1.4;
}

/* =========================================================
   ROW A: KPI METRIC CARDS
   ========================================================= */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.65rem;
    margin-bottom: 0.75rem;
}

@media (max-width: 1100px) { .kpi-row { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 680px)  { .kpi-row { grid-template-columns: 1fr; } }

.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-card);
    padding: 0.8rem 0.95rem;
    display: flex; flex-direction: column; justify-content: space-between;
    min-height: 94px;
    box-shadow: var(--shadow-card);
    transition: box-shadow var(--transition-med), border-color var(--transition-med),
                transform var(--transition-med);
    position: relative; overflow: hidden;
}

.kpi-card:hover {
    box-shadow: var(--shadow-hover);
    border-color: var(--border-accent);
    transform: translateY(-2px);
}

.kpi-card-top {
    display: flex; justify-content: space-between; align-items: flex-start;
    margin-bottom: 0.25rem;
}

.kpi-icon-wrap {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.0rem; flex-shrink: 0;
}

.kpi-icon-teal   { background: rgba(20,184,166,0.15);  }
.kpi-icon-cyan   { background: rgba(56,189,248,0.15);  }
.kpi-icon-green  { background: rgba(34,197,94,0.15);   }
.kpi-icon-indigo { background: rgba(129,140,248,0.15); }

.kpi-delta {
    display: inline-flex; align-items: center; gap: 0.2rem;
    font-size: 0.72rem; font-weight: 700;
    font-family: var(--font-mono);
    padding: 0.15rem 0.42rem; border-radius: 5px;
}

.kpi-delta-positive { color: var(--accent-green); background: rgba(34,197,94,0.12); }
.kpi-delta-negative { color: var(--accent-red);   background: rgba(239,68,68,0.12); }
.kpi-delta-neutral  { color: var(--text-secondary); background: rgba(148,163,184,0.1); }

.kpi-label {
    font-size: 0.67rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--text-label); margin-bottom: 0.15rem;
}

.kpi-value {
    font-size: 1.45rem; font-weight: 700;
    font-family: var(--font-mono);
    color: var(--text-primary); line-height: 1.1;
    letter-spacing: -0.025em;
    animation: countUp 0.35s ease-out 0.1s both;
}

.kpi-value-teal   { color: var(--accent-teal);   }
.kpi-value-cyan   { color: var(--accent-cyan);   }
.kpi-value-green  { color: var(--accent-green);  }

.kpi-subtext {
    font-size: 0.66rem; color: var(--text-muted);
    margin-top: 0.18rem; line-height: 1.25;
}

/* =========================================================
   ROW B: BATTERY ACTIONS TABLE
   ========================================================= */
.batt-actions-table {
    width: 100%;
    border-collapse: collapse;
    min-width: 700px;
    font-size: 0.69rem;
    margin-top: 0.1rem;
}

.batt-table-scroll {
    width: 100%;
    overflow-x: auto;
    scrollbar-width: thin;
}

.batt-actions-table th {
    color: var(--text-muted);
    font-size: 0.63rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    padding: 0.3rem 0.35rem;
    border-bottom: 1px solid var(--border-subtle);
    text-align: left;
}

.batt-actions-table th.num-col,
.batt-actions-table td.num-col {
    text-align: right;
}

.batt-actions-table td {
    padding: 0.3rem 0.35rem;
    border-bottom: 1px solid rgba(37, 50, 73, 0.4);
    color: var(--text-secondary);
    vertical-align: middle;
}

.batt-actions-table tr:last-child td {
    border-bottom: none;
}

.batt-id-dot {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--text-primary);
    font-weight: 600;
    font-family: var(--font-mono);
}

.batt-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

.action-badge {
    display: inline-block;
    padding: 0.16rem 0.5rem;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-align: center;
}

.action-charge {
    background: rgba(20, 184, 166, 0.16);
    color: var(--accent-teal);
    border: 1px solid rgba(20, 184, 166, 0.35);
}

.action-discharge {
    background: rgba(168, 85, 247, 0.16);
    color: #C084FC;
    border: 1px solid rgba(168, 85, 247, 0.35);
}

.action-idle {
    background: rgba(148, 163, 184, 0.1);
    color: var(--text-secondary);
    border: 1px solid rgba(148, 163, 184, 0.2);
}

.action-power {
    font-family: var(--font-mono);
    font-size: 0.73rem;
    color: var(--text-primary);
}

.action-soc {
    font-family: var(--font-mono);
    font-size: 0.73rem;
    color: var(--text-primary);
    font-weight: 600;
}

.safety-status {
    display: inline-block;
    white-space: nowrap;
    padding: 0.12rem 0.32rem;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 0.59rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}

.safety-status-breach {
    color: #FCA5A5;
    background: rgba(239,68,68,0.12);
}

.safety-status-near {
    color: #FCD34D;
    background: rgba(245,158,11,0.12);
}

.safety-status-normal {
    color: #86EFAC;
    background: rgba(34,197,94,0.1);
}

/* =========================================================
   ROW C: COST BREAKDOWN COMPONENT LIST
   ========================================================= */
.cost-legend-list {
    display: flex;
    flex-direction: column;
    justify-content: center;
    height: 100%;
    gap: 0.15rem;
    padding: 0.2rem 0;
}

.cost-legend-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.32rem 0;
    border-bottom: 1px solid rgba(37, 50, 73, 0.4);
    font-size: 0.72rem;
}

.cost-legend-row:last-child {
    border-bottom: none;
}

.cost-legend-left {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    color: var(--text-secondary);
}

.cost-legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

.cost-legend-right {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.cost-legend-pct {
    font-family: var(--font-mono);
    color: var(--text-muted);
    font-size: 0.69rem;
    min-width: 38px;
    text-align: right;
}

.cost-legend-val {
    font-family: var(--font-mono);
    color: var(--text-primary);
    font-weight: 600;
    font-size: 0.72rem;
    min-width: 58px;
    text-align: right;
}

/* =========================================================
   ROW C: KEY INSIGHTS LIST
   ========================================================= */
.insights-list {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    padding-top: 0.1rem;
}

.insight-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.42rem 0.55rem;
    border-radius: var(--radius-sm);
    background: rgba(26, 37, 64, 0.5);
    border: 1px solid rgba(37, 50, 73, 0.4);
    transition: background var(--transition-fast);
}

.insight-item:hover {
    background: var(--bg-hover);
}

.insight-icon-wrap {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    flex-shrink: 0;
}

.insight-icon-green  { background: rgba(34,197,94,0.18);  }
.insight-icon-amber  { background: rgba(245,158,11,0.18); }
.insight-icon-cyan   { background: rgba(56,189,248,0.18); }
.insight-icon-indigo { background: rgba(129,140,248,0.18); }
.insight-icon-red    { background: rgba(239,68,68,0.18);  }

.insight-text {
    font-size: 0.72rem;
    color: var(--text-secondary);
    line-height: 1.35;
    flex: 1;
}

.insight-text strong {
    color: var(--text-primary);
    font-weight: 600;
}

/* =========================================================
   ROW D: RECENT ACTIVITY LOG
   ========================================================= */
.activity-list {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
}

.activity-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.32rem 0.35rem;
    border-bottom: 1px solid rgba(37, 50, 73, 0.35);
    transition: background var(--transition-fast);
}

.activity-item:last-child {
    border-bottom: none;
}

.activity-item:hover {
    background: rgba(255,255,255,0.02);
    border-radius: 4px;
}

.activity-time {
    font-size: 0.67rem;
    font-family: var(--font-mono);
    color: var(--accent-cyan);
    min-width: 36px;
    font-weight: 600;
}

.activity-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--text-muted);
    flex-shrink: 0;
}

.activity-msg {
    font-size: 0.71rem;
    color: var(--text-secondary);
    line-height: 1.3;
    flex: 1;
}

/* =========================================================
   BATTERY FLEET (DIAGNOSTICS VIEW)
   ========================================================= */
.bess-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
    gap: 0.6rem; margin-bottom: 0.8rem;
}

.bess-unit-card {
    background: var(--bg-card);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-card);
    padding: 0.7rem 0.85rem;
    box-shadow: var(--shadow-card);
    transition: border-color var(--transition-med);
}

.bess-unit-card:hover {
    border-color: var(--border-accent);
}

.bess-unit-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.4rem;
}

.bess-unit-id { font-size: 0.8rem; font-weight: 700; color: var(--text-primary); }

.bess-soc-bar-bg {
    width: 100%; height: 5px; background: var(--bg-secondary);
    border-radius: 3px; margin: 0.35rem 0; overflow: hidden;
}

.bess-soc-bar-val {
    height: 100%; border-radius: 3px;
    transition: width 0.6s ease;
}

.bess-meta-row {
    display: flex; justify-content: space-between;
    font-size: 0.67rem; color: var(--text-secondary); margin-top: 0.2rem;
}

.bess-meta-num {
    font-family: var(--font-mono); color: var(--text-primary); font-weight: 500;
}

/* =========================================================
   DATAFRAME OVERRIDE
   ========================================================= */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-card) !important;
    overflow: hidden !important;
}

/* =========================================================
   TABS
   ========================================================= */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.2rem; background-color: transparent;
    border-bottom: 1px solid var(--border-strong); padding-bottom: 0;
}

.stTabs [data-baseweb="tab"] {
    padding: 0.4rem 0.85rem !important;
    font-weight: 500 !important; font-size: 0.8rem !important;
    color: var(--text-secondary) !important;
    border-radius: 5px 5px 0 0 !important;
    background-color: var(--bg-secondary) !important;
    border: 1px solid transparent !important; border-bottom: none !important;
}

.stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    background-color: var(--bg-card) !important;
    border-color: var(--border-strong) !important;
    border-top: 2px solid var(--accent-teal) !important;
    font-weight: 600 !important;
}

/* =========================================================
   FOOTER
   ========================================================= */
.industrial-footer {
    border-left: 2px solid var(--accent-cyan);
    background: var(--bg-card); padding: 0.5rem 0.85rem;
    font-size: 0.7rem; color: var(--text-muted); margin-top: 1.5rem;
    line-height: 1.45; border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    box-shadow: var(--shadow-card);
}

/* =========================================================
   REDUCED MOTION
   ========================================================= */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
"""
