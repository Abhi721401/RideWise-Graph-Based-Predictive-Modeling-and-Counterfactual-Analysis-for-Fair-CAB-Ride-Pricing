"""
RideWise — Graph-Based Pricing Fairness & Counterfactual Intelligence Engine
Streamlit App v2  |  Loads artifacts from Kaggle run
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib, json, os, math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
import folium
from streamlit_folium import st_folium

# ── PyG imports (must match training)
from torch_geometric.nn import SAGEConv, GATConv

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RideWise · Fare Intelligence",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — dark terminal-meets-data-dashboard aesthetic
# Inspired by Bloomberg Terminal + Vercel dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: #080B10;
    color: #C8CDD8;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem; max-width: 1400px; }

/* ── Top nav bar ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 0 24px 0;
    border-bottom: 1px solid #1C2130;
    margin-bottom: 28px;
}
.topbar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem; font-weight: 800;
    color: #F0F4FF; letter-spacing: -0.5px;
}
.topbar-logo span { color: #3B82F6; }
.topbar-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem; color: #3B82F6;
    border: 1px solid #1D3461;
    background: #0D1829;
    padding: 3px 10px; border-radius: 20px;
    letter-spacing: 1.5px; text-transform: uppercase;
}

/* ── Stat strip ── */
.stat-strip {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1px;
    background: #1C2130;
    border: 1px solid #1C2130;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 28px;
}
.stat-cell {
    background: #0D1117;
    padding: 16px 20px;
    display: flex; flex-direction: column; gap: 4px;
}
.stat-cell:hover { background: #111827; }
.stat-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.55rem; font-weight: 500;
    color: #F0F4FF; line-height: 1;
}
.stat-val.green { color: #22C55E; }
.stat-val.blue  { color: #3B82F6; }
.stat-val.amber { color: #F59E0B; }
.stat-lbl {
    font-size: 0.72rem; color: #4B5563;
    text-transform: uppercase; letter-spacing: 1px;
}

/* ── Two-column layout ── */
.input-panel, .result-panel {
    background: #0D1117;
    border: 1px solid #1C2130;
    border-radius: 12px;
    padding: 24px;
}

/* ── Section label ── */
.sec-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem; color: #3B82F6;
    text-transform: uppercase; letter-spacing: 2px;
    margin-bottom: 12px; display: block;
}

/* ── Coord inputs ── */
.coord-row { display: flex; gap: 8px; margin-bottom: 8px; }

/* ── Fare display ── */
.fare-block {
    background: linear-gradient(135deg, #0D1829 0%, #111827 100%);
    border: 1px solid #1D3461;
    border-radius: 10px;
    padding: 24px 28px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 16px;
}
.fare-amount {
    font-family: 'DM Mono', monospace;
    font-size: 3.2rem; font-weight: 500;
    color: #F0F4FF; line-height: 1;
}
.fare-sub {
    font-size: 0.78rem; color: #4B5563;
    text-transform: uppercase; letter-spacing: 1px;
    margin-top: 6px;
}
.fare-right { text-align: right; }

/* ── Fairness badge ── */
.badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem; font-weight: 500;
    padding: 5px 14px; border-radius: 4px;
    text-transform: uppercase; letter-spacing: 1.5px;
}
.badge-fair      { background: #052E16; color: #4ADE80; border: 1px solid #166534; }
.badge-over      { background: #2D0A0A; color: #F87171; border: 1px solid #7F1D1D; }
.badge-under     { background: #0C1E3C; color: #60A5FA; border: 1px solid #1E3A8A; }

/* ── Score dial ── */
.score-block { text-align: center; padding: 8px 0; }
.score-num {
    font-family: 'DM Mono', monospace;
    font-size: 2.8rem; font-weight: 500;
    line-height: 1;
}
.score-unit { font-size: 1rem; color: #4B5563; }
.score-lbl  { font-size: 0.68rem; color: #4B5563; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }

/* ── Info row ── */
.info-row {
    display: flex; gap: 6px; flex-wrap: wrap;
    margin-bottom: 16px;
}
.info-chip {
    background: #0D1117;
    border: 1px solid #1C2130;
    border-radius: 6px;
    padding: 6px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem; color: #6B7280;
}
.info-chip b { color: #9CA3AF; }

/* ── Counterfactual cards ── */
.cf-card {
    background: #0D1117;
    border: 1px solid #1C2130;
    border-left: 3px solid #3B82F6;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex; justify-content: space-between; align-items: center;
}
.cf-card:hover { border-color: #2563EB; background: #111827; }
.cf-left { flex: 1; }
.cf-type {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem; color: #3B82F6;
    text-transform: uppercase; letter-spacing: 2px; margin-bottom: 4px;
}
.cf-text { font-size: 0.88rem; color: #9CA3AF; }
.cf-right { text-align: right; flex-shrink: 0; padding-left: 16px; }
.cf-fare { font-family: 'DM Mono', monospace; font-size: 1.1rem; color: #F0F4FF; }
.cf-save { font-family: 'DM Mono', monospace; font-size: 0.78rem; color: #22C55E; margin-top: 2px; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1C2130;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #4B5563;
    padding: 10px 20px;
    border: none;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #3B82F6 !important;
    border-bottom: 2px solid #3B82F6 !important;
    background: transparent !important;
}

/* ── Importance bar ── */
.imp-row {
    display: flex; align-items: center; gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid #0D1117;
}
.imp-name { font-size: 0.82rem; color: #6B7280; width: 160px; flex-shrink: 0; }
.imp-bar-bg {
    flex: 1; height: 6px; background: #1C2130; border-radius: 3px; overflow: hidden;
}
.imp-bar-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #1D4ED8, #3B82F6); }
.imp-pct { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #374151; width: 36px; text-align: right; }

/* ── Presets ── */
.preset-btn { /* handled by streamlit */ }

/* ── Divider ── */
.divider {
    height: 1px; background: #1C2130;
    margin: 24px 0;
}

/* ── Analytics header ── */
.analytics-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem; font-weight: 700;
    color: #F0F4FF;
    margin-bottom: 4px;
}
.analytics-sub {
    font-size: 0.8rem; color: #4B5563;
    margin-bottom: 20px;
}

/* ── Empty state ── */
.empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 60px 20px; text-align: center;
    min-height: 420px;
}
.empty-icon { font-size: 3rem; margin-bottom: 16px; opacity: 0.5; }
.empty-title { font-family: 'Syne', sans-serif; font-size: 1.1rem; color: #374151; margin-bottom: 8px; }
.empty-sub   { font-size: 0.85rem; color: #1F2937; max-width: 280px; line-height: 1.6; }

/* ── Streamlit widget tweaks ── */
.stSlider > div > div > div { background: #1C2130 !important; }
.stSlider > div > div > div > div { background: #3B82F6 !important; }
div[data-testid="stNumberInput"] input {
    background: #111827; border: 1px solid #1C2130; color: #C8CDD8;
    border-radius: 6px; font-family: 'DM Mono', monospace; font-size: 0.85rem;
}
div[data-testid="stSelectbox"] > div > div {
    background: #111827; border: 1px solid #1C2130; color: #C8CDD8;
}
.stButton button {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem; letter-spacing: 1.5px; text-transform: uppercase;
    border-radius: 6px;
}
.stButton button[kind="primary"] {
    background: #1D4ED8; border: none; color: white;
}
.stButton button[kind="primary"]:hover { background: #2563EB; }
.stButton button[kind="secondary"] {
    background: #111827; border: 1px solid #1C2130; color: #6B7280;
}
.stButton button[kind="secondary"]:hover { border-color: #3B82F6; color: #3B82F6; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB dark theme
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0D1117',
    'axes.facecolor':    '#0D1117',
    'axes.edgecolor':    '#1C2130',
    'axes.labelcolor':   '#4B5563',
    'xtick.color':       '#374151',
    'ytick.color':       '#374151',
    'text.color':        '#9CA3AF',
    'grid.color':        '#1C2130',
    'grid.alpha':        1.0,
    'font.family':       'monospace',
    'figure.dpi':        130,
})

# ─────────────────────────────────────────────────────────────────────────────
# GNN MODEL  (must match training exactly)
# ─────────────────────────────────────────────────────────────────────────────
class RideWiseGNN(nn.Module):
    def __init__(self, node_in, edge_in, hidden=128, gat_out=64, heads=4, dropout=0.2):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(node_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.sage1 = SAGEConv(hidden, hidden, aggr='mean')
        self.sage2 = SAGEConv(hidden, hidden, aggr='mean')
        self.gat   = GATConv(hidden, gat_out // heads, heads=heads,
                              dropout=dropout, concat=True, add_self_loops=True)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.ln3 = nn.LayerNorm(gat_out)
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU()
        )
        head_in = gat_out + gat_out + hidden // 2
        self.head = nn.Sequential(
            nn.Linear(head_in, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1)
        )
        self.drop = nn.Dropout(dropout)

    def encode_nodes(self, x, edge_index):
        h = self.node_enc(x)
        h = self.ln1(F.gelu(self.sage1(h, edge_index)) + h)
        h = self.drop(h)
        h = self.ln2(F.gelu(self.sage2(h, edge_index)) + h)
        h = self.drop(h)
        h = self.ln3(F.elu(self.gat(h, edge_index)))
        return h

    def forward(self, x, edge_index, edge_attr):
        node_emb = self.encode_nodes(x, edge_index)
        edge_emb = self.edge_enc(edge_attr)
        src, dst = edge_index
        combined = torch.cat([node_emb[src], node_emb[dst], edge_emb], dim=-1)
        return self.head(combined).squeeze(-1), node_emb

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def haversine(la1, lo1, la2, lo2):
    R = 6371
    la1, lo1, la2, lo2 = map(math.radians, [la1, lo1, la2, lo2])
    dlat, dlon = la2 - la1, lo2 - lo1
    a = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(max(0, min(1, a))))

def get_peak(h):
    if 7 <= h <= 9 or 17 <= h <= 20: return 2
    if h >= 22 or h <= 5:             return 1
    return 0

def classify_fairness(dev):
    if dev > 0.20:  return 'Overpriced'
    if dev < -0.20: return 'Underpriced'
    return 'Fair'

def fairness_score(dev):
    return round(max(0, min(100, 100 - abs(dev) * 100)), 1)

DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
RIDE_FEAT_COLS = [
    'distance_km','passenger_count','fare_per_km',
    'hour_sin','hour_cos','dow_sin','dow_cos',
    'mon_sin','mon_cos','peak_type','is_weekend'
]

# ─────────────────────────────────────────────────────────────────────────────
# ARTIFACT LOADER
# ─────────────────────────────────────────────────────────────────────────────
ARTIFACT_DIR = './ridewise_artifacts'

@st.cache_resource(show_spinner="Loading RideWise model…")
def load_artifacts():
    try:
        with open(f'{ARTIFACT_DIR}/model_config.json') as f: cfg = json.load(f)
        with open(f'{ARTIFACT_DIR}/summary.json')      as f: smry = json.load(f)

        node_scaler  = joblib.load(f'{ARTIFACT_DIR}/node_scaler.pkl')
        edge_scaler  = joblib.load(f'{ARTIFACT_DIR}/edge_scaler.pkl')
        label_scaler = joblib.load(f'{ARTIFACT_DIR}/label_scaler.pkl')
        kmeans       = joblib.load(f'{ARTIFACT_DIR}/kmeans_zones.pkl')

        node_feats_scaled = np.load(f'{ARTIFACT_DIR}/node_feats_scaled.npy')
        edge_df   = pd.read_csv(f'{ARTIFACT_DIR}/edge_data.csv')
        fair_df   = pd.read_csv(f'{ARTIFACT_DIR}/fairness_results.csv')
        zone_df   = pd.read_csv(f'{ARTIFACT_DIR}/zone_centers.csv')
        zone_stats = pd.read_csv(f'{ARTIFACT_DIR}/zone_stats.csv')

        model = RideWiseGNN(
            node_in = cfg['node_in'], edge_in = cfg['edge_in'],
            hidden  = cfg['hidden'],  gat_out = cfg['gat_out'],
            heads   = cfg['heads'],   dropout = cfg['dropout']
        )
        model.load_state_dict(
            torch.load(f'{ARTIFACT_DIR}/best_model.pt', map_location='cpu')
        )
        model.eval()

        X_node = torch.tensor(node_feats_scaled, dtype=torch.float32)

        return dict(cfg=cfg, smry=smry, model=model, kmeans=kmeans,
                    node_scaler=node_scaler, edge_scaler=edge_scaler,
                    label_scaler=label_scaler, X_node=X_node,
                    node_feats_scaled=node_feats_scaled,
                    edge_df=edge_df, fair_df=fair_df,
                    zone_df=zone_df, zone_stats=zone_stats)
    except Exception as e:
        return {'error': str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE  — single ride
# ─────────────────────────────────────────────────────────────────────────────
def predict_ride(art, pu_lat, pu_lon, do_lat, do_lon,
                 hour, dow, passengers, month=6):
    kmeans       = art['kmeans']
    edge_scaler  = art['edge_scaler']
    label_scaler = art['label_scaler']
    model        = art['model']
    X_node       = art['X_node']

    dist         = haversine(pu_lat, pu_lon, do_lat, do_lon)
    peak         = get_peak(hour)
    is_wknd      = float(dow >= 5)
    fare_per_km_est = 2.5 + (0.8 if peak == 2 else 0)

    pu_zone = int(kmeans.predict([[pu_lat, pu_lon]])[0])
    do_zone = int(kmeans.predict([[do_lat, do_lon]])[0])

    ef_raw = np.array([[
        dist, float(passengers), fare_per_km_est,
        np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
        np.sin(2*np.pi*dow/7),   np.cos(2*np.pi*dow/7),
        np.sin(2*np.pi*month/12),np.cos(2*np.pi*month/12),
        float(peak), is_wknd
    ]], dtype=np.float32)

    ef_scaled  = edge_scaler.transform(ef_raw)
    edge_index = torch.tensor([[pu_zone],[do_zone]], dtype=torch.long)
    edge_attr  = torch.tensor(ef_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        node_emb = model.encode_nodes(X_node, edge_index)
        edge_emb = model.edge_enc(edge_attr)
        src_emb  = node_emb[0:1]
        dst_emb  = node_emb[-1:]
        combined = torch.cat([src_emb, dst_emb, edge_emb], dim=-1)
        pred_sc   = model.head(combined).squeeze().item()
        pred_fare = float(label_scaler.inverse_transform([[pred_sc]])[0][0])

    pred_fare = max(2.50, pred_fare)
    return pred_fare, pu_zone, do_zone, dist, peak, ef_raw, ef_scaled

# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────
FEAT_LABELS = {
    'distance_km':      'Distance',
    'passenger_count':  'Passengers',
    'fare_per_km':      'Fare/km Signal',
    'hour_sin':         'Hour (sin)',
    'hour_cos':         'Hour (cos)',
    'dow_sin':          'Day of Week',
    'dow_cos':          'Day of Week (phase)',
    'mon_sin':          'Month',
    'mon_cos':          'Month (phase)',
    'peak_type':        'Peak Hour',
    'is_weekend':       'Weekend',
}

def explain_prediction(art, X_node, edge_index, ef_scaled_base):
    model        = art['model']
    label_scaler = art['label_scaler']

    ef_t = torch.tensor(ef_scaled_base, dtype=torch.float32)

    with torch.no_grad():
        node_emb = model.encode_nodes(X_node, edge_index)
        edge_emb_base = model.edge_enc(ef_t)
        combined_base = torch.cat([node_emb[0:1], node_emb[-1:], edge_emb_base], dim=-1)
        base_sc  = model.head(combined_base).squeeze().item()
        base_fare  = float(label_scaler.inverse_transform([[base_sc]])[0][0])

    importances = {}
    for i, col in enumerate(RIDE_FEAT_COLS):
        ef_perturbed = ef_t.clone()
        ef_perturbed[0, i] = 0.0
        with torch.no_grad():
            ee = model.edge_enc(ef_perturbed)
            cc = torch.cat([node_emb[0:1], node_emb[-1:], ee], dim=-1)
            p_sc = model.head(cc).squeeze().item()
            p_fare = float(label_scaler.inverse_transform([[p_sc]])[0][0])
        importances[FEAT_LABELS.get(col, col)] = abs(base_fare - p_fare)

    total = sum(importances.values()) + 1e-8
    return {k: round(v/total*100, 1) for k, v in importances.items()}, base_fare

# ─────────────────────────────────────────────────────────────────────────────
# COUNTERFACTUALS
# ─────────────────────────────────────────────────────────────────────────────
def generate_counterfactuals(art, pu_lat, pu_lon, do_lat, do_lon,
                              hour, dow, passengers, current_fare):
    kmeans = art['kmeans']
    scenarios = []
    curr_peak = get_peak(hour)

    # 1. Time alternatives
    for dh in [-3, -2, -1, 1, 2, 3]:
        ah = (hour + dh) % 24
        ap = get_peak(ah)
        if ap < curr_peak:
            try:
                af, *_ = predict_ride(art, pu_lat, pu_lon, do_lat, do_lon, ah, dow, passengers)
                sv = current_fare - af
                if sv > 0.50:
                    scenarios.append({
                        'type': 'Time Shift',
                        'suggestion': f'Leave at {ah:02d}:00 — less busy ({["Off-peak","Night","Rush"][ap]})',
                        'estimated_fare': round(af, 2),
                        'savings': round(sv, 2)
                    })
            except: pass

    # 2. Nearby pickup
    pu_zone   = int(kmeans.predict([[pu_lat, pu_lon]])[0])
    pu_center = kmeans.cluster_centers_[pu_zone]
    dists_all = np.linalg.norm(kmeans.cluster_centers_ - pu_center, axis=1)
    nearby    = np.argsort(dists_all)[1:5]
    for z in nearby:
        alt_lat, alt_lon = kmeans.cluster_centers_[z]
        walk_m = int(dists_all[z] * 111_000)
        try:
            af, *_ = predict_ride(art, alt_lat, alt_lon, do_lat, do_lon, hour, dow, passengers)
            sv = current_fare - af
            if sv > 1.0:
                scenarios.append({
                    'type': 'Nearby Pickup',
                    'suggestion': f'Walk ~{walk_m}m to alternate pickup zone',
                    'estimated_fare': round(af, 2),
                    'savings': round(sv, 2)
                })
        except: pass

    # 3. Ride pool
    if passengers <= 3:
        pf = round(current_fare * 0.70, 2)
        scenarios.append({
            'type': 'Pool / Share',
            'suggestion': 'UberPool — share with other riders going your way',
            'estimated_fare': pf,
            'savings': round(current_fare - pf, 2)
        })

    # 4. Fewer passengers edge-case warning
    if passengers > 1:
        try:
            af, *_ = predict_ride(art, pu_lat, pu_lon, do_lat, do_lon, hour, dow, 1)
            if current_fare - af > 0.5:
                scenarios.append({
                    'type': 'Single Passenger',
                    'suggestion': 'Estimated fare if only 1 passenger (model baseline)',
                    'estimated_fare': round(af, 2),
                    'savings': round(current_fare - af, 2)
                })
        except: pass

    return sorted(scenarios, key=lambda x: -x['savings'])[:5]

# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def make_importance_html(importances):
    sorted_items = sorted(importances.items(), key=lambda x: -x[1])
    rows = ""
    for name, pct in sorted_items:
        if pct < 0.5: continue
        rows += f"""
        <div class="imp-row">
            <div class="imp-name">{name}</div>
            <div class="imp-bar-bg">
                <div class="imp-bar-fill" style="width:{min(pct,100)}%"></div>
            </div>
            <div class="imp-pct">{pct:.0f}%</div>
        </div>"""
    return rows

def plot_fairness_distribution(fair_df):
    counts = fair_df['fairness_label'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors  = {'Fair':'#22C55E','Overpriced':'#EF4444','Underpriced':'#3B82F6'}
    bars    = ax.bar(counts.index, counts.values,
                     color=[colors.get(l,'#6B7280') for l in counts.index],
                     width=0.55, zorder=2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+500,
                f'{val:,}', ha='center', va='bottom', fontsize=8, color='#6B7280')
    ax.set_title('Fairness Label Distribution', fontsize=9, color='#6B7280', pad=10)
    ax.set_ylabel('Rides', fontsize=8)
    ax.yaxis.grid(True, lw=0.5); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig

def plot_deviation_histogram(fair_df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    devs = fair_df['deviation'].clip(-1, 1)
    ax.hist(devs, bins=60, color='#1D4ED8', alpha=0.85, edgecolor='none', zorder=2)
    ax.axvline( 0.20, color='#EF4444', ls='--', lw=1.5, alpha=0.8, label='+20%')
    ax.axvline(-0.20, color='#3B82F6', ls='--', lw=1.5, alpha=0.8, label='-20%')
    ax.axvline(    0, color='#9CA3AF', ls='-',  lw=0.8, alpha=0.4)
    ax.set_title('Deviation Distribution', fontsize=9, color='#6B7280', pad=10)
    ax.set_xlabel('Relative Deviation', fontsize=8)
    ax.legend(fontsize=7, facecolor='#0D1117', edgecolor='#1C2130', labelcolor='#6B7280')
    ax.yaxis.grid(True, lw=0.5); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig

def plot_zone_fairness_map(zone_stats, zone_df, highlight_zones=None):
    # zone_stats already has z_lat/z_lon from notebook merge; fall back to zone_df if not
    if 'z_lat' in zone_stats.columns and 'z_lon' in zone_stats.columns:
        zs = zone_stats.dropna(subset=['z_lat','z_lon'])
    else:
        zc = zone_df.rename(columns={'zone_id':'pu_zone','lat':'z_lat','lon':'z_lon'})
        zs = zone_stats.merge(zc, on='pu_zone', how='left').dropna(subset=['z_lat','z_lon'])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(zs['z_lon'], zs['z_lat'],
                    c=zs['avg_fairness'], cmap='RdYlGn',
                    s=zs['ride_count'].clip(10,500)/5,
                    alpha=0.75, vmin=60, vmax=100,
                    edgecolors='none', zorder=2)
    if highlight_zones:
        for z, col, label in highlight_zones:
            row = zs[zs['pu_zone']==z]
            if not row.empty:
                ax.scatter(row['z_lon'], row['z_lat'],
                           s=180, c=col, zorder=5,
                           edgecolors='white', linewidths=1.5, label=label)
        ax.legend(fontsize=7, facecolor='#0D1117', edgecolor='#1C2130', labelcolor='#9CA3AF')

    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=7, colors='#4B5563')
    cbar.set_label('Fairness Score', fontsize=7, color='#4B5563')
    ax.set_title('Zone-Level Fairness Map', fontsize=9, color='#6B7280', pad=10)
    ax.set_xlabel('Longitude', fontsize=7); ax.set_ylabel('Latitude', fontsize=7)
    ax.tick_params(labelsize=7)
    ax.yaxis.grid(True, lw=0.4); ax.xaxis.grid(True, lw=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig

def plot_route_on_map(zone_stats, zone_df, pu_zone, do_zone, pu_lat, pu_lon, do_lat, do_lon):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    # background zones
    if 'z_lat' in zone_stats.columns and 'z_lon' in zone_stats.columns:
        zs = zone_stats.dropna(subset=['z_lat','z_lon'])
    else:
        zc = zone_df.rename(columns={'zone_id':'pu_zone','lat':'z_lat','lon':'z_lon'})
        zs = zone_stats.merge(zc, on='pu_zone', how='left').dropna(subset=['z_lat','z_lon'])
    sc = ax.scatter(zs['z_lon'], zs['z_lat'],
                    c=zs['avg_fairness'], cmap='RdYlGn',
                    s=30, alpha=0.35, vmin=60, vmax=100, edgecolors='none')

    # Route arrow
    ax.annotate("",
        xy=(do_lon, do_lat), xytext=(pu_lon, pu_lat),
        arrowprops=dict(
            arrowstyle="-|>", color='#3B82F6', lw=2,
            connectionstyle="arc3,rad=0.15"
        )
    )
    # Pickup
    ax.scatter([pu_lon],[pu_lat], s=150, c='#22C55E', zorder=6,
               edgecolors='white', linewidths=1.5, label='Pickup')
    # Dropoff
    ax.scatter([do_lon],[do_lat], s=150, c='#EF4444', zorder=6,
               edgecolors='white', linewidths=1.5, label='Dropoff')
    ax.legend(fontsize=7, facecolor='#0D1117', edgecolor='#1C2130', labelcolor='#9CA3AF')
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=7, colors='#4B5563')
    cbar.set_label('Zone Fairness', fontsize=7, color='#4B5563')
    ax.set_title('Your Route', fontsize=9, color='#6B7280', pad=10)
    ax.set_xlabel('Longitude', fontsize=7); ax.set_ylabel('Latitude', fontsize=7)
    ax.tick_params(labelsize=7)
    ax.yaxis.grid(True, lw=0.4); ax.xaxis.grid(True, lw=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
art = load_artifacts()

# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">RideWise <span style="font-weight:400;font-size:1rem;color:#6B7280">with Abhijnan</span></div>
    <div class="topbar-badge">GNN · Fairness · Counterfactuals</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ERROR GUARD
# ─────────────────────────────────────────────────────────────────────────────
if 'error' in art:
    st.error(f"**Could not load artifacts:** {art['error']}")
    st.markdown("""
    ### Setup Instructions

    **Step 1 — Download from Kaggle:**  
    After running the notebook, go to the Kaggle output tab and download the entire  
    `ridewise_artifacts/` folder (or download as ZIP and unzip it).

    **Step 2 — Place it here:**
    ```
    ridewise_streamlit/
    ├── app.py              ← this file
    ├── requirements.txt
    └── ridewise_artifacts/ ← paste the downloaded folder here
        ├── best_model.pt
        ├── kmeans_zones.pkl
        ├── node_scaler.pkl
        ├── edge_scaler.pkl
        ├── label_scaler.pkl
        ├── model_config.json
        ├── summary.json
        ├── node_feats_scaled.npy
        ├── edge_data.csv
        ├── fairness_results.csv
        ├── zone_centers.csv
        └── zone_stats.csv
    ```

    **Step 3 — Run:**
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
    """)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# STAT STRIP
# ─────────────────────────────────────────────────────────────────────────────
s = art['smry']
st.markdown(f"""
<div class="stat-strip">
    <div class="stat-cell">
        <div class="stat-val blue">{s['total_rides']:,}</div>
        <div class="stat-lbl">Training Rides</div>
    </div>
    <div class="stat-cell">
        <div class="stat-val">{s['n_zones']}</div>
        <div class="stat-lbl">Spatial Zones</div>
    </div>
    <div class="stat-cell">
        <div class="stat-val green">${s['test_mae']:.2f}</div>
        <div class="stat-lbl">Model MAE</div>
    </div>
    <div class="stat-cell">
        <div class="stat-val">{s['test_r2']:.3f}</div>
        <div class="stat-lbl">R² Score</div>
    </div>
    <div class="stat-cell">
        <div class="stat-val amber">{s['pct_fair']:.0f}%</div>
        <div class="stat-lbl">Fair Routes</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT: LEFT = inputs  |  RIGHT = results
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.55], gap="large")

# ── PRESETS


with left:
    st.markdown('<span class="sec-label">Trip Details</span>', unsafe_allow_html=True)
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    # Which point are we setting?
    if 'click_step' not in st.session_state:
        st.session_state['click_step'] = 'pickup'
    if 'pu_lat' not in st.session_state:
        st.session_state['pu_lat'] = 40.7580
        st.session_state['pu_lon'] = -73.9855
    if 'do_lat' not in st.session_state:
        st.session_state['do_lat'] = 40.6413
        st.session_state['do_lon'] = -73.7781

    step = st.session_state['click_step']

    # Status indicator
    if step == 'pickup':
        step_html = '<span style="font-family:DM Mono,monospace;font-size:0.72rem;color:#22C55E;letter-spacing:1.5px">● CLICK MAP TO SET PICKUP</span>'
    elif step == 'dropoff':
        step_html = '<span style="font-family:DM Mono,monospace;font-size:0.72rem;color:#EF4444;letter-spacing:1.5px">● CLICK MAP TO SET DROPOFF</span>'
    else:
        step_html = '<span style="font-family:DM Mono,monospace;font-size:0.72rem;color:#3B82F6;letter-spacing:1.5px">✓ ROUTE SET — READY TO ANALYZE</span>'

    st.markdown(step_html, unsafe_allow_html=True)
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    # Build the click map — locked to NYC training data bounds
    NYC_BOUNDS = [[40.4, -74.5], [41.0, -72.8]]  # matches training data filter

    click_map = folium.Map(
        location=[40.7300, -73.9350],
        zoom_start=11,
        tiles='CartoDB dark_matter',
        height=280,
        max_bounds=True,
        min_zoom=10,
    )

    # Lock map to NYC bbox — user cannot pan outside training data coverage
    click_map.fit_bounds(NYC_BOUNDS)
    click_map.options['maxBounds'] = [[40.2, -74.8], [41.2, -72.5]]
    click_map.options['maxBoundsViscosity'] = 1.0

    # Draw NYC coverage boundary as a subtle rectangle
    folium.Rectangle(
        bounds=NYC_BOUNDS,
        color='#1D4ED8',
        fill=True,
        fill_opacity=0.03,
        weight=1.5,
        dash_array='6',
        tooltip='NYC Training Data Coverage'
    ).add_to(click_map)

    # Instruction overlay on map
    step_instruction = (
        'Click to set <b style="color:#22C55E">Pickup</b>' if st.session_state['click_step'] == 'pickup'
        else 'Click to set <b style="color:#EF4444">Dropoff</b>' if st.session_state['click_step'] == 'dropoff'
        else '<b style="color:#3B82F6">✓ Route set</b>'
    )
    instruction_html = (
        '<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
        'z-index:1000;background:rgba(13,17,23,0.88);border:1px solid #1C2130;'
        'border-radius:6px;padding:6px 14px;font-family:monospace;font-size:12px;'
        f'color:#9CA3AF;pointer-events:none">{step_instruction}</div>'
    )
    click_map.get_root().html.add_child(folium.Element(instruction_html))

    # Show existing pickup pin if set
    pu_lat_def = st.session_state['pu_lat']
    pu_lon_def = st.session_state['pu_lon']
    do_lat_def = st.session_state['do_lat']
    do_lon_def = st.session_state['do_lon']

    if step in ('dropoff', 'done'):
        folium.Marker(
            location=[pu_lat_def, pu_lon_def],
            tooltip="📍 Pickup",
            icon=folium.Icon(color='green', icon='record', prefix='fa')
        ).add_to(click_map)
        folium.CircleMarker(
            location=[pu_lat_def, pu_lon_def],
            radius=10, color='#22C55E',
            fill=True, fill_opacity=0.2, weight=2
        ).add_to(click_map)

    if step == 'done':
        folium.Marker(
            location=[do_lat_def, do_lon_def],
            tooltip="🏁 Dropoff",
            icon=folium.Icon(color='red', icon='record', prefix='fa')
        ).add_to(click_map)
        folium.CircleMarker(
            location=[do_lat_def, do_lon_def],
            radius=10, color='#EF4444',
            fill=True, fill_opacity=0.2, weight=2
        ).add_to(click_map)
        folium.PolyLine(
            locations=[[pu_lat_def, pu_lon_def], [do_lat_def, do_lon_def]],
            color='#3B82F6', weight=3, opacity=0.7, dash_array='6'
        ).add_to(click_map)
        click_map.fit_bounds([
            [min(pu_lat_def, do_lat_def) - 0.01, min(pu_lon_def, do_lon_def) - 0.01],
            [max(pu_lat_def, do_lat_def) + 0.01, max(pu_lon_def, do_lon_def) + 0.01]
        ])

    # Render click map — capture clicks
    map_result = st_folium(
        click_map,
        use_container_width=True,
        height=280,
        returned_objects=["last_clicked"],
        key="click_map"
    )

    # Handle click — with NYC bounds validation
    if map_result and map_result.get("last_clicked"):
        clicked_lat = map_result["last_clicked"]["lat"]
        clicked_lng = map_result["last_clicked"]["lng"]

        # Validate within NYC training data coverage
        in_bounds = (40.4 <= clicked_lat <= 41.0) and (-74.5 <= clicked_lng <= -72.8)

        if not in_bounds:
            st.warning("📍 Click within NYC bounds (the blue dashed area) — model is trained on NYC data only.")
        elif step == 'pickup':
            st.session_state['pu_lat'] = clicked_lat
            st.session_state['pu_lon'] = clicked_lng
            st.session_state['click_step'] = 'dropoff'
            st.rerun()
        elif step == 'dropoff':
            st.session_state['do_lat'] = clicked_lat
            st.session_state['do_lon'] = clicked_lng
            st.session_state['click_step'] = 'done'
            st.rerun()

    # Reset button
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        if st.button("↺ Reset Points", use_container_width=True, key="reset_pts"):
            st.session_state['click_step'] = 'pickup'
            st.rerun()
    with rcol2:
        if st.button("✎ Edit Manually", use_container_width=True, key="edit_manual"):
            st.session_state['show_manual'] = not st.session_state.get('show_manual', False)
            st.rerun()

    # Manual coordinate override (collapsed by default)
    if st.session_state.get('show_manual', False):
        st.markdown('<span class="sec-label" style="margin-top:8px">Pickup</span>', unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        manual_pu_lat = mc1.number_input("Lat", value=float(st.session_state['pu_lat']), format="%.4f", step=0.001, key="m_pu_lat")
        manual_pu_lon = mc2.number_input("Lon", value=float(st.session_state['pu_lon']), format="%.4f", step=0.001, key="m_pu_lon")
        st.markdown('<span class="sec-label" style="margin-top:4px">Dropoff</span>', unsafe_allow_html=True)
        mc3, mc4 = st.columns(2)
        manual_do_lat = mc3.number_input("Lat", value=float(st.session_state['do_lat']), format="%.4f", step=0.001, key="m_do_lat")
        manual_do_lon = mc4.number_input("Lon", value=float(st.session_state['do_lon']), format="%.4f", step=0.001, key="m_do_lon")
        if st.button("Apply", key="apply_manual", use_container_width=True):
            st.session_state['pu_lat'] = manual_pu_lat
            st.session_state['pu_lon'] = manual_pu_lon
            st.session_state['do_lat'] = manual_do_lat
            st.session_state['do_lon'] = manual_do_lon
            st.session_state['click_step'] = 'done'
            st.rerun()

    # Final coord values used for prediction
    pu_lat = float(st.session_state['pu_lat'])
    pu_lon = float(st.session_state['pu_lon'])
    do_lat = float(st.session_state['do_lat'])
    do_lon = float(st.session_state['do_lon'])

    # Show current coords as chips
    st.markdown(
        f'<div style="display:flex;gap:6px;margin:8px 0;flex-wrap:wrap">'
        f'<div style="background:#052E16;border:1px solid #166534;border-radius:5px;'
        f'padding:4px 10px;font-family:DM Mono,monospace;font-size:0.7rem;color:#4ADE80">'
        f'📍 {pu_lat:.4f}, {pu_lon:.4f}</div>'
        f'<div style="background:#2D0A0A;border:1px solid #7F1D1D;border-radius:5px;'
        f'padding:4px 10px;font-family:DM Mono,monospace;font-size:0.7rem;color:#F87171">'
        f'🏁 {do_lat:.4f}, {do_lon:.4f}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Time + passengers
    st.markdown('<span class="sec-label" style="margin-top:8px">Time & Passengers</span>', unsafe_allow_html=True)
    c5, c6, c7 = st.columns([2, 2, 1])

    hour = c5.slider("Hour", 0, 23, 8, format="%02d:00")
    peak = get_peak(hour)
    peak_labels = {0: "✓ Off-peak", 1: "⬡ Night", 2: "⚡ Rush hour"}
    peak_colors = {0: "#22C55E", 1: "#6B7280", 2: "#F59E0B"}
    c5.markdown(
        f'<span style="font-family:DM Mono,monospace;font-size:0.7rem;color:{peak_colors[peak]}">'
        f'{peak_labels[peak]}</span>',
        unsafe_allow_html=True
    )

    dow = c6.selectbox("Day", DAYS, index=0)
    dow_idx = DAYS.index(dow)

    passengers = c7.selectbox("Passengers", [1,2,3,4,5,6], index=0)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    run_btn = st.button("→ Analyze Fare", type="primary", use_container_width=True,
                        disabled=(st.session_state.get('click_step') != 'done'))

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with right:
    if run_btn:
        with st.spinner(""):
            pred_fare, pu_zone, do_zone, dist_km, peak_type, ef_raw, ef_scaled = predict_ride(
                art, pu_lat, pu_lon, do_lat, do_lon, hour, dow_idx, passengers
            )

            # For explainability we need X_node and edge_index
            X_node_t   = art['X_node']
            edge_index = torch.tensor([[pu_zone],[do_zone]], dtype=torch.long)
            importances, _ = explain_prediction(art, X_node_t, edge_index, ef_scaled)

            # Fairness — compare to zone route average if available
            edge_df = art['edge_df']
            route_mask = (edge_df['pu_zone']==pu_zone) & (edge_df['do_zone']==do_zone)
            if route_mask.sum() > 0:
                route_avg = float(edge_df.loc[route_mask, 'avg_fare'].values[0])
                dev = (route_avg - pred_fare) / (pred_fare + 1e-6)
            else:
                dev = 0.0
                route_avg = pred_fare

            f_label = classify_fairness(dev)
            f_score = fairness_score(dev)

            badge_class = {'Fair':'badge-fair','Overpriced':'badge-over','Underpriced':'badge-under'}[f_label]
            score_color = {'Fair':'#22C55E','Overpriced':'#EF4444','Underpriced':'#3B82F6'}[f_label]

            cfs = generate_counterfactuals(
                art, pu_lat, pu_lon, do_lat, do_lon, hour, dow_idx, passengers, pred_fare
            )

        # ── Fare block
        st.markdown(f"""
        <div class="fare-block">
            <div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                            color:#4B5563;letter-spacing:2px;text-transform:uppercase;
                            margin-bottom:8px">Estimated Fair Fare</div>
                <div class="fare-amount">${pred_fare:.2f}</div>
                <div style="margin-top:10px">
                    <span class="badge {badge_class}">{f_label}</span>
                </div>
            </div>
            <div class="fare-right">
                <div class="score-num" style="color:{score_color}">{f_score}<span class="score-unit">/100</span></div>
                <div class="score-lbl">Fairness Score</div>
                <div style="margin-top:10px">
                    <div class="info-chip"><b>{dist_km:.1f} km</b></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Info chips row
        direction_labels = ["N","NE","E","SE","S","SW","W","NW","N"]
        bearing = (math.degrees(math.atan2(
            do_lon - pu_lon, do_lat - pu_lat
        )) + 360) % 360
        dir_label = direction_labels[int((bearing + 22.5) / 45) % 8]
        peak_str  = ["Off-peak","Night","Rush hour"][peak_type]

        st.markdown(f"""
        <div class="info-row">
            <div class="info-chip">Zone <b>{pu_zone}</b> → <b>{do_zone}</b></div>
            <div class="info-chip">Direction: <b>{dir_label}</b></div>
            <div class="info-chip">Time: <b>{peak_str}</b></div>
            <div class="info-chip">Dev: <b>{dev*100:+.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Tabs
        tab_exp, tab_cf, tab_map = st.tabs(["Explanation", "Alternatives", "Map"])

        with tab_exp:
            st.markdown("**What drives this prediction?**")
            imp_html = make_importance_html(importances)
            st.markdown(f'<div style="padding:4px 0">{imp_html}</div>', unsafe_allow_html=True)

            # Context callout
            ctx_msgs = {
                2: "⚡ Rush hour detected — fares typically run 15–25% higher between 7–9 AM and 5–8 PM.",
                1: "🌙 Night time — late-night surcharges may apply after 10 PM.",
                0: "✓ Off-peak window — this is the most price-competitive time to ride."
            }
            st.markdown(
                f'<div style="background:#0D1829;border:1px solid #1D3461;border-radius:6px;'
                f'padding:12px 14px;font-size:0.82rem;color:#60A5FA;margin-top:16px">'
                f'{ctx_msgs[peak_type]}</div>',
                unsafe_allow_html=True
            )

        with tab_cf:
            if cfs:
                for cf in cfs:
                    st.markdown(f"""
                    <div class="cf-card">
                        <div class="cf-left">
                            <div class="cf-type">{cf['type']}</div>
                            <div class="cf-text">{cf['suggestion']}</div>
                        </div>
                        <div class="cf-right">
                            <div class="cf-fare">${cf['estimated_fare']:.2f}</div>
                            <div class="cf-save">save ${cf['savings']:.2f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="color:#374151;font-size:0.85rem;padding:20px 0">'
                    'No cheaper alternatives found for this route.</div>',
                    unsafe_allow_html=True
                )

        with tab_map:
            mid_lat = (pu_lat + do_lat) / 2
            mid_lon = (pu_lon + do_lon) / 2

            m = folium.Map(
                location=[mid_lat, mid_lon],
                zoom_start=12,
                tiles='CartoDB dark_matter'
            )

            # Pickup marker
            folium.Marker(
                location=[pu_lat, pu_lon],
                popup=folium.Popup(
                    f"<b>Pickup</b><br>Zone {pu_zone}<br>({pu_lat:.4f}, {pu_lon:.4f})",
                    max_width=200
                ),
                tooltip="📍 Pickup",
                icon=folium.Icon(color='green', icon='record', prefix='fa')
            ).add_to(m)

            # Dropoff marker
            folium.Marker(
                location=[do_lat, do_lon],
                popup=folium.Popup(
                    f"<b>Dropoff</b><br>Zone {do_zone}<br>Fare: ${pred_fare:.2f}",
                    max_width=200
                ),
                tooltip=f"🏁 Dropoff — ${pred_fare:.2f}",
                icon=folium.Icon(color='red', icon='record', prefix='fa')
            ).add_to(m)

            # Route line
            folium.PolyLine(
                locations=[[pu_lat, pu_lon], [do_lat, do_lon]],
                color='#3B82F6',
                weight=4,
                opacity=0.85,
                tooltip=f"{dist_km:.1f} km — ${pred_fare:.2f} estimated fair fare"
            ).add_to(m)

            # Pickup glow circle
            folium.CircleMarker(
                location=[pu_lat, pu_lon],
                radius=14, color='#22C55E',
                fill=True, fill_opacity=0.2, weight=2
            ).add_to(m)

            # Dropoff glow circle
            folium.CircleMarker(
                location=[do_lat, do_lon],
                radius=14, color='#EF4444',
                fill=True, fill_opacity=0.2, weight=2
            ).add_to(m)

            # Fare legend overlay
            fare_color = '#22C55E' if f_label == 'Fair' else '#EF4444' if f_label == 'Overpriced' else '#3B82F6'
            legend_html = (
                '<div style="'
                'position:fixed;bottom:20px;left:20px;z-index:1000;'
                'background:rgba(13,17,23,0.92);'
                'border:1px solid #1C2130;'
                f'border-left:3px solid {fare_color};'
                'border-radius:8px;padding:12px 16px;'
                'font-family:monospace;font-size:13px;color:#C8CDD8;">'
                f'<div style="color:{fare_color};font-weight:700;font-size:16px">${pred_fare:.2f}</div>'
                '<div style="color:#6B7280;font-size:11px;margin-top:2px">Estimated fair fare</div>'
                f'<div style="margin-top:6px;color:{fare_color}">{f_label}</div>'
                f'<div style="color:#6B7280;font-size:11px">{dist_km:.1f} km &nbsp;·&nbsp; Score: {f_score}/100</div>'
                '</div>'
            )
            m.get_root().html.add_child(folium.Element(legend_html))

            st_folium(m, use_container_width=True, height=430, returned_objects=[])

    else:
        # Empty state
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🚖</div>
            <div class="empty-title">Enter trip details to analyze</div>
            <div class="empty-sub">GNN-powered fare estimation with fairness scoring and counterfactual suggestions for NYC Uber rides</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL ANALYTICS  (always visible)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="analytics-header">Global Fairness Analytics</div>
<div class="analytics-sub">System-wide statistics from training data — 
GraphSAGE + GAT · 100 zones · Huber loss · StandardScaler labels</div>
""", unsafe_allow_html=True)

ac1, ac2, ac3 = st.columns(3)
fair_df   = art['fair_df']
zone_stats = art['zone_stats']
zone_df    = art['zone_df']

with ac1:
    fig1 = plot_fairness_distribution(fair_df)
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

with ac2:
    fig2 = plot_deviation_histogram(fair_df)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

with ac3:
    fig3 = plot_zone_fairness_map(zone_stats, zone_df)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 8px 0;
            font-family:'DM Mono',monospace;font-size:0.68rem;
            color:#1F2937;letter-spacing:0.5px">
    RIDEWISE · GraphSAGE + GAT · PyTorch Geometric · Uber Fares Dataset<br>
    MAE $2.02 · RMSE $3.12 · R² 0.889 · 100 Zones · AdamW + Cosine LR
</div>
""", unsafe_allow_html=True)

