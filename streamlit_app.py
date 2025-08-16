# app.py — Telegram Strategy Simulator (Solana, Simulated, Fully Dynamic)
# Scope-aware metrics: All vs Selected token; Auto threshold tuning by PnL on validation

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Telegram Strategy Simulator", layout="wide")
st.title("Telegram Strategy Simulator")

# -----------------------------
# Helpers
# -----------------------------
def backtest_tp_sl_df(df, TP=0.02, SL=0.01, H=6, fee=0.001):
    """Backtest long-only entries on 'signal' with take-profit, stop-loss, timed exit."""
    trades = []
    if df.empty:
        return pd.DataFrame(columns=["token","entry","exit","entry_p","exit_p","net","reason","equity"]), 0.0
    for tok, g in df.groupby("token"):
        g = g.sort_values("t").reset_index(drop=True)
        price = g["price"].values
        sig   = g["signal"].values
        tt    = g["t"].values
        i=0
        while i < len(g)-1:
            if sig[i]==1:
                entry_i=i+1
                if entry_i>=len(g): break
                entry=price[entry_i]
                exit_i=min(entry_i+H, len(g)-1)
                exit_p=price[exit_i]; reason="timed"
                for j in range(entry_i+1, exit_i+1):
                    r=(price[j]-entry)/max(entry,1e-9)
                    if r>=TP: exit_i=j; exit_p=price[j]; reason="tp"; break
                    if r<=-SL: exit_i=j; exit_p=price[j]; reason="sl"; break
                net=((exit_p-entry)/max(entry,1e-9))-fee
                trades.append((tok, tt[entry_i], tt[exit_i], entry, exit_p, net, reason))
                i=exit_i+1
            else:
                i+=1
    if not trades:
        return pd.DataFrame(columns=["token","entry","exit","entry_p","exit_p","net","reason","equity"]), 0.0
    tdf = pd.DataFrame(trades, columns=["token","entry","exit","entry_p","exit_p","net","reason"])
    tdf = tdf.sort_values("entry").reset_index(drop=True)
    tdf["equity"] = (1.0 + tdf["net"]).cumprod()
    total_return = float(tdf["equity"].iloc[-1]-1.0)
    return tdf, total_return

def tune_threshold_by_pnl(valid_df, valid_probs, tp=0.02, sl=0.01, H=6, fee=0.001):
    """Pick threshold that maximizes validation PnL under backtest rules."""
    if valid_df.empty:
        return 0.5, 0.0
    grid = np.linspace(0.05, 0.95, 37)  # step=0.025
    best_thr, best_ret = 0.5, -1e9
    tmp = valid_df.copy()
    tmp["p"] = valid_probs
    for thr in grid:
        tmp["signal"] = ((tmp["p"]>=thr) & (tmp["dv_z"]>0) & (tmp["tx_z"]>0) & (tmp["whale_z"]>0.5)).astype(int)
        _, ret = backtest_tp_sl_df(tmp, TP=tp, SL=sl, H=H, fee=fee)
        if ret > best_ret:
            best_thr, best_ret = float(thr), float(ret)
    return best_thr, best_ret

def max_drawdown(equity_series: pd.Series) -> float:
    if equity_series is None or equity_series.empty:
        return 0.0
    roll_max = equity_series.cummax()
    drawdown = equity_series/roll_max - 1.0
    return float(drawdown.min())

def hold_return_over_scope(scope_df: pd.DataFrame) -> float:
    """Equal-weight buy&hold across tokens for the rows present in scope_df."""
    if scope_df.empty:
        return 0.0
    rets = []
    for tok, g in scope_df.groupby("token"):
        g = g.sort_values("t")
        if len(g) < 2: 
            continue
        rets.append(g["price"].iloc[-1]/g["price"].iloc[0] - 1.0)
    return float(np.mean(rets)) if rets else 0.0

def top_trigger_label(signal_df: pd.DataFrame) -> str:
    if signal_df.empty:
        return "N/A"
    cats = []
    for _, r in signal_df.iterrows():
        zvals = {"VOLUME_SURGE": r["dv_z"], "TX_BURST": r["tx_z"], "WHALE_SPIKE": r["whale_z"]}
        cats.append(max(zvals, key=zvals.get))
    vc = pd.Series(cats).value_counts()
    return f"{vc.index[0]} ({vc.iloc[0]} trades)"

# -----------------------------
# 1) Simulate hourly data with learnable 'alpha'
# -----------------------------
np.random.seed(42)
N_TOKENS, HOURS = 30, 24*30  # 30 days hourly
tokens = [f"TOK{str(i).zfill(2)}" for i in range(N_TOKENS)]
mu = np.random.uniform(0.0002, 0.0010, size=N_TOKENS)
sigma = np.random.uniform(0.01, 0.04, size=N_TOKENS)
base_liq = np.random.uniform(2e5, 2e6, size=N_TOKENS)
base_vol = np.random.uniform(5e4, 1e6, size=N_TOKENS)
base_tx  = np.random.uniform(50, 2000, size=N_TOKENS)
base_wallet_growth = np.random.uniform(1, 25, size=N_TOKENS)

rows=[]
for tid, tok in enumerate(tokens):
    p0 = np.random.uniform(0.5, 10.0)
    prices=[p0]; vol=[]; tx=[]; uw=[]; whale=[]; liq=[]
    uw_series = 10000 + np.cumsum(np.random.poisson(base_wallet_growth[tid]/24, HOURS))
    alpha = 0.0
    for t in range(HOURS):
        spike = np.random.rand() < 0.035
        whale.append(np.random.gamma(2.0,5000)*(3.0 if spike else 1.0)+np.random.rand()*1000)
        tx.append(np.random.poisson(base_tx[tid]/24)+(30 if spike else 0)+np.random.randint(0,10))
        v = max(0, np.random.normal(base_vol[tid]/24, base_vol[tid]/100))*(2.0 if spike else 1.0)
        vol.append(v)
        li = base_liq[tid]*(1.0+0.01*np.sin(2*np.pi*t/24/7)) + np.random.normal(0, base_liq[tid]/300)
        liq.append(max(li, 1e4))
        uw.append(uw_series[t])
        if spike:
            alpha += np.random.uniform(0.003, 0.012)
        eps = np.random.normal(0,1)
        nextp = prices[-1]*np.exp(mu[tid] + sigma[tid]*eps + alpha)
        prices.append(max(0.01, nextp))
        alpha *= 0.6
    prices=prices[:-1]
    rows.append(pd.DataFrame({
        "token": tok,
        "t": pd.date_range("2025-05-01", periods=HOURS, freq="H"),
        "price": prices, "dex_volume": vol, "tx_count": tx,
        "unique_wallets": uw, "whale_inflow": whale, "liquidity_usd": liq
    }))

df = pd.concat(rows).sort_values(["token","t"]).reset_index(drop=True)

# -----------------------------
# 2) Feature engineering
# -----------------------------
feat = df.copy()
g = feat.groupby("token", sort=False)

feat["ret_1h"]   = g["price"].pct_change(1,  fill_method=None)
feat["ret_6h"]   = g["price"].pct_change(6,  fill_method=None)
feat["ret_24h"]  = g["price"].pct_change(24, fill_method=None)

feat["vol_24h"]  = g["ret_1h"].rolling(24).std().reset_index(level=0, drop=True)
feat["ema_6"]     = g["price"].transform(lambda s: s.ewm(span=6,  adjust=False).mean())
feat["ema_24"]    = g["price"].transform(lambda s: s.ewm(span=24, adjust=False).mean())
feat["ema_ratio"] = feat["ema_6"]/(feat["ema_24"]+1e-9)

def zscore_rolling(s):
    m = s.rolling(24).mean()
    sd = s.rolling(24).std()
    return (s - m) / (sd + 1e-9)

feat["dv_z"]     = g["dex_volume"].transform(zscore_rolling)
feat["tx_z"]     = g["tx_count"].transform(zscore_rolling)
feat["whale_z"]  = g["whale_inflow"].transform(zscore_rolling)
feat["liq_chg_24h"] = g["liquidity_usd"].pct_change(24, fill_method=None)

feat["fwd_ret_1h"]  = g["price"].pct_change(-1, fill_method=None) * -1
feat["label_up1h"]  = (feat["fwd_ret_1h"] > 0.004).astype(int)

feat = feat.dropna().sort_values(["t","token"]).reset_index(drop=True)

# -----------------------------
# 3) Split
# -----------------------------
times = feat["t"].sort_values().unique()
t_train_end = times[int(0.70*len(times))]
t_valid_end = times[int(0.85*len(times))]

train = feat[feat["t"]<=t_train_end].copy()
valid = feat[(feat["t"]>t_train_end)&(feat["t"]<=t_valid_end)].copy()
test  = feat[feat["t"]>t_valid_end].copy()

cols  = ["ret_1h","ret_6h","ret_24h","vol_24h","ema_ratio","dv_z","tx_z","whale_z","liq_chg_24h"]

# -----------------------------
# 4) Model
# -----------------------------
clf = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=42
)
clf.fit(train[cols], train["label_up1h"])

# Store probabilities for later scoping
valid_probs_all = clf.predict_proba(valid[cols])[:,1]
test_probs_all  = clf.predict_proba(test[cols])[:,1]

# -----------------------------
# 5) UI Controls (make everything dynamic)
# -----------------------------
top_row = st.columns([1,1,1,1,2])
with top_row[0]:
    scope = st.radio("Scope", ["All tokens", "Selected token"], index=0)
with top_row[1]:
    token_sel = st.selectbox("Token", sorted(df["token"].unique()))
with top_row[2]:
    TP_ui = st.slider("TP (%)", 0.5, 10.0, 2.0, 0.5)/100.0
with top_row[3]:
    SL_ui = st.slider("SL (%)", 0.5, 10.0, 1.0, 0.5)/100.0
with top_row[4]:
    H_ui = st.slider("Timed Exit (hours)", 1, 24, 6, 1)

mode_row = st.columns([2,2,2])
with mode_row[0]:
    thr_mode = st.radio("Threshold Mode", ["Auto (tune by PnL on validation)", "Manual"], index=0)
with mode_row[1]:
    thr_manual = st.slider("Manual Threshold", 0.05, 0.95, 0.50, 0.01)

# Scope the data and probabilities
if scope == "Selected token":
    valid_scope = valid[valid["token"]==token_sel].copy()
    test_scope  = test[test["token"]==token_sel].copy()
    p_valid = valid_probs_all[valid.index.get_indexer(valid_scope.index)]
    p_test  = test_probs_all[test.index.get_indexer(test_scope.index)]
else:
    valid_scope = valid.copy()
    test_scope  = test.copy()
    p_valid = valid_probs_all
    p_test  = test_probs_all

# Safety: empty scopes
if valid_scope.empty or test_scope.empty:
    st.warning("Not enough rows for the chosen scope to compute metrics. Try another token or switch to All tokens.")
    st.stop()

# Threshold selection (Auto tunes on scoped validation)
if thr_mode.startswith("Auto"):
    thr_star, valid_ret = tune_threshold_by_pnl(valid_scope.copy(), p_valid, tp=TP_ui, sl=SL_ui, H=H_ui, fee=0.001)
else:
    thr_star = float(thr_manual)
    # compute valid_ret for display with the manual threshold too
    tmpv = valid_scope.copy()
    tmpv["p"] = p_valid
    tmpv["signal"] = ((tmpv["p"]>=thr_star) & (tmpv["dv_z"]>0) & (tmpv["tx_z"]>0) & (tmpv["whale_z"]>0.5)).astype(int)
    _, valid_ret = backtest_tp_sl_df(tmpv, TP=TP_ui, SL=SL_ui, H=H_ui, fee=0.001)

# Compute validation metrics for the scope
valid_auc = roc_auc_score(valid_scope["label_up1h"], p_valid) if valid_scope["label_up1h"].nunique()>1 else np.nan
valid_ap  = average_precision_score(valid_scope["label_up1h"], p_valid)

# Build test signals for the scope with chosen thr*
tmp_test = test_scope.copy()
tmp_test["p"] = p_test
tmp_test["signal"] = ((tmp_test["p"]>=thr_star) & (tmp_test["dv_z"]>0) & (tmp_test["tx_z"]>0) & (tmp_test["whale_z"]>0.5)).astype(int)

# Label “trigger” for signal rows
sig_rows = tmp_test[tmp_test["signal"]==1].copy()
if not sig_rows.empty:
    trig = []
    for _, r in sig_rows.iterrows():
        zvals = {"VOLUME_SURGE": r["dv_z"], "TX_BURST": r["tx_z"], "WHALE_SPIKE": r["whale_z"]}
        trig.append(max(zvals, key=zvals.get))
    sig_rows["trigger"] = trig

# Backtest on scope
trades_ui, ret_ui = backtest_tp_sl_df(tmp_test, TP=TP_ui, SL=SL_ui, H=H_ui, fee=0.001)
wr_ui  = float((trades_ui["net"]>0).mean()) if not trades_ui.empty else float("nan")
mdd_ui = max_drawdown(trades_ui["equity"]) if not trades_ui.empty else 0.0
hold_ui = hold_return_over_scope(tmp_test)
top_trigger_ui = top_trigger_label(sig_rows) if not sig_rows.empty else "N/A"

# -----------------------------
# 6) Layout
# -----------------------------
tab1, tab2 = st.tabs(["🤖 Summary & Backtest", "📈 Trend Detection"])

with tab1:
    # TG-style scoped summary
    scope_name = "ALL" if scope=="All tokens" else token_sel
    st.markdown(
        f"""
**✅ RESULTS ({scope_name}):**  
**📈 {ret_ui*100:+.1f}%** vs **HOLD ({hold_ui*100:+.1f}%)**  
**🏆 Win Rate:** {wr_ui*100:.1f}%  
**⚠️ Max Drawdown:** {mdd_ui*100:.1f}%  
**💡 Top trigger:** {top_trigger_ui}
""".strip()
    )

    # Scoped validation metrics (dynamic)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best threshold (PnL)", f"{thr_star:.2f}")
    m2.metric("Valid ROC AUC", f"{valid_auc:.3f}" if not np.isnan(valid_auc) else "—")
    m3.metric("Valid PR AUC",  f"{valid_ap:.3f}")
    m4.metric("Valid PnL @ thr*", f"{valid_ret:.2%}")

    # Price + signals (for selected token only, chart-wise)
    c1, c2 = st.columns([1,3])
    with c1:
        st.caption("Signals require dv_z>0, tx_z>0, whale_z>0.5")
    with c2:
        tok_plot = token_sel if scope=="Selected token" else st.selectbox("Chart token", sorted(df["token"].unique()), index=0, key="chart_tok")
        dftok = feat[feat["token"]==tok_plot].copy()
        fig = px.line(dftok, x="t", y="price", title=f"{tok_plot} — Price")
        just = tmp_test[tmp_test["token"]==tok_plot] if scope=="All tokens" else tmp_test
        if not just.empty:
            ss = just[just["signal"]==1]
            if not ss.empty:
                fig.add_scatter(x=ss["t"], y=ss["price"], mode="markers", name="signal",
                                marker=dict(size=7, symbol="triangle-up"))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Backtest (TP={TP_ui:.1%}, SL={SL_ui:.1%}, H={H_ui}h, thr={thr_star:.2f}) — Return: {ret_ui:.2%}")
    if not trades_ui.empty:
        st.plotly_chart(px.line(trades_ui, x="exit", y="equity", title="Equity Curve"), use_container_width=True)
        st.dataframe(trades_ui.tail(50), use_container_width=True, height=320)
    else:
        st.info("No trades generated with the current settings and scope.")

# Trend detection (unchanged; always uses full data — you can scope it if you prefer)
daily = df.copy()
daily["date"]=daily["t"].dt.floor("D")
daily = daily.groupby(["token","date"], as_index=False).agg(
    tx_count=("tx_count","sum"),
    unique_wallets=("unique_wallets","last"),
    liquidity_usd=("liquidity_usd","mean")
)
trend=[]
for tok, g2 in daily.groupby("token"):
    g2=g2.sort_values("date")
    recent=g2.tail(14)
    if len(recent)<2: continue
    tx_tr=(recent["tx_count"].iloc[-1]-recent["tx_count"].iloc[0])/(recent["tx_count"].iloc[0]+1e-9)
    w_tr =(recent["unique_wallets"].iloc[-1]-recent["unique_wallets"].iloc[0])/(recent["unique_wallets"].iloc[0]+1e-9)
    li_tr=(recent["liquidity_usd"].iloc[-1]-recent["liquidity_usd"].iloc[0])/(recent["liquidity_usd"].iloc[0]+1e-9)
    trend.append([tok,tx_tr,w_tr,li_tr])

clusters=pd.DataFrame(trend, columns=["token","tx_trend","wallets_trend","liq_trend"])
if not clusters.empty:
    X=clusters[["tx_trend","wallets_trend","liq_trend"]].values
    Xs=StandardScaler().fit_transform(X)
    km=KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs)
    clusters["cluster"]=km.labels_
    labels={}
    for c in sorted(clusters["cluster"].unique()):
        score = clusters.loc[clusters["cluster"]==c, ["tx_trend","wallets_trend"]].mean().sum()
        labels[c] = "Emerging" if score>0.05 else ("Declining" if score<-0.05 else "Stable")
    clusters["cluster_label"]=clusters["cluster"].map(labels)

tab2 = st.tabs(["📈 Trend Detection"])[0]
with tab2:
    if not clusters.empty:
        st.subheader("14-day Activity Trend Clusters")
        st.plotly_chart(px.scatter(clusters, x="tx_trend", y="wallets_trend",
                                   color="cluster_label", hover_data=["token","liq_trend"],
                                   title="Emerging / Stable / Declining"),
                        use_container_width=True)
        tok2 = st.selectbox("Inspect token", sorted(df["token"].unique()), index=0, key="tok2")
        d2 = df[df["token"]==tok2]
        c1,c2 = st.columns(2)
        c1.plotly_chart(px.line(d2, x="t", y="tx_count", title=f"{tok2} — TX Count"), use_container_width=True)
        c1.plotly_chart(px.line(d2, x="t", y="dex_volume", title=f"{tok2} — DEX Volume"), use_container_width=True)
        c2.plotly_chart(px.line(d2, x="t", y="unique_wallets", title=f"{tok2} — Unique Wallets"), use_container_width=True)
        c2.plotly_chart(px.line(d2, x="t", y="liquidity_usd", title=f"{tok2} — Liquidity"), use_container_width=True)
    else:
        st.info("Not enough daily data to build trend clusters yet.")

st.caption("Everything is scope-aware now: Validation metrics, thr* tuning, and Test backtest all reflect the selected scope & parameters.")

