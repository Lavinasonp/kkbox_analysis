import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="KKBox Churn Analytics", page_icon="",
                   layout="wide", initial_sidebar_state="expanded")

CYAN, RED, GOLD, GREEN, PURPLE = "#09CEF6", "#FF4B4B", "#FFD700", "#00D68F", "#A855F7"
CHURN_MAP = {0: "Retained", 1: "Churned"}

st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap");
html,body,[class*="css"]{font-family:"Inter",sans-serif;}
.kpi{border-left:4px solid #09CEF6;background:rgba(9,206,246,.07);border-radius:10px;
     padding:16px 20px;margin-bottom:12px;}
.kpi.r{border-left-color:#FF4B4B;background:rgba(255,75,75,.07);}
.kpi.g{border-left-color:#00D68F;background:rgba(0,214,143,.07);}
.kpi.y{border-left-color:#FFD700;background:rgba(255,215,0,.07);}
.kt{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;opacity:.6;margin-bottom:4px;}
.kv{font-size:26px;font-weight:800;color:#09CEF6;}
.kpi.r .kv{color:#FF4B4B;} .kpi.g .kv{color:#00D68F;} .kpi.y .kv{color:#FFD700;}
.ks{font-size:11px;opacity:.45;margin-top:2px;}
.box{border-left:4px solid #A855F7;background:rgba(168,85,247,.06);
     border-radius:8px;padding:12px 16px;margin:10px 0;font-size:13px;line-height:1.6;}
.box.r{border-left-color:#FF4B4B;background:rgba(255,75,75,.06);}
.box.g{border-left-color:#00D68F;background:rgba(0,214,143,.06);}
.stTabs [data-baseweb="tab"]{padding:8px 16px;font-size:13px;}
.stTabs [aria-selected="true"]{border-bottom:3px solid #09CEF6!important;
  color:#09CEF6!important;font-weight:700!important;}
</style>""", unsafe_allow_html=True)

# ── DATA LOAD ─────────────────────────────────────────
ADIR = "data/analytics"

@st.cache_data(ttl=3600)
def load():
    d = {}
    def r(k, f, t="csv"):
        p = os.path.join(ADIR, f)
        if os.path.exists(p):
            try:
                d[k] = pd.read_csv(p) if t == "csv" else pd.read_parquet(p)
            except Exception:
                pass
    r("g",        "global_kpis.csv")
    r("demo",     "demographics_churn.parquet", "parquet")
    r("trans",    "user_transaction_kpis.parquet", "parquet")
    r("eng",      "user_engagement.parquet", "parquet")
    # pre-aggregated tiny CSVs (used instead of parquets on deploy)
    r("donut",    "churn_donut.csv")
    r("obars",    "overview_bars.csv")
    r("age_c",    "age_churn.csv")
    r("gen_c",    "gender_churn.csv")
    r("city_c",   "city_churn.csv")
    r("reg_c",    "reg_method_churn.csv")
    r("coh",      "cohort_year_churn.csv")
    r("seas",     "seasonal_reg_churn.csv")
    r("age_gen",  "age_gender_churn.csv")
    r("age_dist", "age_distribution.parquet", "parquet")
    r("mon_rev",  "monthly_revenue.csv")
    r("plan_rev", "plan_type_revenue.csv")
    r("pay_c",    "payment_method_churn.csv")
    r("auto_r",   "auto_renew_churn.csv")
    r("promo_s",  "promo_segment_churn.csv")
    r("cancel_s", "cancel_segment_churn.csv")
    r("rev_b",    "revenue_bucket_churn.csv")
    r("skip_s",   "skip_rate_segment_churn.csv")
    r("compl_s",  "completion_ratio_churn.csv")
    r("active_s", "active_days_segment_churn.csv")
    r("listen_s", "listening_segment_churn.csv")
    r("song_bk",  "song_completion_breakdown.csv")
    r("cf",       "churn_feature_summary.csv")
    r("rfm",      "rfm_segment_churn.csv")
    r("funnel",   "churn_funnel_metrics.csv")
    r("age_reg",  "age_reg_churn_heatmap.csv")
    return d

data = load()
if not data:
    st.error("Run `python src/generate_full_analytics.py` first.")
    st.stop()

g    = data.get("g",     pd.DataFrame())
demo = data.get("demo",  pd.DataFrame())
trx  = data.get("trans", pd.DataFrame())
eng  = data.get("eng",   pd.DataFrame())

# ── HELPERS ───────────────────────────────────────────
def fmt(n):
    if n >= 1e6: return f"{n/1e6:.1f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return f"{n:.0f}"

def L(fig, title="", h=370):
    """Apply consistent dark transparent chart layout."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#ddd"), x=0.01),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#bbb"), height=h,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(bgcolor="rgba(0,0,0,.2)", borderwidth=0))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.05)", showline=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.05)", showline=False, zeroline=False)
    return fig

def kcard(label, val, sub="", cls=""):
    s = f'<div class="ks">{sub}</div>' if sub else ""
    return f'<div class="kpi {cls}"><div class="kt">{label}</div><div class="kv">{val}</div>{s}</div>'

def note(txt, cls=""):
    return f'<div class="box {cls}">{txt}</div>'

# ── SIDEBAR ───────────────────────────────────────────
st.sidebar.markdown(
    f'<h2 style="color:{CYAN};font-weight:800;margin:0;">KK<span style="color:{RED};">BOX</span></h2>',
    unsafe_allow_html=True)
st.sidebar.caption("Churn Analytics Dashboard  ·  2015–2017")
st.sidebar.divider()
st.sidebar.markdown("**Filters**")
st.sidebar.caption("Select one or more filters to narrow the analysis across all tabs.")

if not demo.empty:
    age_opts  = ["All"] + sorted([str(x) for x in demo["age_group"].dropna().unique()])
    gen_opts  = ["All"] + sorted(demo["gender"].dropna().unique().tolist())
    city_opts = ["All"] + sorted(demo["city_label"].dropna().unique().tolist())
    reg_opts  = ["All"] + sorted(demo["reg_method_label"].dropna().unique().tolist())
    yr_opts   = ["All"] + [str(y) for y in sorted(demo["reg_year"].dropna().astype(int).unique().tolist())]
else:
    age_opts = gen_opts = city_opts = reg_opts = yr_opts = ["All"]

sel_age  = st.sidebar.selectbox("Age Group",             age_opts)
sel_gen  = st.sidebar.selectbox("Gender",                gen_opts)
sel_city = st.sidebar.selectbox("City",                  city_opts)
sel_reg  = st.sidebar.selectbox("Registration Channel",  reg_opts)
sel_yr   = st.sidebar.selectbox("Registration Year",     yr_opts)

# Show active filter count
active = sum(1 for v in [sel_age, sel_gen, sel_city, sel_reg, sel_yr] if v != "All")
if active:
    st.sidebar.markdown(
        f'<div style="background:rgba(9,206,246,.1);border-left:3px solid {CYAN};'
        f'border-radius:6px;padding:8px 12px;margin-top:8px;font-size:12px;">'
        f'<b>{active} filter{"s" if active>1 else ""} active</b> — '
        f'charts reflect filtered subset</div>',
        unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.markdown("**About**")
st.sidebar.caption(
    "Data source: KKBox WSDM 2018 churn prediction challenge. "
    "Observation window: Feb–Mar 2017. "
    "Filters apply to the demographics parquet; if parquet is not present locally, "
    "aggregated CSVs are used and filters have no effect.")

def filt(df):
    d = df.copy()
    if sel_age  != "All" and "age_group"       in d.columns: d = d[d["age_group"].astype(str) == sel_age]
    if sel_gen  != "All" and "gender"          in d.columns: d = d[d["gender"]          == sel_gen]
    if sel_city != "All" and "city_label"      in d.columns: d = d[d["city_label"]      == sel_city]
    if sel_reg  != "All" and "reg_method_label" in d.columns: d = d[d["reg_method_label"] == sel_reg]
    if sel_yr   != "All" and "reg_year"        in d.columns: d = d[d["reg_year"].astype(str) == sel_yr]
    return d

fd   = filt(demo)
mset = set(fd["msno"]) if not fd.empty and "msno" in fd.columns else None
ft   = trx[trx["msno"].isin(mset)] if mset and "msno" in trx.columns else trx.copy()
fe   = eng[eng["msno"].isin(mset)] if mset and "msno" in eng.columns else eng.copy()

# ── HEADER ────────────────────────────────────────────
st.markdown(f"""
<h1 style="font-size:1.9rem;font-weight:800;margin:0;
background:linear-gradient(90deg,{CYAN},{PURPLE});
-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
KKBox Subscriber Churn Analytics
</h1>
<p style="margin:4px 0 16px;opacity:.5;font-size:13px;">
970K+ subscribers  -  Taiwan  -  2015-2017
</p>
""", unsafe_allow_html=True)

# ── KPI CARDS ─────────────────────────────────────────
tu  = int(g["total_users"].iloc[0])      if not g.empty else 0
chu = int(g["churned_users"].iloc[0])    if not g.empty else 0
cr  = float(g["churn_rate"].iloc[0])*100 if not g.empty else 0
rev      = ft["total_revenue"].sum()      if not ft.empty and "total_revenue"      in ft.columns else 0
rev_lost = ft["total_revenue_lost"].sum() if not ft.empty and "total_revenue_lost" in ft.columns else 0
avg_rev  = ft["total_revenue"].mean()     if not ft.empty and "total_revenue"      in ft.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.markdown(kcard("Total Subscribers", fmt(tu),        "users analysed"),              unsafe_allow_html=True)
with c2: st.markdown(kcard("Churned",           fmt(chu),       f"{cr:.1f}% churn rate", "r"),  unsafe_allow_html=True)
with c3: st.markdown(kcard("Retained",          fmt(tu-chu),    f"{100-cr:.1f}% retention","g"),unsafe_allow_html=True)
with c4: st.markdown(kcard("Total Revenue",     f"${fmt(rev)}", "from subscriptions"),           unsafe_allow_html=True)
with c5: st.markdown(kcard("Promo Leakage",     f"${fmt(rev_lost)}","given away via promos","r"),unsafe_allow_html=True)

st.divider()

# ── TABS ──────────────────────────────────────────────
T = st.tabs(["Overview", "Demographics", "Revenue & Plans", "Engagement", "Insights & Actions"])


# ════════════════════════════════════════════════════
# TAB 0 — OVERVIEW
# ════════════════════════════════════════════════════
with T[0]:
    st.markdown("#### Overview")
    c1, c2 = st.columns(2)

    # Churn donut — use pre-aggregated CSV (tiny) or fall back to demographics parquet
    with c1:
        donut_df = data.get("donut")
        if donut_df is not None and not donut_df.empty:
            cc = donut_df.copy()
            cc.columns = ["Status", "Count"]
        elif not fd.empty and "is_churn" in fd.columns:
            cc = fd["is_churn"].value_counts().reset_index()
            cc.columns = ["Status", "Count"]
        else:
            cc = None
        if cc is not None:
            cc["Label"] = cc["Status"].map({0: "Retained", 1: "Churned"})
            fig = px.pie(cc, values="Count", names="Label", hole=.55,
                         color="Label", color_discrete_map={"Retained": CYAN, "Churned": RED})
            fig.update_traces(textinfo="percent+label", pull=[0, .04],
                              marker=dict(line=dict(color="rgba(0,0,0,.3)", width=2)))
            fig.add_annotation(text=f"<b>{cr:.1f}%</b><br>Churn",
                               x=.5, y=.5, font_size=15, showarrow=False, font_color=RED)
            L(fig, "Churn Distribution")
            st.plotly_chart(fig, use_container_width=True)

    # Radar profile
    with c2:
        cf = data.get("cf")
        if cf is not None and len(cf) >= 2:
            rr  = cf[cf["is_churn"] == 0].iloc[0]
            cr2 = cf[cf["is_churn"] == 1].iloc[0]
            cats = ["Active Days", "Revenue", "Listen Hrs", "Completion %", "Auto-Renew %"]
            pairs = [
                (rr.get("avg_active_days", 0),    cr2.get("avg_active_days", 0)),
                (rr.get("avg_revenue", 0),         cr2.get("avg_revenue", 0)),
                (rr.get("avg_secs", 0) / 3600,     cr2.get("avg_secs", 0) / 3600),
                (rr.get("avg_completion", 0) * 100, cr2.get("avg_completion", 0) * 100),
                (rr.get("avg_auto_renew", 0) * 100, cr2.get("avg_auto_renew", 0) * 100),
            ]
            def norm(a, b):
                mx = max(abs(a), abs(b), 1e-9)
                return a / mx * 10, b / mx * 10
            rv, cv = [], []
            for a, b in pairs:
                x, y = norm(a, b); rv.append(x); cv.append(y)
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=rv + [rv[0]], theta=cats + [cats[0]],
                fill="toself", name="Retained", line_color=CYAN, fillcolor="rgba(9,206,246,.15)"))
            fig.add_trace(go.Scatterpolar(r=cv + [cv[0]], theta=cats + [cats[0]],
                fill="toself", name="Churned", line_color=RED, fillcolor="rgba(255,75,75,.15)"))
            fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 10], gridcolor="rgba(255,255,255,.1)")))
            L(fig, "Retained vs Churned — Normalised Behavioural Profile")
            st.plotly_chart(fig, use_container_width=True)

    # 3 quick metric bars — use pre-aggregated overview_bars.csv (tiny) or fall back to parquets
    ob = data.get("obars")
    c3, c4, c5 = st.columns(3)
    with c3:
        if ob is not None and not ob.empty and "avg_total_secs" in ob.columns:
            ec = ob[["is_churn","avg_total_secs"]].copy()
            ec["label"] = ec["is_churn"].map(CHURN_MAP)
            ec["hours"] = ec["avg_total_secs"] / 3600
        elif not fe.empty and "total_secs" in fe.columns:
            ec = fe.groupby("is_churn")["total_secs"].mean().reset_index()
            ec["label"] = ec["is_churn"].map(CHURN_MAP)
            ec["hours"] = ec["total_secs"] / 3600
        else:
            ec = None
        if ec is not None:
            fig = px.bar(ec, x="label", y="hours", color="label",
                         color_discrete_map={"Retained": CYAN, "Churned": RED},
                         text=ec["hours"].apply(lambda h: f"{h:.0f}h"))
            fig.update_traces(textposition="outside")
            L(fig, "Avg Listening Hours", 300)
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        if ob is not None and not ob.empty and "avg_revenue" in ob.columns:
            rc = ob[["is_churn","avg_revenue"]].copy()
            rc["label"] = rc["is_churn"].map(CHURN_MAP)
            rc.rename(columns={"avg_revenue": "total_revenue"}, inplace=True)
        elif not ft.empty and "total_revenue" in ft.columns:
            rc = ft.groupby("is_churn")["total_revenue"].mean().reset_index()
            rc["label"] = rc["is_churn"].map(CHURN_MAP)
        else:
            rc = None
        if rc is not None:
            fig = px.bar(rc, x="label", y="total_revenue", color="label",
                         color_discrete_map={"Retained": CYAN, "Churned": RED},
                         text=rc["total_revenue"].apply(lambda v: f"${v:.0f}"))
            fig.update_traces(textposition="outside")
            L(fig, "Avg Revenue / User ($)", 300)
            st.plotly_chart(fig, use_container_width=True)

    with c5:
        if ob is not None and not ob.empty and "avg_auto_renew" in ob.columns:
            ar = ob[["is_churn","avg_auto_renew"]].copy()
            ar["label"] = ar["is_churn"].map(CHURN_MAP)
            ar["pct"]   = ar["avg_auto_renew"] * 100
        elif not ft.empty and "auto_renew_rate" in ft.columns:
            ar = ft.groupby("is_churn")["auto_renew_rate"].mean().reset_index()
            ar["label"] = ar["is_churn"].map(CHURN_MAP)
            ar["pct"]   = ar["auto_renew_rate"] * 100
        else:
            ar = None
        if ar is not None:
            fig = px.bar(ar, x="label", y="pct", color="label",
                         color_discrete_map={"Retained": CYAN, "Churned": RED},
                         text=ar["pct"].apply(lambda v: f"{v:.0f}%"))
            fig.update_traces(textposition="outside")
            L(fig, "Auto-Renew Rate (%)", 300)
            st.plotly_chart(fig, use_container_width=True)

    # Summary table
    cf = data.get("cf")
    if cf is not None and not cf.empty:
        st.markdown("#### Retained vs Churned — All Key Metrics")
        d2 = cf.copy()
        d2["Group"]       = d2["is_churn"].map(CHURN_MAP)
        d2["Users"]       = d2["user_count"].apply(lambda v: f"{v:,}")
        d2["Avg Rev ($)"] = d2["avg_revenue"].apply(lambda v: f"${v:.0f}")
        d2["Active Days"] = d2["avg_active_days"].apply(lambda v: f"{v:.1f}")
        d2["Listen Hrs"]  = d2["avg_secs"].apply(lambda v: f"{v/3600:.0f}h")
        d2["Skip Rate"]   = d2["avg_skip"].apply(lambda v: f"{v*100:.1f}%")
        d2["Completion"]  = d2["avg_completion"].apply(lambda v: f"{v*100:.1f}%")
        d2["Auto-Renew"]  = d2["avg_auto_renew"].apply(lambda v: f"{v*100:.0f}%")
        st.dataframe(
            d2[["Group","Users","Avg Rev ($)","Active Days","Listen Hrs","Skip Rate","Completion","Auto-Renew"]],
            use_container_width=True, hide_index=True)

    st.markdown(note(
        "The overall churn rate sits at roughly 9%, which means roughly 1 in every 11 subscribers left "
        "within the observation window. The most striking difference between retained and churned users "
        "is not how much they listened, but whether they had auto-renew enabled. Retained users auto-renew "
        "at 92%, whereas churned users auto-renew at just 47%. That 45-point gap suggests that the act of "
        "setting up auto-renew is itself a signal of long-term intent. Revenue follows the same pattern — "
        "retained users spend nearly four times more on average, partly because they stay longer and partly "
        "because they are on more committed payment plans."
    ), unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# TAB 1 — DEMOGRAPHICS
# ════════════════════════════════════════════════════
with T[1]:
    st.markdown("#### Demographics")
    c1, c2 = st.columns(2)

    with c1:
        df = data.get("age_c")
        if df is not None and not df.empty:
            fig = px.bar(df, x="age_group", y="churn_rate", color="churn_rate",
                color_continuous_scale=[GREEN, GOLD, RED],
                text=df["churn_rate"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_coloraxes(showscale=False)
            L(fig, "Churn Rate by Age Group")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        df = data.get("gen_c")
        if df is not None and not df.empty:
            df = df[df["gender"].notna()]
            fig = px.bar(df, x="gender", y="churn_rate", color="gender",
                color_discrete_sequence=[CYAN, RED, GOLD],
                text=df["churn_rate"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            L(fig, "Churn Rate by Gender")
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        df = data.get("city_c")
        if df is not None and not df.empty:
            top = df.sort_values("total", ascending=False).head(12)
            fig = px.bar(top, x="churn_rate", y="city_label", orientation="h",
                color="churn_rate", color_continuous_scale=[GREEN, GOLD, RED],
                text=top["churn_rate"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_coloraxes(showscale=False)
            fig.update_layout(yaxis=dict(autorange="reversed"))
            L(fig, "Churn Rate by City (Top 12 by Volume)", 400)
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        df = data.get("reg_c")
        if df is not None and not df.empty:
            fig = px.bar(df, x="reg_method_label", y="churn_rate",
                color="churn_rate", color_continuous_scale=[GREEN, GOLD, RED],
                text=df["churn_rate"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_coloraxes(showscale=False)
            L(fig, "Churn Rate by Registration Channel", 400)
            st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        df = data.get("coh")
        if df is not None and not df.empty:
            df = df.dropna(subset=["reg_year"])
            df["reg_year"] = df["reg_year"].astype(int)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["reg_year"], y=df["churn_rate"], name="Churn Rate %",
                marker=dict(color=df["churn_rate"].tolist(),
                            colorscale=[[0, GREEN], [.5, GOLD], [1, RED]]),
                text=[f"{v:.1f}%" for v in df["churn_rate"]], textposition="outside"))
            fig.add_trace(go.Scatter(x=df["reg_year"], y=df["total"], name="Cohort Size",
                yaxis="y2", mode="lines+markers",
                line=dict(color=CYAN, width=2, dash="dot"), marker=dict(size=6)))
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Users"))
            L(fig, "Churn Rate by Registration Year Cohort", 400)
            st.plotly_chart(fig, use_container_width=True)

    with c6:
        # seasonal — uses 'total' (not 'new_users') and 'month_name'
        df = data.get("seas")
        if df is not None and not df.empty:
            mo = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            df["month_name"] = pd.Categorical(df["month_name"], categories=mo, ordered=True)
            df = df.sort_values("month_name")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["month_name"], y=df["total"],
                name="Registrations", marker_color=CYAN, opacity=.7))
            fig.add_trace(go.Scatter(x=df["month_name"], y=df["churn_rate"],
                name="Churn Rate %", mode="lines+markers", yaxis="y2",
                line=dict(color=RED, width=2.5), marker=dict(size=7, symbol="diamond")))
            fig.update_layout(yaxis=dict(title="Registrations"),
                              yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Churn %"))
            L(fig, "Seasonality — Registrations vs Churn Rate by Month", 400)
            st.plotly_chart(fig, use_container_width=True)

    # Age x Gender heatmap
    df = data.get("age_gen")
    if df is not None and not df.empty:
        piv = df.pivot_table(index="age_group", columns="gender",
                             values="churn_rate", aggfunc="mean")
        fig = px.imshow(piv, color_continuous_scale=[GREEN, GOLD, RED],
            aspect="auto", text_auto=".1f", labels={"color": "Churn %"})
        L(fig, "Churn Rate Heatmap — Age Group × Gender", 280)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(note(
        "City 1, which is the largest metropolitan area in the dataset, has a churn rate of just 6.4% "
        "compared to 12–15% across most other cities. This is a meaningful gap that deserves investigation — "
        "whether it comes from better brand awareness, carrier partnerships, or simply a more engaged urban "
        "demographic. On the acquisition side, users who registered through the Android app churn at 23%, "
        "which is nearly four times the rate of users who registered via a web browser (4.5%). This points "
        "to a quality problem with mobile acquisition, not the app itself. Looking at cohorts, users who "
        "joined in 2009 and 2010 still show the highest churn rates at around 11%, likely reflecting a less "
        "mature product and fewer retention tools at the time. Cohorts from 2015 onwards have stabilised "
        "closer to 8.5%, suggesting improvements in onboarding and plan design have had a real effect.", "r"
    ), unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# TAB 2 — REVENUE & PLANS
# ════════════════════════════════════════════════════
with T[2]:
    st.markdown("#### Revenue & Plans")
    c1, c2 = st.columns(2)

    with c1:
        df = data.get("mon_rev")
        if df is not None and not df.empty:
            df = df[df["trans_year_month"].str.match(r"^\d{4}-\d{2}$", na=False)].sort_values("trans_year_month")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["trans_year_month"], y=df["total_revenue"],
                mode="lines", name="Revenue", line=dict(color=GREEN, width=2),
                fill="tozeroy", fillcolor="rgba(0,214,143,.1)"))
            fig.add_trace(go.Scatter(x=df["trans_year_month"], y=df["revenue_lost"],
                mode="lines", name="Promo Leakage", line=dict(color=RED, width=2, dash="dot")))
            L(fig, "Monthly Revenue vs Promo Leakage (2015–2017)")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        df = data.get("rev_b")
        if df is not None and not df.empty:
            fig = px.bar(df, x="revenue_bucket", y="churn_rate", color="churn_rate",
                color_continuous_scale=[GREEN, GOLD, RED],
                text=df["churn_rate"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_coloraxes(showscale=False)
            L(fig, "Churn Rate by Lifetime Revenue Spent")
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        df = data.get("auto_r")
        if df is not None and not df.empty:
            df["status"] = df["is_churn"].map(CHURN_MAP)
            fig = px.bar(df, x="auto_renew_label", y="users", color="status",
                barmode="group", color_discrete_map={"Retained": CYAN, "Churned": RED},
                text="users")
            fig.update_traces(textposition="outside")
            L(fig, "Auto-Renew Setting vs Churn Count", 330)
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        df = data.get("promo_s")
        if df is not None and not df.empty:
            fig = px.bar(df, x="promo_segment", y="churn_rate", color="churn_rate",
                color_continuous_scale=[GREEN, GOLD, RED],
                text=df["churn_rate"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_coloraxes(showscale=False)
            L(fig, "Churn Rate by Promo Usage %", 330)
            st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        df = data.get("plan_rev")
        if df is not None and not df.empty:
            df["status"] = df["is_churn"].map(CHURN_MAP)
            fig = px.bar(df, x="plan_type", y="total_revenue", color="status",
                barmode="group", color_discrete_map={"Retained": CYAN, "Churned": RED},
                text=df["total_revenue"].apply(
                    lambda v: f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1000:.0f}K"))
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_tickangle=-20)
            L(fig, "Revenue by Plan Type — Retained vs Churned", 330)
            st.plotly_chart(fig, use_container_width=True)

    with c6:
        df = data.get("pay_c")
        if df is not None and not df.empty:
            df["status"] = df["is_churn"].map(CHURN_MAP)
            fig = px.bar(df, x="payment_method_label", y="revenue", color="status",
                barmode="group", color_discrete_map={"Retained": CYAN, "Churned": RED},
                text=df["revenue"].apply(lambda v: f"${v/1000:.0f}K"))
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_tickangle=-25)
            L(fig, "Revenue by Payment Method", 330)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown(note(
        "There is a clear and consistent relationship between how much a subscriber has spent over their "
        "lifetime and how likely they are to churn. Users who have spent less than $50 in total are among "
        "the highest-risk groups, while those who have crossed the $600 mark almost never leave. This makes "
        "intuitive sense — higher spend reflects both longer tenure and a deeper connection to the product. "
        "Auto-renew tells a similar story. Subscribers who have it switched on churn at roughly half the "
        "rate of those who manage renewals manually, and this holds across plan types. Perhaps the most "
        "concerning finding is the promo data: users who received free or heavily discounted transactions "
        "more than half the time churn at 20–25%, which is about double the overall rate. These users are "
        "not being converted into paying customers — they are cycling through promotions and leaving when "
        "they run out.", "g"
    ), unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# TAB 3 — ENGAGEMENT
# ════════════════════════════════════════════════════
with T[3]:
    st.markdown("#### Engagement")
    c1, c2 = st.columns(2)

    with c1:
        df = data.get("skip_s")
        if df is not None and not df.empty:
            fig = px.bar(df, x="skip_segment", y="churn_rate", color="churn_rate",
                color_continuous_scale=[GREEN, GOLD, RED],
                text=df["churn_rate"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_coloraxes(showscale=False)
            L(fig, "Churn Rate by Skip Rate Segment")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        df = data.get("compl_s")
        if df is not None and not df.empty:
            fig = px.bar(df, x="compl_segment", y="churn_rate", color="churn_rate",
                color_continuous_scale=[RED, GOLD, GREEN],
                text=df["churn_rate"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_coloraxes(showscale=False)
            L(fig, "Churn Rate by Song Completion Ratio")
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        df = data.get("active_s")
        if df is not None and not df.empty:
            fig = px.bar(df, x="active_segment", y="churn_rate", color="churn_rate",
                color_continuous_scale=[RED, GOLD, GREEN],
                text=df["churn_rate"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_coloraxes(showscale=False)
            L(fig, "Churn Rate by Active Days Segment")
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        df = data.get("listen_s")
        if df is not None and not df.empty:
            fig = px.bar(df, x="listening_segment", y="churn_rate", color="churn_rate",
                color_continuous_scale=[RED, GOLD, GREEN],
                text=df["churn_rate"].apply(lambda v: f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_coloraxes(showscale=False)
            L(fig, "Churn Rate by Total Listening Hours")
            st.plotly_chart(fig, use_container_width=True)

    # Song depth stacked bar
    sb = data.get("song_bk")
    if sb is not None and not sb.empty:
        sb["status"] = sb["is_churn"].map(CHURN_MAP)
        dcols   = ["songs_25_pct", "songs_50_pct", "songs_75_pct", "songs_985_pct", "songs_completed_pct"]
        dlabels = ["25% Played", "50%", "75%", "98.5%", "Fully Completed (100%)"]
        dcolors = [RED, "#F97316", GOLD, "#A3E635", GREEN]
        fig = go.Figure()
        for col, lbl, clr in zip(dcols, dlabels, dcolors):
            if col in sb.columns:
                fig.add_trace(go.Bar(x=sb["status"], y=sb[col], name=lbl, marker_color=clr,
                    text=[f"{v:.0f}%" for v in sb[col]], textposition="inside"))
        fig.update_layout(barmode="stack")
        L(fig, "Song Listening Depth — Retained vs Churned (% breakdown of all plays)", 320)
        st.plotly_chart(fig, use_container_width=True)

    # RFM heatmap
    rfm = data.get("rfm")
    if rfm is not None and not rfm.empty:
        piv = rfm.pivot_table(index="days_active_bucket", columns="revenue_bucket",
                              values="churn_rate", aggfunc="mean")
        fig = px.imshow(piv, color_continuous_scale=[GREEN, GOLD, RED],
            aspect="auto", text_auto=".0f", labels={"color": "Churn %"})
        L(fig, "RFM Heatmap — Churn Rate by Activity × Revenue Tier", 310)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(note(
        "Skip rate is one of the clearest early signals of an at-risk subscriber. Users who skip more than "
        "80% of the songs they start are churning at nearly three times the base rate. This is not a "
        "lagging indicator — it shows up before the user cancels, which means there is a window to act on it. "
        "The song completion breakdown makes this even more visible: churned users have a disproportionately "
        "high share of plays that stopped at the 25% mark, meaning they are opening tracks and abandoning them "
        "almost immediately. That pattern points to a content relevance problem, not a UX one — the "
        "recommendations are not landing. The RFM heatmap in the lower section reinforces this: users who are "
        "rarely active and have spent very little are churning at rates above 35%, while highly active, "
        "higher-spending users are some of the most loyal in the entire subscriber base.", "r"
    ), unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# TAB 4 — INSIGHTS
# ════════════════════════════════════════════════════
with T[4]:
    st.markdown("#### Insights & Recommended Actions")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        recs = [
            ("Reduce reliance on promotions", "r",
             "Users who received discounted or free transactions on more than half of their renewals are "
             "churning at 20 to 25 percent, roughly double the overall rate. The data suggests that heavy "
             "promo users were never genuinely committed subscribers — they came for the discount and left "
             "when it stopped. A practical fix is to restrict repeat promotions: users who have already "
             "received a free or heavily discounted period should be moved to a modest paid option, such as "
             "a discounted three-month plan, rather than another full free trial. Introducing a 90-day "
             "cooldown between promotional offers per account would also help separate genuine new subscribers "
             "from those cycling the system."),
            ("Make auto-renew the default, not the exception", "",
             "Auto-renew is the single strongest structural predictor of retention in this dataset. "
             "Users with it enabled churn at roughly half the rate of those who manage renewals manually, "
             "yet a significant portion of the subscriber base still has it switched off. The most direct "
             "way to close that gap is an incentive-based enrollment push: a targeted in-app prompt and "
             "follow-up email for users who have had two or more transactions but never enabled auto-renew, "
             "offering one extra month free in exchange for switching it on. The cost of that extra month "
             "is almost certainly lower than the cost of re-acquiring a churned subscriber through "
             "paid channels."),
            ("Use skip rate as an early warning, not an afterthought", "r",
             "By the time a user cancels, the decision to leave has usually already been made. Skip rate "
             "gives an earlier read on that intent. When a user's 7-day rolling skip rate climbs above "
             "75 percent, the data shows they are on a fast path toward churning. The right response at "
             "that moment is not a discount — it is a better content experience. A triggered playlist "
             "recommendation sent within 24 hours of the threshold being crossed is a low-cost intervention "
             "that addresses the actual problem: the user is not finding music they want to listen to, "
             "and the existing recommendation engine is not catching it."),
            ("Rethink how Android users are acquired and onboarded", "r",
             "Android-registered users churn at 23 percent, which is nearly four times the rate of users "
             "who signed up through a web browser. It is unlikely that the app itself is the issue. More "
             "probably, the advertising channels driving Android installs are reaching a less "
             "subscription-minded audience. Adding a short personalisation flow on first launch — three "
             "questions about listening preferences and genres — has been shown in comparable products to "
             "meaningfully improve 90-day retention, because it creates an immediate sense that the product "
             "understands the user. The ad targeting strategy should also be reviewed to shift optimisation "
             "away from cost per install and toward cost per second renewal."),
            ("Study what City 1 is doing differently and replicate it", "g",
             "City 1's 6.4 percent churn rate is not a statistical outlier — it is roughly half the rate "
             "of every other city in the dataset. Something structural is working there, whether it is "
             "stronger carrier integration, higher brand recognition, or a demographic that naturally skews "
             "toward longer-term subscription behaviour. Before spending on retention campaigns in "
             "high-churn cities, it is worth investing in understanding what is already working in City 1. "
             "Those findings should inform how KKBox approaches product positioning and partnerships in "
             "Cities 13, 5, and 4, which consistently show the highest churn across the dataset."),
            ("Convert monthly subscribers to annual plans after the third renewal", "g",
             "Annual and semi-annual subscribers churn at dramatically lower rates and are worth three to "
             "five times more in lifetime value compared to month-to-month users. Many subscribers never "
             "make the move to a longer plan because no one prompts them at the right moment. After a user "
             "completes three consecutive monthly renewals — a reasonable signal of genuine intent — a "
             "targeted message offering a switch to an annual plan at 30 percent off, sent via both push "
             "notification and email, tends to convert well. The framing should lead with the saving rather "
             "than the commitment, since price sensitivity runs high across this subscriber base."),
        ]
        for title, cls, body in recs:
            st.markdown(f"**{title}**")
            st.markdown(note(body, cls), unsafe_allow_html=True)


    with col_r:
        # ── Priority Matrix (manual go.Figure for clean labels) ──
        acts = [
            ("Stop Promos",    9.0, 8.0, RED,    "Revenue Impact →"),
            ("Auto-Renew",     8.0, 9.0, CYAN,   ""),
            ("Skip Alert",     7.0, 7.0, GOLD,   ""),
            ("Android Fix",    6.0, 8.0, PURPLE, ""),
            ("City 1 Formula", 5.0, 6.0, "#F97316", ""),
            ("Annual Upgrade", 8.5, 9.5, GREEN,  ""),
        ]
        fig = go.Figure()
        # Quadrant shading
        fig.add_shape(type="rect", x0=7, x1=10.3, y0=7, y1=10.3,
            fillcolor="rgba(0,214,143,.07)",
            line=dict(color=GREEN, dash="dot", width=1))
        fig.add_annotation(x=8.6, y=10.1, text="High Priority Zone",
            font=dict(color=GREEN, size=11), showarrow=False,
            bgcolor="rgba(0,0,0,.3)", borderpad=4)
        # Bubbles + labels
        for name, rx, ry, col, _ in acts:
            fig.add_trace(go.Scatter(
                x=[rx], y=[ry], mode="markers+text",
                marker=dict(size=30, color=col, opacity=.85,
                            line=dict(color="white", width=1.5)),
                text=[name], textposition="top center",
                textfont=dict(size=11, color="#ddd"),
                name=name, showlegend=False
            ))
        fig.update_layout(
            xaxis=dict(title="Revenue Impact",  range=[4, 10.5],
                       gridcolor="rgba(255,255,255,.06)", zeroline=False),
            yaxis=dict(title="Retention Impact", range=[5, 10.5],
                       gridcolor="rgba(255,255,255,.06)", zeroline=False),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#bbb"),
            height=400, margin=dict(l=10, r=10, t=45, b=10),
            title=dict(text="Initiative Priority Matrix  (Revenue vs Retention Impact)",
                       font=dict(size=14, color="#ddd"), x=0.01))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Quick Wins — Effort vs Impact**")
        qw = {
            "Initiative": ["Auto-Renew Push", "Annual Upgrade", "Promo Cooldown", "Skip Alert"],
            "Effort":     ["Low", "Low", "Medium", "Medium"],
            "Impact":     ["Very High", "Very High", "High", "Medium"],
            "Timeline":   ["2 weeks", "1 month", "1 month", "2 months"],
        }
        st.dataframe(pd.DataFrame(qw), use_container_width=True, hide_index=True)