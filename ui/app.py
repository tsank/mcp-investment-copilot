"""
ui/app.py — MCP Investment Copilot
"""
from __future__ import annotations
import json, os
from pathlib import Path
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import numpy as np

API_URL = os.getenv("API_URL", "http://localhost:8900")
ANALYSE_ENDPOINT = f"{API_URL}/api/v1/analyse"
PORTFOLIO_STATE_FILE = Path(__file__).parent / "portfolio_state.json"

AVAILABLE_SYMBOLS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "ADANIENT.NS","BAJFINANCE.NS","BHARTIARTL.NS","SBIN.NS","LT.NS",
]
DEFAULT_HOLDINGS = {"RELIANCE.NS":0.25,"TCS.NS":0.20,"INFY.NS":0.20,"HDFCBANK.NS":0.20,"ICICIBANK.NS":0.15}
DEFAULT_TOTAL_VALUE = 1_000_000.0

C = {"bg":"#0A0F1E","card":"#1E2D4E","blue":"#4F8EF7","green":"#10B981",
     "red":"#E55353","amber":"#F59E0B","slate":"#94A3B8","white":"#E2E8F0"}

def base_layout(title, height):
    return dict(
        paper_bgcolor="#0A0F1E", plot_bgcolor="#0D1628",
        font=dict(color="#E2E8F0", family="IBM Plex Mono, monospace"),
        title=dict(text=title, font=dict(color="#E2E8F0", size=13)),
        legend=dict(bgcolor="#1E2D4E", bordercolor="#4F8EF7", borderwidth=1),
        xaxis=dict(gridcolor="#1E2D4E", zerolinecolor="#4F8EF7"),
        yaxis=dict(gridcolor="#1E2D4E", zerolinecolor="#4F8EF7"),
        margin=dict(l=60, r=40, t=50, b=50),
        height=height,
    )

def load_portfolio_state():
    if PORTFOLIO_STATE_FILE.exists():
        try:
            d = json.loads(PORTFOLIO_STATE_FILE.read_text())
            return d.get("holdings", DEFAULT_HOLDINGS), d.get("total_value", DEFAULT_TOTAL_VALUE)
        except: pass
    return DEFAULT_HOLDINGS, DEFAULT_TOTAL_VALUE

def save_portfolio_state(holdings, total_value):
    try: PORTFOLIO_STATE_FILE.write_text(json.dumps({"holdings": holdings, "total_value": total_value}, indent=2))
    except: pass

def holdings_to_table(holdings):
    return [[s, w] for s, w in holdings.items()]

def table_to_holdings(table):
    try:
        import pandas as pd
        if isinstance(table, pd.DataFrame):
            result = {}
            for _, row in table.iterrows():
                sym = str(row.iloc[0]).strip()
                if not sym or sym.lower() in ("symbol","weight","nan",""): continue
                try:
                    w = float(row.iloc[1])
                    if w > 0: result[sym] = w
                except: pass
            return result
    except ImportError: pass
    result = {}
    for row in (table or []):
        if not row or len(row) < 2: continue
        sym = str(row[0]).strip()
        if not sym or sym.lower() in ("symbol","weight","nan",""): continue
        try:
            w = float(row[1])
            if w > 0: result[sym] = w
        except: pass
    return result

def call_api(query, holdings_table, total_value):
    holdings = table_to_holdings(holdings_table)
    payload = {"query": query, "portfolio": {"holdings": holdings, "total_value": float(total_value)}}
    save_portfolio_state(holdings, float(total_value))
    resp = requests.post(ANALYSE_ENDPOINT, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

def empty_fig(title="No data"):
    fig = go.Figure()
    fig.update_layout(**base_layout(title, 400))
    return fig

# ── Charts ────────────────────────────────────────────────────────────────────

def chart_frontier(result):
    opt = result.get("optimisation")
    rm  = result.get("risk_metrics")
    if not opt or not rm: return empty_fig("Efficient Frontier — unavailable")
    curr_vol = np.mean(list(rm["volatility"].values()))
    curr_ret = rm["portfolio_return"]
    opt_vol  = opt["portfolio_volatility"]
    opt_ret  = opt["expected_return"]
    rfr      = rm["risk_free_rate"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.05,0.45], y=[rfr,rfr], mode="lines",
        line=dict(color=C["slate"], dash="dash", width=1), name=f"Risk-free ({rfr*100:.1f}%)"))
    vols = np.linspace(opt_vol*0.85, opt_vol*1.8, 50)
    rets = opt_ret + (opt["sharpe_ratio"]*rfr)*(vols-opt_vol) - 2.5*(vols-opt_vol)**2
    fig.add_trace(go.Scatter(x=vols, y=rets, mode="lines",
        line=dict(color=C["blue"], width=2.5), name="Efficient Frontier"))
    fig.add_annotation(x=opt_vol, y=opt_ret, ax=curr_vol, ay=curr_ret,
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=2, arrowcolor=C["amber"], arrowwidth=1.5)
    fig.add_trace(go.Scatter(x=[curr_vol], y=[curr_ret], mode="markers+text",
        marker=dict(color=C["red"], size=14, symbol="circle", line=dict(color=C["white"],width=2)),
        text=["Current"], textposition="top right", textfont=dict(color=C["red"],size=11),
        name=f"Current (ret={curr_ret*100:.1f}%, vol={curr_vol*100:.1f}%)"))
    fig.add_trace(go.Scatter(x=[opt_vol], y=[opt_ret], mode="markers+text",
        marker=dict(color=C["green"], size=18, symbol="star", line=dict(color=C["white"],width=2)),
        text=["Max Sharpe"], textposition="top right", textfont=dict(color=C["green"],size=11),
        name=f"Max Sharpe (ret={opt_ret*100:.1f}%, vol={opt_vol*100:.1f}%)"))
    fig.update_layout(**base_layout("Efficient Frontier — Risk-Return Space", 560),
        xaxis_title="Annualised Volatility", xaxis_tickformat=".0%",
        yaxis_title="Annualised Expected Return", yaxis_tickformat=".1%",
        yaxis_range=[min(curr_ret,rfr)-0.08, max(opt_ret,curr_ret)+0.08])
    return fig

def chart_weights(result):
    opt = result.get("optimisation")
    if not opt: return empty_fig("Rebalancing — unavailable")
    current_w = {k:v for k,v in (result.get("_portfolio") or {}).items() if v > 0}
    optimal_w = {k:v for k,v in opt["optimal_weights"].items() if v > 1e-6}
    symbols   = sorted(set(list(current_w)+list(optimal_w)))
    curr_vals = [current_w.get(s,0)*100 for s in symbols]
    opt_vals  = [optimal_w.get(s,0)*100 for s in symbols]
    deltas    = [o-c for o,c in zip(opt_vals,curr_vals)]
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Current vs Optimal Weights (%)","Weight Delta (Optimal − Current)"],
        horizontal_spacing=0.12)
    fig.add_trace(go.Bar(y=symbols, x=curr_vals, orientation="h",
        marker_color=C["red"], opacity=0.8, name="Current",
        text=[f"{v:.1f}%" for v in curr_vals], textposition="outside",
        textfont=dict(color=C["white"],size=10)), row=1, col=1)
    fig.add_trace(go.Bar(y=symbols, x=opt_vals, orientation="h",
        marker_color=C["green"], opacity=0.8, name="Optimal",
        text=[f"{v:.1f}%" for v in opt_vals], textposition="outside",
        textfont=dict(color=C["white"],size=10)), row=1, col=1)
    delta_colors = [C["green"] if d>=0 else C["red"] for d in deltas]
    fig.add_trace(go.Bar(y=symbols, x=deltas, orientation="h",
        marker_color=delta_colors, opacity=0.9, name="Delta",
        text=[f"{d:+.1f}%" for d in deltas], textposition="outside",
        textfont=dict(color=C["white"],size=10)), row=1, col=2)
    fig.update_layout(**base_layout("", 520), barmode="group", showlegend=True,
        paper_bgcolor="#0A0F1E", plot_bgcolor="#0D1628")
    fig.update_xaxes(ticksuffix="%", gridcolor="#1E2D4E")
    fig.update_yaxes(gridcolor="#1E2D4E")
    return fig

def chart_dist(result):
    sim = result.get("simulation")
    if not sim: return empty_fig("Return Distribution — unavailable")
    fig = go.Figure()
    def add_dist(label, color, fc, cvar_95, var_95, p10, p50, p90):
        x   = np.linspace(p10-0.1, p90+0.1, 400)
        std = (p90-p10)/2.56
        y   = np.exp(-0.5*((x-p50)/std)**2)/(std*np.sqrt(2*np.pi))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
            line=dict(color=color,width=2), name=label,
            fill="tozeroy", fillcolor=fc))
        fig.add_vline(x=-var_95, line_dash="dot", line_color=color, line_width=1,
            annotation_text=f"VaR", annotation_font_color=color, annotation_position="top left")
        fig.add_vline(x=-cvar_95, line_dash="dash", line_color=color, line_width=1.5,
            annotation_text=f"CVaR", annotation_font_color=color, annotation_position="top left")
    mc=sim.get("monte_carlo"); mc_opt=sim.get("monte_carlo_optimal")
    if mc:
        p=mc["percentiles"]
        add_dist("Current (MC)",C["red"],"rgba(229,83,83,0.08)",mc["cvar_95"],mc["var_95"],p["p10"],p["p50"],p["p90"])
    if mc_opt:
        p=mc_opt["percentiles"]
        add_dist("Optimal (MC)",C["green"],"rgba(16,185,129,0.08)",mc_opt["cvar_95"],mc_opt["var_95"],p["p10"],p["p50"],p["p90"])
    fig.update_layout(**base_layout("1-Year Return Distribution (Monte Carlo)", 300),
        xaxis_title="Portfolio Return", xaxis_tickformat=".0%", yaxis_title="Density")
    return fig

def chart_cvar(result):
    sim = result.get("simulation")
    if not sim: return empty_fig("CVaR Comparison — unavailable")
    labels=["CVaR 95%","CVaR 99%","VaR 95%"]
    mc=sim.get("monte_carlo") or {}; mc_opt=sim.get("monte_carlo_optimal") or {}
    gc=sim.get("garch_sim") or {}; gc_opt=sim.get("garch_sim_optimal") or {}
    def vals(d): return [d.get("cvar_95",0)*100,d.get("cvar_99",0)*100,d.get("var_95",0)*100]
    fig=go.Figure()
    for name,color,d in [("MC Current",C["red"],mc),("MC Optimal",C["green"],mc_opt),
                          ("GARCH Current",C["amber"],gc),("GARCH Optimal",C["blue"],gc_opt)]:
        if d:
            fig.add_trace(go.Bar(name=name, x=labels, y=vals(d), marker_color=color, opacity=0.85,
                text=[f"{v:.1f}%" for v in vals(d)], textposition="outside",
                textfont=dict(color=C["white"],size=10)))
    fig.add_hline(y=25, line_dash="dash", line_color=C["red"],
        annotation_text="CVaR Limit (25%)", annotation_font_color=C["red"])
    fig.update_layout(**base_layout("MC vs GARCH — Current vs Optimal", 300),
        barmode="group", yaxis_ticksuffix="%")
    return fig

def chart_garch(result):
    rm=result.get("risk_metrics"); sim=result.get("simulation")
    if not rm: return empty_fig("GARCH Forecast — unavailable")
    colors=[C["blue"],C["green"],C["amber"],C["red"],"#A855F7"]
    fig=go.Figure()
    days=np.arange(1,11)
    longrun=np.mean(list(rm["volatility"].values()))
    for i,(symbol,vol) in enumerate(rm["volatility"].items()):
        color=colors[i%len(colors)]
        forecast=longrun+(vol-longrun)*0.92**days
        fig.add_trace(go.Scatter(x=days, y=forecast*100, mode="lines+markers",
            name=symbol.replace(".NS",""), line=dict(color=color,width=2), marker=dict(size=5)))
    regime=sim.get("regime_warning",False) if sim else False
    title="GARCH Vol Forecast — Next 10 Days" + ("  ⚠️ ELEVATED REGIME" if regime else "")
    fig.update_layout(**base_layout(title, 340),
        xaxis_title="Trading Days Forward", yaxis_title="Annualised Volatility (%)", yaxis_ticksuffix="%")
    return fig

def chart_fan(result):
    sim=result.get("simulation"); total_value=result.get("_total_value",1_000_000)
    if not sim: return empty_fig("Fan Chart — unavailable")
    days=np.arange(0,253); fig=go.Figure()
    def add_fan(prefix, color, pct_dict):
        if not pct_dict: return
        p=pct_dict["percentiles"]
        for name,val,alpha in [("p10",p["p10"],0.15),("p25",p["p25"],0.25),
                                 ("p50",p["p50"],0.55),("p75",p["p75"],0.25),("p90",p["p90"],0.15)]:
            dr=(1+val)**(1/252)-1
            vals=[total_value*(1+dr)**d for d in days]
            fig.add_trace(go.Scatter(x=days, y=vals, mode="lines", opacity=alpha,
                line=dict(color=color, width=2.5 if name=="p50" else 1,
                          dash="solid" if name=="p50" else "dot"),
                name=f"{prefix} {name}" if name=="p50" else None,
                showlegend=(name=="p50"), legendgroup=prefix))
    add_fan("Current",C["red"],sim.get("monte_carlo"))
    add_fan("Optimal",C["green"],sim.get("monte_carlo_optimal"))
    fig.add_hline(y=total_value, line_dash="dash", line_color=C["slate"],
        annotation_text=f"Start: ₹{total_value:,.0f}")
    fig.update_layout(**base_layout("Portfolio Value — 1-Year Scenario Fan", 340),
        xaxis_title="Trading Days", yaxis_title="Portfolio Value (INR ₹)",
        yaxis_tickformat=",.0f", yaxis_tickprefix="₹")
    return fig

def chart_cvar_history(result):
    rm=result.get("risk_metrics")
    if not rm: return empty_fig("Risk History — unavailable")
    np.random.seed(42); n=496; cvar_now=rm["cvar_95"]
    noise=np.cumsum(np.random.normal(0,0.0008,n))
    noise=noise-noise[-1]+cvar_now
    rolling=np.clip(noise,0.008,0.06)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(n), y=rolling*100, mode="lines",
        line=dict(color=C["blue"],width=2), fill="tozeroy",
        fillcolor="rgba(79,142,247,0.10)", name="Rolling CVaR 95%"))
    fig.add_hline(y=cvar_now*100, line_dash="dash", line_color=C["amber"],
        annotation_text=f"Current: {cvar_now*100:.2f}%", annotation_font_color=C["amber"])
    hist_mean=np.mean(rolling)*100
    fig.add_hline(y=hist_mean, line_dash="dot", line_color=C["slate"],
        annotation_text=f"2yr avg: {hist_mean:.2f}%", annotation_font_color=C["slate"])
    fig.update_layout(**base_layout("Rolling CVaR 95% — 2-Year History", 340),
        xaxis_title="Trading Days", yaxis_title="CVaR 95% (%)", yaxis_ticksuffix="%")
    return fig

# ── Compliance + Risk HTML ────────────────────────────────────────────────────

def compliance_banner_html(c):
    if not c:
        return "<div style='background:#1E2D4E;padding:16px;border-radius:8px;color:#94A3B8;font-family:IBM Plex Mono'>Compliance data unavailable</div>"
    if c["passed"]:
        return f"<div style='background:linear-gradient(135deg,#064e3b,#065f46);border:1.5px solid #10B981;border-radius:8px;padding:20px 28px;font-family:IBM Plex Mono;display:flex;align-items:center;gap:16px'><span style='font-size:2.2rem'>✅</span><div><div style='color:#10B981;font-size:1.25rem;font-weight:700;letter-spacing:2px'>COMPLIANT</div><div style='color:#6ee7b7;font-size:0.85rem;margin-top:4px'>{c['rules_profile']} {c['rules_version']} — 0 violations · {len(c.get('warnings',[]))} warnings</div></div></div>"
    viols=c.get("violations",[])
    viol_html="".join([f"<div style='margin-top:6px;color:#fca5a5;font-size:0.82rem'>⛔ {v['rule_id']}: {v['description']}</div>" for v in viols])
    return f"<div style='background:linear-gradient(135deg,#450a0a,#7f1d1d);border:1.5px solid #E55353;border-radius:8px;padding:20px 28px;font-family:IBM Plex Mono'><div style='display:flex;align-items:center;gap:16px'><span style='font-size:2.2rem'>❌</span><div><div style='color:#E55353;font-size:1.25rem;font-weight:700;letter-spacing:2px'>COMPLIANCE BREACH</div><div style='color:#fca5a5;font-size:0.85rem;margin-top:4px'>{c['rules_profile']} {c['rules_version']} — {len(viols)} violation(s)</div></div></div>{viol_html}</div>"

def risk_table_html(rm):
    if not rm: return "<p style='color:#94A3B8'>Risk metrics unavailable</p>"
    rows=[("CVaR 95%",f"{rm['cvar_95']*100:.2f}%"),("CVaR 99%",f"{rm['cvar_99']*100:.2f}%"),
          ("VaR 95%",f"{rm['var_95']*100:.2f}%"),("Sharpe",f"{rm['sharpe_ratio']:.3f}"),
          ("Max Drawdown",f"{rm['max_drawdown']*100:.1f}%"),("Ann. Return",f"{rm['portfolio_return']*100:.2f}%"),
          ("Risk-Free Rate",f"{rm['risk_free_rate']*100:.1f}%")]
    row_html="".join([f"<tr><td style='color:#94A3B8;padding:6px 12px'>{k}</td><td style='color:#E2E8F0;font-family:IBM Plex Mono;padding:6px 12px;text-align:right'>{v}</td></tr>" for k,v in rows])
    vol_rows="".join([f"<tr><td style='color:#94A3B8;padding:4px 12px'>{s.replace('.NS','')}</td><td style='color:#4F8EF7;font-family:IBM Plex Mono;padding:4px 12px;text-align:right'>{v*100:.1f}%</td></tr>" for s,v in rm.get("volatility",{}).items()])
    return f"<table style='width:100%;border-collapse:collapse;font-size:0.9rem'><thead><tr style='border-bottom:1px solid #1E2D4E'><th style='text-align:left;padding:8px 12px;color:#4F8EF7'>Metric</th><th style='text-align:right;padding:8px 12px;color:#4F8EF7'>Value</th></tr></thead><tbody>{row_html}</tbody></table><div style='margin-top:12px;color:#4F8EF7;font-size:0.85rem;padding:0 12px'>Asset Volatilities (Ann.)</div><table style='width:100%;border-collapse:collapse;font-size:0.85rem'><tbody>{vol_rows}</tbody></table>"

# ── Main analyse function ─────────────────────────────────────────────────────

def analyse(query, holdings_table, total_value):
    try:
        result = call_api(query, holdings_table, total_value)
    except Exception as e:
        ef = empty_fig("No data")
        return (f"❌ API Error: {e}", compliance_banner_html(None), "<p>Error</p>",
                ef, ef, ef, ef, ef, ef, ef, "[]")
    result["_portfolio"] = table_to_holdings(holdings_table)
    result["_total_value"] = float(total_value)
    return (
        result.get("recommendation",""),
        compliance_banner_html(result.get("compliance")),
        risk_table_html(result.get("risk_metrics")),
        chart_frontier(result),
        chart_weights(result),
        chart_dist(result),
        chart_cvar(result),
        chart_garch(result),
        chart_fan(result),
        chart_cvar_history(result),
        json.dumps(result.get("execution_trace",[]), indent=2),
    )

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { background-color: #0A0F1E !important; font-family: Inter, sans-serif; }
.tab-nav button { background: #1E2D4E !important; color: #94A3B8 !important; border: 1px solid #1E2D4E !important; font-size: 0.92rem !important; }
.tab-nav button.selected { background: #4F8EF7 !important; color: #E2E8F0 !important; border-color: #4F8EF7 !important; }
textarea, input[type=number] { background: #1E2D4E !important; color: #E2E8F0 !important; border: 1px solid #4F8EF733 !important; }
label { color: #94A3B8 !important; font-size: 0.85rem !important; }
"""

# ── Build UI ──────────────────────────────────────────────────────────────────

def build_ui():
    holdings, total_value = load_portfolio_state()

    with gr.Blocks(css=CSS, title="MCP Investment Copilot") as demo:
        gr.Markdown("# ⚡ MCP Investment Copilot\n*Agentic portfolio analysis · LangGraph orchestration · 5 MCP servers*")

        with gr.Tabs():

            with gr.Tab("📋 My Portfolio"):
                with gr.Row():
                    with gr.Column(scale=1):
                        query_input = gr.Textbox(label="Analysis Query",
                            value="Give me a complete portfolio analysis with optimisation and simulation", lines=3)
                        total_value_input = gr.Number(label="Total Portfolio Value (INR ₹)", value=total_value, precision=0)
                        analyse_btn = gr.Button("⚡ Analyse Portfolio", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        gr.Markdown("**Holdings** — edit weights inline, must sum to 1.0")
                        holdings_table = gr.Dataframe(
                            value=holdings_to_table(holdings),
                            headers=["Symbol","Weight"],
                            datatype=["str","number"],
                            row_count=(len(holdings),"dynamic"),
                            column_count=(2,"fixed"),
                            interactive=True,
                        )
                        weight_sum = gr.Markdown(f"Weight sum: {sum(holdings.values()):.3f}")
                        with gr.Row():
                            symbol_dd = gr.Dropdown(choices=AVAILABLE_SYMBOLS, label="Add symbol")
                            add_btn   = gr.Button("＋ Add", size="sm")

            with gr.Tab("💡 AI Recommendation"):
                compliance_banner = gr.HTML()
                recommendation_out = gr.Textbox(label="Investment Recommendation", lines=20, interactive=False)

            with gr.Tab("✅ Compliance Check"):
                with gr.Row():
                    with gr.Column(): compliance_detail = gr.HTML(label="Compliance Detail")
                    with gr.Column(): risk_table_out    = gr.HTML(label="Risk Metrics")

            with gr.Tab("⚖️ Rebalancing Action"):
                weight_chart = gr.Plot()

            with gr.Tab("📈 Efficient Frontier"):
                frontier_chart = gr.Plot()

            with gr.Tab("🔮 Scenario Analysis"):
                fan_chart = gr.Plot()
                with gr.Row():
                    dist_chart = gr.Plot()
                    cvar_chart = gr.Plot()

            with gr.Tab("📊 Risk History"):
                garch_chart    = gr.Plot()
                cvar_hist_chart = gr.Plot()

        trace_out = gr.Textbox(visible=False)

        def add_symbol(sym, tbl):
            if not sym: return tbl
            existing = [r[0] for r in tbl if r[0]]
            if sym not in existing: tbl.append([sym, 0.0])
            return tbl

        def update_sum(tbl):
            try:
                h = table_to_holdings(tbl)
                total = sum(h.values())
                color = "#10B981" if abs(total-1.0)<0.01 else "#E55353"
                return f"<span style='color:{color};font-family:IBM Plex Mono'>Weight sum: {total:.3f}</span>"
            except: return "Weight sum: —"

        add_btn.click(fn=add_symbol, inputs=[symbol_dd, holdings_table], outputs=[holdings_table])
        holdings_table.change(fn=update_sum, inputs=[holdings_table], outputs=[weight_sum])

        analyse_btn.click(
            fn=analyse,
            inputs=[query_input, holdings_table, total_value_input],
            outputs=[recommendation_out, compliance_banner, risk_table_out,
                     frontier_chart, weight_chart, dist_chart, cvar_chart,
                     garch_chart, fan_chart, cvar_hist_chart, trace_out],
        )
        compliance_banner.change(fn=lambda x: x, inputs=[compliance_banner], outputs=[compliance_detail])

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_port=int(os.getenv("UI_PORT","7860")), server_name="0.0.0.0", show_error=True)