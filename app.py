"""
天盾 · A股投资风控网页（真实数据优先）

严格对照你的规格：
- 侧边栏：持仓管理 + 预警中心（st.session_state 持久化，同会话生效）
- 主页面：组合仪表盘 + 持仓明细（含“查看详情”触发个股深度分析）
- 个股深度分析：5 个标签页（财务/ESG/供应链/舆情/量化）
- 风险趋势图：近 30 天综合风险分趋势（基于滚动量化近似 + 最新常数项）
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from data_fetcher import TianDunDataFetcher, normalize_stock_code
from risk_engine_v2 import (
    calculate_comprehensive_risk,
    calculate_daily_composite_risk_trend,
    calculate_esg_risk_combined,
    calculate_financial_risk,
    calculate_portfolio_metrics,
    calculate_quant_risk,
    calculate_sentiment_daily_risk,
    calculate_supply_chain_risk_simulated,
    score_level,
)


st.set_page_config(page_title="天盾·投资风控", layout="wide")

# 缩小指标数字字号、修正列宽，避免「组合总市值」等 st.metric 出现省略号截断
st.markdown(
    """
<style>
/* 多列布局下子项默认可伸缩，否则数字会被挤成「2,81…」 */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
div[data-testid="stMetricContainer"] {
    min-width: 0 !important;
}
/* 指标数值：略小字号，必要时换行，完整显示 */
div[data-testid="stMetricValue"] {
    font-size: 1.05rem !important;
    line-height: 1.25 !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: break-word;
}
div[data-testid="stMetricLabel"] p {
    font-size: 0.78rem !important;
    line-height: 1.2 !important;
}
</style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _get_fetcher() -> TianDunDataFetcher:
    return TianDunDataFetcher(request_timeout_s=8)


fetcher = _get_fetcher()


# -------------------- session 状态 --------------------
if "holdings" not in st.session_state:
    st.session_state.holdings = {}  # code -> {shares, cost}
if "alerts" not in st.session_state:
    st.session_state.alerts = []  # list[dict]
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = None
if "tushare_token" not in st.session_state:
    st.session_state.tushare_token = ""


def _seed_sentiment_daily(news_items: List[Dict]) -> pd.DataFrame:
    """
    由新闻标题计算“情绪得分”（关键词打分：基于真实标题数据，但情绪映射规则为合理模拟）。
    若标题为空，则返回空 DataFrame，app 层会做中性降级。
    """
    pos_kw = [
        "增长",
        "上涨",
        "利好",
        "突破",
        "盈利",
        "订单",
        "中标",
        "上调",
        "增持",
    ]
    neg_kw = [
        "风险",
        "下滑",
        "下跌",
        "亏损",
        "处罚",
        "违规",
        "诉讼",
        "减持",
        "暴雷",
        "停牌",
        "污染",
        "排放",
        "事故",
        "造假",
        "信披",
    ]

    if not news_items:
        return pd.DataFrame(columns=["date", "sentiment_score"])

    rows = []
    for it in news_items:
        title = str(it.get("title", "")).strip()
        dt = it.get("date", None)
        if not title or dt is None:
            continue
        pos_cnt = sum(1 for kw in pos_kw if kw in title)
        neg_cnt = sum(1 for kw in neg_kw if kw in title)
        denom = pos_cnt + neg_cnt + 1
        raw = (pos_cnt - neg_cnt) / denom  # 约略在 [-1,1]
        score = float(np.clip(raw, -1.0, 1.0))
        rows.append({"date": pd.to_datetime(dt), "sentiment_score": score})

    if not rows:
        return pd.DataFrame(columns=["date", "sentiment_score"])

    df = pd.DataFrame(rows).groupby("date", as_index=False)["sentiment_score"].mean()
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(ttl=60, show_spinner=False)
def _load_spot_quotes() -> pd.DataFrame:
    df = fetcher.fetch_spot_quotes()
    return df if df is not None else pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def _load_stock_ohlc(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    return fetcher.fetch_stock_daily_ohlc(code, start_date=start_date, end_date=end_date, adjust="qfq")


@st.cache_data(ttl=300, show_spinner=False)
def _load_index_ohlc(start_date: str, end_date: str) -> pd.DataFrame:
    # 沪深300
    return fetcher.fetch_index_daily(index_symbol="000300", start_date=start_date, end_date=end_date)


@st.cache_data(ttl=1800, show_spinner=False)
def _load_financial_history(code: str) -> pd.DataFrame:
    return fetcher.fetch_financial_ratio_history(code, max_reports=12)


@st.cache_data(ttl=1800, show_spinner=False)
def _load_latest_financial_metrics(code: str) -> Dict:
    return fetcher.fetch_latest_financial_metrics(code)


@st.cache_data(ttl=21600, show_spinner=False)
def _load_esg_hz_table() -> pd.DataFrame:
    """华证 ESG 全市场表（新浪源），首次较慢，缓存约 6 小时。"""
    df = fetcher.fetch_esg_hz_sina_table()
    return df if df is not None else pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def _load_news_items(code: str, name: str, days: int = 30) -> List[Dict]:
    items = fetcher.fetch_stock_news_titles(code, name, days=days) or []
    out = []
    for it in items:
        out.append({"date": it.date, "title": it.title, "source": it.source})
    return out


def _last_close_from_hist(hist: pd.DataFrame) -> Optional[float]:
    """日线最后一根收盘价，用于实时行情缺失时的市值与组合指标兜底。"""
    if hist is None or hist.empty or "close" not in hist.columns:
        return None
    s = pd.to_numeric(hist["close"], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _validate_add_holding(code_in: str, shares: float, cost: float) -> Optional[str]:
    code = normalize_stock_code(code_in)
    if not code:
        return "股票代码格式不正确，请输入如 `600519` 或 `002415`（6位数字）。"
    if shares <= 0:
        return "持股数量必须大于 0。"
    if cost <= 0:
        return "成本价必须大于 0。"
    return None


def _get_stock_name_industry_from_spot(spot_df: pd.DataFrame, code: str) -> Dict:
    if spot_df is None or spot_df.empty:
        try:
            nm = fetcher.fetch_stock_name_from_eastmoney(code)
            return {"name": nm or code, "industry": ""}
        except Exception:
            return {"name": code, "industry": ""}

    code = normalize_stock_code(code)
    df = spot_df.copy()
    code_col_candidates = ["代码", "code", "股票代码", "Symbol", "sec_code", "证券代码"]
    name_col_candidates = ["名称", "name", "股票简称", "简称", "证券简称", "stock_name"]
    ind_col_candidates = ["所属行业", "industry", "行业"]

    def _first_present(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    code_col = _first_present(code_col_candidates)
    name_col = _first_present(name_col_candidates)
    ind_col = _first_present(ind_col_candidates)

    if code_col is None:
        return {"name": code, "industry": ""}

    one = df.copy()
    one["_norm_code"] = one[code_col].apply(lambda x: normalize_stock_code(str(x).replace("sh", "").replace("sz", "")))
    one = one[one["_norm_code"] == code]
    if one.empty:
        return {"name": code, "industry": ""}
    one = one.iloc[0]
    name = code
    if name_col and name_col in one.index:
        v = one.get(name_col)
        name = str(v).strip() if v is not None else code

    industry = ""
    if ind_col and ind_col in one.index:
        v = one.get(ind_col)
        industry = str(v).strip() if v is not None else ""
    return {"name": name, "industry": industry}


def _compute_volatility_increase_alert(price_df: pd.DataFrame) -> Dict:
    """
    检测“波动率较前30日上升20%”：
    - recent_vol = last 30 日年化波动率
    - prev_vol = 前 30 日年化波动率
    """
    if price_df is None or price_df.empty or len(price_df) < 70:
        return {"trigger": False}
    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    returns = df["close"].pct_change().dropna()
    if len(returns) < 60:
        return {"trigger": False}

    recent = returns.tail(30)
    prev = returns.tail(60).head(30)
    if prev.std() == 0:
        return {"trigger": False}
    vol_recent = recent.std() * np.sqrt(252)
    vol_prev = prev.std() * np.sqrt(252)
    if vol_prev <= 0:
        return {"trigger": False}
    ratio = vol_recent / vol_prev
    return {
        "trigger": ratio >= 1.2,
        "vol_recent_annual": float(vol_recent),
        "vol_prev_annual": float(vol_prev),
        "ratio": float(ratio),
    }


def _generate_alerts_for_stocks(
    holdings_codes: List[str],
    stock_latest: Dict[str, Dict],
    price_history: Dict[str, pd.DataFrame],
    comprehensive_risk_map: Dict[str, float],
    detailed_map: Dict[str, Dict],
) -> None:
    """
    更新 st.session_state.alerts
    规则：
    - 综合风险分 > 70 -> 红色
    - 单日跌幅 > 5% -> 红色
    - 波动率较前30日上升20% -> 黄色
    """
    today = datetime.now().strftime("%Y-%m-%d")
    existing_keys = set(a.get("key") for a in st.session_state.alerts)

    new_alerts = []
    for code in holdings_codes:
        info = stock_latest.get(code, {})
        name = info.get("name", code)
        change_pct = info.get("change_pct")
        comp = comprehensive_risk_map.get(code, None)
        level = None
        msg = None
        rule = None
        detail = {}

        if comp is not None and comp > 70:
            level = "red"
            # 归因：取综合风险中分数最高的维度
            dims = {}
            if detailed_map and code in detailed_map:
                d = detailed_map[code]
                dims = {
                    "财务风险": d.get("fin_risk"),
                    "ESG风险": d.get("esg_risk"),
                    "供应链风险": d.get("supply_risk"),
                    "舆情风险": d.get("sentiment_risk"),
                    "量化风险": d.get("quant_risk"),
                }
            top_dim = None
            try:
                top_dim = max(dims, key=lambda k: float(dims.get(k, 0.0)) if dims.get(k) is not None else -1.0) if dims else None
            except Exception:
                top_dim = None
            msg = f"综合风险分 {comp:.1f} > 70" + (f"，主要由 {top_dim} 触发" if top_dim else "")
            rule = "risk_gt70"
            detail = {"comprehensive_risk": comp, "dimension_risks": dims}

        if change_pct is not None and change_pct <= -5.0:
            # 跌幅条件优先级：红色
            level = "red"
            msg = f"单日跌幅 {change_pct:.2f}% <= -5%"
            rule = "drop_gt5"
            detail = {"daily_change_pct": change_pct}

        # 波动率上升是黄色条件（若红色条件已触发，不再降级）
        if level != "red":
            pd_df = price_history.get(code)
            vinfo = _compute_volatility_increase_alert(pd_df) if pd_df is not None else {"trigger": False}
            if pd_df is not None and not pd_df.empty and vinfo.get("trigger", False):
                level = "yellow"
                msg = "波动率较前30日上升20%"
                rule = "vol_up_20"
                detail = {
                    "vol_recent_annual": vinfo.get("vol_recent_annual"),
                    "vol_prev_annual": vinfo.get("vol_prev_annual"),
                    "ratio": vinfo.get("ratio"),
                }

        if level and rule and msg:
            key = f"{code}_{today}_{rule}"
            if key in existing_keys:
                continue
            new_alerts.append(
                {
                    "key": key,
                    "code": code,
                    "name": name,
                    "level": level,
                    "rule": rule,
                    "message": msg,
                    "time": datetime.now().strftime("%m-%d %H:%M"),
                    "detail": detail,
                    "status": "new",
                }
            )

    if new_alerts:
        st.session_state.alerts = (new_alerts + st.session_state.alerts)[:200]


def _build_supply_chain_plot(graph_data: Dict) -> go.Figure:
    """
    用 networkx + plotly 绘制“力导向”供应链图（模拟图谱：注释说明见 risk_engine）。
    """
    nodes, edges = graph_data.get("nodes", []), graph_data.get("edges", [])

    G = nx.Graph()
    for n in nodes:
        ntype = n.get("type", "peer")
        size = 18
        if ntype == "center":
            size = 28
        if "weight" in n:
            size = int(12 + float(n["weight"]) * 30)
        G.add_node(n["name"], node_type=ntype, size=size)

    for u, v, w in edges:
        G.add_edge(u, v, weight=float(w))

    pos = nx.spring_layout(G, seed=42, k=0.6)

    edge_x = []
    edge_y = []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="rgba(100,100,100,0.35)"),
        hoverinfo="none",
    )

    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_colors = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        ntype = G.nodes[node].get("node_type", "peer")
        size = G.nodes[node].get("size", 18)
        node_sizes.append(size)
        color = "rgba(66,165,245,0.95)" if ntype == "center" else "rgba(171,71,188,0.65)" if ntype == "upstream" else "rgba(102,187,106,0.65)"
        node_colors.append(color)
        node_text.append(f"{node} ({ntype})")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color="rgba(0,0,0,0.2)")),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="供应链知识图谱（典型行业节点示例）",
        showlegend=False,
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


st.title("🛡️ 天盾 · 投资风控")
st.caption("行情/财务/TuShare 优先；ESG 含华证评级（akshare 新浪源）+ 新闻事件层；供应链等在接口缺失时按注释说明示例化。")

with st.sidebar:
    st.header("持仓管理")
    st.caption("同会话持久化：`st.session_state`")

    st.markdown("### TuShare 数据源")
    token_in = st.text_input(
        "TuShare Token（建议粘贴后回车）",
        value=st.session_state.tushare_token,
        type="password",
        help="用于调取 TuShare 真实数据；不写入代码文件。",
    )
    if token_in != st.session_state.tushare_token:
        st.session_state.tushare_token = token_in
        ok = fetcher.set_tushare_token(token_in)
        if ok:
            st.success("TuShare 已连接")
        else:
            st.warning("TuShare Token 无效或网络受限，已自动回退 AkShare。")
    elif st.session_state.tushare_token:
        # 页面刷新后保持连接
        fetcher.set_tushare_token(st.session_state.tushare_token)

    spot_df = _load_spot_quotes()

    # 数据刷新按钮（避免缓存导致一直是空表/缺失）
    if st.button("刷新行情/指数/财务/新闻", help="清除缓存并重新拉取数据"):
        try:
            _load_spot_quotes.clear()
            _load_index_ohlc.clear()
            _load_financial_history.clear()
            _load_latest_financial_metrics.clear()
            _load_news_items.clear()
            _load_esg_hz_table.clear()
        except Exception:
            pass
        st.rerun()

    with st.form("add_holding_form", clear_on_submit=True):
        code_in = st.text_input("股票代码（如 `600519`）", value="")
        shares_in = st.number_input("持股数量", min_value=0, step=100, value=0)
        cost_in = st.number_input("成本价（元）", min_value=0.0, step=0.01, value=0.0)
        submitted = st.form_submit_button("添加持仓")

    if submitted:
        err = _validate_add_holding(code_in, float(shares_in), float(cost_in))
        if err:
            st.error(err)
        else:
            code = normalize_stock_code(code_in)
            meta = _get_stock_name_industry_from_spot(spot_df, code)
            st.session_state.holdings[code] = {
                "shares": float(shares_in),
                "cost": float(cost_in),
                "name": meta.get("name", code),
                "industry": meta.get("industry", ""),
            }
            st.success(f"已添加：{meta.get('name', code)} ({code})")
            if st.session_state.selected_stock is None:
                st.session_state.selected_stock = code

    if st.session_state.holdings:
        st.markdown("### 现有持仓")
        for c, h in list(st.session_state.holdings.items()):
            meta = _get_stock_name_industry_from_spot(spot_df, c)
            st.write(f"{meta.get('name', c)} ({c})：{int(h['shares'])} 股，成本 @{h['cost']:.2f}")
            if st.button("删除", key=f"del_{c}"):
                st.session_state.holdings.pop(c, None)
                if st.session_state.selected_stock == c:
                    st.session_state.selected_stock = next(iter(st.session_state.holdings.keys()), None)
                st.rerun()
    else:
        st.info("请先添加持仓。")

    st.divider()
    st.caption("组合指标与实时预警会在页面下方侧边栏继续展示。")


holdings: Dict[str, Dict] = st.session_state.holdings
codes = list(holdings.keys())

if not codes:
    st.stop()

if st.session_state.selected_stock is None:
    st.session_state.selected_stock = codes[0]

selected_code = st.session_state.selected_stock

now = datetime.now()
end_date = now.strftime("%Y%m%d")
start_date = (now - timedelta(days=420)).strftime("%Y%m%d")

with st.spinner("拉取行情/财务/指数/华证ESG全表/新闻并计算风控指标（ESG 全表首次可能较慢）..."):
    # index (HS300)
    index_df = _load_index_ohlc(start_date=start_date, end_date=end_date)
    if index_df is None or index_df.empty:
        st.warning("指数基准（沪深300）数据获取失败：Beta/对比指标可能缺失。")

    # 计算 index returns
    if index_df is not None and not index_df.empty and "close" in index_df.columns:
        idx_ret = index_df.copy()
        idx_ret["date"] = pd.to_datetime(idx_ret["date"])
        index_returns = idx_ret.set_index("date")["close"].pct_change().dropna()
    else:
        index_returns = pd.Series(dtype=float)

    spot_df = _load_spot_quotes()
    esg_table = _load_esg_hz_table()

    stock_latest: Dict[str, Dict] = {}
    price_history: Dict[str, pd.DataFrame] = {}
    financial_latest: Dict[str, Dict] = {}
    financial_risk_map: Dict[str, float] = {}
    sentiment_daily_map: Dict[str, pd.DataFrame] = {}
    esg_events_map: Dict[str, List[Dict]] = {}
    supply_chain_graph_map: Dict[str, Dict] = {}
    quant_metrics_map: Dict[str, Dict] = {}
    comprehensive_risk_map: Dict[str, float] = {}
    detailed_map: Dict[str, Dict] = {}

    # 逐个持仓拉取
    for code in codes:
        try:
            quote = fetcher.get_realtime_quote_from_spot(spot_df=spot_df, stock_code=code)
            hist = _load_stock_ohlc(code, start_date=start_date, end_date=end_date)
            price_history[code] = hist
            # 实时行情常因网络/接口为空：用日线最后收盘价兜底，否则组合市值、夏普等会全部为 0 / N/A
            if quote.get("price") is None:
                lc = _last_close_from_hist(hist)
                if lc is not None:
                    quote = dict(quote)
                    quote["price"] = lc
                    quote["price_source"] = "ohlc_last_close"

            stock_latest[code] = quote

            fin_hist = _load_financial_history(code)
            fin_latest = _load_latest_financial_metrics(code)
            # 用实时行情兜底 PE/PB（来自市盈率-动态/市净率字段）
            if fin_latest is None:
                fin_latest = {}
            if fin_latest.get("pe_ratio") is None and quote.get("pe_ratio") is not None:
                fin_latest["pe_ratio"] = quote.get("pe_ratio")
            if fin_latest.get("pb_ratio") is None and quote.get("pb_ratio") is not None:
                fin_latest["pb_ratio"] = quote.get("pb_ratio")
            financial_latest[code] = fin_latest

            fin_risk, fin_risk_detail = calculate_financial_risk(fin_latest)
            financial_risk_map[code] = fin_risk

            # 新闻/ESG/舆情
            news_items = _load_news_items(code, name=quote.get("name", code), days=30)
            news_items_for_model = [{"date": it["date"], "title": it["title"]} for it in news_items]

            esg_row_dict = fetcher.find_esg_hz_row(esg_table, code)
            esg_risk, esg_events, esg_detail = calculate_esg_risk_combined(news_items_for_model, esg_row_dict)
            esg_events_map[code] = esg_events

            sentiment_df_raw = _seed_sentiment_daily(news_items_for_model)
            sentiment_risk, sentiment_df = calculate_sentiment_daily_risk(sentiment_df_raw)
            sentiment_daily_map[code] = sentiment_df.assign(sentiment_risk=sentiment_df["sentiment_risk"]) if not sentiment_df.empty else sentiment_df

            # 供应链风险（模拟逻辑，基于行业）
            industry = quote.get("industry", "") or holdings[code].get("industry", "")
            supply_risk, supply_graph = calculate_supply_chain_risk_simulated(
                industry=industry,
                stock_name=quote.get("name", code),
                stock_code=code,
            )
            supply_chain_graph_map[code] = supply_graph

            # 量化风险（1年数据）
            price_1y = price_history[code]
            if price_1y is not None and not price_1y.empty and len(price_1y) > 260:
                # 尽量取最后一年交易日
                price_1y = price_1y.tail(260).reset_index(drop=True)

            quant_risk, quant_detail = calculate_quant_risk(
                price_df=price_1y if price_1y is not None else pd.DataFrame(),
                index_df=index_df,
                financial_history=fin_hist,
            )
            quant_metrics_map[code] = quant_detail

            comprehensive, comp_detail = calculate_comprehensive_risk(
                financial_risk=fin_risk,
                esg_risk=esg_risk,
                supply_chain_risk=supply_risk,
                sentiment_risk=sentiment_risk,
                quant_risk=quant_risk,
            )
            comprehensive_risk_map[code] = comprehensive

            detailed_map[code] = {
                "quote": quote,
                "fin_latest": fin_latest,
                "fin_risk": fin_risk,
                "fin_risk_detail": fin_risk_detail,
                "esg_risk": esg_risk,
                "esg_events": esg_events,
                "esg_detail": esg_detail,
                "esg_rating_row": esg_row_dict,
                "sentiment_risk": sentiment_risk,
                "sentiment_df": sentiment_df,
                "supply_risk": supply_risk,
                "supply_graph": supply_graph,
                "quant_risk": quant_risk,
                "quant_detail": quant_detail,
                "comprehensive_risk": comprehensive,
            }
        except Exception as e:
            # 保底：该股票的部分维度可能缺失，但不让整个 app 崩溃
            st.warning(f"计算失败：{code}，错误：{e}")
            continue

    # 组合指标：按市值加权的综合风险分
    # 权重使用最新实时价与用户持股数
    mv_map = {}
    for code in codes:
        p = stock_latest.get(code, {}).get("price")
        if p is None:
            continue
        mv_map[code] = float(holdings[code]["shares"]) * float(p)
    total_mv = sum(mv_map.values()) if mv_map else 0.0
    weights = {c: mv_map[c] / total_mv for c in mv_map} if total_mv > 0 else {}
    weighted_comprehensive_risk = 0.0
    for code, w in weights.items():
        weighted_comprehensive_risk += w * float(comprehensive_risk_map.get(code, 0.0))

    # 组合 Sharpe / Beta / 最大回撤
    portfolio_metrics = calculate_portfolio_metrics(
        holdings=holdings,
        quotes=stock_latest,
        price_history=price_history,
        index_returns=index_returns,
        rf_annual=0.0,
    )

    # 生成实时预警（写入 session_state，仅同会话生效）
    _generate_alerts_for_stocks(
        holdings_codes=codes,
        stock_latest=stock_latest,
        price_history=price_history,
        comprehensive_risk_map=comprehensive_risk_map,
        detailed_map=detailed_map,
    )


if total_mv <= 0:
    st.warning(
        "**组合总市值为 ￥0**：当前未能得到有效成交价（东财全市场快照、TuShare 实时均失败，且日线收盘价也无法解析时会出现）。"
        "请检查网络、在侧栏填写 **TuShare Token** 并点击「刷新行情/指数/财务/新闻」，或确认持仓代码为正确 A 股 6 位代码。"
    )
elif any((stock_latest.get(c) or {}).get("price_source") == "ohlc_last_close" for c in codes):
    st.info("提示：当前至少一只股票的市值按 **最新日线收盘价** 估算（实时行情接口暂不可用）。")

if total_mv > 0:
    icon, _ = score_level(weighted_comprehensive_risk)
    wcr_display = f"{icon} {weighted_comprehensive_risk:.1f}"
else:
    icon, wcr_display = "—", "N/A（无市值权重）"

top_c1, top_c2, top_c3, top_c4 = st.columns(4)
top_c1.metric("组合总市值", f"￥{portfolio_metrics.get('total_market_value', 0.0):,.2f}")
top_c2.metric("组合加权综合风险分", wcr_display)
shv = portfolio_metrics.get("sharpe", None)
bhv = portfolio_metrics.get("beta", None)
sharpe_str = (
    "N/A（组合日收益序列无法构建：检查各股历史 K 线是否为空）"
    if total_mv > 0 and shv is None
    else (f"{shv}" if shv is not None else "N/A")
)
beta_str = (
    "N/A（需沪深300指数日线与组合收益对齐；或指数数据获取失败）"
    if total_mv > 0 and bhv is None
    else (f"{bhv}" if bhv is not None else "N/A")
)
top_c3.metric("组合夏普比率", sharpe_str)
top_c4.metric("组合Beta(沪深300)", beta_str)

with st.expander("综合风险分算法（可解释公式）", expanded=False):
    st.markdown(
        """
        ### 一句话总结
        综合风险分 = `0.6 * max(五维风险) + 0.4 * 加权平均(五维风险)`

        ### 五维风险（每个维度 0~100）
        - 财务风险（由 PE/PB/ROE/营收增长映射得到）
        - ESG风险（**华证 ESG 评级层** + **新闻关键词事件层** 融合，见下「个股深度分析 → ESG风险」展开说明）
        - 供应链风险（行业基准 + 噪声；图谱节点为“典型行业链条示例”）
        - 舆情情绪风险（日度：`sentiment_risk_day = 50*(1 - sentiment_score)`；总分：last 7 days 平均）
        - 量化风险（由波动率/最大回撤/Beta/Sharpe/下行风险映射后加权）

        ### 融合公式（你要求展示的核心算法）
        设五维风险为：`R = {financial, esg, supply, sentiment, quant}`
        - `max_risk = max(R)`
        - `weighted_avg = 0.25*financial + 0.15*esg + 0.15*supply + 0.20*sentiment + 0.25*quant`
        - `comprehensive_risk = clamp(0.6*max_risk + 0.4*weighted_avg)`

        ### 风险等级
        - `red >= 70`
        - `yellow >= 40 且 < 70`
        - `green < 40`
        """
    )

# -------------------- 持仓明细表 --------------------
rows = []
for code in codes:
    d = detailed_map.get(code)
    if not d:
        continue
    q = d["quote"]
    shares = float(holdings[code]["shares"])
    cost = float(holdings[code]["cost"])
    cur_price = q.get("price")
    change_pct = q.get("change_pct")
    mv = shares * cur_price if cur_price is not None else 0.0
    pnl = (cur_price - cost) * shares if cur_price is not None else None
    pnl_pct = (cur_price - cost) / cost * 100.0 if (cur_price is not None and cost > 0) else None
    rows.append(
        {
            "股票代码": code,
            "名称": holdings.get(code, {}).get("name") or q.get("name", code),
            "股数": int(shares),
            "成本价": round(cost, 2),
            "当前价": round(cur_price, 4) if cur_price is not None else None,
            "市值": round(mv, 2),
            "涨跌幅%": round(change_pct, 3) if change_pct is not None else None,
            "持仓盈亏": round(pnl, 2) if pnl is not None else None,
            "持仓盈亏%": round(pnl_pct, 3) if pnl_pct is not None else None,
            "综合风险分": float(d["comprehensive_risk"]),
        }
    )

df_positions = pd.DataFrame(rows)
if df_positions.empty:
    st.warning("持仓数据不足，无法展示明细。")
    st.stop()

st.markdown("### 持仓明细")
st.dataframe(df_positions.sort_values("综合风险分", ascending=False), use_container_width=True)

st.markdown("#### 查看详情")
for code in df_positions["股票代码"].tolist():
    name = df_positions.loc[df_positions["股票代码"] == code, "名称"].iloc[0]
    if st.button(f"查看详情：{name} ({code})", key=f"detail_{code}"):
        st.session_state.selected_stock = code
        st.rerun()


# -------------------- 风险趋势图：选中股票 --------------------
st.markdown("### 风险趋势（近 30 天）")
trend_code = selected_code if selected_code in detailed_map else df_positions["股票代码"].iloc[0]
trend_data = detailed_map.get(trend_code)
if trend_data and trend_data.get("sentiment_df") is not None:
    hist = price_history.get(trend_code, pd.DataFrame())
    if hist is not None and not hist.empty:
        last_60 = hist.tail(200).reset_index(drop=True)
        # sentiment_df 已由 app 计算出 sentiment_risk 字段（calculate_sentiment_daily_risk 输出）
        sentiment_df = trend_data.get("sentiment_df", pd.DataFrame())
        risk_trend_df = calculate_daily_composite_risk_trend(
            price_df=last_60[["date", "close"]].rename(columns={"close": "close"}),
            sentiment_daily_df=sentiment_df,
            financial_risk=trend_data["fin_risk"],
            esg_risk=trend_data["esg_risk"],
            supply_chain_risk=trend_data["supply_risk"],
        )
        if not risk_trend_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=risk_trend_df["date"], y=risk_trend_df["risk_score"], mode="lines+markers", line=dict(width=2)))
            fig.update_layout(
                title=f"{trend_data['quote'].get('name', trend_code)} 综合风险分趋势（近30交易日）",
                xaxis_title="日期",
                yaxis_title="综合风险分(0-100)",
                height=360,
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)


# -------------------- 个股深度分析 --------------------
st.divider()
st.markdown("## 个股深度分析")
detail_code = trend_code
detail_data = detailed_map.get(detail_code)

if detail_data:
    # 综合风险计算过程（把你要的“具体算法 + 各维度数值”落到页面上）
    with st.expander("该股票综合风险分计算过程（维度分解）", expanded=False):
        fin_risk = float(detail_data.get("fin_risk", 0.0))
        esg_risk = float(detail_data.get("esg_risk", 0.0))
        supply_risk = float(detail_data.get("supply_risk", 0.0))
        sent_risk = float(detail_data.get("sentiment_risk", 0.0))
        quant_risk = float(detail_data.get("quant_risk", 0.0))
        comp_risk = float(detail_data.get("comprehensive_risk", 0.0))

        risks_list = {
            "财务风险": fin_risk,
            "ESG风险": esg_risk,
            "供应链风险": supply_risk,
            "舆情情绪风险": sent_risk,
            "量化风险": quant_risk,
        }
        max_r = max(risks_list.values()) if risks_list else 0.0
        weighted_avg = (
            0.25 * fin_risk
            + 0.15 * esg_risk
            + 0.15 * supply_risk
            + 0.20 * sent_risk
            + 0.25 * quant_risk
        )
        calc = 0.6 * max_r + 0.4 * weighted_avg

        st.write(
            f"维度风险：{risks_list}"
        )
        st.write(f"`max_risk = {max_r:.1f}`")
        st.write(f"`weighted_avg = {weighted_avg:.1f}`")
        st.write(f"`comprehensive_risk = 0.6*max_risk + 0.4*weighted_avg = {calc:.1f}`")
        st.write(f"页面显示的综合风险分：`{comp_risk:.1f}`")

    tabs = st.tabs(["财务风险", "ESG风险", "供应链风险", "舆情情绪", "量化风险"])

    # 1) 财务风险
    with tabs[0]:
        fin = detail_data["fin_latest"]
        fin_risk = detail_data["fin_risk"]
        icon_fin, _ = score_level(fin_risk)
        st.metric("财务风险得分", f"{icon_fin} {fin_risk:.1f}")

        pe = fin.get("pe_ratio")
        pb = fin.get("pb_ratio")
        roe = fin.get("roe")
        rev_g = fin.get("revenue_growth")
        def _fmt(x):
            return "N/A（数据受限）" if x is None else f"{x}"

        st.write(
            f"PE(TTM)：`{_fmt(pe)}`  |  PB：`{_fmt(pb)}`  |  ROE：`{_fmt(roe)}`  |  营收增长率：`{_fmt(rev_g)}`"
        )

        fin_detail = detail_data.get("fin_risk_detail", {})
        pe_score = fin_detail.get("pe_score", None)
        pb_score = fin_detail.get("pb_score", None)
        roe_score = fin_detail.get("roe_score", None)
        rev_score = fin_detail.get("revenue_growth_score", None)
        st.caption("以下评价为基于指标分段的解释性规则（不等同于行业严格估值结论）。")
        st.info("PE/PB/ROE/营收增长来自 akshare 财务接口；若接口受限或字段解析失败会显示 N/A。综合风险分仍会继续计算：缺失项在打分映射中按中性值 50 处理。")
        if pe_score is not None and pe_score >= 60:
            st.error("PE 偏高：估值压力可能较大。")
        elif pe_score is not None and pe_score >= 40:
            st.warning("PE 中等偏高：需关注后续盈利兑现。")
        else:
            st.success("PE 相对可控。")

        if roe_score is not None and roe_score >= 60:
            st.error("ROE 较弱：盈利能力/资本效率需重点跟踪。")

        if rev_score is not None and rev_score >= 60:
            st.warning("营收增长偏弱或为负：基本面弹性不足可能增加风险。")

        # 估值相对行业（抽样同行业：最多5只）
        industry = detail_data["quote"].get("industry", "") or holdings[detail_code].get("industry", "")
        if industry and not spot_df.empty and "所属行业" in spot_df.columns and "代码" in spot_df.columns:
            df_sp = spot_df.copy()
            mask = df_sp["所属行业"].astype(str) == str(industry)
            peer_codes = []
            for c in df_sp.loc[mask, "代码"].astype(str).tolist():
                cc = normalize_stock_code(c.replace("sh", "").replace("sz", ""))
                if cc and cc != detail_code:
                    peer_codes.append(cc)
            peer_codes = list(dict.fromkeys(peer_codes))[:5]

            pe_peers = []
            pb_peers = []
            for pc in peer_codes:
                fin_pc = _load_latest_financial_metrics(pc)
                if fin_pc.get("pe_ratio") is not None:
                    pe_peers.append(float(fin_pc["pe_ratio"]))
                if fin_pc.get("pb_ratio") is not None:
                    pb_peers.append(float(fin_pc["pb_ratio"]))

            cur_pe = pe
            cur_pb = pb
            if pe_peers:
                peer_pe_mean = float(np.nanmean(pe_peers))
                if cur_pe is not None and cur_pe > peer_pe_mean:
                    st.warning(f"PE 相对行业（样本均值）偏高：{cur_pe:.2f} vs {peer_pe_mean:.2f}")
                elif cur_pe is not None:
                    st.success(f"PE 相对行业（样本均值）偏低/接近：{cur_pe:.2f} vs {peer_pe_mean:.2f}")
            if pb_peers:
                peer_pb_mean = float(np.nanmean(pb_peers))
                if cur_pb is not None and cur_pb > peer_pb_mean:
                    st.warning(f"PB 相对行业（样本均值）偏高：{cur_pb:.2f} vs {peer_pb_mean:.2f}")
                elif cur_pb is not None:
                    st.success(f"PB 相对行业（样本均值）偏低/接近：{cur_pb:.2f} vs {peer_pb_mean:.2f}")

    # 2) ESG风险
    with tabs[1]:
        esg_risk = detail_data["esg_risk"]
        icon_esg, _ = score_level(esg_risk)
        st.metric("ESG风险得分", f"{icon_esg} {esg_risk:.1f}")
        esg_meta = detail_data.get("esg_detail") or {}
        esg_row = detail_data.get("esg_rating_row")

        with st.expander("ESG 风险得分计算方法（数据与公式）", expanded=True):
            st.markdown(
                """
**一、数据来源（均为公开信息抓取，非手工编造）**
- **华证 ESG 评级**：新浪财经源，经 `akshare.stock_esg_hz_sina` 获取全市场表后，按 `股票代码`（如 `600519.SH`）匹配当前个股；字段包括综合分「ESG评分」及「环境 / 社会 / 公司治理」分项。**得分越高表示 ESG 表现越好**。
- **新闻标题**：东方财富等渠道（见数据模块 `fetch_stock_news_titles`），用于捕捉近期与 ESG 相关的负面舆情事件。

**二、分层计算**
1. **评级风险** `R_rating`（0~100，越高越不利）：将「表现分」转为「风险分」  
   - 若环境、社会、治理三项齐全：  
     `R_rating = 0.35×(100−S_env) + 0.30×(100−S_soc) + 0.35×(100−S_gov)`  
   - 否则若仅有综合分 `S_all`：  
     `R_rating = 100 − S_all`  
   - 若仅部分分项：对已有分项取 `(100−S)` 的**简单平均**。
2. **新闻风险** `R_news`：对标题匹配预设负面关键词，权重累加为 `W`，则  
   `R_news = clamp(40 + min(W, 60))`；无匹配时 `W=0` → `R_news=40`（中性）。

**三、融合（本系统采用的当下口径）**
- 评级与新闻**均可用时**：`R_ESG = clamp(0.55×R_rating + 0.45×R_news)`（评级偏中长期结构，新闻偏短期事件）。
- **仅新闻可用**：`R_ESG = R_news`；**仅评级可用**（新闻层仍为中性 40 时效果同上式，以评级为主）。

**说明**：第三方评级更新频率与覆盖范围以数据源为准；若表中无该股票，则自动退回「仅新闻层」。
                """
            )

        if esg_row:
            st.subheader("华证 ESG 原始评级（匹配结果）")
            show = {
                "股票": f"{esg_row.get('stock_name', '')} ({esg_row.get('symbol', '')})",
                "ESG评分(越高越好)": esg_row.get("esg_score"),
                "ESG等级": esg_row.get("esg_grade"),
                "环境": esg_row.get("env_score"),
                "社会": esg_row.get("social_score"),
                "公司治理": esg_row.get("gov_score"),
                "数据源": esg_row.get("data_source", ""),
            }
            st.json(show)
        else:
            st.warning("当前全市场 ESG 表中未匹配到该股票代码（或接口暂时失败）。已仅使用新闻层计算。")

        rd = esg_meta.get("rating_detail") or {}
        c1, c2, c3 = st.columns(3)
        if esg_meta.get("r_rating") is not None:
            c1.metric("评级层 R_rating", f"{esg_meta['r_rating']:.1f}")
        else:
            c1.metric("评级层 R_rating", "N/A")
        c2.metric("新闻层 R_news", f"{esg_meta.get('r_news', 0):.1f}")
        c3.metric("融合方式", esg_meta.get("fusion", "—"))

        events = detail_data.get("esg_events", [])
        st.subheader("ESG 相关负面关键词命中（新闻事件层）")
        if not events:
            st.info("近期新闻标题中未命中预设 ESG 负面关键词（或新闻数据源受限）。评级层仍可能来自华证表。")
        else:
            for ev in events:
                title = ev.get("title", "")
                kw = ev.get("keyword", "")
                dt = ev.get("date", "")
                w = ev.get("weight", 0)
                with st.expander(f"{dt.date() if hasattr(dt,'date') else dt} · 关键词：{kw} · 权重 {w}"):
                    st.write(title)

    # 3) 供应链风险
    with tabs[2]:
        supply_risk = detail_data["supply_risk"]
        icon_supply, _ = score_level(supply_risk)
        st.metric("供应链风险得分", f"{icon_supply} {supply_risk:.1f}")
        st.caption(
            "供应链图谱与评分：由于稳定的公司级上下游穿透数据接口缺失，这里使用“典型行业链条节点示例”（节点是具体公司名称，但不代表该公司真实合同/股权穿透关系）。"
        )
        graph_data = detail_data["supply_graph"]
        fig_graph = _build_supply_chain_plot(graph_data)
        st.plotly_chart(fig_graph, use_container_width=True)
        nodes = graph_data.get("nodes", []) if graph_data else []
        ups = [n["name"] for n in nodes if n.get("type") == "upstream"]
        downs = [n["name"] for n in nodes if n.get("type") == "downstream"]
        if ups:
            st.write("上游（示例）：" + "，".join(ups))
        if downs:
            st.write("下游（示例）：" + "，".join(downs))

    # 4) 舆情情绪
    with tabs[3]:
        sent_df = detail_data.get("sentiment_df", pd.DataFrame())
        sent_risk = detail_data["sentiment_risk"]
        icon_sent, _ = score_level(sent_risk)
        st.metric("舆情情绪风险得分", f"{icon_sent} {sent_risk:.1f}")
        if sent_df is None or sent_df.empty:
            st.info("暂无可用新闻标题，情绪趋势无法计算。")
        else:
            plot_df = sent_df.copy()
            plot_df["date"] = pd.to_datetime(plot_df["date"])
            plot_df = plot_df.tail(30)
            fig_sent = px.line(plot_df, x="date", y="sentiment_score", markers=True, title="情感得分趋势（新闻关键词打分）")
            fig_sent.update_layout(yaxis_title="情感得分(-1~1)", height=320)
            st.plotly_chart(fig_sent, use_container_width=True)

    # 5) 量化风险
    with tabs[4]:
        quant_risk = detail_data["quant_risk"]
        icon_quant, _ = score_level(quant_risk)
        st.metric("量化风险得分", f"{icon_quant} {quant_risk:.1f}")
        qd = detail_data.get("quant_detail", {}) or {}
        annual_vol = qd.get("annual_volatility", None)
        max_dd = qd.get("max_drawdown_pct", None)
        sharpe = qd.get("sharpe", None)
        beta = qd.get("beta", None)
        downside = qd.get("downside_risk_annual", None)
        val_pct = qd.get("valuation_percentile", None)

        max_dd_str = f"{max_dd:.2f}%" if isinstance(max_dd, (int, float)) and max_dd is not None else "N/A"
        st.write(f"年化波动率：`{annual_vol}`  |  最大回撤：`{max_dd_str}`  |  Sharpe：`{sharpe}`  |  Beta：`{beta}`")
        st.write(f"下行风险(年化)：`{downside}`  |  估值分位(PE/PB)：`{val_pct}`")

        st.caption("估值分位基于财务摘要中提取到的 PE/PB 历史点数计算（点数不足则显示 N/A）。")

        # 与行业基准对比（轻量：抽样同行业最多5只）
        industry = detail_data["quote"].get("industry", "") or holdings[detail_code].get("industry", "")
        if industry:
            st.subheader("与行业基准对比（抽样）")
            peer_codes = []
            if "所属行业" in spot_df.columns and "代码" in spot_df.columns:
                df_sp = spot_df.copy()
                mask = df_sp["所属行业"].astype(str) == str(industry)
                tmp_codes = df_sp.loc[mask, "代码"].astype(str).tolist()
                # 去掉 sh/sz 前缀并归一化
                for c in tmp_codes:
                    cc = normalize_stock_code(c.replace("sh", "").replace("sz", ""))
                    if cc and cc != detail_code:
                        peer_codes.append(cc)
                peer_codes = list(dict.fromkeys(peer_codes))[:5]
            if not peer_codes:
                st.info("行业同类样本不足，无法对比。")
            else:
                # 简化展示：仅对比波动率与Beta（减少 API 次数）
                vol_list = []
                beta_list = []
                for pc in peer_codes:
                    try:
                        ph = _load_stock_ohlc(pc, start_date=start_date, end_date=end_date)
                        fin_hist_pc = _load_financial_history(pc)
                        # 量化只取近1年
                        ph_1y = ph.tail(260).reset_index(drop=True) if ph is not None and not ph.empty else pd.DataFrame()
                        quant_r, quant_d = calculate_quant_risk(ph_1y, index_df, fin_hist_pc)
                        vol_list.append(float(quant_d.get("annual_volatility", np.nan)))
                        beta_list.append(float(quant_d.get("beta", np.nan)))
                    except Exception:
                        continue

                vol_peer = np.nanmean(vol_list) if vol_list else np.nan
                beta_peer = np.nanmean(beta_list) if beta_list else np.nan
                st.write(f"行业基准（样本均值）年化波动率：`{vol_peer}`  |  Beta：`{beta_peer}`")
                if "annual_volatility" in qd and pd.notna(qd.get("annual_volatility")) and pd.notna(vol_peer):
                    if float(qd["annual_volatility"]) > float(vol_peer):
                        st.warning("本股波动率高于行业抽样基准。")
                    else:
                        st.success("本股波动率低于行业抽样基准。")

    st.caption("提示：本页面用于投资风控展示，不构成投资建议。")

with st.sidebar:
    st.header("组合概览（侧边栏）")
    st.caption("市值加权风险/夏普/Beta/最大回撤来自持仓历史计算。")
    md_col1, md_col2 = st.columns(2)
    md_col1.metric("总市值", f"￥{portfolio_metrics.get('total_market_value', 0.0):,.2f}")
    md_col2.metric("加权综合风险分", wcr_display)
    md_col3, md_col4 = st.columns(2)
    md_col3.metric("组合夏普比率", sharpe_str)
    md_col4.metric("组合Beta(沪深300)", beta_str)
    st.metric("组合最大回撤", f"{portfolio_metrics.get('max_drawdown_pct', None) if portfolio_metrics.get('max_drawdown_pct', None) is not None else 'N/A'}%")

    st.divider()

    # 在页面底部补一个“预警中心”渲染段（保证其在侧边栏底部）
    st.header("实时预警中心")
    st.caption("按时间倒序展示；每条可展开归因。")
    if st.session_state.alerts:
        st.markdown("### 今日记录")
        alerts_sorted = sorted(st.session_state.alerts, key=lambda x: x.get("time", ""), reverse=True)
        for a in alerts_sorted[:50]:
            level = a.get("level", "yellow")
            icon_a = "🔴" if level == "red" else "🟡"
            msg = a.get("message", "")
            with st.expander(f"{icon_a} {a.get('name', a.get('code'))} · {a.get('time','')} · {msg}"):
                st.caption(f"规则：{a.get('rule')}")
                detail = a.get("detail", {}) or {}
                if detail:
                    st.json(detail)
                else:
                    st.info("暂无细化归因字段。")

        # 一键清空（仅同会话）
        if st.button("清空预警记录（仅同会话）"):
            st.session_state.alerts = []
            st.rerun()
    else:
        st.info("暂无预警记录。")

