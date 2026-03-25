"""
天盾 - 风险引擎（v2：干净实现版）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float(lo)
    return float(max(lo, min(hi, x)))


def safe_pct(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return float(x)
    except Exception:
        return None


def score_level(score: float) -> Tuple[str, str]:
    if score >= 70:
        return "🔴", "red"
    if score >= 40:
        return "🟡", "yellow"
    return "🟢", "green"


def _risk_from_pe(pe: Optional[float]) -> float:
    if pe is None:
        return 50.0
    if pe <= 15:
        return 10.0 + (pe / 15.0) * 20.0
    if pe <= 30:
        return 35.0 + ((pe - 15) / 15.0) * 25.0
    return 65.0 + min((pe - 30) / 20.0 * 35.0, 35.0)


def _risk_from_pb(pb: Optional[float]) -> float:
    if pb is None:
        return 50.0
    if pb <= 1.5:
        return 15.0 + (pb / 1.5) * 20.0
    if pb <= 4.0:
        return 35.0 + ((pb - 1.5) / 2.5) * 25.0
    return 60.0 + min((pb - 4.0) / 6.0 * 40.0, 40.0)


def _risk_from_roe(roe: Optional[float]) -> float:
    if roe is None:
        return 50.0
    if roe <= 0:
        return 90.0
    if roe < 10:
        return 60.0 + (10.0 - roe) / 10.0 * 30.0
    if roe < 20:
        return 40.0 + (20.0 - roe) / 10.0 * 15.0
    return 15.0


def _risk_from_revenue_growth(g: Optional[float]) -> float:
    if g is None:
        return 50.0
    if g <= -10:
        return 90.0
    if g < 0:
        return 70.0 + (0.0 - g) / 10.0 * 20.0
    if g < 10:
        return 55.0 - (g / 10.0) * 20.0
    if g < 30:
        return 35.0 - ((g - 10.0) / 20.0) * 15.0
    return 20.0


def calculate_financial_risk(financials: Dict) -> Tuple[float, Dict]:
    pe = safe_pct(financials.get("pe_ratio"))
    pb = safe_pct(financials.get("pb_ratio"))
    roe = safe_pct(financials.get("roe"))
    rev_g = safe_pct(financials.get("revenue_growth"))

    pe_score = _risk_from_pe(pe)
    pb_score = _risk_from_pb(pb)
    roe_score = _risk_from_roe(roe)
    rev_score = _risk_from_revenue_growth(rev_g)

    total = 0.3 * pe_score + 0.2 * pb_score + 0.3 * roe_score + 0.2 * rev_score
    return clamp(total), {
        "pe_score": pe_score,
        "pb_score": pb_score,
        "roe_score": roe_score,
        "revenue_growth_score": rev_score,
        "pe": pe,
        "pb": pb,
        "roe": roe,
        "revenue_growth": rev_g,
    }


def _esg_news_layer(news_items: List[Dict], esg_negative_keywords: Dict[str, float]) -> Tuple[float, List[Dict], float]:
    """
    新闻事件层：对标题做 E/S/G 相关负面关键词加权，得到 R_news。
    公式：W = sum(w_k)；R_news = clamp(40 + min(W, 60))，无匹配时 W=0 → R_news=40（中性）。
    """
    events: List[Dict] = []
    total_weight = 0.0
    for it in news_items or []:
        title = str(it.get("title", ""))
        dt = it.get("date", None)
        for kw, w in esg_negative_keywords.items():
            if kw in title:
                total_weight += w
                events.append({"date": dt, "title": title, "keyword": kw, "weight": w})

    r_news = clamp(40.0 + min(total_weight, 60.0))
    events_sorted = sorted(events, key=lambda x: x.get("date", pd.Timestamp.min), reverse=True)
    return r_news, events_sorted[:8], total_weight


def _esg_rating_risk(esg_rating_row: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    评级层：华证 ESG 综合分与各支柱分（0~100，越高表示 ESG 表现越好）。
    转为「风险分」：R_pillar = 100 - S_pillar；再按 E/S/G 加权为 R_rating。

    权重（与常见 ESG 披露结构一致，可在界面展示）：环境 0.35、社会 0.30、治理 0.35。
    若缺少分项但存在综合分 S_all：R_rating = 100 - S_all。
    """
    detail: Dict[str, Any] = {}
    s_all = esg_rating_row.get("esg_score")
    s_e = esg_rating_row.get("env_score")
    s_s = esg_rating_row.get("social_score")
    s_g = esg_rating_row.get("gov_score")

    def _safe(x) -> Optional[float]:
        if x is None:
            return None
        try:
            v = float(x)
            if np.isnan(v):
                return None
            return v
        except Exception:
            return None

    s_all = _safe(s_all)
    s_e, s_s, s_g = _safe(s_e), _safe(s_s), _safe(s_g)

    if s_all is not None:
        detail["r_from_overall"] = float(100.0 - s_all)
    if s_e is not None:
        detail["r_env"] = float(100.0 - s_e)
    if s_s is not None:
        detail["r_social"] = float(100.0 - s_s)
    if s_g is not None:
        detail["r_governance"] = float(100.0 - s_g)

    w_e, w_s, w_g = 0.35, 0.30, 0.35
    if s_e is not None and s_s is not None and s_g is not None:
        r_rating = w_e * detail["r_env"] + w_s * detail["r_social"] + w_g * detail["r_governance"]
        detail["blend_mode"] = "pillar_weighted"
    elif s_all is not None:
        r_rating = float(detail["r_from_overall"])
        detail["blend_mode"] = "overall_only"
    else:
        parts = []
        if s_e is not None:
            parts.append(100.0 - s_e)
        if s_s is not None:
            parts.append(100.0 - s_s)
        if s_g is not None:
            parts.append(100.0 - s_g)
        if not parts:
            return None, detail
        r_rating = float(np.mean(parts))
        detail["blend_mode"] = "partial_pillars_mean"

    detail["r_rating_raw"] = float(clamp(r_rating))
    return float(clamp(r_rating)), detail


def calculate_esg_risk_combined(
    news_items: List[Dict],
    esg_rating_row: Optional[Dict[str, Any]] = None,
    esg_negative_keywords: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[Dict], Dict[str, Any]]:
    """
    ESG 风险得分（0~100，越高越不利）：

    1) **评级层 R_rating**（可选）：来自华证 ESG 公开评级；将「得分越高越好」转为风险补数后再按 E/S/G 加权。
    2) **新闻层 R_news**：负面关键词累计权重 W，R_news = clamp(40 + min(W,60))。
    3) **融合**：两者皆有时 R = clamp(0.55 * R_rating + 0.45 * R_news)；仅新闻则 R=R_news；仅评级则 R=R_rating。
       （评级偏中长期结构，新闻偏短期事件冲击，故略提高评级权重。）
    """
    if esg_negative_keywords is None:
        esg_negative_keywords = {
            "环保": 8,
            "污染": 10,
            "排放": 7,
            "碳": 5,
            "处罚": 10,
            "违规": 9,
            "诉讼": 7,
            "虚假": 12,
            "信披": 8,
            "员工伤亡": 12,
            "安全": 7,
        }

    r_news, events_sorted, w_sum = _esg_news_layer(news_items or [], esg_negative_keywords)

    meta: Dict[str, Any] = {
        "r_news": r_news,
        "news_weight_sum": w_sum,
        "news_event_count": len(events_sorted),
        "keywords_version": "esg_negative_v1",
    }

    if esg_rating_row:
        meta["esg_rating_source"] = esg_rating_row.get("data_source", "")
        meta["esg_rating_symbol"] = esg_rating_row.get("symbol", "")
        meta["esg_grade"] = esg_rating_row.get("esg_grade", "")
        r_rating, r_detail = _esg_rating_risk(esg_rating_row)
        meta["rating_detail"] = r_detail
        if r_rating is not None:
            meta["r_rating"] = r_rating
            meta["fusion"] = "0.55*r_rating + 0.45*r_news"
            final = clamp(0.55 * r_rating + 0.45 * r_news)
            meta["final_risk"] = final
            return final, events_sorted, meta
        meta["rating_unavailable"] = "华证分项与综合分均缺失或无法解析，已退回新闻层"

    meta["fusion"] = "news_only"
    meta["final_risk"] = r_news
    return r_news, events_sorted, meta


def calculate_esg_risk_from_news(news_items: List[Dict], esg_negative_keywords: Optional[Dict[str, float]] = None) -> Tuple[float, List[Dict]]:
    r, ev, _ = calculate_esg_risk_combined(news_items, None, esg_negative_keywords)
    return r, ev


def calculate_sentiment_daily_risk(sentiment_df: pd.DataFrame, negative_threshold: float = 0.0) -> Tuple[float, pd.DataFrame]:
    if sentiment_df is None or sentiment_df.empty or "sentiment_score" not in sentiment_df.columns:
        out = pd.DataFrame(columns=["date", "sentiment_score"])
        return 50.0, out

    df = sentiment_df.copy()
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    df = df.dropna(subset=["sentiment_score", "date"]).sort_values("date")
    if df.empty:
        out = pd.DataFrame(columns=["date", "sentiment_score"])
        return 50.0, out

    df["sentiment_risk"] = 50.0 * (1.0 - df["sentiment_score"])
    last_7 = df.tail(7)
    score = clamp(float(last_7["sentiment_risk"].mean()))
    return score, df[["date", "sentiment_score", "sentiment_risk"]]


def calculate_quant_risk(price_df: pd.DataFrame, index_df: pd.DataFrame, financial_history: pd.DataFrame) -> Tuple[float, Dict]:
    if price_df is None or price_df.empty or "close" not in price_df.columns or len(price_df) < 30:
        return 50.0, {}

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close", "date"]).sort_values("date").set_index("date")
    returns = df["close"].pct_change().dropna()
    if returns.empty:
        return 50.0, {}

    vol_annual = float(returns.std() * np.sqrt(252))
    equity = (1.0 + returns).cumprod()
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_drawdown_pct = float(-float(dd.min()) * 100.0)
    sharpe = float((returns.mean() / (returns.std() + 1e-12)) * np.sqrt(252))

    beta = 1.0
    downside_risk = float(returns[returns < 0].std() * np.sqrt(252)) if (returns < 0).any() else 0.0
    if index_df is not None and not index_df.empty and "close" in index_df.columns:
        idx = index_df.copy()
        idx["date"] = pd.to_datetime(idx["date"])
        idx = idx.dropna(subset=["close", "date"]).sort_values("date").set_index("date")
        idx_ret = idx["close"].pct_change().dropna()
        aligned = pd.concat([returns.rename("stk"), idx_ret.rename("idx")], axis=1, join="inner").dropna()
        if not aligned.empty:
            cov = float(np.cov(aligned["stk"].values, aligned["idx"].values)[0][1])
            var = float(np.var(aligned["idx"].values))
            beta = cov / var if var > 0 else 1.0

    valuation_percentile = None
    if financial_history is not None and not financial_history.empty:
        hist = financial_history.copy()
        for col in ["pe_ratio", "pb_ratio"]:
            if col in hist.columns:
                hist[col] = pd.to_numeric(hist[col], errors="coerce")
        pe_hist = hist["pe_ratio"].dropna() if "pe_ratio" in hist.columns else pd.Series(dtype=float)
        pb_hist = hist["pb_ratio"].dropna() if "pb_ratio" in hist.columns else pd.Series(dtype=float)
        pe_cur = float(hist["pe_ratio"].dropna().iloc[-1]) if "pe_ratio" in hist.columns and len(pe_hist) >= 1 else None
        pb_cur = float(hist["pb_ratio"].dropna().iloc[-1]) if "pb_ratio" in hist.columns and len(pb_hist) >= 1 else None
        if pe_cur is not None and len(pe_hist) >= 3:
            valuation_percentile = float((pe_hist < pe_cur).sum() / len(pe_hist) * 100.0)
        elif pb_cur is not None and len(pb_hist) >= 3:
            valuation_percentile = float((pb_hist < pb_cur).sum() / len(pb_hist) * 100.0)

    vol_score = clamp((vol_annual * 100.0 - 20.0) / 20.0 * 40.0, 0.0, 40.0)
    dd_score = clamp((max_drawdown_pct - 10.0) / 30.0 * 40.0, 0.0, 40.0)
    beta_score = clamp(abs(beta - 1.0) * 10.0, 0.0, 20.0)
    sharpe_score = clamp((-sharpe) * 10.0, 0.0, 20.0)
    downside_score = clamp((downside_risk * 100.0 - 10.0) / 30.0 * 10.0, 0.0, 10.0)

    quant_risk = clamp(0.35 * vol_score + 0.35 * dd_score + 0.15 * beta_score + 0.1 * sharpe_score + 0.05 * downside_score)
    return quant_risk, {
        "annual_volatility": vol_annual,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "beta": beta,
        "downside_risk_annual": downside_risk,
        "valuation_percentile": valuation_percentile,
        "vol_score": vol_score,
        "dd_score": dd_score,
        "beta_score": beta_score,
    }


def calculate_comprehensive_risk(financial_risk: float, esg_risk: float, supply_chain_risk: float, sentiment_risk: float, quant_risk: float) -> Tuple[float, Dict]:
    weights = {"financial_risk": 0.25, "esg_risk": 0.15, "supply_chain_risk": 0.15, "sentiment_risk": 0.20, "quant_risk": 0.25}
    risks = {
        "financial_risk": clamp(financial_risk),
        "esg_risk": clamp(esg_risk),
        "supply_chain_risk": clamp(supply_chain_risk),
        "sentiment_risk": clamp(sentiment_risk),
        "quant_risk": clamp(quant_risk),
    }
    max_risk = max(risks.values())
    weighted_avg = sum(risks[k] * weights[k] for k in risks)
    comprehensive = clamp(0.6 * max_risk + 0.4 * weighted_avg)
    icon, _ = score_level(comprehensive)
    return comprehensive, {**risks, "icon": icon}


def calculate_supply_chain_risk_simulated(industry: str, stock_name: str, stock_code: str = "") -> Tuple[float, Dict]:
    """
    供应链图谱（典型链条示例）：节点为真实公司名称（示例），风险评分为行业基准+噪声。
    """
    industry = (industry or "").strip()
    center = stock_name or stock_code or "未知公司"

    code_map: Dict[str, Dict] = {
        "600519": {
            "base": 48.0,
            "upstream": [("中粮集团", 0.30), ("茅台上游原料供应（示例）", 0.35), ("包装材料龙头（示例）", 0.35)],
            "downstream": [("京东零售", 0.40), ("天猫/阿里零售（示例）", 0.35), ("线下经销渠道（示例）", 0.25)],
        },
        "300750": {
            "base": 72.0,
            "upstream": [("赣锋锂业", 0.30), ("华友钴业", 0.20), ("中伟股份", 0.25), ("恩捷股份", 0.25)],
            "downstream": [("比亚迪", 0.40), ("宁德时代装机生态伙伴（示例）", 0.30), ("储能集成商（示例）", 0.30)],
        },
        "000858": {
            "base": 55.0,
            "upstream": [("中粮集团", 0.25), ("包装材料（示例）", 0.25), ("物流仓储（示例）", 0.25), ("酒类原料（示例）", 0.25)],
            "downstream": [("京东零售", 0.30), ("经销商网络（示例）", 0.35), ("商超与终端（示例）", 0.35)],
        },
        "601318": {
            "base": 35.0,
            "upstream": [("中国平安", 0.30), ("腾讯", 0.20), ("企业客户（示例）", 0.30), ("同业机构（示例）", 0.20)],
            "downstream": [("招商银行/同业合作（示例）", 0.40), ("个人客户（示例）", 0.30), ("企业客户（示例）", 0.30)],
        },
    }

    if stock_code in code_map:
        item = code_map[stock_code]
        base = float(item["base"])
        upstream = item["upstream"]
        downstream = item["downstream"]
        seed = abs(hash(center + stock_code)) % (2**32)
        np.random.seed(seed)
        risk = clamp(base + float(np.random.uniform(-5.0, 5.0)), 0.0, 100.0)
    else:
        industry_base = {"新能源汽车": 70.0, "动力电池": 72.0, "汽车": 68.0, "白酒": 48.0, "银行": 35.0, "保险": 40.0}
        base = float(industry_base.get(industry, 55.0))
        seed = abs(hash(center)) % (2**32)
        np.random.seed(seed)
        risk = clamp(base + float(np.random.uniform(-5.0, 5.0)), 0.0, 100.0)
        upstream = [("上游原材料供应商（示例）", 0.30), ("关键零部件供给方（示例）", 0.35), ("工艺/设备供应商（示例）", 0.35)]
        downstream = [("下游客户/渠道（示例）", 0.45), ("终端消费（示例）", 0.30), ("经销商体系（示例）", 0.25)]

    nodes: List[Dict] = [{"name": center, "type": "center"}]
    edges: List[Tuple[str, str, float]] = []
    for n, w in upstream:
        nodes.append({"name": n, "type": "upstream", "weight": float(w)})
        edges.append((center, n, float(w)))
    for n, w in downstream:
        nodes.append({"name": n, "type": "downstream", "weight": float(w)})
        edges.append((center, n, float(w)))
    return risk, {"nodes": nodes, "edges": edges, "industry": industry}


def calculate_daily_composite_risk_trend(price_df: pd.DataFrame, sentiment_daily_df: pd.DataFrame, financial_risk: float, esg_risk: float, supply_chain_risk: float, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    if price_df is None or price_df.empty or len(price_df) < 10:
        return pd.DataFrame()

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close", "date"]).sort_values("date")
    returns = df.set_index("date")["close"].pct_change().dropna()

    sentiment_df = sentiment_daily_df.copy() if sentiment_daily_df is not None else pd.DataFrame()
    if not sentiment_df.empty and "sentiment_risk" not in sentiment_df.columns and "sentiment_score" in sentiment_df.columns:
        sentiment_df["sentiment_risk"] = 50.0 * (1.0 - sentiment_df["sentiment_score"])
    if not sentiment_df.empty:
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
        sentiment_df = sentiment_df.sort_values("date").set_index("date")

    last_dates = returns.index[-30:]
    comp_rows = []
    win_vol = 20
    win_dd = 30

    for dt in last_dates:
        sub_ret = returns.loc[:dt].tail(win_vol)
        vol_annual = float(sub_ret.std() * np.sqrt(252)) if len(sub_ret) > 3 else 0.2

        sub_close = df.set_index("date").loc[:dt, "close"].tail(win_dd)
        equity = (sub_close.pct_change().fillna(0.0) + 1.0).cumprod()
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd_pct = float(-dd.min() * 100.0) if len(dd) > 0 else 0.0

        vol_score = clamp((vol_annual * 100.0 - 20.0) / 20.0 * 40.0, 0.0, 40.0)
        dd_score = clamp((max_dd_pct - 10.0) / 30.0 * 40.0, 0.0, 40.0)
        quant_daily_risk = clamp(0.65 * vol_score + 0.35 * dd_score)

        sentiment_risk_daily = 50.0
        if not sentiment_df.empty:
            if dt in sentiment_df.index:
                sentiment_risk_daily = float(sentiment_df.loc[dt, "sentiment_risk"])
            else:
                nearest = sentiment_df.index[sentiment_df.index <= dt]
                if len(nearest) > 0:
                    sentiment_risk_daily = float(sentiment_df.loc[nearest[-1], "sentiment_risk"])

        comp, _ = calculate_comprehensive_risk(
            financial_risk=financial_risk,
            esg_risk=esg_risk,
            supply_chain_risk=supply_chain_risk,
            sentiment_risk=sentiment_risk_daily,
            quant_risk=quant_daily_risk,
        )
        comp_rows.append({"date": dt, "risk_score": comp})

    return pd.DataFrame(comp_rows).sort_values("date").reset_index(drop=True)


def calculate_portfolio_metrics(holdings: Dict[str, Dict], quotes: Dict[str, Dict], price_history: Dict[str, pd.DataFrame], index_returns: pd.Series, rf_annual: float = 0.0) -> Dict:
    mv: Dict[str, float] = {}
    for code, h in holdings.items():
        p = quotes.get(code, {}).get("price")
        if p is None:
            continue
        mv[code] = float(h.get("shares", 0)) * float(p)

    total_mv = sum(mv.values()) if mv else 0.0
    if total_mv <= 0:
        return {
            "total_market_value": 0.0,
            "weighted_composite_risk": 0.0,
            "portfolio_equity": pd.Series(dtype=float),
            "sharpe": None,
            "beta": None,
            "max_drawdown_pct": None,
            "weights": {},
        }

    weights = {code: mv[code] / total_mv for code in mv}

    port_ret = None
    for code, w in weights.items():
        df = price_history.get(code)
        if df is None or df.empty or "close" not in df.columns:
            continue
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.dropna(subset=["date", "close"]).sort_values("date")
        r = tmp.set_index("date")["close"].pct_change().dropna()
        port_ret = r * w if port_ret is None else port_ret.add(r * w, fill_value=0.0)

    if port_ret is None or port_ret.empty:
        return {
            "total_market_value": float(total_mv),
            "weighted_composite_risk": 0.0,
            "portfolio_equity": pd.Series(dtype=float),
            "sharpe": None,
            "beta": None,
            "max_drawdown_pct": None,
            "weights": weights,
        }

    rf_daily = rf_annual / 252.0
    excess = port_ret - rf_daily
    sharpe = float(excess.mean() / (excess.std() + 1e-12) * np.sqrt(252))

    idx_r = index_returns.copy().dropna().sort_index() if index_returns is not None else pd.Series(dtype=float)
    aligned = pd.concat([port_ret.rename("port"), idx_r.rename("idx")], axis=1, join="inner").dropna()
    if not aligned.empty:
        cov = float(np.cov(aligned["port"].values, aligned["idx"].values)[0][1])
        var = float(np.var(aligned["idx"].values))
        beta = cov / var if var > 0 else None
    else:
        beta = None

    equity = (1.0 + port_ret.fillna(0.0)).cumprod()
    peak = np.maximum.accumulate(equity.values)
    dd = (equity.values - peak) / peak
    max_drawdown_pct = float(-dd.min() * 100.0) if len(dd) > 0 else None

    return {
        "total_market_value": float(total_mv),
        "weighted_composite_risk": None,
        "portfolio_equity": equity,
        "sharpe": sharpe,
        "beta": beta,
        "max_drawdown_pct": max_drawdown_pct,
        "weights": weights,
    }

