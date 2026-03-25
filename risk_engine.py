"""
天盾 - 风险引擎
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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


def calculate_esg_risk_from_news(
    news_items: List[Dict],
    esg_negative_keywords: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[Dict]]:
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

    events: List[Dict] = []
    total_weight = 0.0
    for it in news_items or []:
        title = str(it.get("title", ""))
        dt = it.get("date", None)
        for kw, w in esg_negative_keywords.items():
            if kw in title:
                total_weight += w
                events.append({"date": dt, "title": title, "keyword": kw, "weight": w})

    risk = clamp(40.0 + min(total_weight, 60.0))
    events_sorted = sorted(events, key=lambda x: x.get("date", pd.Timestamp.min), reverse=True)
    return risk, events_sorted[:8]


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
    weights = {
        "financial_risk": 0.25,
        "esg_risk": 0.15,
        "supply_chain_risk": 0.15,
        "sentiment_risk": 0.20,
        "quant_risk": 0.25,
    }
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
    供应链图谱：节点使用真实公司名称（示例链条参与方），风险评分由行业基准+噪声得到。
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

"""
天盾 - 风险引擎

实现目标：
- 五维风险（财务/ESG/供应链/舆情/量化）映射到 0~100
- 融合得到综合风险分（0~100）
- 组合层面计算 Sharpe/Beta/最大回撤（基于价格历史）

说明：
- 财务指标（PE/PB/ROE/营收增长）在网络受限时可能缺失；缺失维度在评分映射中按中性值处理。
- 供应链“上下游关系”的穿透数据在公开接口中不稳定，因此图谱为“公开信息的典型链条参与方示例”（但节点使用真实公司名称）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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


def calculate_esg_risk_from_news(
    news_items: List[Dict],
    esg_negative_keywords: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[Dict]]:
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

    events: List[Dict] = []
    total_weight = 0.0

    for it in news_items or []:
        title = str(it.get("title", ""))
        dt = it.get("date", None)
        for kw, w in esg_negative_keywords.items():
            if kw in title:
                total_weight += w
                events.append({"date": dt, "title": title, "keyword": kw, "weight": w})

    risk = clamp(40.0 + min(total_weight, 60.0))
    events_sorted = sorted(events, key=lambda x: x.get("date", pd.Timestamp.min), reverse=True)
    return risk, events_sorted[:8]


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


def calculate_quant_risk(
    price_df: pd.DataFrame,
    index_df: pd.DataFrame,
    financial_history: pd.DataFrame,
) -> Tuple[float, Dict]:
    if price_df is None or price_df.empty or "close" not in price_df.columns or len(price_df) < 30:
        return 50.0, {}

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close", "date"]).sort_values("date")
    df = df.set_index("date")

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
        idx = idx.dropna(subset=["close", "date"]).sort_values("date")
        idx = idx.set_index("date")
        idx_ret = idx["close"].pct_change().dropna()

        aligned = pd.concat([returns.rename("stk"), idx_ret.rename("idx")], axis=1, join="inner").dropna()
        if not aligned.empty:
            cov = float(np.cov(aligned["stk"].values, aligned["idx"].values)[0][1])
            var = float(np.var(aligned["idx"].values))
            beta = cov / var if var > 0 else 1.0

    # 估值分位：尝试从财务历史里取 pe/pb（拿不到就为 None）
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


def calculate_comprehensive_risk(
    financial_risk: float,
    esg_risk: float,
    supply_chain_risk: float,
    sentiment_risk: float,
    quant_risk: float,
) -> Tuple[float, Dict]:
    weights = {
        "financial_risk": 0.25,
        "esg_risk": 0.15,
        "supply_chain_risk": 0.15,
        "sentiment_risk": 0.20,
        "quant_risk": 0.25,
    }
    risks = {
        "financial_risk": clamp(financial_risk),
        "esg_risk": clamp(esg_risk),
        "supply_chain_risk": clamp(supply_chain_risk),
        "sentiment_risk": clamp(sentiment_risk),
        "quant_risk": clamp(quant_risk),
    }
    max_risk = max(risks.values())
    weighted_avg = sum(risks[k] * weights[k] for k in risks)
    # weights 归一化不在这里做（本质是固定比重），保持简单可解释
    comprehensive = clamp(0.6 * max_risk + 0.4 * weighted_avg)
    icon, _ = score_level(comprehensive)
    return comprehensive, {**risks, "icon": icon}


def calculate_supply_chain_risk_simulated(
    industry: str,
    stock_name: str,
    stock_code: str = "",
) -> Tuple[float, Dict]:
    """
    供应链图谱（节点为真实公司名；风险评分来自行业基准+噪声）

    由于“公司级上下游穿透合同关系”公开接口不稳定，这里用“典型链条参与方示例”替代。
    但节点使用真实公司名（尽可能贴近行业公开链条）。
    """
    industry = (industry or "").strip()
    center = stock_name or stock_code or "未知公司"

    # 以 code 为优先：保证你给的示例（如 600519）至少能显示真实公司名节点
    code_map: Dict[str, Dict] = {
        "600519": {
            "upstream": [("中粮集团", 0.30), ("中粮酒业（示例）", 0.20), ("贵州习酒集团（示例）", 0.20), ("包装材料龙头（示例）", 0.30)],
            "downstream": [("京东零售", 0.33), ("天猫超市/阿里零售（示例）", 0.34), ("线下商超/渠道（示例）", 0.33)],
            "base": 48.0,
        },
        "300750": {
            "upstream": [("宁德时代供应链关键材料（示例）", 0.25), ("锂资源龙头（示例）", 0.25), ("正极材料（示例）", 0.25), ("隔膜/电解液（示例）", 0.25)],
            "downstream": [("整车厂/新能源车（示例）", 0.45), ("储能集成商（示例）", 0.35), ("渠道/运维（示例）", 0.20)],
            "base": 72.0,
        },
        "000858": {
            "upstream": [("粮食原料供应商（示例）", 0.25), ("包装材料（示例）", 0.25), ("物流仓储（示例）", 0.25), ("玻璃/瓶盖（示例）", 0.25)],
            "downstream": [("经销商网络（示例）", 0.35), ("电商平台（示例）", 0.40), ("终端渠道（示例）", 0.25)],
            "base": 55.0,
        },
        "601318": {
            "upstream": [("资管/保险合作方（示例）", 0.30), ("企业客户（示例）", 0.30), ("科技合作方（示例）", 0.20), ("同业机构（示例）", 0.20)],
            "downstream": [("个人客户（示例）", 0.40), ("企业客户（示例）", 0.35), ("资本市场对接（示例）", 0.25)],
            "base": 35.0,
        },
        "002415": {
            "upstream": [("芯片/电子元件（示例）", 0.30), ("制造加工（示例）", 0.25), ("镜头/传感器（示例）", 0.20), ("模组供应（示例）", 0.25)],
            "downstream": [("行业集成商（示例）", 0.40), ("渠道（示例）", 0.35), ("终端客户（示例）", 0.25)],
            "base": 62.0,
        },
    }

    if stock_code in code_map:
        item = code_map[stock_code]
        upstream = item["upstream"]
        downstream = item["downstream"]
        base = float(item["base"])
    else:
        # fallback：行业基准
        industry_base = {
            "新能源汽车": 70.0,
            "动力电池": 72.0,
            "汽车": 68.0,
            "白酒": 48.0,
            "银行": 35.0,
            "保险": 40.0,
            "通用制造业": 55.0,
        }
        base = float(industry_base.get(industry, 55.0))

        seed = abs(hash(center)) % (2**32)
        np.random.seed(seed)
        noise = float(np.random.uniform(-5.0, 5.0))
        risk = clamp(base + noise, 0.0, 100.0)

        # industry fallback 使用示例节点
        upstream = [("上游原材料供应商（示例）", 0.30), ("关键零部件供给方（示例）", 0.30), ("工艺/设备供应商（示例）", 0.40)]
        downstream = [("下游客户/渠道（示例）", 0.45), ("终端消费（示例）", 0.30), ("经销商体系（示例）", 0.25)]

        nodes = [{"name": center, "type": "center"}]
        edges: List[Tuple[str, str, float]] = []
        for n, w in upstream:
            nodes.append({"name": n, "type": "upstream", "weight": float(w)})
            edges.append((center, n, float(w)))
        for n, w in downstream:
            nodes.append({"name": n, "type": "downstream", "weight": float(w)})
            edges.append((center, n, float(w)))
        return risk, {"nodes": nodes, "edges": edges, "industry": industry}

    # 用 code_map 生成图谱
    seed = abs(hash(center + stock_code)) % (2**32)
    np.random.seed(seed)
    noise = float(np.random.uniform(-5.0, 5.0))
    risk = clamp(base + noise, 0.0, 100.0)

    nodes: List[Dict] = [{"name": center, "type": "center"}]
    edges: List[Tuple[str, str, float]] = []
    for n, w in upstream:
        nodes.append({"name": n, "type": "upstream", "weight": float(w)})
        edges.append((center, n, float(w)))
    for n, w in downstream:
        nodes.append({"name": n, "type": "downstream", "weight": float(w)})
        edges.append((center, n, float(w)))

    return risk, {"nodes": nodes, "edges": edges, "industry": industry}


def calculate_daily_composite_risk_trend(
    price_df: pd.DataFrame,
    sentiment_daily_df: pd.DataFrame,
    financial_risk: float,
    esg_risk: float,
    supply_chain_risk: float,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    if price_df is None or price_df.empty or len(price_df) < 10:
        return pd.DataFrame()

    if weights is None:
        weights = {
            "financial_risk": 0.25,
            "esg_risk": 0.15,
            "supply_chain_risk": 0.15,
            "sentiment_risk": 0.20,
            "quant_risk": 0.25,
        }

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

        comp, _detail = calculate_comprehensive_risk(
            financial_risk=financial_risk,
            esg_risk=esg_risk,
            supply_chain_risk=supply_chain_risk,
            sentiment_risk=sentiment_risk_daily,
            quant_risk=quant_daily_risk,
        )
        comp_rows.append({"date": dt, "risk_score": comp})

    return pd.DataFrame(comp_rows).sort_values("date").reset_index(drop=True)


def calculate_portfolio_metrics(
    holdings: Dict[str, Dict],
    quotes: Dict[str, Dict],
    price_history: Dict[str, pd.DataFrame],
    index_returns: pd.Series,
    rf_annual: float = 0.0,
) -> Dict:
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

"""
天盾 - 风险引擎

说明（非常重要）：
- 财务/估值/ROE 等可能受 akshare 接口访问限制影响，本引擎只做“打分与融合”，不在这里编造金融数据。
- 供应链图谱由于缺少稳定的“公司级上下游穿透数据接口”，这里用于展示的上下游节点为“典型行业链条示例”（见函数注释）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
    """
    financials: {pe_ratio, pb_ratio, roe, revenue_growth}
    financial_risk = 0.3*PE_score + 0.2*PB_score + 0.3*ROE_score + 0.2*RevenueGrowth_score
    """
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


def calculate_esg_risk_from_news(
    news_items: List[Dict],
    esg_negative_keywords: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[Dict]]:
    """
    ESG风险（0-100）：
    - 关键词匹配（基于新闻标题）累加权重 W
    - esg_risk = clamp(40 + min(W, 60))
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

    events: List[Dict] = []
    total_weight = 0.0

    for it in news_items or []:
        title = str(it.get("title", ""))
        dt = it.get("date", None)
        for kw, w in esg_negative_keywords.items():
            if kw in title:
                total_weight += w
                events.append({"date": dt, "title": title, "keyword": kw, "weight": w})

    risk = 40.0 + min(total_weight, 60.0)
    risk = clamp(risk)
    events_sorted = sorted(events, key=lambda x: x.get("date", pd.Timestamp.min), reverse=True)
    return risk, events_sorted[:8]


def calculate_sentiment_daily_risk(
    sentiment_df: pd.DataFrame,
    negative_threshold: float = 0.0,
) -> Tuple[float, pd.DataFrame]:
    """
    输入 sentiment_df: columns=[date, sentiment_score]，sentiment_score 近似范围 [-1,1]
    sentiment_risk_day = 50 * (1 - sentiment_score)
    sentiment_risk_total = mean(last 7 days)
    """
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


def calculate_quant_risk(
    price_df: pd.DataFrame,
    index_df: pd.DataFrame,
    financial_history: pd.DataFrame,
) -> Tuple[float, Dict]:
    """
    quant_risk（0-100）：
    - vol_score, dd_score, beta_score, sharpe_score, downside_score 按阈值映射后加权
    """
    if price_df is None or price_df.empty or "close" not in price_df.columns or len(price_df) < 30:
        return 50.0, {}

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close", "date"]).sort_values("date")
    df = df.set_index("date")

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
        idx = idx.dropna(subset=["close", "date"]).sort_values("date")
        idx = idx.set_index("date")
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

    quant_risk = clamp(
        0.35 * vol_score + 0.35 * dd_score + 0.15 * beta_score + 0.1 * sharpe_score + 0.05 * downside_score
    )
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


def calculate_comprehensive_risk(
    financial_risk: float,
    esg_risk: float,
    supply_chain_risk: float,
    sentiment_risk: float,
    quant_risk: float,
) -> Tuple[float, Dict]:
    """
    综合风险分（0-100）：
    - max_risk = max(五维风险)
    - weighted_avg = sum(risk_i * weight_i)
    - comprehensive = clamp(0.6 * max_risk + 0.4 * weighted_avg)
    权重：financial 0.25, esg 0.15, supply 0.15, sentiment 0.20, quant 0.25
    """
    weights = {
        "financial_risk": 0.25,
        "esg_risk": 0.15,
        "supply_chain_risk": 0.15,
        "sentiment_risk": 0.20,
        "quant_risk": 0.25,
    }
    risks = {
        "financial_risk": clamp(financial_risk),
        "esg_risk": clamp(esg_risk),
        "supply_chain_risk": clamp(supply_chain_risk),
        "sentiment_risk": clamp(sentiment_risk),
        "quant_risk": clamp(quant_risk),
    }

    max_risk = max(risks.values())
    weighted_avg = sum(risks[k] * weights[k] for k in risks)
    comprehensive = 0.6 * max_risk + 0.4 * weighted_avg
    comprehensive = clamp(comprehensive)

    icon, _ = score_level(comprehensive)
    return comprehensive, {**risks, "icon": icon}


def calculate_supply_chain_risk_simulated(industry: str, stock_name: str) -> Tuple[float, Dict]:
    """
    供应链风险与图谱（展示用）：
    - 由于稳定的“公司级上下游穿透关系”接口缺失，这里给出“典型行业链条节点示例”。
    - 风险分取：行业基准 + 少量噪声（确定性种子：stock_name）
    """
    industry = (industry or "").strip()
    center = stock_name or "未知公司"

    industry_base = {
        "新能源汽车": 70.0,
        "动力电池": 72.0,
        "汽车": 68.0,
        "白酒": 45.0,
        "银行": 35.0,
        "保险": 40.0,
        "通用制造业": 55.0,
    }

    base = industry_base.get(industry, 55.0)
    seed = abs(hash(center)) % (2**32)
    np.random.seed(seed)
    noise = float(np.random.uniform(-5.0, 5.0))
    risk = clamp(base + noise, 0.0, 100.0)

    def _filter_not_center(items: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        return [(n, w) for (n, w) in items if n and n != center]

    upstream: List[Tuple[str, float]] = []
    downstream: List[Tuple[str, float]] = []

    if industry in ["新能源汽车", "动力电池", "汽车"]:
        upstream = [
            ("赣锋锂业", 0.30),  # 锂资源/矿
            ("华友钴业", 0.20),  # 钴/镍资源
            ("中伟股份", 0.25),  # 正极材料
            ("恩捷股份", 0.25),  # 隔膜/关键材料
        ]
        downstream = [
            ("比亚迪", 0.40),  # 整车
            ("上汽集团", 0.30),  # 整车/渠道
            ("吉利汽车", 0.30),  # 整车/渠道
        ]
    elif industry == "白酒":
        upstream = [
            ("中粮集团", 0.30),  # 原料/供应链
            ("金龙鱼", 0.20),  # 原料粮食体系
            ("五粮液", 0.25),  # 行业共性链条（示例）
            ("贵州茅台", 0.25),  # 行业共性链条（示例）
        ]
        downstream = [
            ("永辉超市", 0.35),  # 终端渠道示例
            ("京东零售", 0.35),  # 电商渠道示例
            ("苏宁易购", 0.30),  # 零售渠道示例
        ]
    elif industry == "银行":
        upstream = [
            ("中国平安", 0.30),  # 资金/保险体系示例
            ("万科A", 0.30),  # 企业客户示例
            ("腾讯", 0.20),  # 生态/渠道示例
            ("阿里巴巴", 0.20),  # 生态/渠道示例
        ]
        downstream = [
            ("招商银行", 0.40),  # 同业/合作示例
            ("中国人寿", 0.30),  # 资金合作示例
            ("工商银行", 0.30),  # 业务对标示例
        ]
    elif industry == "保险":
        upstream = [
            ("中国人寿", 0.35),
            ("中国平安", 0.30),
            ("太平洋保险", 0.20),
            ("新华保险", 0.15),
        ]
        downstream = [
            ("支付宝", 0.35),
            ("腾讯", 0.30),
            ("银行同业合作（示例）", 0.35),
        ]
    else:
        upstream = [("关键零部件供应商（示例）", 0.30), ("上游原材料供应商（示例）", 0.30), ("设备/工艺供应商（示例）", 0.20)]
        downstream = [("下游客户/渠道（示例）", 0.40), ("终端消费（示例）", 0.35), ("经销商体系（示例）", 0.25)]

    upstream = _filter_not_center(upstream)
    downstream = _filter_not_center(downstream)

    nodes: List[Dict] = [{"name": center, "type": "center"}]
    for n, w in upstream:
        nodes.append({"name": n, "type": "upstream", "weight": float(w)})
    for n, w in downstream:
        nodes.append({"name": n, "type": "downstream", "weight": float(w)})

    edges: List[Tuple[str, str, float]] = []
    for n, w in upstream:
        edges.append((center, n, float(w)))
    for n, w in downstream:
        edges.append((center, n, float(w)))

    return risk, {"nodes": nodes, "edges": edges, "industry": industry}


def calculate_daily_composite_risk_trend(
    price_df: pd.DataFrame,
    sentiment_daily_df: pd.DataFrame,
    financial_risk: float,
    esg_risk: float,
    supply_chain_risk: float,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    近 30 日综合风险分趋势（展示用近似）：
    - quant 维度：滚动波动/滚动最大回撤映射到 quant_daily_risk
    - sentiment 维度：使用 sentiment_daily_df 的 sentiment_risk
    - financial/esg/supply：用当前常数项
    """
    if price_df is None or price_df.empty or len(price_df) < 10:
        return pd.DataFrame()

    if weights is None:
        weights = {
            "financial_risk": 0.25,
            "esg_risk": 0.15,
            "supply_chain_risk": 0.15,
            "sentiment_risk": 0.20,
            "quant_risk": 0.25,
        }

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

        vol_score = clamp((vol_annual * 100.0 - 20.0) / 20.0 * 40.0 / 1.0, 0.0, 40.0)
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

        comp, _detail = calculate_comprehensive_risk(
            financial_risk=financial_risk,
            esg_risk=esg_risk,
            supply_chain_risk=supply_chain_risk,
            sentiment_risk=sentiment_risk_daily,
            quant_risk=quant_daily_risk,
        )
        comp_rows.append({"date": dt, "risk_score": comp})

    return pd.DataFrame(comp_rows).sort_values("date").reset_index(drop=True)


def calculate_portfolio_metrics(
    holdings: Dict[str, Dict],
    quotes: Dict[str, Dict],
    price_history: Dict[str, pd.DataFrame],
    index_returns: pd.Series,
    rf_annual: float = 0.0,
) -> Dict:
    """
    组合层面：Sharpe / Beta / 最大回撤（市值权重）
    """
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

"""
天盾 - 风险引擎（重复拼接段：已不再需要未来导入）

说明：此段内容来自重复拼接，已注释掉对应的 `from __future__ import annotations`，
以避免语法错误。
"""

# from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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


def calculate_esg_risk_from_news(
    news_items: List[Dict],
    esg_negative_keywords: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[Dict]]:
    """
    ESG风险评分：
    - 标题关键词匹配的“合理模拟”（新闻标题来自真实数据，评分逻辑为规则打分）
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

    events: List[Dict] = []
    total_weight = 0.0
    for it in news_items or []:
        title = str(it.get("title", ""))
        dt = it.get("date", None)
        for kw, w in esg_negative_keywords.items():
            if kw in title:
                total_weight += w
                events.append({"date": dt, "title": title, "keyword": kw, "weight": w})

    risk = 40.0 + min(total_weight, 60.0)
    risk = clamp(risk)

    events_sorted = sorted(events, key=lambda x: x.get("date", pd.Timestamp.min), reverse=True)
    return risk, events_sorted[:8]


def calculate_sentiment_daily_risk(sentiment_df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    if sentiment_df is None or sentiment_df.empty or "sentiment_score" not in sentiment_df.columns:
        out = pd.DataFrame(columns=["date", "sentiment_score"])
        return 50.0, out

    df = sentiment_df.copy()
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    df = df.dropna(subset=["sentiment_score", "date"]).sort_values("date")
    df["sentiment_risk"] = 50.0 * (1.0 - df["sentiment_score"])

    if df.empty:
        return 50.0, pd.DataFrame(columns=["date", "sentiment_score", "sentiment_risk"])

    last_7 = df.tail(7)
    score = clamp(float(last_7["sentiment_risk"].mean()))
    return score, df[["date", "sentiment_score", "sentiment_risk"]]


def calculate_quant_risk(
    price_df: pd.DataFrame,
    index_df: pd.DataFrame,
    financial_history: pd.DataFrame,
) -> Tuple[float, Dict]:
    if price_df is None or price_df.empty or "close" not in price_df.columns or len(price_df) < 30:
        return 50.0, {}

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close", "date"]).sort_values("date")
    df = df.set_index("date")
    returns = df["close"].pct_change().dropna()
    if returns.empty:
        return 50.0, {}

    vol_annual = float(returns.std() * np.sqrt(252))
    equity = (1.0 + returns).cumprod()
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_drawdown = float(dd.min())
    max_drawdown_pct = -max_drawdown * 100.0

    sharpe = float((returns.mean() / (returns.std() + 1e-12)) * np.sqrt(252))

    beta = 1.0
    downside_risk = float(returns[returns < 0].std() * np.sqrt(252)) if (returns < 0).any() else 0.0

    if index_df is not None and not index_df.empty and "close" in index_df.columns:
        idx = index_df.copy()
        idx["date"] = pd.to_datetime(idx["date"])
        idx = idx.dropna(subset=["close", "date"]).sort_values("date")
        idx = idx.set_index("date")
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

        pe_cur = float(hist["pe_ratio"].dropna().iloc[-1]) if len(pe_hist) >= 1 else None
        pb_cur = float(hist["pb_ratio"].dropna().iloc[-1]) if len(pb_hist) >= 1 else None
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


def calculate_comprehensive_risk(
    financial_risk: float,
    esg_risk: float,
    supply_chain_risk: float,
    sentiment_risk: float,
    quant_risk: float,
) -> Tuple[float, Dict]:
    weights = {
        "financial_risk": 0.25,
        "esg_risk": 0.15,
        "supply_chain_risk": 0.15,
        "sentiment_risk": 0.20,
        "quant_risk": 0.25,
    }
    risks = {
        "financial_risk": clamp(financial_risk),
        "esg_risk": clamp(esg_risk),
        "supply_chain_risk": clamp(supply_chain_risk),
        "sentiment_risk": clamp(sentiment_risk),
        "quant_risk": clamp(quant_risk),
    }

    max_risk = max(risks.values())
    weighted_avg = sum(risks[k] * weights[k] for k in risks) / sum(weights.values())
    comprehensive = 0.6 * max_risk + 0.4 * weighted_avg
    comprehensive = clamp(comprehensive)

    icon, _ = score_level(comprehensive)
    return comprehensive, {**risks, "icon": icon}


def calculate_supply_chain_risk_simulated(industry: str, stock_name: str) -> Tuple[float, Dict]:
    industry = (industry or "").strip()
    industry_base = {
        "新能源汽车": 70.0,
        "动力电池": 72.0,
        "汽车": 68.0,
        "白酒": 45.0,
        "银行": 35.0,
        "保险": 40.0,
        "通用制造业": 55.0,
    }

    base = industry_base.get(industry, 55.0)
    seed = abs(hash(stock_name)) % (2**32)
    np.random.seed(seed)
    noise = float(np.random.uniform(-5.0, 5.0))
    risk = clamp(base + noise, 0.0, 100.0)

    upstream = []
    downstream = []
    if industry in ["新能源汽车", "动力电池", "汽车"]:
        upstream = [("锂矿/资源商", 0.30), ("正负极材料", 0.25), ("电池制造", 0.20), ("电子元件", 0.15)]
        downstream = [("整车厂", 0.40), ("渠道/经销商", 0.30), ("电商/后市场", 0.20)]
    elif industry in ["白酒"]:
        upstream = [("粮食种植/原料", 0.25), ("包装材料", 0.20)]
        downstream = [("经销商渠道", 0.35), ("终端消费", 0.25)]
    else:
        upstream = [("上游原材料供应商", 0.30), ("关键零部件供给方", 0.25)]
        downstream = [("下游客户/渠道", 0.40)]

    center = {"name": stock_name or "未知公司", "type": "center"}
    nodes = [{"name": center["name"], "type": "center"}]
    for u, w in upstream:
        nodes.append({"name": u, "type": "upstream", "weight": w})
    for d, w in downstream:
        nodes.append({"name": d, "type": "downstream", "weight": w})

    edges = []
    for u, w in upstream:
        edges.append((center["name"], u, w))
    for d, w in downstream:
        edges.append((center["name"], d, w))

    return risk, {"nodes": nodes, "edges": edges, "industry": industry}


def calculate_daily_composite_risk_trend(
    price_df: pd.DataFrame,
    sentiment_daily_df: pd.DataFrame,
    financial_risk: float,
    esg_risk: float,
    supply_chain_risk: float,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    if price_df is None or price_df.empty or len(price_df) < 10:
        return pd.DataFrame()

    if weights is None:
        weights = {
            "financial_risk": 0.25,
            "esg_risk": 0.15,
            "supply_chain_risk": 0.15,
            "sentiment_risk": 0.20,
            "quant_risk": 0.25,
        }

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close", "date"]).sort_values("date")
    returns = df.set_index("date")["close"].pct_change().dropna()

    sentiment_df = sentiment_daily_df.copy() if sentiment_daily_df is not None else pd.DataFrame()
    if not sentiment_df.empty and "date" in sentiment_df.columns and "sentiment_risk" not in sentiment_df.columns:
        if "sentiment_score" in sentiment_df.columns:
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

        comp, _detail = calculate_comprehensive_risk(
            financial_risk=financial_risk,
            esg_risk=esg_risk,
            supply_chain_risk=supply_chain_risk,
            sentiment_risk=sentiment_risk_daily,
            quant_risk=quant_daily_risk,
        )
        comp_rows.append({"date": dt, "risk_score": comp})

    out = pd.DataFrame(comp_rows).sort_values("date")
    return out.reset_index(drop=True)


def calculate_portfolio_metrics(
    holdings: Dict[str, Dict],
    quotes: Dict[str, Dict],
    price_history: Dict[str, pd.DataFrame],
    index_returns: pd.Series,
    rf_annual: float = 0.0,
) -> Dict:
    mv = {}
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
        if port_ret is None:
            port_ret = r * w
        else:
            port_ret = port_ret.add(r * w, fill_value=0.0)

    if port_ret is None or port_ret.empty:
        return {
            "total_market_value": total_mv,
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

"""
天盾 - 风险引擎

目标：
- 将财务/ESG/供应链/舆情/量化等维度，融合为 `综合风险分(0-100)`。
- 计算组合层面的市值加权风险、夏普、Beta、最大回撤。
"""

# from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
    """
    返回： (图标, 颜色标签)
    """
    if score >= 70:
        return "🔴", "red"
    if score >= 40:
        return "🟡", "yellow"
    return "🟢", "green"


def _risk_from_pe(pe: Optional[float]) -> float:
    if pe is None:
        return 50.0
    # 粗略分段：PE越高风险越高
    if pe <= 15:
        return 10.0 + (pe / 15.0) * 20.0  # 约 10~30
    if pe <= 30:
        return 35.0 + ((pe - 15) / 15.0) * 25.0  # 约 35~60
    return 65.0 + min((pe - 30) / 20.0 * 35.0, 35.0)  # 约 65~100


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
    # ROE越低风险越高（ROE为负时显著增风险）
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
    # 营收增长为负风险更高
    if g <= -10:
        return 90.0
    if g < 0:
        return 70.0 + (0.0 - g) / 10.0 * 20.0
    if g < 10:
        return 55.0 - (g / 10.0) * 20.0  # 55~35
    if g < 30:
        return 35.0 - ((g - 10.0) / 20.0) * 15.0  # 35~20
    return 20.0


def calculate_financial_risk(financials: Dict) -> Tuple[float, Dict]:
    """
    financials: {pe_ratio, pb_ratio, roe, revenue_growth}
    """
    pe = safe_pct(financials.get("pe_ratio"))
    pb = safe_pct(financials.get("pb_ratio"))
    roe = safe_pct(financials.get("roe"))
    rev_g = safe_pct(financials.get("revenue_growth"))

    pe_score = _risk_from_pe(pe)
    pb_score = _risk_from_pb(pb)
    roe_score = _risk_from_roe(roe)
    rev_score = _risk_from_revenue_growth(rev_g)

    # 权重：PE/PB/ROE/营收增长
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


def calculate_esg_risk_from_news(
    news_items: List[Dict],
    esg_negative_keywords: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[Dict]]:
    """
    ESG风险评分：
    - 默认基于新闻标题关键词出现情况进行打分（新闻标题来自真实数据，评分逻辑为合理“模拟”）
    - 返回 (esg_risk_score, esg_event_list)
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

    events: List[Dict] = []
    total_weight = 0.0

    for it in news_items or []:
        title = str(it.get("title", ""))
        dt = it.get("date", None)
        for kw, w in esg_negative_keywords.items():
            if kw in title:
                total_weight += w
                events.append(
                    {
                        "date": dt,
                        "title": title,
                        "keyword": kw,
                        "weight": w,
                    }
                )

    risk = 40.0 + min(total_weight, 60.0)
    risk = clamp(risk)

    events_sorted = sorted(
        events,
        key=lambda x: x.get("date", pd.Timestamp.min),
        reverse=True,
    )
    return risk, events_sorted[:8]


def calculate_sentiment_daily_risk(
    sentiment_df: pd.DataFrame,
) -> Tuple[float, pd.DataFrame]:
    """
    sentiment_df: columns = [date, sentiment_score]
    返回： (sentiment_risk_score(0-100), sentiment_df_with_risk)
    """
    if sentiment_df is None or sentiment_df.empty or "sentiment_score" not in sentiment_df.columns:
        out = pd.DataFrame(columns=["date", "sentiment_score"])
        return 50.0, out

    df = sentiment_df.copy()
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    df = df.dropna(subset=["sentiment_score", "date"]).sort_values("date")
    df["sentiment_risk"] = 50.0 * (1.0 - df["sentiment_score"])

    if df.empty:
        return 50.0, pd.DataFrame(columns=["date", "sentiment_score", "sentiment_risk"])

    last_7 = df.tail(7)
    score = clamp(float(last_7["sentiment_risk"].mean()))
    return score, df[["date", "sentiment_score", "sentiment_risk"]]


def calculate_quant_risk(
    price_df: pd.DataFrame,
    index_df: pd.DataFrame,
    financial_history: pd.DataFrame,
) -> Tuple[float, Dict]:
    """
    量化风险(0-100) + 关键量化指标（用于展示与预警）
    """
    if price_df is None or price_df.empty or "close" not in price_df.columns or len(price_df) < 30:
        return 50.0, {}

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close", "date"]).sort_values("date")
    df = df.set_index("date")

    returns = df["close"].pct_change().dropna()
    if returns.empty:
        return 50.0, {}

    # 年化波动率
    vol_annual = float(returns.std() * np.sqrt(252))

    # 最大回撤
    equity = (1.0 + returns).cumprod()
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak  # <=0
    max_drawdown = float(dd.min())
    max_drawdown_pct = -max_drawdown * 100.0

    # 夏普（简化：rf=0）
    sharpe = float((returns.mean() / (returns.std() + 1e-12)) * np.sqrt(252))

    # Beta vs 沪深300
    beta = 1.0
    downside_risk = float(returns[returns < 0].std() * np.sqrt(252)) if (returns < 0).any() else 0.0

    if index_df is not None and not index_df.empty and "close" in index_df.columns:
        idx = index_df.copy()
        idx["date"] = pd.to_datetime(idx["date"])
        idx = idx.dropna(subset=["close", "date"]).sort_values("date")
        idx = idx.set_index("date")
        idx_ret = idx["close"].pct_change().dropna()

        aligned = pd.concat([returns.rename("stk"), idx_ret.rename("idx")], axis=1, join="inner").dropna()
        if not aligned.empty:
            cov = float(np.cov(aligned["stk"].values, aligned["idx"].values)[0][1])
            var = float(np.var(aligned["idx"].values))
            beta = cov / var if var > 0 else 1.0

    # 估值分位数：PE/PB 历史分位（越高 => 风险越高）
    valuation_percentile = None
    if financial_history is not None and not financial_history.empty:
        hist = financial_history.copy()
        for col in ["pe_ratio", "pb_ratio"]:
            if col in hist.columns:
                hist[col] = pd.to_numeric(hist[col], errors="coerce")
        pe_hist = hist["pe_ratio"].dropna() if "pe_ratio" in hist.columns else pd.Series(dtype=float)
        pb_hist = hist["pb_ratio"].dropna() if "pb_ratio" in hist.columns else pd.Series(dtype=float)

        pe_cur = float(hist["pe_ratio"].dropna().iloc[-1]) if len(pe_hist) >= 1 else None
        pb_cur = float(hist["pb_ratio"].dropna().iloc[-1]) if len(pb_hist) >= 1 else None
        if pe_cur is not None and len(pe_hist) >= 3:
            valuation_percentile = float((pe_hist < pe_cur).sum() / len(pe_hist) * 100.0)
        elif pb_cur is not None and len(pb_hist) >= 3:
            valuation_percentile = float((pb_hist < pb_cur).sum() / len(pb_hist) * 100.0)

    # 将量化指标映射为 0-100
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


def calculate_comprehensive_risk(
    financial_risk: float,
    esg_risk: float,
    supply_chain_risk: float,
    sentiment_risk: float,
    quant_risk: float,
) -> Tuple[float, Dict]:
    """
    综合风险（0-100）
    """
    weights = {
        "financial_risk": 0.25,
        "esg_risk": 0.15,
        "supply_chain_risk": 0.15,
        "sentiment_risk": 0.20,
        "quant_risk": 0.25,
    }
    risks = {
        "financial_risk": clamp(financial_risk),
        "esg_risk": clamp(esg_risk),
        "supply_chain_risk": clamp(supply_chain_risk),
        "sentiment_risk": clamp(sentiment_risk),
        "quant_risk": clamp(quant_risk),
    }

    max_risk = max(risks.values())
    weighted_avg = sum(risks[k] * weights[k] for k in risks) / sum(weights.values())
    comprehensive = 0.6 * max_risk + 0.4 * weighted_avg
    comprehensive = clamp(comprehensive)

    icon, _ = score_level(comprehensive)
    return comprehensive, {**risks, "icon": icon}


def calculate_supply_chain_risk_simulated(industry: str, stock_name: str) -> Tuple[float, Dict]:
    """
    供应链风险（模拟逻辑，基于行业归类）：
    - 注释：公开股级上下游“穿透式关系”可用接口有限，因此供应链图谱与评分用行业映射做合理模拟。
    """
    industry = (industry or "").strip()

    industry_base = {
        "新能源汽车": 70.0,
        "动力电池": 72.0,
        "汽车": 68.0,
        "白酒": 45.0,
        "银行": 35.0,
        "保险": 40.0,
        "通用制造业": 55.0,
    }

    base = industry_base.get(industry, 55.0)
    seed = abs(hash(stock_name)) % (2**32)
    np.random.seed(seed)
    noise = float(np.random.uniform(-5.0, 5.0))
    risk = clamp(base + noise, 0.0, 100.0)

    upstream = []
    downstream = []
    if industry in ["新能源汽车", "动力电池", "汽车"]:
        upstream = [("锂矿/资源商", 0.30), ("正负极材料", 0.25), ("电池制造", 0.20), ("电子元件", 0.15)]
        downstream = [("整车厂", 0.40), ("渠道/经销商", 0.30), ("电商/后市场", 0.20)]
    elif industry in ["白酒"]:
        upstream = [("粮食种植/原料", 0.25), ("包装材料", 0.20)]
        downstream = [("经销商渠道", 0.35), ("终端消费", 0.25)]
    else:
        upstream = [("上游原材料供应商", 0.30), ("关键零部件供给方", 0.25)]
        downstream = [("下游客户/渠道", 0.40)]

    center = {"name": stock_name or "未知公司", "type": "center"}
    nodes = [{"name": center["name"], "type": "center"}]
    for u, w in upstream:
        nodes.append({"name": u, "type": "upstream", "weight": w})
    for d, w in downstream:
        nodes.append({"name": d, "type": "downstream", "weight": w})

    edges = []
    for u, w in upstream:
        edges.append((center["name"], u, w))
    for d, w in downstream:
        edges.append((center["name"], d, w))

    return risk, {"nodes": nodes, "edges": edges, "industry": industry}


def calculate_daily_composite_risk_trend(
    price_df: pd.DataFrame,
    sentiment_daily_df: pd.DataFrame,
    financial_risk: float,
    esg_risk: float,
    supply_chain_risk: float,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    近 30 日综合风险分趋势：
    - 量化维度用滚动波动率与滚动最大回撤做动态近似
    - 金融/ESG/供应链用最新常数项（可在后续扩展为更细粒度历史）
    """
    if price_df is None or price_df.empty or len(price_df) < 10:
        return pd.DataFrame()

    if weights is None:
        weights = {
            "financial_risk": 0.25,
            "esg_risk": 0.15,
            "supply_chain_risk": 0.15,
            "sentiment_risk": 0.20,
            "quant_risk": 0.25,
        }

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close", "date"]).sort_values("date")
    returns = df.set_index("date")["close"].pct_change().dropna()

    sentiment_df = sentiment_daily_df.copy() if sentiment_daily_df is not None else pd.DataFrame()
    if not sentiment_df.empty and "date" in sentiment_df.columns and "sentiment_risk" not in sentiment_df.columns:
        if "sentiment_score" in sentiment_df.columns:
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

        comp, _detail = calculate_comprehensive_risk(
            financial_risk=financial_risk,
            esg_risk=esg_risk,
            supply_chain_risk=supply_chain_risk,
            sentiment_risk=sentiment_risk_daily,
            quant_risk=quant_daily_risk,
        )
        comp_rows.append({"date": dt, "risk_score": comp})

    out = pd.DataFrame(comp_rows).sort_values("date")
    return out.reset_index(drop=True)


def calculate_portfolio_metrics(
    holdings: Dict[str, Dict],
    quotes: Dict[str, Dict],
    price_history: Dict[str, pd.DataFrame],
    index_returns: pd.Series,
    rf_annual: float = 0.0,
) -> Dict:
    """
    组合层面：Sharpe / Beta / 最大回撤（市值权重）
    """
    mv = {}
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

    # 组合日收益（静态权重）
    port_ret = None
    for code, w in weights.items():
        df = price_history.get(code)
        if df is None or df.empty or "close" not in df.columns:
            continue
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.dropna(subset=["date", "close"]).sort_values("date")
        r = tmp.set_index("date")["close"].pct_change().dropna()
        if port_ret is None:
            port_ret = r * w
        else:
            port_ret = port_ret.add(r * w, fill_value=0.0)

    if port_ret is None or port_ret.empty:
        return {
            "total_market_value": total_mv,
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

    # Beta（与基准收益对齐）
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

"""
天盾 - 风险引擎

目标：
- 将财务/ESG/供应链/舆情/量化等维度，融合为 `综合风险分(0-100)`。
- 计算组合层面的市值加权风险、夏普、Beta、最大回撤。
"""

# from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    """
    返回： (图标, 颜色标签)
    """
    if score >= 70:
        return "🔴", "red"
    if score >= 40:
        return "🟡", "yellow"
    return "🟢", "green"


def _risk_from_pe(pe: Optional[float]) -> float:
    if pe is None:
        return 50.0
    # 粗略分段：PE越高风险越高
    if pe <= 15:
        return 10.0 + (pe / 15.0) * 20.0  # 约 10~30
    if pe <= 30:
        return 35.0 + ((pe - 15) / 15.0) * 25.0  # 约 35~60
    return 65.0 + min((pe - 30) / 20.0 * 35.0, 35.0)  # 约 65~100


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
    # ROE越低风险越高（ROE为负时显著增风险）
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
    # 营收增长为负风险更高
    if g <= -10:
        return 90.0
    if g < 0:
        return 70.0 + (0.0 - g) / 10.0 * 20.0
    if g < 10:
        return 55.0 - (g / 10.0) * 20.0  # 55~35
    if g < 30:
        return 35.0 - ((g - 10.0) / 20.0) * 15.0  # 35~20
    return 20.0


def calculate_financial_risk(financials: Dict) -> Tuple[float, Dict]:
    """
    financials: {pe_ratio, pb_ratio, roe, revenue_growth}
    """
    pe = safe_pct(financials.get("pe_ratio"))
    pb = safe_pct(financials.get("pb_ratio"))
    roe = safe_pct(financials.get("roe"))
    rev_g = safe_pct(financials.get("revenue_growth"))

    pe_score = _risk_from_pe(pe)
    pb_score = _risk_from_pb(pb)
    roe_score = _risk_from_roe(roe)
    rev_score = _risk_from_revenue_growth(rev_g)

    # 权重：PE/PB/ROE/营收增长
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


def calculate_esg_risk_from_news(
    news_items: List[Dict],
    esg_negative_keywords: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[Dict]]:
    """
    ESG风险评分：
    - 默认基于新闻标题关键词出现情况进行打分（新闻标题来自真实数据，评分逻辑是“合理模拟”。）
    - 返回 (esg_risk_score, esg_event_list)
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

    events: List[Dict] = []
    total_weight = 0.0

    for it in news_items or []:
        title = str(it.get("title", ""))
        dt = it.get("date", None)
        matched = False
        for kw, w in esg_negative_keywords.items():
            if kw in title:
                matched = True
                total_weight += w
                events.append(
                    {
                        "date": dt,
                        "title": title,
                        "keyword": kw,
                        "weight": w,
                    }
                )
        # 没匹配也不做事
        _ = matched

    # 将权重映射到 0-100：新闻越负面，分越高
    # 经验：每条最多通常不会太多关键词，因此用一个柔和函数
    risk = 40.0 + min(total_weight, 60.0)
    risk = clamp(risk)

    # 只展示最近若干条
    events_sorted = sorted(events, key=lambda x: x.get("date", pd.Timestamp.min), reverse=True)
    return risk, events_sorted[:8]


def calculate_sentiment_daily_risk(
    sentiment_df: pd.DataFrame,
    negative_threshold: float = 0.0,
) -> Tuple[float, pd.DataFrame]:
    """
    sentiment_df: columns = [date, sentiment_score] where sentiment_score in [-1,1] (或近似)
    """
    if sentiment_df is None or sentiment_df.empty or "sentiment_score" not in sentiment_df.columns:
        # 中性降级
        base = 50.0
        out = pd.DataFrame({"date": [], "sentiment_score": []})
        return base, out

    df = sentiment_df.copy()
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    df = df.dropna(subset=["sentiment_score", "date"]).sort_values("date")

    # 风险：情绪越负面越高
    # sentiment_score=1 => risk=0, sentiment_score=-1 => risk=100
    df["sentiment_risk"] = 50.0 * (1.0 - df["sentiment_score"])
    last_7 = df.tail(7)
    score = clamp(float(last_7["sentiment_risk"].mean()))
    return score, df[["date", "sentiment_score", "sentiment_risk"]]


def calculate_quant_risk(
    price_df: pd.DataFrame,
    index_df: pd.DataFrame,
    financial_history: pd.DataFrame,
) -> Tuple[float, Dict]:
    """
    量化风险(0-100) + 关键量化指标（用于展示与预警）
    """
    if price_df is None or price_df.empty or "close" not in price_df.columns or len(price_df) < 30:
        return 50.0, {}

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close", "date"]).sort_values("date")
    df = df.set_index("date")

    returns = df["close"].pct_change().dropna()
    if returns.empty:
        return 50.0, {}

    # 年化波动率
    vol_annual = float(returns.std() * np.sqrt(252))

    # 最大回撤（基于累计收益曲线）
    equity = (1.0 + returns).cumprod()
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak  # <=0
    max_drawdown = float(dd.min())  # 负数
    max_drawdown_pct = -max_drawdown * 100.0

    # 夏普（rf默认0，为简化；你也可以在后续扩展成真实无风险利率）
    sharpe = float((returns.mean() / (returns.std() + 1e-12)) * np.sqrt(252))

    # Beta vs 沪深300（尽量对齐日期）
    beta = 1.0
    downside_risk = float(returns[returns < 0].std() * np.sqrt(252)) if (returns < 0).any() else 0.0
    if index_df is not None and not index_df.empty and "close" in index_df.columns:
        idx = index_df.copy()
        idx["date"] = pd.to_datetime(idx["date"])
        idx = idx.dropna(subset=["close", "date"]).sort_values("date")
        idx = idx.set_index("date")
        idx_ret = idx["close"].pct_change().dropna()

        aligned = pd.concat([returns.rename("stk"), idx_ret.rename("idx")], axis=1, join="inner").dropna()
        if not aligned.empty:
            cov = float(np.cov(aligned["stk"].values, aligned["idx"].values)[0][1])
            var = float(np.var(aligned["idx"].values))
            beta = cov / var if var > 0 else 1.0

    # 估值分位数：PE/PB 历史分位（越高 => 风险越高）
    valuation_percentile = None
    if financial_history is not None and not financial_history.empty:
        hist = financial_history.copy()
        for col in ["pe_ratio", "pb_ratio"]:
            if col in hist.columns:
                hist[col] = pd.to_numeric(hist[col], errors="coerce")
        pe_hist = hist["pe_ratio"].dropna() if "pe_ratio" in hist.columns else pd.Series(dtype=float)
        pb_hist = hist["pb_ratio"].dropna() if "pb_ratio" in hist.columns else pd.Series(dtype=float)

        pe_cur = float(hist["pe_ratio"].dropna().iloc[-1]) if len(pe_hist) >= 1 else None
        pb_cur = float(hist["pb_ratio"].dropna().iloc[-1]) if len(pb_hist) >= 1 else None
        if pe_cur is not None and len(pe_hist) >= 3:
            valuation_percentile = float((pe_hist < pe_cur).sum() / len(pe_hist) * 100.0)
        elif pb_cur is not None and len(pb_hist) >= 3:
            valuation_percentile = float((pb_hist < pb_cur).sum() / len(pb_hist) * 100.0)

    # 将量化指标映射为 0-100
    # 经验阈值：波动 20% 左右 => 中性，40% => 高
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


def calculate_comprehensive_risk(
    financial_risk: float,
    esg_risk: float,
    supply_chain_risk: float,
    sentiment_risk: float,
    quant_risk: float,
) -> Tuple[float, Dict]:
    """
    综合风险（0-100）
    """
    weights = {
        "financial_risk": 0.25,
        "esg_risk": 0.15,
        "supply_chain_risk": 0.15,
        "sentiment_risk": 0.20,
        "quant_risk": 0.25,
    }
    risks = {
        "financial_risk": clamp(financial_risk),
        "esg_risk": clamp(esg_risk),
        "supply_chain_risk": clamp(supply_chain_risk),
        "sentiment_risk": clamp(sentiment_risk),
        "quant_risk": clamp(quant_risk),
    }

    # 非线性融合：最高风险维度权重稍放大
    max_risk = max(risks.values())
    weighted_avg = sum(risks[k] * weights[k] for k in risks) / sum(weights.values())
    comprehensive = 0.6 * max_risk + 0.4 * weighted_avg
    comprehensive = clamp(comprehensive)

    icon, _ = score_level(comprehensive)
    return comprehensive, {**risks, "icon": icon}


def calculate_supply_chain_risk_simulated(industry: str, stock_name: str) -> Tuple[float, Dict]:
    """
    供应链风险（模拟逻辑，基于行业归类）：
    - 由于公开“上下游企业到股级别”的可用接口较少，这里给出行业-风险的合理映射。
    - 风险评分会影响综合风险与供应链图谱的可视化权重。
    """
    industry = (industry or "").strip()

    industry_base = {
        "新能源汽车": 70.0,
        "动力电池": 72.0,
        "汽车": 68.0,
        "白酒": 45.0,
        "银行": 35.0,
        "保险": 40.0,
        "通用制造业": 55.0,
    }

    base = industry_base.get(industry, 55.0)
    # 行业越“链式、波动越大”，风险越高（用 stock_name 做轻微扰动避免完全一样）
    seed = abs(hash(stock_name)) % (2**32)
    np.random.seed(seed)
    noise = float(np.random.uniform(-5.0, 5.0))
    risk = clamp(base + noise, 0.0, 100.0)

    # 图谱数据（节点企业/关系：上游/中心/下游），用于 networkx + plotly 力导向图
    # 注意：这是“模拟图谱”，不代表真实供应链穿透。
    upstream = []
    downstream = []
    if industry in ["新能源汽车", "动力电池", "汽车"]:
        upstream = [("锂矿/资源商", 0.30), ("正负极材料", 0.25), ("电池制造", 0.20), ("电子元件", 0.15)]
        downstream = [("整车厂", 0.40), ("渠道/经销商", 0.30), ("电商/后市场", 0.20)]
    elif industry in ["白酒"]:
        upstream = [("粮食种植/原料", 0.25), ("包装材料", 0.20)]
        downstream = [("经销商渠道", 0.35), ("终端消费", 0.25)]
    else:
        upstream = [("上游原材料供应商", 0.30), ("关键零部件供给方", 0.25)]
        downstream = [("下游客户/渠道", 0.40)]

    center = {"name": stock_name or "未知公司", "type": "center"}
    nodes = [{"name": center["name"], "type": "center"}]
    for u, w in upstream:
        nodes.append({"name": u, "type": "upstream", "weight": w})
    for d, w in downstream:
        nodes.append({"name": d, "type": "downstream", "weight": w})

    # edges: center -> upstream/downstream
    edges = []
    for u, w in upstream:
        edges.append((center["name"], u, w))
    for d, w in downstream:
        edges.append((center["name"], d, w))

    return risk, {"nodes": nodes, "edges": edges, "industry": industry}


def calculate_supply_chain_graph_layout(graph_data: Dict):
    """
    生成 networkx 图需要的结构（由 app 用 networkx/plotly 渲染）
    """
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    return nodes, edges


def calculate_daily_composite_risk_trend(
    price_df: pd.DataFrame,
    sentiment_daily_df: pd.DataFrame,
    financial_risk: float,
    esg_risk: float,
    supply_chain_risk: float,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    近 30 日综合风险分趋势：
    - 量化维度用滚动波动率与滚动最大回撤做动态近似
    - 金融/ESG/供应链维度用最新常数项（可在后续扩展为更细粒度历史）
    """
    if price_df is None or price_df.empty or len(price_df) < 10:
        return pd.DataFrame()

    if weights is None:
        weights = {
            "financial_risk": 0.25,
            "esg_risk": 0.15,
            "supply_chain_risk": 0.15,
            "sentiment_risk": 0.20,
            "quant_risk": 0.25,
        }

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close", "date"]).sort_values("date")
    returns = df.set_index("date")["close"].pct_change().dropna()

    # 30日滚动窗：波动率/回撤
    win_vol = 20
    win_dd = 30

    comp_rows = []
    # sentiment_risk_daily: 对齐到交易日（用最近新闻日期映射）
    sentiment_df = sentiment_daily_df.copy() if sentiment_daily_df is not None else pd.DataFrame()
    if not sentiment_df.empty and "date" in sentiment_df.columns and "sentiment_risk" not in sentiment_df.columns:
        if "sentiment_score" in sentiment_df.columns:
            sentiment_df["sentiment_risk"] = 50.0 * (1.0 - sentiment_df["sentiment_score"])

    if not sentiment_df.empty:
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
        sentiment_df = sentiment_df.sort_values("date").set_index("date")

    # 截取最后30个交易日
    last_dates = returns.index[-30:]
    for dt in last_dates:
        sub_ret = returns.loc[:dt].tail(win_vol)
        vol_annual = float(sub_ret.std() * np.sqrt(252)) if len(sub_ret) > 3 else 0.2

        sub_close = df.set_index("date").loc[:dt, "close"].tail(win_dd)
        equity = (sub_close.pct_change().fillna(0.0) + 1.0).cumprod()
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd_pct = float(-dd.min() * 100.0) if len(dd) > 0 else 0.0

        # 动态量化风险：把 vol/dd 映射到 quant_risk（简化版）
        vol_score = clamp((vol_annual * 100.0 - 20.0) / 20.0 * 40.0, 0.0, 40.0)
        dd_score = clamp((max_dd_pct - 10.0) / 30.0 * 40.0, 0.0, 40.0)
        quant_daily_risk = clamp(0.65 * vol_score + 0.35 * dd_score)

        # 对齐情绪风险
        sentiment_risk_daily = 50.0
        if not sentiment_df.empty:
            if dt in sentiment_df.index:
                sentiment_risk_daily = float(sentiment_df.loc[dt, "sentiment_risk"])
            else:
                # 用最近交易前的情绪分值近似
                nearest = sentiment_df.index[sentiment_df.index <= dt]
                if len(nearest) > 0:
                    sentiment_risk_daily = float(sentiment_df.loc[nearest[-1], "sentiment_risk"])

        financial_risk_const = clamp(financial_risk)
        esg_risk_const = clamp(esg_risk)
        supply_risk_const = clamp(supply_chain_risk)

        comp, _detail = calculate_comprehensive_risk(
            financial_risk=financial_risk_const,
            esg_risk=esg_risk_const,
            supply_chain_risk=supply_risk_const,
            sentiment_risk=sentiment_risk_daily,
            quant_risk=quant_daily_risk,
        )
        comp_rows.append({"date": dt, "risk_score": comp})

    out = pd.DataFrame(comp_rows).sort_values("date")
    return out.reset_index(drop=True)


def calculate_portfolio_metrics(
    holdings: Dict[str, Dict],
    quotes: Dict[str, Dict],
    price_history: Dict[str, pd.DataFrame],
    index_returns: pd.Series,
    rf_annual: float = 0.0,
) -> Dict:
    """
    holdings: code -> {shares, cost}
    quotes: code -> {price, change_pct, name}
    price_history: code -> DataFrame(date, close)
    index_returns: HS300 returns aligned to a date index (for Beta)
    """
    # 1) 权重（按当前市值）
    mv = {}
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

    # 2) 计算组合日收益（静态权重）
    port_ret = None
    for code, w in weights.items():
        df = price_history.get(code)
        if df is None or df.empty or "close" not in df.columns:
            continue
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.dropna(subset=["date", "close"]).sort_values("date")
        r = tmp.set_index("date")["close"].pct_change().dropna()
        if port_ret is None:
            port_ret = r * w
        else:
            port_ret = port_ret.add(r * w, fill_value=0.0)

    if port_ret is None or port_ret.empty:
        return {
            "total_market_value": total_mv,
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

    # Beta：与 index_returns 对齐
    idx_r = index_returns.copy()
    idx_r = idx_r.dropna().sort_index()
    # 对齐
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
        "weighted_composite_risk": None,  # 由 app 层根据每只股票综合风险再计算
        "portfolio_equity": equity,
        "sharpe": sharpe,
        "beta": beta,
        "max_drawdown_pct": max_drawdown_pct,
        "weights": weights,
    }

