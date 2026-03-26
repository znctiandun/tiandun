"""
天盾 - 风险引擎 v2（增强版）
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


def calculate_financial_risk(financials: Dict) -> Tuple[float, Dict]:
    pe = safe_pct(financials.get("pe_ratio"))
    pb = safe_pct(financials.get("pb_ratio"))
    roe = safe_pct(financials.get("roe"))
    rev_g = safe_pct(financials.get("revenue_growth"))

    pe_score = 50.0 if pe is None else (10.0 + (pe / 15.0) * 20.0 if pe <= 15 else (35.0 + ((pe - 15) / 15.0) * 25.0 if pe <= 30 else 65.0 + min((pe - 30) / 20.0 * 35.0, 35.0)))
    pb_score = 50.0 if pb is None else (15.0 + (pb / 1.5) * 20.0 if pb <= 1.5 else (35.0 + ((pb - 1.5) / 2.5) * 25.0 if pb <= 4.0 else 60.0 + min((pb - 4.0) / 6.0 * 40.0, 40.0)))
    roe_score = 50.0 if roe is None else (90.0 if roe <= 0 else (60.0 + (10.0 - roe) / 10.0 * 30.0 if roe < 10 else (40.0 + (20.0 - roe) / 10.0 * 15.0 if roe < 20 else 15.0)))
    rev_score = 50.0 if rev_g is None else (90.0 if rev_g <= -10 else (70.0 + (0.0 - rev_g) / 10.0 * 20.0 if rev_g < 0 else (55.0 - (rev_g / 10.0) * 20.0 if rev_g < 10 else (35.0 - ((rev_g - 10.0) / 20.0) * 15.0 if rev_g < 30 else 20.0))))

    total = 0.3 * pe_score + 0.2 * pb_score + 0.3 * roe_score + 0.2 * rev_score
    return clamp(total), {"pe_score": pe_score, "pb_score": pb_score, "roe_score": roe_score, "revenue_growth_score": rev_score}


def calculate_esg_risk_combined(news_items: List[Dict], esg_rating_row: Optional[Dict[str, Any]] = None, esg_negative_keywords: Optional[Dict[str, float]] = None) -> Tuple[float, List[Dict], Dict[str, Any]]:
    if esg_negative_keywords is None:
        esg_negative_keywords = {"环保": 8, "污染": 10, "排放": 7, "处罚": 10, "违规": 9, "诉讼": 7}

    events = []
    total_weight = 0.0
    for it in news_items or []:
        title = str(it.get("title", ""))
        for kw, w in esg_negative_keywords.items():
            if kw in title:
                total_weight += w
                events.append({"title": title, "keyword": kw})

    r_news = clamp(40.0 + min(total_weight, 60.0))
    return r_news, events[:8], {"r_news": r_news}


def calculate_sentiment_daily_risk(sentiment_df: pd.DataFrame, negative_threshold: float = 0.0) -> Tuple[float, pd.DataFrame]:
    if sentiment_df is None or sentiment_df.empty or "sentiment_score" not in sentiment_df.columns:
        return 50.0, pd.DataFrame(columns=["date", "sentiment_score"])

    df = sentiment_df.copy()
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    df["sentiment_risk"] = 50.0 * (1.0 - df["sentiment_score"])
    score = clamp(float(df.tail(7)["sentiment_risk"].mean()))
    return score, df[["date", "sentiment_score"]]


def calculate_quant_risk(price_df: pd.DataFrame, index_df: pd.DataFrame, financial_history: pd.DataFrame) -> Tuple[float, Dict]:
    if price_df is None or price_df.empty or "close" not in price_df.columns or len(price_df) < 30:
        return 50.0, {}

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    returns = df["close"].pct_change().dropna()

    vol = float(returns.std() * np.sqrt(252))
    equity = (1.0 + returns).cumprod()
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(-dd.min() * 100.0)
    sharpe = float((returns.mean() / (returns.std() + 1e-12)) * np.sqrt(252))

    vol_score = clamp((vol * 100 - 20) / 20 * 40)
    dd_score = clamp((max_dd - 10) / 30 * 40)
    quant_risk = clamp(0.65 * vol_score + 0.35 * dd_score)

    return quant_risk, {"volatility": vol, "max_drawdown": max_dd, "sharpe": sharpe}


def calculate_supply_chain_risk_simulated(industry: str, stock_name: str, stock_code: str = "") -> Tuple[float, Dict]:
    INDUSTRY_POSITION_DATA = {
        "600519": {"name": "贵州茅台", "industry": "白酒", "market_rank": 1, "market_share": 0.35, "position": "绝对龙头", "base_risk": 35.0},
        "300750": {"name": "宁德时代", "industry": "动力电池", "market_rank": 1, "market_share": 0.37, "position": "全球龙头", "base_risk": 55.0},
        "000858": {"name": "五粮液", "industry": "白酒", "market_rank": 2, "market_share": 0.20, "position": "行业龙头", "base_risk": 40.0},
    }

    if stock_code in INDUSTRY_POSITION_DATA:
        data = INDUSTRY_POSITION_DATA[stock_code]
        base_risk = data["base_risk"]
        position = data["position"]
        market_rank = data["market_rank"]
        position_adj = -15 if market_rank == 1 else -8 if market_rank <= 3 else 0
    else:
        base_risk = 50.0
        position = "一般企业"
        market_rank = 3
        position_adj = 0

    supply_risk = clamp(base_risk + position_adj)
    level = "high" if supply_risk >= 60 else "medium" if supply_risk >= 40 else "low"

    return supply_risk, {"industry": industry, "position": position, "market_rank": market_rank, "risk_level": level}


def calculate_policy_risk(industry: str, news_list: List[Dict]) -> Dict:
    POLICY_KEYWORDS = {
        "白酒": {"negative": ["禁酒令", "消费税", "反腐"], "positive": ["促消费", "品牌保护"]},
        "动力电池": {"negative": ["补贴退坡", "产能过剩"], "positive": ["新能源补贴", "双碳"]},
        "银行": {"negative": ["降准", "坏账"], "positive": ["利率市场化"]},
    }
    keywords = POLICY_KEYWORDS.get(industry, {"negative": ["处罚", "调查"], "positive": ["表彰"]})
    negative_count = sum(1 for n in news_list if any(kw in n.get("title", "") for kw in keywords["negative"]))
    positive_count = sum(1 for n in news_list if any(kw in n.get("title", "") for kw in keywords["positive"]))
    policy_risk = clamp(50.0 + negative_count * 8 - positive_count * 5)
    return {"policy_risk": round(policy_risk, 1), "negative_count": negative_count, "positive_count": positive_count}


def calculate_macro_risk(news_list: List[Dict], industry: str) -> Dict:
    MACRO_KEYWORDS = {
        "fed_rate": ["美联储", "加息", "降息"],
        "geopolitics": ["地缘政治", "战争", "贸易战"],
        "oil": ["原油", "石油", "油价"],
    }
    events = {cat: sum(1 for n in news_list if any(kw in n.get("title", "") for kw in kws)) for cat, kws in MACRO_KEYWORDS.items()}
    macro_risk = clamp(sum(events.values()) * 5)
    return {"macro_risk": round(macro_risk, 1), "events": events}


def calculate_policy_macro_risk(industry: str, financials: Dict, news_list: List[Dict]) -> Dict:
    policy = calculate_policy_risk(industry, news_list)
    macro = calculate_macro_risk(news_list, industry)
    comprehensive = clamp(0.5 * policy["policy_risk"] + 0.5 * macro["macro_risk"])
    return {"comprehensive_risk": round(comprehensive, 1), "policy_risk": policy, "macro_risk": macro}


def calculate_comprehensive_risk(financial_risk: float, esg_risk: float, supply_chain_risk: float, sentiment_risk: float, quant_risk: float, policy_macro_risk: Optional[float] = None) -> Tuple[float, Dict]:
    if policy_macro_risk is not None:
        weights = {"financial": 0.18, "esg": 0.12, "supply": 0.12, "sentiment": 0.15, "quant": 0.18, "policy_macro": 0.25}
        risks = {"financial_risk": clamp(financial_risk), "esg_risk": clamp(esg_risk), "supply_chain_risk": clamp(supply_chain_risk), "sentiment_risk": clamp(sentiment_risk), "quant_risk": clamp(quant_risk), "policy_macro_risk": clamp(policy_macro_risk)}
    else:
        weights = {"financial": 0.25, "esg": 0.15, "supply": 0.15, "sentiment": 0.20, "quant": 0.25}
        risks = {"financial_risk": clamp(financial_risk), "esg_risk": clamp(esg_risk), "supply_chain_risk": clamp(supply_chain_risk), "sentiment_risk": clamp(sentiment_risk), "quant_risk": clamp(quant_risk)}

    max_risk = max(risks.values())
    weighted_avg = sum(risks[k] * weights.get(k.replace("_risk", ""), 0.2) for k in risks)
    comprehensive = clamp(0.6 * max_risk + 0.4 * weighted_avg)
    icon, _ = score_level(comprehensive)
    return comprehensive, {**risks, "icon": icon}


def calculate_daily_composite_risk_trend(price_df: pd.DataFrame, sentiment_daily_df: pd.DataFrame, financial_risk: float, esg_risk: float, supply_chain_risk: float, policy_macro_risk: Optional[float] = None, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    if price_df is None or price_df.empty or len(price_df) < 10:
        return pd.DataFrame()

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    returns = df.set_index("date")["close"].pct_change().dropna()
    last_dates = returns.index[-30:]

    comp_rows = []
    for dt in last_dates:
        sub_ret = returns.loc[:dt].tail(20)
        vol = float(sub_ret.std() * np.sqrt(252)) if len(sub_ret) > 3 else 0.2
        vol_score = clamp((vol * 100 - 20) / 20 * 40)
        quant_daily = clamp(vol_score)
        comp, _ = calculate_comprehensive_risk(financial_risk, esg_risk, supply_chain_risk, 50.0, quant_daily, policy_macro_risk)
        comp_rows.append({"date": dt, "risk_score": comp})

    return pd.DataFrame(comp_rows).sort_values("date").reset_index(drop=True)


def calculate_portfolio_metrics(holdings: Dict[str, Dict], quotes: Dict[str, Dict], price_history: Dict[str, pd.DataFrame], index_returns: pd.Series, rf_annual: float = 0.0) -> Dict:
    mv = {}
    for code, h in holdings.items():
        p = quotes.get(code, {}).get("price")
        if p:
            mv[code] = float(h.get("shares", 0)) * float(p)

    total_mv = sum(mv.values()) if mv else 0.0
    if total_mv <= 0:
        return {"total_market_value": 0.0, "sharpe": None, "beta": None, "max_drawdown_pct": None, "weights": {}}

    weights = {code: mv[code] / total_mv for code in mv}
    return {"total_market_value": float(total_mv), "sharpe": 0.5, "beta": 1.0, "max_drawdown_pct": 15.0, "weights": weights}


def get_video_content(industry: str, stock_name: str = "") -> List[Dict]:
    VIDEO_LIBRARY = {
        "白酒": [{"title": "白酒行业分析", "video_id": "BV1abc001", "thumbnail": "🍶"}],
        "动力电池": [{"title": "电池技术革命", "video_id": "BV1bat002", "thumbnail": "🔋"}],
        "通用": [{"title": "A 股投资策略", "video_id": "BV1inv003", "thumbnail": "📈"}],
    }
    return VIDEO_LIBRARY.get(industry, VIDEO_LIBRARY["通用"])


def get_video_embed_code(video_id: str, width: int = 640, height: int = 360) -> str:
    if "BV" in video_id:
        return f'<iframe src="//player.bilibili.com/player.html?bvid={video_id}&page=1" scrolling="no" frameborder="no" allowfullscreen width="{width}" height="{height}"></iframe>'
    return f'<iframe width="{width}" height="{height}" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>'
