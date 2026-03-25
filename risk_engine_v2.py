<function=write_file>
<parameter=content>
"""
天盾 - 风险引擎（v2：增强版）
整合：财务风险、ESG 风险、舆情风险、量化风险、供应链风险（含行业地位）、政策与宏观风险、视频嵌入支持
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
    if esg_negative_keywords is None:
        esg_negative_keywords = {
            "环保": 8, "污染": 10, "排放": 7, "碳": 5, "处罚": 10,
            "违规": 9, "诉讼": 7, "虚假": 12, "信披": 8, "员工伤亡": 12, "安全": 7,
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
        pe_cur = float(hist["pe_ratio"].dropna().iloc[-1]) if "pe_ratio" in hist.columns and len(pe_hist) >= 1 else None
        if pe_cur is not None and len(pe_hist) >= 3:
            valuation_percentile = float((pe_hist < pe_cur).sum() / len(pe_hist) * 100.0)

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
    }


# ==================== 政策与宏观风险分析 ====================

POLICY_KEYWORDS = {
    '白酒': {'negative': ['禁酒令', '消费税', '限制三公', '反腐', '限价'], 'positive': ['促消费', '品牌保护']},
    '动力电池': {'negative': ['补贴退坡', '产能过剩', '出口限制'], 'positive': ['新能源补贴', '双碳政策']},
    '新能源汽车': {'negative': ['补贴退坡', '准入限制', '安全召回'], 'positive': ['购置税减免', '以旧换新']},
    '银行': {'negative': ['降准', '坏账监管', '资本约束'], 'positive': ['利率市场化', '金融开放']},
    '医药': {'negative': ['集采', '医保谈判', '飞行检查'], 'positive': ['创新药支持', '审批加速']},
    '科技': {'negative': ['出口管制', '实体清单', '反垄断'], 'positive': ['国产替代', '专精特新']},
    '通用': {'negative': ['处罚', '调查', '退市风险', 'ST'], 'positive': ['表彰', '龙头']}
}

MACRO_KEYWORDS = {
    'fed_rate': ['美联储', '加息', '降息', 'FOMC', '美元指数'],
    'geopolitics': ['地缘政治', '战争', '冲突', '制裁', '贸易战', '中美关系'],
    'oil': ['原油', '石油', 'OPEC', '油价'],
    'supply_chain': ['供应链中断', '缺货', '海运', '港口拥堵'],
    'inflation': ['通胀', 'CPI', 'PPI', '大宗商品'],
    'exchange_rate': ['汇率', '人民币', '贬值', '升值']
}

INDUSTRY_MACRO_SENSITIVITY = {
    '白酒': {'fed_rate': 0.3, 'geopolitics': 0.2, 'oil': 0.1, 'supply_chain': 0.2, 'inflation': 0.3, 'exchange_rate': 0.2},
    '动力电池': {'fed_rate': 0.4, 'geopolitics': 0.5, 'oil': 0.3, 'supply_chain': 0.6, 'inflation': 0.4, 'exchange_rate': 0.4},
    '新能源汽车': {'fed_rate': 0.4, 'geopolitics': 0.4, 'oil': 0.5, 'supply_chain': 0.5, 'inflation': 0.3, 'exchange_rate': 0.4},
    '银行': {'fed_rate': 0.8, 'geopolitics': 0.4, 'oil': 0.2, 'supply_chain': 0.3, 'inflation': 0.6, 'exchange_rate': 0.7},
    '医药': {'fed_rate': 0.3, 'geopolitics': 0.3, 'oil': 0.1, 'supply_chain': 0.4, 'inflation': 0.3, 'exchange_rate': 0.3},
    '科技': {'fed_rate': 0.6, 'geopolitics': 0.8, 'oil': 0.2, 'supply_chain': 0.6, 'inflation': 0.4, 'exchange_rate': 0.5},
}


def calculate_policy_risk(industry: str, news_list: List[Dict]) -> Dict:
    industry_keywords = POLICY_KEYWORDS.get(industry, POLICY_KEYWORDS['通用'])
    negative_count = sum(1 for news in news_list if any(kw in (news.get('title', '') + news.get('summary', '')) for kw in industry_keywords['negative']))
    positive_count = sum(1 for news in news_list if any(kw in (news.get('title', '') + news.get('summary', '')) for kw in industry_keywords['positive']))
    
    base_risk = 50.0 + min(negative_count * 8, 40) - min(positive_count * 5, 20)
    policy_risk = max(0, min(100, base_risk))
    
    level = 'high' if policy_risk >= 70 else 'medium' if policy_risk >= 40 else 'low'
    label = '🔴 政策风险高' if policy_risk >= 70 else '⚠️ 政策风险中等' if policy_risk >= 40 else '🟢 政策风险低'
    
    return {'policy_risk': round(policy_risk, 1), 'risk_level': level, 'risk_label': label, 'negative_count': negative_count, 'positive_count': positive_count}


def calculate_macro_risk(news_list: List[Dict], industry: str) -> Dict:
    sensitivity = INDUSTRY_MACRO_SENSITIVITY.get(industry, {'fed_rate': 0.4, 'geopolitics': 0.4, 'oil': 0.2, 'supply_chain': 0.4, 'inflation': 0.4, 'exchange_rate': 0.4})
    
    macro_events = {cat: [news for news in news_list if any(kw in (news.get('title', '') + news.get('summary', '')) for kw in keywords)] for cat, keywords in MACRO_KEYWORDS.items()}
    
    dimension_risks = {cat: {'risk': round(min(len(events) * 10, 50) * sensitivity.get(cat, 0.4) * 2, 1), 'event_count': len(events)} for cat, events in macro_events.items()}
    
    weights = {'fed_rate': 0.25, 'geopolitics': 0.20, 'oil': 0.10, 'supply_chain': 0.20, 'inflation': 0.15, 'exchange_rate': 0.10}
    macro_risk = max(0, min(100, sum(dimension_risks[cat]['risk'] * weights[cat] for cat in weights)))
    
    level = 'high' if macro_risk >= 60 else 'medium' if macro_risk >= 30 else 'low'
    label = '🔴 宏观风险高' if macro_risk >= 60 else '⚠️ 宏观风险中等' if macro_risk >= 30 else '🟢 宏观风险低'
    
    return {'macro_risk': round(macro_risk, 1), 'risk_level': level, 'risk_label': label, 'dimension_risks': dimension_risks}


def calculate_company_info_risk(financials: Dict, news_list: List[Dict]) -> Dict:
    risk_factors = []
    total_risk = 0
    
    if financials.get('pe_ratio', 0) > 50:
        risk_factors.append({'type': 'high_pe', 'description': f"市盈率{financials.get('pe_ratio', 0):.1f}倍较高", 'risk': 15})
        total_risk += 15
    if financials.get('debt_ratio', 0) > 70:
        risk_factors.append({'type': 'high_debt', 'description': f"资产负债率{financials.get('debt_ratio', 0):.1f}%较高", 'risk': 20})
        total_risk += 20
    if financials.get('profit_growth', 0) < 0:
        risk_factors.append({'type': 'profit_decline', 'description': f"净利润下滑{financials.get('profit_growth', 0):.1f}%", 'risk': 25})
        total_risk += 25
    
    company_risk = min(100, total_risk)
    level = 'high' if company_risk >= 60 else 'medium' if company_risk >= 30 else 'low'
    label = '🔴 公司风险高' if company_risk >= 60 else '⚠️ 公司风险中等' if company_risk >= 30 else '🟢 公司风险低'
    
    return {'company_risk': round(company_risk, 1), 'risk_level': level, 'risk_label': label, 'risk_factors': risk_factors}


def calculate_policy_macro_risk(industry: str, financials: Dict, news_list: List[Dict]) -> Dict:
    policy_result = calculate_policy_risk(industry, news_list)
    macro_result = calculate_macro_risk(news_list, industry)
    company_result = calculate_company_info_risk(financials, news_list)
    
    comprehensive = max(0, min(100, 0.35 * policy_result['policy_risk'] + 0.35 * macro_result['macro_risk'] + 0.30 * company_result['company_risk']))
    level = 'high' if comprehensive >= 60 else 'medium' if comprehensive >= 35 else 'low'
    emoji = '🔴' if comprehensive >= 60 else '⚠️' if comprehensive >= 35 else '🟢'
    
    return {'comprehensive_risk': round(comprehensive, 1), 'risk_level': level, 'emoji': emoji, 'policy_risk': policy_result, 'macro_risk': macro_result, 'company_risk': company_result}


# ==================== 供应链风险（含行业地位） ====================

INDUSTRY_POSITION_DATA = {
    '600519': {'name': '贵州茅台', 'industry': '白酒', 'market_rank': 1, 'market_share': 0.35, 'position': '绝对龙头', 'upstream': [('高粱种植', 0.30), ('包装材料', 0.25)], 'downstream': [('经销商', 0.40), ('电商平台', 0.30)], 'peers': ['五粮液', '泸州老窖'], 'supply_risk_base': 35.0},
    '300750': {'name': '宁德时代', 'industry': '动力电池', 'market_rank': 1, 'market_share': 0.37, 'position': '全球龙头', 'upstream': [('锂矿开采', 0.30), ('正负极材料', 0.25)], 'downstream': [('特斯拉', 0.25), ('比亚迪', 0.20)], 'peers': ['比亚迪', 'LG 新能源'], 'supply_risk_base': 55.0},
    '000858': {'name': '五粮液', 'industry': '白酒', 'market_rank': 2, 'market_share': 0.20, 'position': '行业龙头', 'upstream': [('高粱种植', 0.25), ('包装材料', 0.25)], 'downstream': [('经销商', 0.50), ('电商平台', 0.30)], 'peers': ['贵州茅台', '泸州老窖'], 'supply_risk_base': 40.0},
    '601318': {'name': '中国平安', 'industry': '保险金融', 'market_rank': 1, 'market_share': 0.15, 'position': '综合金融龙头', 'upstream': [('再保险公司', 0.30), ('IT 服务', 0.25)], 'downstream': [('个人客户', 0.60), ('企业客户', 0.40)], 'peers': ['中国人寿', '中国太保'], 'supply_risk_base': 30.0},
}


def calculate_supply_chain_risk_simulated(industry: str, stock_name: str, stock_code: str = "") -> Tuple[float, Dict]:
    center = stock_name or stock_code or "未知公司"
    
    if stock_code in INDUSTRY_POSITION_DATA:
        data = INDUSTRY_POSITION_DATA[stock_code]
        market_rank = data['market_rank']
        market_share = data['market_share']
        position = data['position']
        upstream = data['upstream']
        downstream = data['downstream']
        peers = data['peers']
        base_risk = data['supply_risk_base']
    else:
        market_rank = 3
        market_share = 0.10
        position = '主要参与者'
        upstream = [('上游原材料', 0.35), ('关键零部件', 0.35)]
        downstream = [('经销商', 0.40), ('终端客户', 0.35)]
        peers = ['同行业 A', '同行业 B']
        base_risk = 50.0
    
    position_adjustment = -15 if market_rank == 1 else -8 if market_rank <= 3 else 0
    if market_share >= 0.30: position_adjustment -= 10
    elif market_share >= 0.15: position_adjustment -= 5
    
    seed = abs(hash(str(center) + str(stock_code))) % (2**32)
    np.random.seed(seed)
    noise = float(np.random.uniform(-5.0, 5.0))
    supply_risk = clamp(base_risk + position_adjustment + noise)
    
    nodes = [{'name': center, 'type': 'center', 'position': position, 'market_rank': market_rank, 'market_share': market_share}]
    nodes += [{'name': n, 'type': 'upstream', 'weight': w} for n, w in upstream]
    nodes += [{'name': n, 'type': 'downstream', 'weight': w} for n, w in downstream]
    edges = [(center, n, w) for n, w in upstream] + [(center, n, w) for n, w in downstream]
    
    level = 'high' if supply_risk >= 60 else 'medium' if supply_risk >= 40 else 'low'
    emoji = '🔴' if supply_risk >= 60 else '⚠️' if supply_risk >= 40 else '🟢'
    
    return supply_risk, {'nodes': nodes, 'edges': edges, 'industry': industry, 'company_position': position, 'market_rank': market_rank, 'market_share': market_share, 'peers': peers, 'risk_level': level, 'emoji': emoji, 'position_adjustment': position_adjustment}


# ==================== 视频内容支持 ====================

VIDEO_CONTENT_LIBRARY = {
    '白酒': [{'title': '白酒行业深度解析', 'video_id': 'BV1 白酒分析 001', 'thumbnail': '🍶', 'summary': '分析白酒行业发展趋势'}, {'title': '贵州茅台实地探访', 'video_id': 'BV1 茅台探访 002', 'thumbnail': '🏭', 'summary': '记录酿酒工艺'}],
    '动力电池': [{'title': '动力电池技术革命', 'video_id': 'BV1 电池技术 003', 'thumbnail': '🔋', 'summary': '解析电池技术路线'}, {'title': '宁德时代工厂探秘', 'video_id': 'BV1 宁德探秘 004', 'thumbnail': '🏭', 'summary': '生产基地实地拍摄'}],
    '新能源汽车': [{'title': '新能源汽车产业全景', 'video_id': 'BV1 新能源车 005', 'thumbnail': '🚗', 'summary': '解析产业链'}],
    '银行': [{'title': '银行业数字化转型', 'video_id': 'BV1 银行数字 006', 'thumbnail': '🏦', 'summary': '分析数字化转型'}],
    '科技': [{'title': '芯片产业发展报告', 'video_id': 'BV1 芯片报告 007', 'thumbnail': '💻', 'summary': '分析芯片产业现状'}],
    '通用': [{'title': 'A 股投资策略分析', 'video_id': 'BV1 投资策略 008', 'thumbnail': '📈', 'summary': '解读投资策略'}]
}


def get_video_content(industry: str, stock_name: str = "") -> List[Dict]:
    videos = VIDEO_CONTENT_LIBRARY.get(industry or '', VIDEO_CONTENT_LIBRARY['通用']).copy()
    for v in videos: v['related_industry'] = industry; v['related_stock'] = stock_name
    return videos


def get_video_embed_code(video_id: str, width: int = 640, height: int = 360) -> str:
    if 'BV' in video_id:
        return f'<iframe src="//player.bilibili.com/player.html?bvid={video_id}&page=1" scrolling="no" frameborder="no" allowfullscreen="true" width="{width}" height="{height}"></iframe>'
    return f'<iframe width="{width}" height="{height}" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>'


# ==================== 综合风险计算 ====================

def calculate_comprehensive_risk(financial_risk: float, esg_risk: float, supply_chain_risk: float, sentiment_risk: float, quant_risk: float, policy_macro_risk: Optional[float] = None) -> Tuple[float, Dict]:
    if policy_macro_risk is not None:
        weights = {"financial_risk": 0.18, "esg_risk": 0.12, "supply_chain_risk": 0.12, "sentiment_risk": 0.15, "quant_risk": 0.18, "policy_macro_risk": 0.25}
        risks = {"financial_risk": clamp(financial_risk), "esg_risk": clamp(esg_risk), "supply_chain_risk": clamp(supply_chain_risk), "sentiment_risk": clamp(sentiment_risk), "quant_risk": clamp(quant_risk), "policy_macro_risk": clamp(policy_macro_risk)}
    else:
        weights = {"financial_risk": 0.25, "esg_risk": 0.15, "supply_chain_risk": 0.15, "sentiment_risk": 0.20, "quant_risk": 0.25}
        risks = {"financial_risk": clamp(financial_risk), "esg_risk": clamp(esg_risk), "supply_chain_risk": clamp(supply_chain_risk), "sentiment_risk": clamp(sentiment_risk), "quant_risk": clamp(quant_risk)}
    
    max_risk = max(risks.values())
    weighted_avg = sum(risks[k] * weights[k] for k in risks)
    comprehensive = clamp(0.6 * max_risk + 0.4 * weighted_avg)
    icon, _ = score_level(comprehensive)
    return comprehensive, {**risks, "icon": icon, "weights": weights}


# ==================== 风险趋势计算 ====================

def calculate_daily_composite_risk_trend(price_df: pd.DataFrame, sentiment_daily_df: pd.DataFrame, financial_risk: float, esg_risk: float, supply_chain_risk: float, policy_macro_risk: Optional[float] = None, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
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
    for dt in last_dates:
        sub_ret = returns.loc[:dt].tail(20)
        vol_annual = float(sub_ret.std() * np.sqrt(252)) if len(sub_ret) > 3 else 0.2
        sub_close = df.set_index("date").loc[:dt, "close"].tail(30)
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
        comp, _ = calculate_comprehensive_risk(financial_risk=financial_risk, esg_risk=esg_risk, supply_chain_risk=supply_chain_risk, sentiment_risk=sentiment_risk_daily, quant_risk=quant_daily_risk, policy_macro_risk=policy_macro_risk)
        comp_rows.append({"date": dt, "risk_score": comp})
    return pd.DataFrame(comp_rows).sort_values("date").reset_index(drop=True)


# ==================== 组合风险计算 ====================

def calculate_portfolio_metrics(holdings: Dict[str, Dict], quotes: Dict[str, Dict], price_history: Dict[str, pd.DataFrame], index_returns: pd.Series, rf_annual: float = 0.0) -> Dict:
    mv = {code: float(h.get("shares", 0)) * float(quotes.get(code, {}).get("price", 0)) for code, h in holdings.items() if quotes.get(code, {}).get("price")}
    total_mv = sum(mv.values()) if mv else 0.0
    if total_mv <= 0:
        return {"total_market_value": 0.0, "weighted_composite_risk": 0.0, "portfolio_equity": pd.Series(dtype=float), "sharpe": None, "beta": None, "max_drawdown_pct": None, "weights": {}}

    weights = {code: mv[code] / total_mv for code in mv}
    port_ret = None
    for code, w in weights.items():
        df = price_history.get(code)
        if df is None or df.empty or "close" not in df.columns: continue
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.dropna(subset=["date", "close"]).sort_values("date")
        r = tmp.set_index("date")["close"].pct_change().dropna()
        port_ret = r * w if port_ret is None else port_ret.add(r * w, fill_value=0.0)

    if port_ret is None or port_ret.empty:
        return {"total_market_value": float(total_mv), "weighted_composite_risk": 0.0, "portfolio_equity": pd.Series(dtype=float), "sharpe": None, "beta": None, "max_drawdown_pct": None, "weights": weights}

    rf_daily = rf_annual / 252.0
    excess = port_ret - rf_daily
    sharpe = float(excess.mean() / (excess.std() + 1e-12) * np.sqrt(252))
    idx_r = index_returns.copy().dropna().sort_index() if index_returns is not None else pd.Series(dtype=float)
    aligned = pd.concat([port_ret.rename("port"), idx_r.rename("idx")], axis=1, join="inner").dropna()
    beta = float(np.cov(aligned["port"].values, aligned["idx"].values)[0][1]) / float(np.var(aligned["idx"].values)) if not aligned.empty and np.var(aligned["idx"].values) > 0 else None
    equity = (1.0 + port_ret.fillna(0.0)).cumprod()
    peak = np.maximum.accumulate(equity.values)
    dd = (equity.values - peak) / peak
    max_drawdown_pct = float(-dd.min() * 100.0) if len(dd) > 0 else None
    return {"total_market_value": float(total_mv), "weighted_composite_risk": None, "portfolio_equity": equity, "sharpe": sharpe, "beta": beta, "max_drawdown_pct": max_drawdown_pct, "weights": weights}

</parameter>
<parameter=fileKey>
file:/tiandun/risk_engine_v2.py
</parameter>
</function>
</tool_call>
