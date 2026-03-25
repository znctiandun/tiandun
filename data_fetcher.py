"""
天盾 - 数据获取模块（真实数据优先）

说明：
- 本模块主要使用 `akshare` 获取 A 股真实数据：实时行情、历史日线、财务指标、指数基准等。
- ESG：**华证 ESG 评级**优先通过 `akshare.stock_esg_hz_sina`（新浪财经源）获取；并与公开新闻标题事件层在 `risk_engine_v2` 中融合。供应链/电话会情绪等在接口缺失时仍可能示例化并在界面注明。
- 为避免网络/接口限制导致程序崩溃：所有外部请求都做了 try/except，并返回安全的空结果（app 层会做容错展示）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import re
import os


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def normalize_stock_code(code: str) -> str:
    """
    将用户输入的股票代码归一化为 6 位纯数字（例如 '600519'）。
    支持输入包含 'sh' / 'sz' 前缀的情况。
    """
    if code is None:
        return ""
    code = str(code).strip()
    code = code.replace(" ", "")
    if code.startswith(("sh", "sz")) and len(code) >= 8:
        code = code[2:]
    # 只保留数字，避免用户误输入
    digits = "".join(ch for ch in code if ch.isdigit())
    if len(digits) != 6:
        return ""
    return digits


def to_ak_symbol(stock_code: str) -> str:
    """akshare 某些接口可能要求带市场前缀的 symbol：sh/sz + 代码。"""
    if not stock_code or len(stock_code) != 6:
        return stock_code
    if stock_code.startswith("6"):
        return f"sh{stock_code}"
    return f"sz{stock_code}"


def _safe_to_datetime(x) -> Optional[pd.Timestamp]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return pd.to_datetime(x)
    except Exception:
        return None


def _pick_first_numeric(row: pd.Series, candidates: List[str]) -> Optional[float]:
    for c in candidates:
        if c in row.index:
            v = row.get(c)
            if v is None:
                continue
            if isinstance(v, str):
                v = v.replace(",", "")
            if _is_number(str(v)):
                return float(v)
    return None


@dataclass
class NewsItem:
    date: pd.Timestamp
    title: str
    source: str = ""


class TianDunDataFetcher:
    def __init__(self, request_timeout_s: int = 8):
        self.timeout_s = request_timeout_s
        self.session = requests.Session()
        self.ts = None
        self.ts_token = None
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                )
            }
        )
        # 可从环境变量读取 token
        env_token = os.getenv("TUSHARE_TOKEN", "").strip()
        if env_token:
            self.set_tushare_token(env_token)

    # -------------------- TuShare 初始化 --------------------
    def set_tushare_token(self, token: str) -> bool:
        token = (token or "").strip()
        if not token:
            self.ts = None
            self.ts_token = None
            return False
        try:
            import tushare as ts

            ts.set_token(token)
            self.ts = ts.pro_api(token)
            self.ts_token = token
            return True
        except Exception:
            self.ts = None
            self.ts_token = None
            return False

    def _to_ts_code(self, stock_code: str) -> str:
        code = normalize_stock_code(stock_code)
        if not code:
            return ""
        suffix = ".SH" if code.startswith("6") else ".SZ"
        return f"{code}{suffix}"

    def _fetch_single_quote_tushare(self, stock_code: str) -> Optional[Dict]:
        """
        TuShare 实时行情（优先）
        返回: {code,name,price,change_pct}
        """
        if self.ts is None:
            return None
        ts_code = self._to_ts_code(stock_code)
        if not ts_code:
            return None
        try:
            import tushare as ts

            # realtime_quote 为 tuhsare 的实时接口（pro之外）
            df = ts.realtime_quote(ts_code=ts_code)
            if df is None or df.empty:
                return None
            row = df.iloc[0]
            name = str(row.get("NAME", "") or row.get("name", "")).strip()
            price_raw = row.get("PRICE", row.get("price", None))
            pre_close_raw = row.get("PRE_CLOSE", row.get("pre_close", None))
            price = float(price_raw) if _is_number(str(price_raw)) else None
            pre_close = float(pre_close_raw) if _is_number(str(pre_close_raw)) else None
            if price is not None and pre_close is not None and pre_close > 0:
                change_pct = (price - pre_close) / pre_close * 100.0
            else:
                change_pct = None
            return {
                "code": normalize_stock_code(stock_code),
                "name": name if name else stock_code,
                "industry": "",
                "price": price,
                "change_pct": change_pct,
            }
        except Exception:
            return None

    # -------------------- 行情/价格 --------------------
    def fetch_spot_quotes(self) -> pd.DataFrame:
        """
        获取全市场 A 股实时行情（一次拉取全表，app 内按 code 过滤）。
        返回：包含 '代码'、'名称'、'最新价'、'涨跌幅'、'所属行业' 等列（列名可能因 akshare 版本变化而不同）。
        """
        # TuShare 没有高效“全市场一次性实时行情”标准接口，因此这里仍优先 AkShare 的全市场快照。
        try:
            import akshare as ak

            # 尝试多个来源，尽量避免某个接口偶发/被限导致空表
            candidates = []
            for fn in [getattr(ak, "stock_zh_a_spot_em", None), getattr(ak, "stock_sh_a_spot_em", None), getattr(ak, "stock_sz_a_spot_em", None), getattr(ak, "stock_new_a_spot_em", None)]:
                if fn is None:
                    continue
                try:
                    df = fn()
                    if df is not None and len(df) > 0:
                        candidates.append(df)
                except Exception:
                    continue

            if not candidates:
                return pd.DataFrame()

            if len(candidates) == 1:
                return candidates[0]

            # 合并不同市场的数据
            spot = pd.concat(candidates, ignore_index=True).drop_duplicates()
            return spot
        except Exception:
            return pd.DataFrame()

    def get_realtime_quote_from_spot(self, spot_df: pd.DataFrame, stock_code: str) -> Dict:
        """
        从全市场 spot_df 里取单只股票的实时行情信息。
        返回安全 dict：若数据缺失则给出 name/industry 为空、价格为 None。
        """
        stock_code = normalize_stock_code(stock_code)
        # 先尝试 TuShare 单票实时
        ts_quote = self._fetch_single_quote_tushare(stock_code)
        if ts_quote is not None and ts_quote.get("price") is not None:
            return ts_quote

        if spot_df is None or spot_df.empty or not stock_code:
            name_fallback = ""
            try:
                name_fallback = self.fetch_stock_name_from_eastmoney(stock_code)
            except Exception:
                name_fallback = ""
            return {
                "code": stock_code,
                "name": name_fallback or stock_code,
                "industry": "",
                "price": None,
                "change_pct": None,
            }
        try:
            df = spot_df.copy()

            # 兼容不同版本的列名
            code_col_candidates = ["代码", "code", "股票代码", "Symbol", "sec_code", "证券代码"]
            name_col_candidates = ["名称", "name", "股票简称", "简称", "证券简称", "stock_name"]
            price_col_candidates = ["最新价", "latest_price", "close", "收盘", "当前价"]
            change_col_candidates = ["涨跌幅", "change_pct", "涨跌幅(%)", "涨跌幅%"]
            industry_col_candidates = ["所属行业", "industry", "行业"]

            def _first_present(cols: List[str]) -> Optional[str]:
                for c in cols:
                    if c in df.columns:
                        return c
                return None

            code_col = _first_present(code_col_candidates)
            name_col = _first_present(name_col_candidates)
            price_col = _first_present(price_col_candidates)
            change_col = _first_present(change_col_candidates)
            industry_col = _first_present(industry_col_candidates)

            if code_col is None or price_col is None:
                return {
                    "code": stock_code,
                    "name": stock_code,
                    "industry": "",
                    "price": None,
                    "change_pct": None,
                }

            # 兼容 code 列可能为 "sh600519"/"sz600519" 或 "600519"
            def _norm_code_in_df(x) -> str:
                s = str(x)
                s = s.replace("sh", "").replace("sz", "")
                return normalize_stock_code(s)

            df["_norm_code"] = df[code_col].apply(_norm_code_in_df)
            one = df[df["_norm_code"] == stock_code]
            if one.empty:
                return {
                    "code": stock_code,
                    "name": stock_code,
                    "industry": "",
                    "price": None,
                    "change_pct": None,
                }

            one = one.iloc[0]
            name = stock_code
            if name_col and name_col in one.index:
                v = one.get(name_col)
                name = str(v).strip() if v is not None else stock_code

            industry = ""
            if industry_col and industry_col in one.index:
                v = one.get(industry_col)
                industry = str(v).strip() if v is not None else ""

            price = one.get(price_col)
            if isinstance(price, str):
                price = price.replace(",", "").strip()
            price = float(price) if _is_number(str(price)) else None

            change_pct = None
            if change_col and change_col in one.index:
                change_pct = one.get(change_col)
                if isinstance(change_pct, str):
                    change_pct = change_pct.replace("%", "").strip()
                change_pct = float(change_pct) if _is_number(str(change_pct)) else None

            # 估值（来自实时行情表字段：市盈率-动态 / 市净率）
            pe_ratio = None
            pb_ratio = None
            pe_col_candidates = ["市盈率-动态", "市盈率(动态)", "PE(TTM)", "市盈率TTM"]
            pb_col_candidates = ["市净率", "PB(TTM)", "PB(TTM)"]
            for c in pe_col_candidates:
                if c in one.index:
                    v = one.get(c)
                    pe_ratio = float(str(v).replace(",", "")) if _is_number(str(v)) else None
                    if pe_ratio is not None:
                        break
            for c in pb_col_candidates:
                if c in one.index:
                    v = one.get(c)
                    pb_ratio = float(str(v).replace(",", "")) if _is_number(str(v)) else None
                    if pb_ratio is not None:
                        break

            return {
                "code": stock_code,
                "name": name,
                "industry": industry,
                "price": price,
                "change_pct": change_pct,
                "pe_ratio": pe_ratio,
                "pb_ratio": pb_ratio,
            }
        except Exception:
            return {
                "code": stock_code,
                "name": stock_code,
                "industry": "",
                "price": None,
                "change_pct": None,
            }

    def fetch_stock_daily_ohlc(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """
        获取股票历史日线 OHLC（真实数据）。
        返回列：date/open/high/low/close/volume（尽量统一列名）。
        """
        stock_code = normalize_stock_code(stock_code)
        if not stock_code:
            return pd.DataFrame()

        # 先尝试 TuShare 日线
        if self.ts is not None:
            try:
                ts_code = self._to_ts_code(stock_code)
                if ts_code:
                    df_ts = self.ts.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                    if df_ts is not None and not df_ts.empty:
                        # tushare: trade_date/open/high/low/close/vol
                        out = df_ts.copy()
                        out = out.rename(
                            columns={
                                "trade_date": "date",
                                "open": "open",
                                "high": "high",
                                "low": "low",
                                "close": "close",
                                "vol": "volume",
                            }
                        )
                        if "date" in out.columns:
                            out["date"] = pd.to_datetime(out["date"])
                        keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in out.columns]
                        out = out[keep].dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
                        if not out.empty:
                            return out
            except Exception:
                pass

        candidates = [stock_code, to_ak_symbol(stock_code)]
        try:
            import akshare as ak

            last_err = None
            for symbol in candidates:
                try:
                    df = ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust,
                    )
                    if df is None or len(df) == 0:
                        continue
                    # 统一列名（akshare 常见：日期/开盘/收盘/最高/最低/成交量）
                    rename_map = {
                        "日期": "date",
                        "开盘": "open",
                        "最高": "high",
                        "最低": "low",
                        "收盘": "close",
                        "成交量": "volume",
                    }
                    for k, v in rename_map.items():
                        if k in df.columns:
                            df = df.rename(columns={k: v})
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                    # 只保留需要列
                    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
                    df = df[keep].dropna(subset=["date", "close"]).sort_values("date")
                    return df.reset_index(drop=True)
                except Exception as e:
                    last_err = e
                    continue
            _ = last_err
        except Exception:
            pass
        return pd.DataFrame()

    # -------------------- 财务指标 --------------------
    def fetch_financial_abstract_raw(self, stock_code: str) -> pd.DataFrame:
        """
        获取财务摘要/关键指标（尽量用 stock_financial_abstract）。
        返回原始 DataFrame，app 层会做列映射与抽取。
        """
        stock_code = normalize_stock_code(stock_code)
        if not stock_code:
            return pd.DataFrame()
        try:
            import akshare as ak

            # stock_financial_abstract 在不同 akshare 版本中参数/字段可能略不同，因此做多候选尝试
            candidates = [stock_code, to_ak_symbol(stock_code)]
            last_err = None
            for symbol in candidates:
                try:
                    df = ak.stock_financial_abstract(symbol=symbol)
                    if df is not None and len(df) > 0:
                        return df
                except Exception as e:
                    last_err = e
                    continue
            _ = last_err
        except Exception:
            pass
        return pd.DataFrame()

    def fetch_financial_ratio_history(self, stock_code: str, max_reports: int = 10) -> pd.DataFrame:
        """
        从财务摘要原始表中抽取 PE/PB/ROE/营收增长等时间序列（尽量真实）。
        返回 DataFrame：report_date, pe_ratio, pb_ratio, roe, revenue_growth
        """
        # 优先 TuShare：日频 pe/pb + 财务指标 roe/or_yoy（尽可能）
        if self.ts is not None:
            try:
                ts_code = self._to_ts_code(stock_code)
                end = datetime.now().strftime("%Y%m%d")
                start = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y%m%d")

                # 估值：daily_basic
                db = self.ts.daily_basic(
                    ts_code=ts_code,
                    start_date=start,
                    end_date=end,
                    fields="ts_code,trade_date,pe_ttm,pb",
                )
                if db is not None and not db.empty:
                    db = db.rename(columns={"trade_date": "report_date", "pe_ttm": "pe_ratio", "pb": "pb_ratio"})
                    db["report_date"] = pd.to_datetime(db["report_date"])
                    # 财务：fina_indicator（报告期）
                    roe_val = None
                    rev_growth_val = None
                    try:
                        fi = self.ts.fina_indicator(ts_code=ts_code, limit=1, fields="ts_code,end_date,roe,or_yoy")
                        if fi is not None and not fi.empty:
                            r = fi.iloc[0]
                            roe_val = float(r["roe"]) if _is_number(str(r.get("roe"))) else None
                            rev_growth_val = float(r["or_yoy"]) if _is_number(str(r.get("or_yoy"))) else None
                    except Exception:
                        pass

                    db["roe"] = roe_val
                    db["revenue_growth"] = rev_growth_val
                    db = db[["report_date", "pe_ratio", "pb_ratio", "roe", "revenue_growth"]]
                    db = db.dropna(subset=["report_date"]).sort_values("report_date").tail(max_reports).reset_index(drop=True)
                    if not db.empty:
                        return db
            except Exception:
                pass

        raw = self.fetch_financial_abstract_raw(stock_code)
        if raw is None or raw.empty:
            return pd.DataFrame()

        df = raw.copy()
        # 尝试识别报告期列
        report_cols = [c for c in df.columns if any(k in str(c) for k in ["报告期", "截止", "time", "date"])]
        report_col = report_cols[0] if report_cols else None
        if report_col is not None:
            df["report_date"] = df[report_col].apply(_safe_to_datetime)
        else:
            df["report_date"] = pd.NaT

        # 识别 PE/PB/ROE/营收增长率列（尽量用字段名匹配）
        pe_candidates = ["市盈率", "PE", "市盈率(动态)", "市盈率(静态)", "PE(TTM)", "市盈率TTM"]
        pb_candidates = ["市净率", "PB", "市净率(TTM)", "PB(TTM)"]
        roe_candidates = ["净资产收益率", "ROE", "ROE(加权)", "ROE(TTM)"]
        rev_growth_candidates = ["营业收入增长率", "营收增长率", "营业收入同比增长率", "Revenue Growth", "收入增长率"]

        def extract_row_metric(row: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
            pe = _pick_first_numeric(row, pe_candidates)
            pb = _pick_first_numeric(row, pb_candidates)
            roe = _pick_first_numeric(row, roe_candidates)
            rev = _pick_first_numeric(row, rev_growth_candidates)
            return pe, pb, roe, rev

        metrics = df.apply(lambda r: extract_row_metric(r), axis=1, result_type="expand")
        metrics.columns = ["pe_ratio", "pb_ratio", "roe", "revenue_growth"]
        df = pd.concat([df, metrics], axis=1)

        out = df[["report_date", "pe_ratio", "pb_ratio", "roe", "revenue_growth"]].copy()
        out = out.dropna(subset=["report_date"])
        out = out.sort_values("report_date").tail(max_reports).reset_index(drop=True)
        return out

    def fetch_latest_financial_metrics(self, stock_code: str) -> Dict:
        """
        获取最新一条财务指标（真实）。
        返回：pe_ratio/pb_ratio/roe/revenue_growth 等字段；缺失时返回 None。
        """
        hist = self.fetch_financial_ratio_history(stock_code, max_reports=5)
        if hist is None or hist.empty:
            return {
                "pe_ratio": None,
                "pb_ratio": None,
                "roe": None,
                "revenue_growth": None,
            }
        latest = hist.iloc[-1]
        return {
            "pe_ratio": float(latest["pe_ratio"]) if pd.notna(latest["pe_ratio"]) else None,
            "pb_ratio": float(latest["pb_ratio"]) if pd.notna(latest["pb_ratio"]) else None,
            "roe": float(latest["roe"]) if pd.notna(latest["roe"]) else None,
            "revenue_growth": float(latest["revenue_growth"]) if pd.notna(latest["revenue_growth"]) else None,
        }

    # -------------------- 指数基准（沪深300） --------------------
    def fetch_index_daily(self, index_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数日线收盘价，用于 Beta/组合风险计算。
        index_symbol 常见：'000300' / 'sh000300'（沪深300）
        返回列：date/close
        """
        # 先尝试 TuShare 指数日线（如 000300.SH）
        if self.ts is not None:
            try:
                digits = "".join(ch for ch in str(index_symbol or "") if ch.isdigit())
                if digits:
                    ts_code = f"{digits}.SH"
                    df_ts = self.ts.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                    if df_ts is not None and not df_ts.empty:
                        out = df_ts.rename(columns={"trade_date": "date", "close": "close"})
                        out["date"] = pd.to_datetime(out["date"])
                        out = out[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
                        if not out.empty:
                            return out
            except Exception:
                pass

        # 指数 symbol 经常不遵循股票 sh/sz 前缀规则。
        # 这里对 HS300 等常用 6 位纯数字编码：优先尝试 "000300"，再尝试 "sh000300"/"sz000300"。
        digits = "".join(ch for ch in str(index_symbol or "") if ch.isdigit())
        symbols: List[str] = []
        if digits:
            symbols.append(digits)
            if len(digits) == 6:
                symbols.append(f"sh{digits}")
                symbols.append(f"sz{digits}")
        else:
            symbols.append(index_symbol)

        # 去重保持顺序
        seen = set()
        deduped: List[str] = []
        for s in symbols:
            if s not in seen:
                deduped.append(s)
                seen.add(s)
        symbols = deduped

        try:
            import akshare as ak

            fn_names = ["stock_zh_index_daily", "index_zh_index_daily"]
            last_err = None
            for fn_name in fn_names:
                if not hasattr(ak, fn_name):
                    continue
                fn = getattr(ak, fn_name)
                for sym in symbols:
                    try:
                        # akshare 不同版本可能参数名不同：symbol / index
                        try:
                            df = fn(symbol=sym, start_date=start_date, end_date=end_date)
                        except TypeError:
                            df = fn(index=sym, start_date=start_date, end_date=end_date)
                        if df is None or len(df) == 0:
                            continue
                        # 常见列：日期/收盘
                        rename_map = {"日期": "date", "收盘": "close"}
                        for k, v in rename_map.items():
                            if k in df.columns:
                                df = df.rename(columns={k: v})
                        if "date" in df.columns:
                            df["date"] = pd.to_datetime(df["date"])
                        if "close" not in df.columns:
                            continue
                        df = df[["date", "close"]].dropna().sort_values("date")
                        return df.reset_index(drop=True)
                    except Exception as e:
                        last_err = e
                        continue
            _ = last_err
        except Exception:
            pass
        if "last_err" in locals() and last_err is not None:
            print(f"[TianDunDataFetcher] fetch_index_daily failed: index_symbol={index_symbol}, err={last_err}")
        return pd.DataFrame()

    # -------------------- 新闻标题（用于 ESG/舆情） --------------------
    def fetch_stock_news_titles(self, stock_code: str, stock_name: str, days: int = 30) -> List[NewsItem]:
        """
        获取股票近 N 天新闻标题（尽量真实）。

        优先策略：
        - 使用东财 search API（requests 爬取公开信息）。
        - 若无法获取/接口字段缺失，则返回空列表；上层会对 ESG/舆情做“合理模拟并在注释中说明”。

        注意：
        - 由于不同接口字段命名可能变化，本函数尽量尝试解析发布时间；无法解析时会把新闻按返回顺序粗略映射到日期。
        """
        stock_code = normalize_stock_code(stock_code)
        if not stock_code:
            return []

        # 方案 1：优先 akshare 的股票新闻接口（一般可直接用 A 股代码）
        try:
            import akshare as ak

            df = ak.stock_news_em(symbol=stock_code)
            if df is not None and not df.empty:
                # 这些列名在不同版本可能略有差异，但常见是以下字段
                title_col = "新闻标题" if "新闻标题" in df.columns else ("title" if "title" in df.columns else None)
                time_col = "发布时间" if "发布时间" in df.columns else ("time" if "time" in df.columns else None)
                source_col = "文章来源" if "文章来源" in df.columns else ("source" if "source" in df.columns else None)

                if title_col and time_col:
                    now = datetime.now()
                    min_dt = now - timedelta(days=int(days))
                    out: List[NewsItem] = []
                    for _, row in df.iterrows():
                        title = str(row.get(title_col, "")).strip()
                        dt = _safe_to_datetime(row.get(time_col))
                        if not title or dt is None:
                            continue
                        if dt < min_dt:
                            continue
                        source = str(row.get(source_col, "")).strip() if source_col else ""
                        out.append(NewsItem(date=dt, title=title, source=source))
                    # 按时间倒序并截断
                    out = sorted(out, key=lambda x: x.date, reverse=True)
                    return out[: max(10, int(days))]
        except Exception:
            pass

        # 方案 2：东财 search API（可能受限时会失败；失败后返回空）
        url = "http://searchapi.eastmoney.com/api/suggest/get"
        params = {
            "cid": stock_code,
            "keyword": stock_name,
            "type": "news",
            "pageIndex": 1,
            "pageSize": max(10, int(days)),
        }
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        if not data or "Quotation" not in data:
            return []
        items = data.get("Quotation", [])
        if not items:
            return []

        now = datetime.now()
        results: List[NewsItem] = []
        for i, it in enumerate(items):
            if i >= days:
                break
            title = str(it.get("Title", "")).strip()
            if not title:
                continue
            source = str(it.get("Source", "")).strip() if it.get("Source") else ""

            dt = None
            for key in ["Time", "PublishTime", "PubDate", "Date", "CreateDate"]:
                if key in it:
                    dt = _safe_to_datetime(it.get(key))
                    if dt is not None:
                        break
            if dt is None:
                dt = pd.Timestamp(now - timedelta(days=i))
            results.append(NewsItem(date=dt, title=title, source=source))

        return sorted(results, key=lambda x: x.date, reverse=True)[: max(10, int(days))]

    # -------------------- 华证 ESG 评级（新浪源，经 akshare） --------------------
    @staticmethod
    def sina_listed_symbol(stock_code: str) -> str:
        """A 股 6 位代码 -> 新浪/华证常用格式，如 600519.SH、000001.SZ。"""
        stock_code = normalize_stock_code(stock_code)
        if not stock_code:
            return ""
        return f"{stock_code}.SH" if stock_code.startswith("6") else f"{stock_code}.SZ"

    def fetch_esg_hz_sina_table(self) -> pd.DataFrame:
        """
        全市场华证 ESG 评级（新浪财经源，akshare：`stock_esg_hz_sina`）。
        首次拉取较慢（接口分页），建议由上层 `st.cache_data` 长周期缓存。
        """
        try:
            import akshare as ak

            df = ak.stock_esg_hz_sina()
            if df is None:
                return pd.DataFrame()
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception as e:
            print(f"[TianDunDataFetcher] fetch_esg_hz_sina_table failed: {e}")
            return pd.DataFrame()

    def find_esg_hz_row(self, esg_df: pd.DataFrame, stock_code: str) -> Optional[Dict]:
        """
        在全市场 ESG 表中查找单只股票，返回可直接用于风险引擎的字典（字段名统一为英文键）。
        """
        if esg_df is None or esg_df.empty:
            return None
        sym = self.sina_listed_symbol(stock_code)
        if not sym:
            return None
        code_col = "股票代码" if "股票代码" in esg_df.columns else None
        if not code_col:
            return None
        sub = esg_df[esg_df[code_col].astype(str).str.strip().str.upper() == sym.upper()]
        if sub.empty:
            return None
        row = sub.iloc[0]

        def _num(cn: str) -> Optional[float]:
            if cn not in row.index:
                return None
            v = row.get(cn)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            try:
                return float(str(v).replace(",", "").replace("%", "").strip())
            except Exception:
                return None

        return {
            "symbol": sym,
            "stock_name": str(row.get("股票名称", "") or "").strip(),
            "esg_score": _num("ESG评分"),
            "esg_grade": str(row.get("ESG等级", "") or "").strip(),
            "env_score": _num("环境"),
            "env_grade": str(row.get("环境等级", "") or "").strip(),
            "social_score": _num("社会"),
            "social_grade": str(row.get("社会等级", "") or "").strip(),
            "gov_score": _num("公司治理"),
            "gov_grade": str(row.get("公司治理等级", "") or "").strip(),
            "report_date": row.get("日期"),
            "data_source": "华证 ESG（新浪财经，akshare stock_esg_hz_sina）",
        }

    # -------------------- 名称兜底（直接访问财经网站） --------------------
    def fetch_stock_name_from_eastmoney(self, stock_code: str) -> str:
        """
        当 akshare spot 表为空/受限时，使用东方财富行情页标题提取公司真实名称。
        例：东方财富标题形如“贵州茅台(600519)_股票价格_行情_走势图—东方财富网”
        """
        stock_code = normalize_stock_code(stock_code)
        if not stock_code:
            return ""

        # 1.x 表示上交所，0.x 表示深交所（东方财富统一页）
        try:
            secid_prefix = "1" if stock_code.startswith("6") else "0"
            secid = f"{secid_prefix}.{stock_code}"
            url = f"http://quote.eastmoney.com/unify/r/{secid}"
            resp = self.session.get(url, timeout=self.timeout_s)
            resp.raise_for_status()
            html = resp.text or ""
            # 从页面 title/文本中提取： 名称(代码)
            m = re.search(r"([^\\(\\)]+)\\(" + re.escape(stock_code) + r"\\)", html)
            if m:
                name = m.group(1).strip()
                # 去掉可能的多余空格/换行
                return re.sub(r"\\s+", "", name)
        except Exception:
            pass

        # 兜底：返回代码本身
        return stock_code

