"""
天盾 - 传统风险指标计算模块
包含：波动率、超额Beta、夏普比率、估值水位、PEG等
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TraditionalRiskMetrics:
    """传统风险指标计算器"""
    
    def __init__(self):
        # 沪深300基准数据（简化，实际应获取真实指数数据）
        self.benchmark_returns = np.random.normal(0.0003, 0.015, 252)  # 年化收益约7.5%
    
    def calculate_volatility(self, price_data, window=252):
        """
        计算波动率（年化）
        基于过去252天（1年）的日收益率
        """
        if len(price_data) < window:
            window = len(price_data)
        
        returns = price_data['close'].pct_change().dropna()
        volatility = returns.tail(window).std() * np.sqrt(252)
        return round(volatility * 100, 2)  # 转为百分比
    
    def calculate_excess_beta(self, price_data, window=63):
        """
        计算超额Beta
        基于过去63天（3个月）相对沪深300的波动
        """
        if len(price_data) < window:
            window = len(price_data)
        
        stock_returns = price_data['close'].pct_change().tail(window)
        
        # 简化：使用随机生成的基准收益（实际应获取沪深300数据）
        np.random.seed(hash(str(len(price_data))) % 2**32)
        benchmark_returns = pd.Series(np.random.normal(0.0003, 0.015, window))
        
        # 计算Beta = Cov(股票，基准) / Var(基准)
        covariance = stock_returns.cov(benchmark_returns)
        variance = benchmark_returns.var()
        
        beta = covariance / variance if variance > 0 else 1.0
        return round(beta, 2)
    
    def calculate_sharpe_ratio(self, price_data, risk_free_rate=0.02, window=252):
        """
        计算夏普比率
        无风险利率默认2%（中国10年期国债收益率）
        """
        if len(price_data) < window:
            window = len(price_data)
        
        returns = price_data['close'].pct_change().tail(window)
        excess_returns = returns - (risk_free_rate / 252)  # 日化无风险利率
        
        if returns.std() == 0:
            return 0.0
        
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252)
        return round(sharpe, 2)
    
    def calculate_sortino_ratio(self, price_data, risk_free_rate=0.02, window=252):
        """
        计算索提诺比率
        只考虑下行波动率
        """
        if len(price_data) < window:
            window = len(price_data)
        
        returns = price_data['close'].pct_change().tail(window)
        excess_returns = returns - (risk_free_rate / 252)
        
        # 只计算负收益的标准差（下行风险）
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
        return round(sortino, 2)
    
    def calculate_valuation_percentile(self, pe_ratio, stock_code):
        """
        计算估值水位（PE历史百分位）
        基于过去5年PE的历史位置
        """
        # 简化：生成模拟的5年PE分布
        np.random.seed(hash(stock_code) % 2**32)
        historical_pe = np.random.normal(pe_ratio, pe_ratio * 0.3, 1260)  # 5年交易日
        historical_pe = np.clip(historical_pe, pe_ratio * 0.3, pe_ratio * 2.5)
        
        # 计算当前PE在历史中的百分位
        percentile = (historical_pe < pe_ratio).sum() / len(historical_pe) * 100
        return round(percentile, 1)
    
    def calculate_peg(self, pe_ratio, profit_growth):
        """
        计算PEG指标
        PEG = PE / 净利润增长率
        """
        if profit_growth <= 0:
            return float('inf')
        
        peg = pe_ratio / profit_growth
        return round(peg, 2)
    
    def get_all_metrics(self, stock_code, pe_ratio, profit_growth, price_data):
        """
        获取所有传统风险指标
        """
        return {
            'volatility': self.calculate_volatility(price_data),
            'beta': self.calculate_excess_beta(price_data),
            'sharpe_ratio': self.calculate_sharpe_ratio(price_data),
            'sortino_ratio': self.calculate_sortino_ratio(price_data),
            'valuation_percentile': self.calculate_valuation_percentile(pe_ratio, stock_code),
            'peg': self.calculate_peg(pe_ratio, profit_growth),
        }
