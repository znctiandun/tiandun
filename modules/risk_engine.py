"""
天盾 - 风险分析引擎
多维度风险计算与融合
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

class RiskAnalysisEngine:
    """风险分析引擎"""
    
    # 风险权重配置
    WEIGHTS = {
        'price_volatility': 0.20,    # 价格波动风险
        'financial_health': 0.25,    # 财务健康风险
        'sentiment_risk': 0.20,      # 舆情风险
        'supply_chain': 0.20,        # 供应链风险
        'esg_risk': 0.15,            # ESG风险
    }
    
    # 预警阈值
    ALERT_THRESHOLDS = {
        'green': 30,      # 0-30 低风险
        'yellow': 60,     # 30-60 中风险
        'red': 100,       # 60-100 高风险
    }
    
    def __init__(self):
        self.risk_history = {}
    
    def calculate_comprehensive_risk(
        self,
        stock_code: str,
        stock_name: str,
        price_data: pd.DataFrame,
        financials: Dict,
        sentiment_data: List[Dict],
        supply_chain_risk: float = 0,
        esg_risk: float = 0
    ) -> Dict:
        """
        计算综合风险评分
        返回：包含各维度风险和综合风险的字典
        """
        # 1. 价格波动风险
        price_risk = self._calculate_price_risk(price_data)
        
        # 2. 财务健康风险
        financial_risk = self._calculate_financial_risk(financials)
        
        # 3. 舆情风险
        sentiment_risk = self._calculate_sentiment_risk(sentiment_data)
        
        # 4. 供应链风险（从外部传入）
        supply_risk = min(supply_chain_risk, 100)
        
        # 5. ESG风险（从外部传入）
        esg = min(esg_risk, 100)
        
        # 加权融合（使用非线性融合，突出最高风险项）
        risks = {
            'price_volatility': price_risk,
            'financial_health': financial_risk,
            'sentiment_risk': sentiment_risk,
            'supply_chain': supply_risk,
            'esg_risk': esg,
        }
        
        # 综合风险 = 0.6 * 最大单项 + 0.4 * 加权平均
        max_risk = max(risks.values())
        weighted_avg = sum(risks[k] * self.WEIGHTS[k] for k in risks)
        comprehensive = 0.6 * max_risk + 0.4 * weighted_avg * 100 / sum(self.WEIGHTS.values())
        comprehensive = min(max(comprehensive, 0), 100)
        
        # 确定风险等级
        if comprehensive >= self.ALERT_THRESHOLDS['red']:
            level = 'red'
            level_name = '高风险'
        elif comprehensive >= self.ALERT_THRESHOLDS['yellow']:
            level = 'yellow'
            level_name = '中风险'
        else:
            level = 'green'
            level_name = '低风险'
        
        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'comprehensive_risk': round(comprehensive, 1),
            'risk_level': level,
            'risk_level_name': level_name,
            'dimension_risks': risks,
            'calculation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    
    def _calculate_price_risk(self, price_data: pd.DataFrame) -> float:
        """计算价格波动风险 (0-100)"""
        if len(price_data) < 5:
            return 50
        
        close_prices = price_data['close'].values
        
        # 波动率（年化）
        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(returns) * np.sqrt(252) * 100
        
        # 最大回撤
        peak = np.maximum.accumulate(close_prices)
        drawdown = (peak - close_prices) / peak * 100
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # 趋势（近期表现）
        if len(close_prices) >= 10:
            recent_return = (close_prices[-1] - close_prices[-10]) / close_prices[-10] * 100
        else:
            recent_return = 0
        
        # 风险评分（波动率越高、回撤越大、趋势越差，风险越高）
        risk_score = (
            min(volatility * 2, 40) +      # 波动率贡献最多40分
            min(max_drawdown, 40) +         # 回撤贡献最多40分
            max(0, -recent_return) * 0.5    # 负收益增加风险
        )
        
        return min(max(risk_score, 0), 100)
    
    def _calculate_financial_risk(self, financials: Dict) -> float:
        """计算财务健康风险 (0-100)"""
        risk_score = 0
        
        # PE过高风险
        pe = financials.get('pe_ratio', 25)
        if pe > 50:
            risk_score += 30
        elif pe > 30:
            risk_score += 15
        
        # 负债率过高
        debt_ratio = financials.get('debt_ratio', 50)
        if debt_ratio > 70:
            risk_score += 30
        elif debt_ratio > 50:
            risk_score += 15
        
        # ROE过低
        roe = financials.get('roe', 15)
        if roe < 5:
            risk_score += 25
        elif roe < 10:
            risk_score += 10
        
        # 利润负增长
        profit_growth = financials.get('profit_growth', 10)
        if profit_growth < 0:
            risk_score += 25
        elif profit_growth < 5:
            risk_score += 10
        
        return min(max(risk_score, 0), 100)
    
    def _calculate_sentiment_risk(self, sentiment_data: List[Dict]) -> float:
        """计算舆情风险 (0-100)"""
        if not sentiment_data:
            return 50
        
        # 近期情感平均分（越负面风险越高）
        recent_sentiments = [d['sentiment_score'] for d in sentiment_data[-7:]]
        avg_sentiment = np.mean(recent_sentiments)
        
        # 情感趋势
        if len(sentiment_data) >= 14:
            prev_sentiment = np.mean([d['sentiment_score'] for d in sentiment_data[-14:-7]])
            sentiment_trend = avg_sentiment - prev_sentiment
        else:
            sentiment_trend = 0
        
        # 负面新闻频率
        negative_count = sum(1 for d in sentiment_data[-7:] if d['sentiment_score'] < -0.2)
        
        # 风险计算
        risk_score = (
            (1 - avg_sentiment) * 30 +          # 情感负面贡献
            max(0, -sentiment_trend) * 30 +      # 趋势恶化贡献
            negative_count * 5                    # 负面新闻数量
        )
        
        return min(max(risk_score, 0), 100)
    
    def generate_alert(self, risk_result: Dict, holding_ratio: float = 0) -> Dict:
        """
        生成风险预警
        holding_ratio: 该股票在用户持仓中的占比
        """
        comprehensive = risk_result['comprehensive_risk']
        level = risk_result['risk_level']
        
        # 根据持仓占比调整预警级别
        if holding_ratio > 0.2:  # 持仓超过20%，预警升级
            if level == 'yellow':
                level = 'red'
        
        # 生成预警信息
        alert = {
            'stock_code': risk_result['stock_code'],
            'stock_name': risk_result['stock_name'],
            'alert_level': level,
            'risk_score': comprehensive,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': self._generate_alert_message(risk_result),
            'suggestions': self._generate_suggestions(risk_result),
        }
        
        return alert
    
    def _generate_alert_message(self, risk_result: Dict) -> str:
        """生成预警消息"""
        level = risk_result['risk_level']
        name = risk_result['stock_name']
        score = risk_result['comprehensive_risk']
        
        dimension_risks = risk_result['dimension_risks']
        max_risk_dim = max(dimension_risks, key=dimension_risks.get)
        
        dim_names = {
            'price_volatility': '价格波动',
            'financial_health': '财务健康',
            'sentiment_risk': '舆情情绪',
            'supply_chain': '供应链',
            'esg_risk': 'ESG风险',
        }
        
        if level == 'red':
            return f"【红色预警】{name}风险评分{score}分，主要风险源：{dim_names[max_risk_dim]}"
        elif level == 'yellow':
            return f"【黄色预警】{name}风险评分{score}分，需关注：{dim_names[max_risk_dim]}"
        else:
            return f"【正常】{name}风险评分{score}分，整体可控"
    
    def _generate_suggestions(self, risk_result: Dict) -> List[str]:
        """生成投资建议"""
        suggestions = []
        dimension_risks = risk_result['dimension_risks']
        
        if dimension_risks.get('price_volatility', 0) > 60:
            suggestions.append("⚠️ 价格波动剧烈，建议设置止损位")
        
        if dimension_risks.get('financial_health', 0) > 60:
            suggestions.append("⚠️ 财务指标承压，建议关注下季度财报")
        
        if dimension_risks.get('sentiment_risk', 0) > 60:
            suggestions.append("⚠️ 舆情偏负面，建议密切关注相关新闻")
        
        if dimension_risks.get('supply_chain', 0) > 60:
            suggestions.append("⚠️ 供应链存在风险传导可能")
        
        if dimension_risks.get('esg_risk', 0) > 60:
            suggestions.append("⚠️ ESG风险较高，长期投资者需谨慎")
        
        if not suggestions:
            suggestions.append("✅ 当前风险可控，可继续持有")
        
        return suggestions
    
    def calculate_30day_risk_history(
        self,
        stock_code: str,
        stock_name: str,
        price_data: pd.DataFrame,
        sentiment_data: List[Dict]
    ) -> List[Dict]:
        """计算30天风险历史（用于动画展示）"""
        history = []
        dates = price_data['date'].values
        
        for i in range(len(dates)):
            # 截取到当前日期的数据
            current_price_data = price_data.iloc[:i+1].copy()
            current_sentiment = sentiment_data[:i+1] if i < len(sentiment_data) else sentiment_data
            
            if len(current_price_data) >= 5:  # 至少5天数据才能计算
                risk_result = self.calculate_comprehensive_risk(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    price_data=current_price_data,
                    financials={'pe_ratio': 30, 'debt_ratio': 50, 'roe': 15, 'profit_growth': 10},
                    sentiment_data=current_sentiment,
                )
                history.append({
                    'date': pd.Timestamp(dates[i]).strftime('%Y-%m-%d'),
                    'risk_score': risk_result['comprehensive_risk'],
                    'price': float(current_price_data['close'].iloc[-1]),
                })
        
        return history
