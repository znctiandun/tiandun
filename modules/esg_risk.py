"""
天盾 - ESG风险动态评估模块
"""

import numpy as np
from datetime import datetime, timedelta
import random

class ESGRiskEvaluator:
    """ESG风险评估器"""
    
    # ESG事件关键词库
    ESG_KEYWORDS = {
        'E': ['环保', '污染', '排放', '碳', '绿色', '能耗', '环境处罚', '清洁生产'],
        'S': ['员工', '安全', '劳动', '社区', '公益', '员工伤亡', '产品质量', '消费者'],
        'G': ['治理', '董事会', '股东', '违规', '处罚', '信披', '关联交易', '腐败']
    }
    
    # 行业ESG基准分
    INDUSTRY_BENCHMARK = {
        '白酒': {'E': 65, 'S': 70, 'G': 75},
        '动力电池': {'E': 60, 'S': 65, 'G': 70},
        '新能源汽车': {'E': 65, 'S': 70, 'G': 75},
        '银行': {'E': 70, 'S': 75, 'G': 65},
        '保险金融': {'E': 70, 'S': 75, 'G': 65},
        '通用制造业': {'E': 55, 'S': 60, 'G': 65},
    }
    
    def __init__(self):
        pass
    
    def extract_esg_events(self, news_sentiment_data):
        """
        从新闻中提取ESG事件
        """
        esg_events = {'E': [], 'S': [], 'G': []}
        
        for news in news_sentiment_data:
            title = news.get('keyword', '')
            sentiment = news.get('sentiment_score', 0)
            
            for category, keywords in self.ESG_KEYWORDS.items():
                for kw in keywords:
                    if kw in title:
                        esg_events[category].append({
                            'date': news.get('date', datetime.now()),
                            'title': title,
                            'sentiment': sentiment,
                            'keyword': kw
                        })
                        break
        
        return esg_events
    
    def calculate_esg_score(self, industry, esg_events):
        """
        计算ESG综合评分（0-100）
        """
        # 获取行业基准分
        benchmark = self.INDUSTRY_BENCHMARK.get(industry, self.INDUSTRY_BENCHMARK['通用制造业'])
        
        # 根据ESG事件调整分数
        adjustments = {'E': 0, 'S': 0, 'G': 0}
        
        for category, events in esg_events.items():
            for event in events:
                # 负面事件扣分，正面事件加分
                if event['sentiment'] < -0.3:
                    adjustments[category] -= 5
                elif event['sentiment'] > 0.3:
                    adjustments[category] += 2
        
        # 计算最终分数
        scores = {
            'E': min(max(benchmark['E'] + adjustments['E'], 0), 100),
            'S': min(max(benchmark['S'] + adjustments['S'], 0), 100),
            'G': min(max(benchmark['G'] + adjustments['G'], 0), 100),
        }
        
        # 综合ESG分（加权平均）
        total_esg = scores['E'] * 0.35 + scores['S'] * 0.35 + scores['G'] * 0.30
        
        return {
            'E_score': round(scores['E'], 1),
            'S_score': round(scores['S'], 1),
            'G_score': round(scores['G'], 1),
            'total_esg': round(total_esg, 1),
            'esg_events': esg_events
        }
    
    def calculate_esg_beta(self, esg_score, industry):
        """
        计算ESG Beta
        ESG风险对股价的传导系数
        """
        # 简化模型：ESG分数越低，Beta越高（风险传导越强）
        base_beta = 0.5
        esg_impact = (100 - esg_score) / 100 * 0.5
        
        return round(base_beta + esg_impact, 2)
    
    def get_esg_risk_assessment(self, industry, news_sentiment_data):
        """
        获取完整ESG风险评估
        """
        esg_events = self.extract_esg_events(news_sentiment_data)
        esg_scores = self.calculate_esg_score(industry, esg_events)
        esg_beta = self.calculate_esg_beta(esg_scores['total_esg'], industry)
        
        # 风险等级
        if esg_scores['total_esg'] >= 75:
            risk_level = 'low'
            risk_label = '✅ ESG风险低'
        elif esg_scores['total_esg'] >= 50:
            risk_level = 'medium'
            risk_label = '⚠️ ESG风险中等'
        else:
            risk_level = 'high'
            risk_label = '🔴 ESG风险高'
        
        return {
            **esg_scores,
            'esg_beta': esg_beta,
            'risk_level': risk_level,
            'risk_label': risk_label,
        }
