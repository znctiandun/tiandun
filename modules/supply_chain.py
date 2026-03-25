"""
天盾 - 供应链风险与行业地位模块（增强版）
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple

class SupplyChainModel:
    """供应链风险传染与行业地位模型"""
    
    # 行业地位数据（实际应用中应从数据库获取）
    INDUSTRY_POSITION = {
        # 贵州茅台
        '600519': {
            'industry': '白酒',
            'market_share': 28.5,  # 市场份额%
            'industry_rank': 1,
            'total_companies': 50,
            'position_label': '行业龙头',
            'position_score': 95,
            'pricing_power': 0.92,  # 定价能力
            'brand_strength': 0.95,  # 品牌强度
            'upstream': [
                {'name': '高粱种植', 'impact': 0.3, 'dependency': 'low', 'suppliers_count': 150},
                {'name': '包装材料', 'impact': 0.2, 'dependency': 'medium', 'suppliers_count': 80},
                {'name': '物流运输', 'impact': 0.15, 'dependency': 'low', 'suppliers_count': 50}
            ],
            'downstream': [
                {'name': '经销商', 'impact': 0.4, 'dependency': 'high', 'channels': 2500},
                {'name': '电商平台', 'impact': 0.25, 'dependency': 'medium', 'channels': 15},
                {'name': '直营店', 'impact': 0.15, 'dependency': 'low', 'channels': 100}
            ],
            'peers': ['五粮液', '泸州老窖', '洋河股份'],
            'competitive_advantage': ['品牌壁垒', '稀缺性', '定价权', '渠道控制']
        },
        # 宁德时代
        '300750': {
            'industry': '动力电池',
            'market_share': 37.2,
            'industry_rank': 1,
            'total_companies': 30,
            'position_label': '全球龙头',
            'position_score': 92,
            'pricing_power': 0.75,
            'brand_strength': 0.82,
            'upstream': [
                {'name': '锂矿开采', 'impact': 0.4, 'dependency': 'high', 'suppliers_count': 20},
                {'name': '正负极材料', 'impact': 0.3, 'dependency': 'high', 'suppliers_count': 40},
                {'name': '隔膜', 'impact': 0.15, 'dependency': 'medium', 'suppliers_count': 15}
            ],
            'downstream': [
                {'name': '新能源汽车', 'impact': 0.6, 'dependency': 'high', 'customers': 30},
                {'name': '储能系统', 'impact': 0.25, 'dependency': 'medium', 'customers': 50},
                {'name': '电动工具', 'impact': 0.1, 'dependency': 'low', 'customers': 100}
            ],
            'peers': ['比亚迪', 'LG 化学', '松下', 'SK On'],
            'competitive_advantage': ['技术领先', '规模效应', '客户绑定', '产能扩张']
        },
        # 比亚迪
        '002594': {
            'industry': '新能源汽车',
            'market_share': 18.5,
            'industry_rank': 2,
            'total_companies': 40,
            'position_label': '行业领先',
            'position_score': 85,
            'pricing_power': 0.68,
            'brand_strength': 0.78,
            'upstream': [
                {'name': '电池供应商', 'impact': 0.35, 'dependency': 'medium', 'suppliers_count': 25},
                {'name': '芯片/电子', 'impact': 0.3, 'dependency': 'high', 'suppliers_count': 60},
                {'name': '钢材/铝材', 'impact': 0.2, 'dependency': 'low', 'suppliers_count': 100}
            ],
            'downstream': [
                {'name': '个人消费者', 'impact': 0.5, 'dependency': 'high', 'channels': 1200},
                {'name': '网约车平台', 'impact': 0.2, 'dependency': 'medium', 'channels': 10},
                {'name': '政府/企业采购', 'impact': 0.15, 'dependency': 'medium', 'channels': 200}
            ],
            'peers': ['特斯拉', '蔚来汽车', '小鹏汽车', '理想汽车'],
            'competitive_advantage': ['全产业链', '成本控制', '品牌认知', '技术积累']
        },
    }
    
    def get_industry_position(self, stock_code: str, stock_name: str) -> Dict:
        """获取行业地位信息"""
        if stock_code in self.INDUSTRY_POSITION:
            return self.INDUSTRY_POSITION[stock_code]
        
        # 默认数据
        return {
            'industry': '通用制造业',
            'market_share': np.random.uniform(5, 15),
            'industry_rank': np.random.randint(3, 10),
            'total_companies': 50,
            'position_label': '行业中游',
            'position_score': np.random.uniform(50, 70),
            'pricing_power': np.random.uniform(0.4, 0.6),
            'brand_strength': np.random.uniform(0.4, 0.6),
            'upstream': [
                {'name': '原材料', 'impact': 0.4, 'dependency': 'medium', 'suppliers_count': 50},
                {'name': '零部件', 'impact': 0.3, 'dependency': 'medium', 'suppliers_count': 30}
            ],
            'downstream': [
                {'name': '经销商', 'impact': 0.4, 'dependency': 'high', 'channels': 200},
                {'name': '终端客户', 'impact': 0.3, 'dependency': 'medium', 'channels': 1000}
            ],
            'peers': ['同行业 A', '同行业 B', '同行业 C'],
            'competitive_advantage': ['成本控制', '产品质量', '渠道优势']
        }
    
    def calculate_industry_position_risk(self, stock_code: str, stock_name: str) -> Dict:
        """
        计算行业地位风险
        地位越高，抗风险能力越强
        """
        position = self.get_industry_position(stock_code, stock_name)
        
        # 行业地位风险评分（地位越高，风险越低）
        base_risk = 100 - position['position_score']
        
        # 定价能力调整
        pricing_adjustment = (1 - position['pricing_power']) * 20
        
        # 品牌强度调整
        brand_adjustment = (1 - position['brand_strength']) * 15
        
        # 市场份额调整
        market_share_risk = 0
        if position['market_share'] < 10:
            market_share_risk = 20
        elif position['market_share'] < 20:
            market_share_risk = 10
        
        # 综合行业地位风险
        position_risk = base_risk * 0.4 + pricing_adjustment * 0.3 + \
                       brand_adjustment * 0.2 + market_share_risk * 0.1
        
        # 确定风险等级
        if position_risk >= 60:
            level = 'high'
            level_name = '风险较高'
            color = '#EF5350'
        elif position_risk >= 35:
            level = 'medium'
            level_name = '风险中等'
            color = '#FFCA28'
        else:
            level = 'low'
            level_name = '风险较低'
            color = '#66BB6A'
        
        return {
            'position_risk_score': round(position_risk, 1),
            'risk_level': level,
            'risk_level_name': level_name,
            'risk_color': color,
            'position_details': position,
            'recommendation': self._generate_position_recommendation(position_risk, position)
        }
    
    def calculate_contagion_risk(self, stock_code: str, stock_name: str) -> Dict:
        """计算供应链传染风险"""
        position = self.get_industry_position(stock_code, stock_name)
        
        # 上游依赖风险
        upstream_risk = 0
        for u in position['upstream']:
            if u['dependency'] == 'high':
                upstream_risk += 25
            elif u['dependency'] == 'medium':
                upstream_risk += 15
            else:
                upstream_risk += 5
        
        # 下游依赖风险
        downstream_risk = 0
        for d in position['downstream']:
            if d['dependency'] == 'high':
                downstream_risk += 20
            elif d['dependency'] == 'medium':
                downstream_risk += 10
            else:
                downstream_risk += 5
        
        # 供应商集中度风险
        supplier_concentration_risk = 0
        total_suppliers = sum(u['suppliers_count'] for u in position['upstream'])
        if total_suppliers < 50:
            supplier_concentration_risk = 30
        elif total_suppliers < 100:
            supplier_concentration_risk = 20
        
        # 综合传染风险
        contagion_risk = (upstream_risk * 0.4 + downstream_risk * 0.4 + 
                         supplier_concentration_risk * 0.2)
        
        # 行业地位越高，抗传染能力越强
        position_adjustment = (100 - position['position_score']) * 0.2
        contagion_risk = contagion_risk + position_adjustment
        
        return {
            'contagion_risk': round(min(contagion_risk, 100), 1),
            'upstream_risk': upstream_risk,
            'downstream_risk': downstream_risk,
            'supplier_concentration_risk': supplier_concentration_risk,
            'position_adjustment': position_adjustment,
            'affected_nodes': len(position['upstream']) + len(position['downstream'])
        }
    
    def get_industry_impact_map(self, stock_code: str, stock_name: str) -> Dict:
        """获取产业链影响地图"""
        position = self.get_industry_position(stock_code, stock_name)
        
        return {
            'center': {
                'name': stock_name,
                'industry': position['industry'],
                'position': position['position_label'],
                'market_share': position['market_share']
            },
            'upstream': position['upstream'],
            'downstream': position['downstream'],
            'peers': position['peers'],
            'competitive_advantage': position['competitive_advantage']
        }
    
    def create_industry_chain_visualization(self, stock_code: str, stock_name: str):
        """创建产业链可视化图（增强版）"""
        data = self.get_industry_impact_map(stock_code, stock_name)
        
        fig = go.Figure()
        
        # 添加行业地位信息
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=data['center']['market_share'],
            delta={'reference': 15, 'increasing': {'color': '#66BB6A'}},
            title={'text': f"{data['center']['name']} - 市场份额 ({data['center']['industry']})"},
            gauge={
                'axis': {'range': [0, 50]},
                'bar': {'color': '#42A5F5'},
                'steps': [
                    {'range': [0, 10], 'color': '#EF5350'},
                    {'range': [10, 25], 'color': '#FFCA28'},
                    {'range': [25, 50], 'color': '#66BB6A'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ))
        
        fig.update_layout(height=350, showlegend=False)
        return fig
    
    def create_position_comparison_chart(self, stock_code: str, stock_name: str):
        """创建行业地位对比图"""
        position = self.get_industry_position(stock_code, stock_name)
        
        fig = go.Figure()
        
        # 雷达图展示各项指标
        categories = ['市场份额', '定价能力', '品牌强度', '技术领先', '成本控制']
        values = [
            position['market_share'] / 50 * 100,  # 归一化
            position['pricing_power'] * 100,
            position['brand_strength'] * 100,
            85,  # 假设技术领先分
            75   # 假设成本控制分
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=stock_name,
            line=dict(color='#42A5F5', width=2)
        ))
        
        # 添加行业平均线
        industry_avg = [40, 55, 55, 55, 60]
        fig.add_trace(go.Scatterpolar(
            r=industry_avg + [industry_avg[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='行业平均',
            line=dict(color='#EF5350', width=1, dash='dash')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            title=f'{stock_name} - 行业竞争力雷达图',
            height=450,
            showlegend=True
        )
        
        return fig
    
    def explain_industry_position_for_common_user(self, stock_name: str) -> str:
        """用通俗语言解释行业地位"""
        position = self.get_industry_position(stock_name[:6] if len(stock_name) == 6 else '600519', stock_name)
        
        explanation = f"## 🏭 {stock_name} - 行业地位分析\n\n"
        explanation += f"**行业**: {position['industry']}\n\n"
        explanation += f"**行业排名**: 第{position['industry_rank']}名 / 共{position['total_companies']}家\n\n"
        explanation += f"**市场份额**: {position['market_share']:.1f}%\n\n"
        explanation += f"**地位标签**: {position['position_label']}\n\n"
        
        explanation += "### 核心竞争优势\n"
        for i, advantage in enumerate(position['competitive_advantage'], 1):
            explanation += f"{i}. {advantage}\n"
        
        explanation += "\n### 上游供应商\n"
        for u in position['upstream']:
            dep_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            explanation += f"- {dep_icon.get(u['dependency'], '⚪')} {u['name']} (依赖度：{u['dependency']}, 供应商数：{u['suppliers_count']})\n"
        
        explanation += "\n### 下游客户\n"
        for d in position['downstream']:
            dep_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            explanation += f"- {dep_icon.get(d['dependency'], '⚪')} {d['name']} (依赖度：{d['dependency']}, 渠道数：{d['channels']})\n"
        
        return explanation
    
    def _generate_position_recommendation(self, risk_score: float, position: Dict) -> str:
        """生成行业地位风险建议"""
        if risk_score >= 60:
            return f"⚠️ 行业地位风险较高，公司{position['position_label']}，市场份额{position['market_share']:.1f}%，" \
                   f"建议关注竞争格局变化。"
        elif risk_score >= 35:
            return f"⚡ 行业地位风险中等，公司{position['position_label']}，需关注市场份额变化。"
        else:
            return f"✅ 行业地位风险较低，公司{position['position_label']}，竞争壁垒较强，可继续持有。"
