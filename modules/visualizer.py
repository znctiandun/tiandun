"""
天盾 - 可视化模块（增强版）
增加产业地图、风险对比等
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List

class RiskVisualizer:
    """风险可视化工具（增强版）"""
    
    COLORS = {
        'red': '#EF5350',
        'yellow': '#FFCA28',
        'green': '#66BB6A',
        'blue': '#42A5F5',
        'purple': '#AB47BC',
        'orange': '#FFA726',
    }
    
    def create_industry_impact_map(self, impact_data: Dict) -> go.Figure:
        """创建产业影响地图（简化版桑基图）"""
        # 准备桑基图数据
        labels = [impact_data['center']['name']]
        
        # 上游标签
        for u in impact_data['upstream']:
            labels.append(f"上游: {u['name']}")
        
        # 下游标签
        for d in impact_data['downstream']:
            labels.append(f"下游: {d['name']}")
        
        # 源和目标
        source = list(range(1, len(impact_data['upstream']) + 1))  # 上游→中心
        target = [0] * len(impact_data['upstream'])
        value = [u['impact'] * 100 for u in impact_data['upstream']]
        
        # 中心→下游
        center_idx = 0
        for i, d in enumerate(impact_data['downstream']):
            source.append(center_idx)
            target.append(len(impact_data['upstream']) + 1 + i)
            value.append(d['impact'] * 100)
        
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=[self.COLORS['purple']] + 
                      [self.COLORS['blue']] * len(impact_data['upstream']) +
                      [self.COLORS['green']] * len(impact_data['downstream'])
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=[self.COLORS['blue']] * len(source)
            )
        ))
        
        fig.update_layout(
            title_text=f"{impact_data['center']['name']} - 产业链影响图",
            font_size=12,
            height=400,
        )
        
        return fig
    
    def create_hidden_risk_comparison(self, surface_score: float, real_score: float, 
                                      stock_name: str) -> go.Figure:
        """创建表面vs真实风险对比图"""
        fig = make_subplots(rows=1, cols=2, 
                           specs=[[{"type": "bar"}, {"type": "indicator"}]],
                           subplot_titles=['健康度对比', '风险差距'])
        
        # 柱状图
        fig.add_trace(go.Bar(
            x=['表面健康度', '真实健康度'],
            y=[surface_score, real_score],
            marker_color=[self.COLORS['green'], 
                         self.COLORS['red'] if surface_score - real_score > 20 else self.COLORS['yellow']],
            text=[f'{surface_score:.0f}分', f'{real_score:.0f}分'],
            textposition='outside',
        ), row=1, col=1)
        
        # 仪表盘
        gap = surface_score - real_score
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=gap,
            delta={'reference': 0, 'increasing': {'color': self.COLORS['red']}},
            number={'font': {'size': 40}},
            title={'text': '差距'},
        ), row=1, col=2)
        
        fig.update_layout(
            title=f'{stock_name} - 表面vs真实风险对比',
            height=350,
            showlegend=False,
        )
        
        # 添加警告标注
        if gap > 20:
            fig.add_annotation(
                text='⚠️ 高风险差异！',
                xref='paper', yref='paper',
                x=0.5, y=0.05,
                showarrow=False,
                font=dict(color=self.COLORS['red'], size=16),
            )
        
        return fig
    
    def create_risk_alert_card(self, risk_result: Dict) -> str:
        """生成风险预警卡片HTML"""
        level = risk_result.get('risk_level', 'green')
        score = risk_result.get('comprehensive_risk', 50)
        name = risk_result.get('stock_name', '未知')
        
        color_map = {
            'red': self.COLORS['red'],
            'yellow': self.COLORS['yellow'],
            'green': self.COLORS['green'],
        }
        
        emoji_map = {
            'red': '🔴',
            'yellow': '🟡',
            'green': '🟢',
        }
        
        return f"""
        <div style="
            background: linear-gradient(135deg, {color_map.get(level, 'gray')} 0%, rgba(255,255,255,0.9) 100%);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 24px; font-weight: bold; color: {color_map.get(level, 'gray')}">
                        {emoji_map.get(level, '⚪')} {name}
                    </div>
                    <div style="font-size: 14px; color: #666; margin-top: 5px;">
                        综合风险评分：{score:.1f}/100
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 36px; font-weight: bold; color: {color_map.get(level, 'gray')}">
                        {score:.0f}
                    </div>
                    <div style="font-size: 12px; color: #666;">风险分</div>
                </div>
            </div>
        </div>
        """
    
    # ... 保留原有方法（create_risk_gauge, create_dimension_radar等）
