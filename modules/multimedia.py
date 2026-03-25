"""
天盾 - 多媒体内容模块
模拟采访、视频、新闻片段展示
"""

import streamlit as st
import pandas as pd
from typing import Dict, List
from datetime import datetime, timedelta
import random

class MultimediaContent:
    """多媒体内容管理器"""
    
    # 模拟采访内容库
    INTERVIEW_TEMPLATES = {
        'positive': [
            {
                'title': '董事长访谈：我们对未来充满信心',
                'speaker': '董事长 张某',
                'duration': '3:45',
                'thumbnail': '👔',
                'summary': '董事长表示公司订单充足，产能利用率达95%，预计下季度业绩将继续增长',
                'key_quotes': [
                    '"我们目前订单已经排到明年上半年"',
                    '"新产品市场反响超出预期"',
                    '"公司现金流非常健康"'
                ],
                'risk_signals': [],
                'sentiment': 'positive',
            },
            {
                'title': 'CFO解读财报：盈利能力持续提升',
                'speaker': '财务总监 李某',
                'duration': '5:20',
                'thumbnail': '📊',
                'summary': 'CFO详细解读了公司财报，强调毛利率和净利率双提升',
                'key_quotes': [
                    '"毛利率同比提升3.5个百分点"',
                    '"费用控制效果显著"',
                    '"经营性现金流创历史新高"'
                ],
                'risk_signals': [],
                'sentiment': 'positive',
            },
        ],
        'negative': [
            {
                'title': '分析师电话会：回应市场关切',
                'speaker': '董秘 王某',
                'duration': '8:15',
                'thumbnail': '📞',
                'summary': '董秘回应了关于应收账款和存货增长的质疑',
                'key_quotes': [
                    '"应收账款增长主要是行业特性..."',
                    '"存货增加是为下季度备货..."',
                    '"请投资者放心，公司经营正常"'
                ],
                'risk_signals': [
                    '回答模糊，未给出具体数据',
                    '多次使用"行业特性"作为解释',
                    '回避了关于现金流的问题'
                ],
                'sentiment': 'negative',
            },
            {
                'title': '高管减持后首度发声',
                'speaker': '副总裁 赵某',
                'duration': '2:30',
                'thumbnail': '🎤',
                'summary': '副总裁解释个人减持原因，强调不影响公司经营',
                'key_quotes': [
                    '"减持是个人财务规划需要"',
                    '"我对公司未来发展依然看好"',
                    '"不会继续减持"'
                ],
                'risk_signals': [
                    '过去6个月已减持3次',
                    '减持金额累计超5000万',
                    '措辞与之前减持时高度相似'
                ],
                'sentiment': 'negative',
            },
        ],
        'neutral': [
            {
                'title': '工厂实地探访：生产一线实录',
                'speaker': '记者 现场报道',
                'duration': '6:40',
                'thumbnail': '🏭',
                'summary': '记者深入工厂生产线，记录实际生产情况',
                'key_quotes': [
                    '"生产线开工率约80%"',
                    '"工人实行两班倒"',
                    '"仓库存货量适中"'
                ],
                'risk_signals': [],
                'sentiment': 'neutral',
            },
        ],
    }
    
    # 模拟新闻视频库
    NEWS_VIDEO_TEMPLATES = [
        {
            'title': '行业龙头财报解析',
            'source': '财经频道',
            'duration': '4:30',
            'thumbnail': '📺',
            'summary': '分析师解读最新财报数据',
            'sentiment': 'neutral',
        },
        {
            'title': '突发！公司遭监管问询',
            'source': '证券市场周刊',
            'duration': '2:15',
            'thumbnail': '⚡',
            'summary': '交易所就财务异常下发问询函',
            'sentiment': 'negative',
        },
        {
            'title': '新产能投产仪式',
            'source': '公司新闻',
            'duration': '3:00',
            'thumbnail': '🎉',
            'summary': '新生产基地正式投产',
            'sentiment': 'positive',
        },
    ]
    
    # 产业影响视频库
    INDUSTRY_IMPACT_TEMPLATES = [
        {
            'title': '上游原材料涨价影响分析',
            'source': '产业研究',
            'duration': '5:45',
            'thumbnail': '📈',
            'summary': '分析原材料价格上涨对下游企业的影响链条',
            'affected_industries': ['制造业', '消费品', '汽车'],
        },
        {
            'title': '供应链中断风险预警',
            'source': '风险研究所',
            'duration': '4:20',
            'thumbnail': '⚠️',
            'summary': '某关键供应商停产可能引发的连锁反应',
            'affected_industries': ['电子', '汽车', '家电'],
        },
    ]
    
    def __init__(self, stock_code: str, stock_name: str):
        self.stock_code = stock_code
        self.stock_name = stock_name
        self.content_cache = {}
    
    def get_interview_content(self, risk_level: str = 'neutral') -> List[Dict]:
        """
        获取采访内容
        risk_level: 'positive', 'negative', 'neutral'
        """
        if risk_level == 'high':
            # 高风险公司返回更多负面采访
            content = self.INTERVIEW_TEMPLATES['negative'] + self.INTERVIEW_TEMPLATES['neutral'][:1]
        elif risk_level == 'low':
            content = self.INTERVIEW_TEMPLATES['positive'] + self.INTERVIEW_TEMPLATES['neutral'][:1]
        else:
            content = (self.INTERVIEW_TEMPLATES['positive'][:1] + 
                      self.INTERVIEW_TEMPLATES['negative'][:1] + 
                      self.INTERVIEW_TEMPLATES['neutral'])
        
        # 添加时间戳
        for item in content:
            item['publish_date'] = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d')
            item['stock_code'] = self.stock_code
            item['stock_name'] = self.stock_name
        
        return content
    
    def get_news_videos(self, days: int = 30) -> List[Dict]:
        """获取新闻视频列表"""
        videos = self.NEWS_VIDEO_TEMPLATES.copy()
        for video in videos:
            video['publish_date'] = (datetime.now() - timedelta(days=random.randint(1, days))).strftime('%Y-%m-%d')
            video['stock_code'] = self.stock_code
            video['stock_name'] = self.stock_name
            video['view_count'] = random.randint(1000, 100000)
        return videos
    
    def get_industry_impact_videos(self, industry: str = '制造业') -> List[Dict]:
        """获取产业影响视频"""
        videos = self.INDUSTRY_IMPACT_TEMPLATES.copy()
        for video in videos:
            video['publish_date'] = (datetime.now() - timedelta(days=random.randint(1, 60))).strftime('%Y-%m-%d')
            video['related_stock'] = self.stock_name
        return videos
    
    def render_video_card(self, video: Dict, key: str):
        """
        渲染视频卡片（Streamlit组件）
        由于实际视频需要视频文件，这里用模拟方式展示
        """
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # 视频封面（用emoji模拟）
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px;
                    padding: 30px;
                    text-align: center;
                    font-size: 48px;
                    margin: 10px 0;
                ">{video.get('thumbnail', '📺')}</div>
                <div style="text-align: center; color: gray; font-size: 12px;">
                    ⏱️ {video.get('duration', '0:00')}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**{video.get('title', '无标题')}**")
                st.markdown(f"📺 {video.get('source', '未知来源')} | 📅 {video.get('publish_date', '未知日期')}")
                st.markdown(f"*{video.get('summary', '无摘要')}*")
                
                # 风险信号（如果有）
                if video.get('risk_signals'):
                    with st.expander("⚠️ AI识别的风险信号"):
                        for signal in video['risk_signals']:
                            st.markdown(f"- {signal}")
                
                # 关键语录
                if video.get('key_quotes'):
                    with st.expander("💬 关键语录"):
                        for quote in video['key_quotes']:
                            st.markdown(f"> {quote}")
                
                # 模拟播放按钮
                if st.button("▶️ 播放视频", key=f"play_{key}"):
                    st.info("🎬 视频播放功能需要实际视频文件支持，此处为演示")
                    st.write(f"播放：{video.get('title')}")
                    st.write(f"时长：{video.get('duration')}")
    
    def create_sentiment_timeline(self, interviews: List[Dict]) -> 'go.Figure':
        """创建情感时间线图"""
        import plotly.graph_objects as go
        import numpy as np
        
        if not interviews:
            return go.Figure()
        
        # 按日期排序
        sorted_interviews = sorted(interviews, key=lambda x: x.get('publish_date', ''))
        
        dates = [i.get('publish_date', '') for i in sorted_interviews]
        
        # 情感分数映射
        sentiment_map = {'positive': 0.8, 'neutral': 0, 'negative': -0.8}
        sentiments = [sentiment_map.get(i.get('sentiment', 'neutral'), 0) for i in sorted_interviews]
        
        # 添加一些随机波动
        sentiments = [s + np.random.uniform(-0.2, 0.2) for s in sentiments]
        sentiments = [max(-1, min(1, s)) for s in sentiments]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=sentiments,
            mode='lines+markers',
            name='媒体情感',
            line=dict(color='#42A5F5', width=2),
            marker=dict(size=10),
            hovertemplate='%{x}<br>情感：%{y:.2f}<extra></extra>',
        ))
        
        # 添加正负区域
        fig.add_hrect(y0=0.3, y1=1, fillcolor='#66BB6A', opacity=0.1, line_width=0)
        fig.add_hrect(y0=-0.3, y1=0.3, fillcolor='#FFCA28', opacity=0.1, line_width=0)
        fig.add_hrect(y0=-1, y1=-0.3, fillcolor='#EF5350', opacity=0.1, line_width=0)
        
        fig.update_layout(
            title='媒体采访情感趋势',
            height=250,
            showlegend=False,
            yaxis_range=[-1, 1],
            yaxis=dict(tickformat='.1f'),
        )
        
        return fig
