"""
天盾 - 隐性风险识别模块
识别"表面平稳实际危险"的公司（财务造假、风险伪装等）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime

class HiddenRiskDetector:
    """隐性风险检测器"""
    
    # 风险伪装识别规则
    RED_FLAGS = {
        'profit_cash_mismatch': {
            'name': '利润与现金流不匹配',
            'description': '公司账面利润很高，但实际现金流入很少，可能存在虚增利润',
            'weight': 0.25,
            'threshold': 0.5,  # 经营现金流/净利润 < 0.5 触发
        },
        'receivables_surge': {
            'name': '应收账款异常增长',
            'description': '应收账款增速远超过收入增速，可能存在虚假销售',
            'weight': 0.20,
            'threshold': 1.5,  # 应收账款增速/收入增速 > 1.5 触发
        },
        'inventory_abnormal': {
            'name': '存货异常积压',
            'description': '存货增速异常，可能产品滞销或存在存货造假',
            'weight': 0.15,
            'threshold': 1.5,
        },
        'high Debt_low Cash': {
            'name': '高负债低现金',
            'description': '负债率高但货币资金少，可能存在资金链断裂风险',
            'weight': 0.20,
            'threshold': 0.3,  # 货币资金/流动负债 < 0.3 触发
        },
        'executive_reduce': {
            'name': '高管密集减持',
            'description': '高管频繁减持股票，内部人不看好公司前景',
            'weight': 0.20,
            'threshold': 3,  # 6个月内减持次数 > 3 触发
        },
    }
    
    def __init__(self):
        self.risk_records = {}
    
    def detect_hidden_risks(self, stock_code: str, stock_name: str, 
                           financials: Dict, price_data: pd.DataFrame = None) -> Dict:
        """
        检测隐性风险
        返回：包含各风险指标和综合风险评分的字典
        """
        flags_triggered = []
        total_risk_score = 0
        
        # 1. 利润与现金流匹配度检测
        if self._check_profit_cash_mismatch(financials):
            flags_triggered.append({
                'type': 'profit_cash_mismatch',
                'severity': 'high',
                'score': 85,
                'detail': f"经营现金流/净利润 = {financials.get('cash_profit_ratio', 0.3):.2f}，远低于安全线0.5",
                'explanation': self._explain_for_common_user('profit_cash_mismatch'),
            })
            total_risk_score += self.RED_FLAGS['profit_cash_mismatch']['weight'] * 100
        
        # 2. 应收账款异常检测
        if self._check_receivables_surge(financials):
            flags_triggered.append({
                'type': 'receivables_surge',
                'severity': 'medium',
                'score': 70,
                'detail': f"应收账款增速 {financials.get('receivables_growth', 60):.1f}% 远超收入增速 {financials.get('revenue_growth', 20):.1f}%",
                'explanation': self._explain_for_common_user('receivables_surge'),
            })
            total_risk_score += self.RED_FLAGS['receivables_surge']['weight'] * 100
        
        # 3. 存货异常检测
        if self._check_inventory_abnormal(financials):
            flags_triggered.append({
                'type': 'inventory_abnormal',
                'severity': 'medium',
                'score': 65,
                'detail': f"存货周转天数从{financials.get('prev_inventory_days', 60)}天增至{financials.get('inventory_days', 120)}天",
                'explanation': self._explain_for_common_user('inventory_abnormal'),
            })
            total_risk_score += self.RED_FLAGS['inventory_abnormal']['weight'] * 100
        
        # 4. 高负债低现金检测
        if self._check_high_debt_low_cash(financials):
            flags_triggered.append({
                'type': 'high_debt_low_cash',
                'severity': 'high',
                'score': 80,
                'detail': f"货币资金仅覆盖{financials.get('cash_coverage', 0.2):.1f}倍流动负债，安全线应>1",
                'explanation': self._explain_for_common_user('high_debt_low_cash'),
            })
            total_risk_score += self.RED_FLAGS['high Debt_low Cash']['weight'] * 100
        
        # 5. 高管减持检测（模拟数据）
        executive_reduce_count = financials.get('executive_reduce_count', 0)
        if executive_reduce_count >= 3:
            flags_triggered.append({
                'type': 'executive_reduce',
                'severity': 'medium',
                'score': 75,
                'detail': f"近6个月高管减持{executive_reduce_count}次，内部人套现{financials.get('executive_reduce_amount', 5000):.0f}万元",
                'explanation': self._explain_for_common_user('executive_reduce'),
            })
            total_risk_score += self.RED_FLAGS['executive_reduce']['weight'] * 100
        
        # 6. 股价与基本面背离检测
        if price_data is not None and len(price_data) >= 30:
            if self._check_price_fundamental_divergence(financials, price_data):
                flags_triggered.append({
                    'type': 'price_fundamental_divergence',
                    'severity': 'medium',
                    'score': 70,
                    'detail': f"股价上涨{financials.get('price_change', 30):.1f}%但利润下滑{abs(financials.get('profit_growth', -10)):.1f}%",
                    'explanation': self._explain_for_common_user('price_fundamental_divergence'),
                })
                total_risk_score += 0.15 * 100
        
        # 综合隐性风险评分
        hidden_risk_score = min(total_risk_score, 100)
        
        # 风险等级判定
        if hidden_risk_score >= 70:
            risk_level = 'high'
            risk_label = '⚠️ 高危预警'
            risk_color = '#EF5350'
        elif hidden_risk_score >= 40:
            risk_level = 'medium'
            risk_label = '⚡ 风险关注'
            risk_color = '#FFCA28'
        else:
            risk_level = 'low'
            risk_label = '✅ 相对安全'
            risk_color = '#66BB6A'
        
        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'hidden_risk_score': round(hidden_risk_score, 1),
            'risk_level': risk_level,
            'risk_label': risk_label,
            'risk_color': risk_color,
            'red_flags': flags_triggered,
            'flag_count': len(flags_triggered),
            'surface_score': self._calculate_surface_score(financials),  # 表面健康度
            'real_score': 100 - hidden_risk_score,  # 真实健康度
            'gap': self._calculate_surface_score(financials) - (100 - hidden_risk_score),  # 差距
            'summary': self._generate_summary(flags_triggered, hidden_risk_score),
        }
    
    def _check_profit_cash_mismatch(self, financials: Dict) -> bool:
        """检查利润与现金流不匹配"""
        cash_profit_ratio = financials.get('cash_profit_ratio', 0.8)
        return cash_profit_ratio < self.RED_FLAGS['profit_cash_mismatch']['threshold']
    
    def _check_receivables_surge(self, financials: Dict) -> bool:
        """检查应收账款异常增长"""
        receivables_growth = financials.get('receivables_growth', 20)
        revenue_growth = financials.get('revenue_growth', 20)
        if revenue_growth <= 0:
            return receivables_growth > 30
        ratio = receivables_growth / revenue_growth
        return ratio > self.RED_FLAGS['receivables_surge']['threshold']
    
    def _check_inventory_abnormal(self, financials: Dict) -> bool:
        """检查存货异常"""
        prev_days = financials.get('prev_inventory_days', 60)
        curr_days = financials.get('inventory_days', 60)
        if prev_days <= 0:
            return False
        ratio = curr_days / prev_days
        return ratio > self.RED_FLAGS['inventory_abnormal']['threshold']
    
    def _check_high_debt_low_cash(self, financials: Dict) -> bool:
        """检查高负债低现金"""
        cash_coverage = financials.get('cash_coverage', 0.5)
        return cash_coverage < self.RED_FLAGS['high Debt_low Cash']['threshold']
    
    def _check_price_fundamental_divergence(self, financials: Dict, price_data: pd.DataFrame) -> bool:
        """检查股价与基本面背离"""
        # 近30天股价涨幅
        price_change = (price_data['close'].iloc[-1] - price_data['close'].iloc[0]) / price_data['close'].iloc[0] * 100
        
        # 利润增长
        profit_growth = financials.get('profit_growth', 10)
        
        # 股价涨但利润降，或股价涨幅远超利润增幅
        return (price_change > 20 and profit_growth < 0) or (price_change - profit_growth > 30)
    
    def _explain_for_common_user(self, flag_type: str) -> str:
        """用通俗语言解释风险（面向普通用户）"""
        explanations = {
            'profit_cash_mismatch': """
                💡 通俗解释：这家公司账面上说赚了很多钱，但实际银行账户里没收到多少现金。
                
                🍎 举个例子：就像你开了一家店，账本上写着卖了100万的货，但实际只收到30万现金，
                剩下70万都是"白条"（别人欠你的钱）。这种情况下，你的"利润"可能是虚的。
                
                ⚠️ 风险：公司可能通过赊销虚增收入，或者收回来的钱都是"假钱"（应收账款无法收回）。
            """,
            
            'receivables_surge': """
                💡 通俗解释：公司卖东西后，别人欠的钱增长速度远超过实际销售增长速度。
                
                🍎 举个例子：你开店今年销售额增长20%，但别人欠你的钱却增长了60%。
                这说明你可能为了"做高"销售额，放宽了收款条件，或者有些销售根本没收到钱。
                
                ⚠️ 风险：这些"欠条"可能永远收不回来，变成坏账。
            """,
            
            'inventory_abnormal': """
                💡 通俗解释：仓库里的货越堆越多，卖不出去。
                
                🍎 举个例子：你开服装店，以前60天能把货卖完，现在要120天。
                说明衣服可能过时了，或者定价太高没人买。
                
                ⚠️ 风险：存货可能贬值，或者根本卖不出去变成废品。
            """,
            
            'high_debt_low_cash': """
                💡 通俗解释：欠了很多债，但手里现金很少，还不起的风险很高。
                
                🍎 举个例子：你欠了100万债，但银行卡里只有20万。
                如果债主突然要你还钱，你就拿不出来。
                
                ⚠️ 风险：可能发生资金链断裂，公司倒闭。
            """,
            
            'executive_reduce': """
                💡 通俗解释：公司高管频繁卖自己公司的股票套现。
                
                🍎 举个例子：船长说这艘船很安全，但自己却偷偷准备救生艇要跑。
                高管比任何人都了解公司真实情况，他们减持说明不看好公司前景。
                
                ⚠️ 风险：内部人都不看好，普通投资者更要小心。
            """,
            
            'price_fundamental_divergence': """
                💡 通俗解释：股价涨得很好，但公司实际赚钱能力在下降。
                
                🍎 举个例子：一个苹果实际只值5块钱，但被炒到了10块。
                价格上涨不是因为苹果变好了，而是因为有人在炒作。
                
                ⚠️ 风险：炒作退潮后，股价会跌回真实价值。
            """,
        }
        
        return explanations.get(flag_type, "该风险因素需要专业人士进一步分析")
    
    def _calculate_surface_score(self, financials: Dict) -> float:
        """计算表面健康度（普通投资者容易看到的指标）"""
        score = 50
        
        # PE看起来合理
        pe = financials.get('pe_ratio', 25)
        if 15 <= pe <= 30:
            score += 15
        elif pe < 15:
            score += 10
        
        # 利润增长率看起来不错
        profit_growth = financials.get('profit_growth', 10)
        if profit_growth > 20:
            score += 20
        elif profit_growth > 10:
            score += 10
        
        # ROE看起来不错
        roe = financials.get('roe', 15)
        if roe > 15:
            score += 15
        elif roe > 10:
            score += 8
        
        return min(score, 100)
    
    def _generate_summary(self, flags: List[Dict], risk_score: float) -> str:
        """生成通俗总结"""
        if not flags:
            return "✅ 未发现明显隐性风险，公司财务状况相对透明健康"
        
        flag_names = [f['type'].replace('_', ' ') for f in flags]
        
        if risk_score >= 70:
            return f"⚠️ **高危预警**：发现{len(flags)}个重大风险信号，包括{flag_names[0]}等。" \
                   f"表面看公司可能还不错，但深层风险很高，建议谨慎投资。"
        elif risk_score >= 40:
            return f"⚡ **风险提示**：发现{len(flags)}个风险信号，需要进一步关注。" \
                   f"建议仔细观察公司后续财报和经营情况。"
        else:
            return f"ℹ️ **轻微关注**：发现{len(flags)}个轻微风险信号，属于正常波动范围。"
    
    def create_risk_comparison_chart(self, hidden_risk_result: Dict):
        """创建表面健康度 vs 真实健康度对比图"""
        import plotly.graph_objects as go
        
        surface = hidden_risk_result['surface_score']
        real = hidden_risk_result['real_score']
        gap = hidden_risk_result['gap']
        
        fig = go.Figure()
        
        # 表面健康度
        fig.add_trace(go.Bar(
            name='表面健康度',
            x=['健康度评分'],
            y=[surface],
            marker_color='#66BB6A',
            text=[f'{surface:.0f}分'],
            textposition='outside',
        ))
        
        # 真实健康度
        fig.add_trace(go.Bar(
            name='真实健康度',
            x=['健康度评分'],
            y=[real],
            marker_color='#EF5350' if gap > 20 else '#FFCA28' if gap > 10 else '#66BB6A',
            text=[f'{real:.0f}分'],
            textposition='outside',
        ))
        
        # 差异标注
        if gap > 20:
            fig.add_annotation(
                x='健康度评分',
                y=(surface + real) / 2,
                text=f'⚠️ 差异{gap:.0f}分',
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-50,
                font=dict(color='red', size=14),
            )
        
        fig.update_layout(
            title='表面健康度 vs 真实健康度对比',
            barmode='group',
            height=350,
            yaxis_range=[0, 100],
            showlegend=True,
        )
        
        return fig
