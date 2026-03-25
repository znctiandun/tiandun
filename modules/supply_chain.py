import numpy as np

def get_plotly():
    import plotly.graph_objects as go
    return go

class SupplyChainModel:
    """供应链风险传染模型"""
    
    INDUSTRY_MAP = {
        '贵州茅台': {'industry': '白酒', 'upstream': [{'name': '高粱种植', 'impact': 0.3}, {'name': '包装材料', 'impact': 0.2}], 'downstream': [{'name': '经销商', 'impact': 0.4}, {'name': '电商平台', 'impact': 0.3}], 'peer': ['五粮液', '泸州老窖']},
        '宁德时代': {'industry': '动力电池', 'upstream': [{'name': '锂矿开采', 'impact': 0.4}, {'name': '正负极材料', 'impact': 0.3}], 'downstream': [{'name': '新能源汽车', 'impact': 0.6}, {'name': '储能系统', 'impact': 0.3}], 'peer': ['比亚迪', '亿纬锂能']},
        '比亚迪': {'industry': '新能源汽车', 'upstream': [{'name': '电池供应商', 'impact': 0.4}, {'name': '芯片/电子', 'impact': 0.3}], 'downstream': [{'name': '个人消费者', 'impact': 0.6}, {'name': '网约车平台', 'impact': 0.2}], 'peer': ['特斯拉', '蔚来汽车']},
    }
    
    def calculate_contagion_risk(self, stock_code):
        """计算供应链传染风险"""
        np.random.seed(hash(stock_code) % 2**32)
        return {
            'stock_code': stock_code,
            'contagion_risk': round(np.random.uniform(20, 60), 1),
            'affected_nodes': np.random.randint(3, 8)
        }
    
    def get_industry_impact_map(self, stock_code, stock_name):
        """获取产业链影响地图"""
        if stock_name in self.INDUSTRY_MAP:
            data = self.INDUSTRY_MAP[stock_name]
            return {
                'center': {'name': stock_name, 'industry': data['industry']},
                'upstream': data['upstream'],
                'downstream': data['downstream'],
                'peer': data['peer']
            }
        return {
            'center': {'name': stock_name, 'industry': '通用制造业'},
            'upstream': [{'name': '原材料', 'impact': 0.4}],
            'downstream': [{'name': '经销商', 'impact': 0.4}],
            'peer': ['同行业 A']
        }
    
    def create_industry_chain_visualization(self, stock_code, stock_name):
        """创建产业链可视化图"""
        go = get_plotly()
        data = self.get_industry_impact_map(stock_code, stock_name)
        
        labels = [data['center']['name']]
        parents = ['']
        values = [100]
        colors = ['#42A5F5']
        
        for u in data['upstream']:
            labels.append(u['name'])
            parents.append(data['center']['name'])
            values.append(u['impact'] * 100)
            colors.append('#AB47BC')
        
        for d in data['downstream']:
            labels.append(d['name'])
            parents.append(data['center']['name'])
            values.append(d['impact'] * 100)
            colors.append('#66BB6A')
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker_colors=colors
        ))
        fig.update_layout(title=f'{stock_name} - 产业链影响地图', height=500)
        return fig
    
    def explain_industry_risk_for_common_user(self, stock_name):
        """用通俗语言解释产业风险"""
        if stock_name in self.INDUSTRY_MAP:
            d = self.INDUSTRY_MAP[stock_name]
            explanation = f"## 🏭 {stock_name}\n\n**行业**: {d['industry']}\n\n**上游供应商**:\n"
            for u in d['upstream']:
                explanation += f"- {u['name']}\n"
            explanation += "\n**下游客户**:\n"
            for dd in d['downstream']:
                explanation += f"- {dd['name']}\n"
            return explanation
        return f"## 🏭 {stock_name}\n\n产业数据暂缺"
