"""
天盾 - 用户持仓管理模块
"""

import numpy as np

class PortfolioManager:
    """持仓管理器"""
    
    def __init__(self):
        self.holdings = {}
    
    def add_stock(self, stock_code, stock_name, shares, cost_basis):
        """添加持仓"""
        self.holdings[stock_code] = {
            'code': stock_code,
            'name': stock_name,
            'shares': shares,
            'cost_basis': cost_basis,
            'current_price': 0,
            'market_value': 0,
            'profit_loss': 0,
            'profit_loss_pct': 0,
            'weight': 0,
        }
    
    def update_prices(self, price_data_dict):
        """更新所有持仓的当前价格"""
        total_value = 0
        
        # 先计算总市值
        for code, holding in self.holdings.items():
            if code in price_data_dict:
                holding['current_price'] = float(price_data_dict[code]['close'].iloc[-1])
                holding['market_value'] = holding['shares'] * holding['current_price']
                holding['profit_loss'] = (holding['current_price'] - holding['cost_basis']) * holding['shares']
                holding['profit_loss_pct'] = (holding['current_price'] - holding['cost_basis']) / holding['cost_basis'] * 100
                total_value += holding['market_value']
        
        # 计算持仓占比
        for holding in self.holdings.values():
            if total_value > 0:
                holding['weight'] = holding['market_value'] / total_value * 100
    
    def get_portfolio_risk(self, risk_results_dict):
        """
        计算组合整体风险
        基于各股票风险分和持仓权重
        """
        total_risk = 0
        max_risk = 0
        high_risk_stocks = []
        
        for code, holding in self.holdings.items():
            if code in risk_results_dict:
                risk_score = risk_results_dict[code]['comprehensive_risk']
                weight = holding['weight'] / 100
                
                total_risk += risk_score * weight
                
                if risk_score > max_risk:
                    max_risk = risk_score
                
                if risk_score >= 60:
                    high_risk_stocks.append({
                        'name': holding['name'],
                        'risk_score': risk_score,
                        'weight': holding['weight']
                    })
        
        return {
            'portfolio_risk': round(total_risk, 1),
            'max_single_risk': round(max_risk, 1),
            'high_risk_stocks': high_risk_stocks,
            'total_value': sum(h['market_value'] for h in self.holdings.values()),
            'total_profit_loss': sum(h['profit_loss'] for h in self.holdings.values()),
        }
    
    def get_alert_adjustment(self, stock_code, base_alert_level):
        """
        根据持仓占比调整预警级别
        持仓超过20%，预警升级
        """
        if stock_code not in self.holdings:
            return base_alert_level
        
        weight = self.holdings[stock_code]['weight']
        
        if weight > 20 and base_alert_level == 'yellow':
            return 'red'
        elif weight > 10 and base_alert_level == 'green':
            return 'yellow'
        
        return base_alert_level
