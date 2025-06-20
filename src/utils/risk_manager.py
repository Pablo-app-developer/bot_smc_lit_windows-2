"""
Gestor de Riesgo Profesional - Risk Manager.

Maneja todos los aspectos relacionados con la gestión de riesgo en el trading:
- Cálculo de tamaño de posición
- Gestión de stop loss y take profit
- Control de drawdown
- Límites por trade y diarios
"""

from typing import Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
from src.utils.logger import log


class RiskManager:
    """
    Gestor de Riesgo Profesional.
    
    Controla el riesgo mediante:
    - Sizing de posiciones basado en volatilidad
    - Límites por trade (% del balance)
    - Límites diarios de pérdidas
    - Drawdown máximo permitido
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Inicializa el gestor de riesgo."""
        
        # Configuración por defecto
        self.config = config or {}
        
        # Límites de riesgo
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.01)  # 1%
        self.max_daily_risk = self.config.get('max_daily_risk', 0.05)  # 5%
        self.max_drawdown = self.config.get('max_drawdown', 0.15)  # 15%
        self.max_positions = self.config.get('max_positions', 3)
        
        # Estado del día
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        
        # Histórico de riesgo
        self.risk_history = []
        
        log.info("RiskManager inicializado")
        log.info(f"Max riesgo por trade: {self.max_risk_per_trade:.1%}")
        log.info(f"Max riesgo diario: {self.max_daily_risk:.1%}")
        log.info(f"Max drawdown: {self.max_drawdown:.1%}")
    
    def calculate_position_size(self, account_balance: float, entry_price: float, 
                              stop_loss: float, symbol: str = "AAPL") -> Dict:
        """
        Calcula el tamaño óptimo de posición.
        
        Args:
            account_balance: Balance de la cuenta
            entry_price: Precio de entrada
            stop_loss: Precio de stop loss
            symbol: Símbolo del instrumento
            
        Returns:
            Dict con información del sizing
        """
        try:
            # Riesgo monetario por trade
            risk_amount = account_balance * self.max_risk_per_trade
            
            # Riesgo por acción
            risk_per_share = abs(entry_price - stop_loss)
            
            if risk_per_share <= 0:
                log.warning("Riesgo por acción es 0 o negativo")
                return {
                    'position_size': 0,
                    'risk_amount': 0,
                    'risk_per_share': 0,
                    'error': 'Riesgo inválido'
                }
            
            # Calcular tamaño base
            base_size = risk_amount / risk_per_share
            
            # Ajustar por liquidez (simplificado)
            liquidity_multiplier = self._get_liquidity_multiplier(symbol)
            adjusted_size = base_size * liquidity_multiplier
            
            # Redondear a entero
            final_size = max(1, int(adjusted_size))
            
            # Verificar límites
            max_size_by_balance = int(account_balance * 0.3 / entry_price)  # Max 30% del balance
            final_size = min(final_size, max_size_by_balance)
            
            # Calcular riesgo real
            actual_risk = final_size * risk_per_share
            actual_risk_pct = actual_risk / account_balance
            
            log.info(f"💼 Position Sizing para {symbol}:")
            log.info(f"   Balance: ${account_balance:,.2f}")
            log.info(f"   Riesgo objetivo: ${risk_amount:.2f} ({self.max_risk_per_trade:.1%})")
            log.info(f"   Tamaño calculado: {final_size} acciones")
            log.info(f"   Riesgo real: ${actual_risk:.2f} ({actual_risk_pct:.2%})")
            
            return {
                'position_size': final_size,
                'risk_amount': actual_risk,
                'risk_percentage': actual_risk_pct,
                'risk_per_share': risk_per_share,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'success': True
            }
            
        except Exception as e:
            log.error(f"Error calculando position size: {str(e)}")
            return {
                'position_size': 0,
                'risk_amount': 0,
                'error': str(e),
                'success': False
            }
    
    def validate_trade(self, account_balance: float, position_size: int, 
                      entry_price: float, stop_loss: float) -> Dict:
        """
        Valida si un trade cumple los criterios de riesgo.
        
        Args:
            account_balance: Balance actual
            position_size: Tamaño de posición propuesto
            entry_price: Precio de entrada
            stop_loss: Stop loss
            
        Returns:
            Dict con resultado de validación
        """
        try:
            # Resetear estadísticas diarias si es necesario
            self._reset_daily_stats_if_needed()
            
            validation_results = {
                'is_valid': True,
                'reasons': [],
                'warnings': []
            }
            
            # 1. Validar riesgo por trade
            risk_per_trade = position_size * abs(entry_price - stop_loss)
            risk_pct = risk_per_trade / account_balance
            
            if risk_pct > self.max_risk_per_trade:
                validation_results['is_valid'] = False
                validation_results['reasons'].append(
                    f"Riesgo por trade muy alto: {risk_pct:.2%} > {self.max_risk_per_trade:.2%}"
                )
            
            # 2. Validar límite diario
            potential_daily_risk = abs(self.daily_pnl) + risk_per_trade
            daily_risk_pct = potential_daily_risk / account_balance
            
            if daily_risk_pct > self.max_daily_risk:
                validation_results['is_valid'] = False
                validation_results['reasons'].append(
                    f"Riesgo diario excedido: {daily_risk_pct:.2%} > {self.max_daily_risk:.2%}"
                )
            
            # 3. Validar que el stop loss sea razonable
            if abs(entry_price - stop_loss) / entry_price > 0.1:  # 10%
                validation_results['warnings'].append(
                    "Stop loss muy amplio (>10%)"
                )
            
            if abs(entry_price - stop_loss) / entry_price < 0.005:  # 0.5%
                validation_results['warnings'].append(
                    "Stop loss muy ajustado (<0.5%)"
                )
            
            # 4. Validar tamaño mínimo
            if position_size < 1:
                validation_results['is_valid'] = False
                validation_results['reasons'].append("Tamaño de posición muy pequeño")
            
            # Log resultado
            if validation_results['is_valid']:
                log.info(f"✅ Trade validado - Riesgo: {risk_pct:.2%}")
            else:
                log.warning(f"❌ Trade rechazado: {', '.join(validation_results['reasons'])}")
            
            if validation_results['warnings']:
                for warning in validation_results['warnings']:
                    log.warning(f"⚠️  {warning}")
            
            return validation_results
            
        except Exception as e:
            log.error(f"Error validando trade: {str(e)}")
            return {
                'is_valid': False,
                'reasons': [f"Error de validación: {str(e)}"],
                'warnings': []
            }
    
    def update_daily_pnl(self, pnl: float):
        """Actualiza el PnL diario."""
        try:
            self._reset_daily_stats_if_needed()
            self.daily_pnl += pnl
            self.daily_trades += 1
            
            log.info(f"📊 PnL diario actualizado: ${self.daily_pnl:+.2f} ({self.daily_trades} trades)")
            
            # Verificar límites diarios
            if self.daily_pnl < -abs(self.max_daily_risk * 1000):  # Simplificado
                log.warning(f"⚠️  Límite de pérdida diaria alcanzado: ${self.daily_pnl:.2f}")
                
        except Exception as e:
            log.error(f"Error actualizando PnL diario: {str(e)}")
    
    def check_daily_limits(self, account_balance: float) -> bool:
        """Verifica si se pueden hacer más trades hoy."""
        try:
            self._reset_daily_stats_if_needed()
            
            # Límite de pérdida diaria
            daily_loss_limit = account_balance * self.max_daily_risk
            
            if abs(self.daily_pnl) >= daily_loss_limit and self.daily_pnl < 0:
                log.warning(f"⛔ Límite de pérdida diaria alcanzado: ${self.daily_pnl:.2f}")
                return False
            
            # Límite de trades diarios (opcional)
            max_daily_trades = 20  # Máximo 20 trades por día
            if self.daily_trades >= max_daily_trades:
                log.warning(f"⛔ Límite de trades diarios alcanzado: {self.daily_trades}")
                return False
            
            return True
            
        except Exception as e:
            log.error(f"Error verificando límites diarios: {str(e)}")
            return False
    
    def calculate_stop_loss(self, entry_price: float, side: str, 
                           atr: Optional[float] = None, 
                           risk_pct: float = 0.02) -> float:
        """
        Calcula stop loss basado en ATR o porcentaje.
        
        Args:
            entry_price: Precio de entrada
            side: 'buy' o 'sell'
            atr: Average True Range (opcional)
            risk_pct: Porcentaje de riesgo por defecto
            
        Returns:
            Precio de stop loss
        """
        try:
            if atr and atr > 0:
                # Stop loss basado en ATR (más profesional)
                atr_multiplier = 1.5  # 1.5x ATR
                stop_distance = atr * atr_multiplier
            else:
                # Stop loss basado en porcentaje
                stop_distance = entry_price * risk_pct
            
            if side.lower() == 'buy':
                stop_loss = entry_price - stop_distance
            else:  # sell
                stop_loss = entry_price + stop_distance
            
            return round(stop_loss, 2)
            
        except Exception as e:
            log.error(f"Error calculando stop loss: {str(e)}")
            # Fallback al porcentaje fijo
            if side.lower() == 'buy':
                return entry_price * (1 - risk_pct)
            else:
                return entry_price * (1 + risk_pct)
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                            side: str, risk_reward_ratio: float = 2.0) -> float:
        """
        Calcula take profit basado en risk-reward ratio.
        
        Args:
            entry_price: Precio de entrada
            stop_loss: Stop loss
            side: 'buy' o 'sell'
            risk_reward_ratio: Ratio riesgo-beneficio (2:1 por defecto)
            
        Returns:
            Precio de take profit
        """
        try:
            risk_distance = abs(entry_price - stop_loss)
            reward_distance = risk_distance * risk_reward_ratio
            
            if side.lower() == 'buy':
                take_profit = entry_price + reward_distance
            else:  # sell
                take_profit = entry_price - reward_distance
            
            return round(take_profit, 2)
            
        except Exception as e:
            log.error(f"Error calculando take profit: {str(e)}")
            # Fallback
            if side.lower() == 'buy':
                return entry_price * 1.04  # 4% profit
            else:
                return entry_price * 0.96
    
    def _get_liquidity_multiplier(self, symbol: str) -> float:
        """Obtiene multiplicador de liquidez por símbolo."""
        # Símbolos de alta liquidez
        high_liquidity = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Forex majors
        forex_majors = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
        
        if symbol in high_liquidity or symbol in forex_majors:
            return 1.0  # Sin ajuste
        else:
            return 0.8  # Reducir tamaño para menor liquidez
    
    def _reset_daily_stats_if_needed(self):
        """Resetea estadísticas diarias si cambió el día."""
        current_date = datetime.now().date()
        
        if current_date != self.last_reset_date:
            log.info(f"📅 Nuevo día de trading - Reseteando estadísticas diarias")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = current_date
    
    def get_risk_summary(self, account_balance: float) -> Dict:
        """Obtiene resumen del estado de riesgo."""
        try:
            self._reset_daily_stats_if_needed()
            
            daily_risk_pct = abs(self.daily_pnl) / account_balance if account_balance > 0 else 0
            
            return {
                'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trades,
                'daily_risk_used': daily_risk_pct,
                'daily_risk_remaining': max(0, self.max_daily_risk - daily_risk_pct),
                'max_risk_per_trade': self.max_risk_per_trade,
                'can_trade': self.check_daily_limits(account_balance),
                'last_reset': self.last_reset_date.isoformat()
            }
            
        except Exception as e:
            log.error(f"Error obteniendo resumen de riesgo: {str(e)}")
            return {} 