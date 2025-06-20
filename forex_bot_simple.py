#!/usr/bin/env python3
"""
Bot de Trading Forex Simplificado - LIT + ML.

Versión funcional inmediata para trading Forex profesional.
"""

import sys
import os
import time
from datetime import datetime
import pandas as pd

# Configurar path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.brokers.forex_data_provider import ForexDataProvider
from src.utils.logger import log
from validate_demo_account import SimpleDemoAccount


class ForexBotSimple:
    """Bot de Trading Forex Simplificado pero Profesional."""
    
    def __init__(self):
        """Inicializa el bot Forex."""
        
        # Configuración
        self.account_balance = 2865.05
        self.risk_per_trade = 0.02  # 2% por operación
        self.max_positions = 3
        self.min_confidence = 0.65
        
        # Pares de divisas principales
        self.forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURGBP']
        
        # Componentes
        self.forex_provider = ForexDataProvider()
        self.demo_account = SimpleDemoAccount()
        
        # Estado
        self.active_positions = {}
        self.total_trades = 0
        self.daily_pnl = 0.0
        
        log.info("🌍 ForexBotSimple inicializado")
        log.info(f"💰 Balance: ${self.account_balance:,.2f}")
        log.info(f"📊 Pares: {', '.join(self.forex_pairs)}")
    
    def initialize(self):
        """Inicializa el bot."""
        try:
            log.info("🔧 Inicializando bot Forex...")
            
            # Conectar cuenta
            self.demo_account.connect()
            balance = self.demo_account.get_account_balance()
            if balance > 0:
                self.account_balance = balance
            
            # Verificar datos
            self._verify_forex_data()
            
            log.info("✅ Bot Forex inicializado correctamente")
            
        except Exception as e:
            log.error(f"❌ Error inicializando: {str(e)}")
    
    def _verify_forex_data(self):
        """Verifica datos de Forex."""
        try:
            valid_pairs = []
            
            for pair in self.forex_pairs:
                data = self.forex_provider.get_forex_data(pair, '1h', 50)
                if len(data) >= 50:
                    valid_pairs.append(pair)
                    log.info(f"✅ {pair}: {len(data)} filas disponibles")
                else:
                    log.warning(f"⚠️  {pair}: Solo {len(data)} filas")
            
            self.forex_pairs = valid_pairs
            log.info(f"📊 Pares válidos: {len(self.forex_pairs)}")
            
        except Exception as e:
            log.error(f"Error verificando datos: {str(e)}")
    
    def analyze_forex_pair(self, pair: str) -> dict:
        """Analiza un par de divisas con lógica simplificada."""
        try:
            # Obtener datos
            data = self.forex_provider.get_forex_data(pair, '1h', 100)
            
            if len(data) < 50:
                return {'signal': 0, 'confidence': 0.0, 'reason': 'Datos insuficientes'}
            
            # Análisis técnico básico
            current_price = data['close'].iloc[-1]
            
            # SMA 20 y 50
            sma_20 = data['close'].tail(20).mean()
            sma_50 = data['close'].tail(50).mean()
            
            # RSI simplificado
            price_changes = data['close'].diff()
            gains = price_changes.where(price_changes > 0, 0)
            losses = -price_changes.where(price_changes < 0, 0)
            avg_gain = gains.tail(14).mean()
            avg_loss = losses.tail(14).mean()
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50
            
            # Volatilidad (ATR simplificado)
            ranges = data['high'] - data['low']
            atr = ranges.tail(14).mean()
            
            # Lógica de señales
            signal = 0
            confidence = 0.0
            reasons = []
            
            # Señal alcista
            if (current_price > sma_20 > sma_50 and 
                rsi < 70 and rsi > 30):
                signal = 1
                confidence += 0.4
                reasons.append("Tendencia alcista")
            
            # Señal bajista
            elif (current_price < sma_20 < sma_50 and 
                  rsi > 30 and rsi < 70):
                signal = -1
                confidence += 0.4
                reasons.append("Tendencia bajista")
            
            # Ajustes por volatilidad
            if 'range_pips' in data.columns:
                current_range = data['range_pips'].iloc[-1]
                avg_range = data['range_pips'].tail(20).mean()
                
                if current_range > avg_range * 1.2:
                    confidence += 0.1
                    reasons.append("Alta volatilidad")
            
            # Ajustes por sesión
            market_status = self.forex_provider.get_market_status()
            if market_status.get('session_overlap', False):
                confidence += 0.15
                reasons.append("Solapamiento de sesiones")
            
            # Calcular niveles
            entry_levels = {}
            if signal != 0 and confidence >= 0.5:
                entry_levels = self._calculate_simple_levels(data, signal, pair)
            
            return {
                'pair': pair,
                'signal': signal,
                'confidence': confidence,
                'reason': ', '.join(reasons) if reasons else 'Sin señal clara',
                'entry_levels': entry_levels,
                'current_price': current_price,
                'rsi': rsi,
                'market_session': market_status.get('active_sessions', [])
            }
            
        except Exception as e:
            log.error(f"Error analizando {pair}: {str(e)}")
            return {'signal': 0, 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    def _calculate_simple_levels(self, data: pd.DataFrame, signal: int, pair: str) -> dict:
        """Calcula niveles de entrada simplificados."""
        try:
            current_price = data['close'].iloc[-1]
            
            # ATR para niveles dinámicos
            ranges = data['high'] - data['low']
            atr = ranges.tail(14).mean()
            
            # Obtener información del par
            pair_info = self.forex_provider.all_pairs.get(pair, {})
            pip_value = pair_info.get('pip_value', 0.0001)
            spread = pair_info.get('spread', 2.0)
            
            # Convertir ATR a pips
            atr_pips = atr / pip_value
            
            if signal == 1:  # BUY
                entry_price = current_price + (spread * pip_value / 2)
                stop_loss_pips = min(max(atr_pips * 0.8, 20), 50)
                take_profit_pips = min(max(atr_pips * 1.5, 30), 80)
                
                stop_loss = entry_price - (stop_loss_pips * pip_value)
                take_profit = entry_price + (take_profit_pips * pip_value)
                
            else:  # SELL
                entry_price = current_price - (spread * pip_value / 2)
                stop_loss_pips = min(max(atr_pips * 0.8, 20), 50)
                take_profit_pips = min(max(atr_pips * 1.5, 30), 80)
                
                stop_loss = entry_price + (stop_loss_pips * pip_value)
                take_profit = entry_price - (take_profit_pips * pip_value)
            
            return {
                'entry_price': round(entry_price, 5),
                'stop_loss': round(stop_loss, 5),
                'take_profit': round(take_profit, 5),
                'stop_loss_pips': round(stop_loss_pips, 1),
                'take_profit_pips': round(take_profit_pips, 1),
                'risk_reward_ratio': round(take_profit_pips / stop_loss_pips, 2),
                'atr_pips': round(atr_pips, 1)
            }
            
        except Exception as e:
            log.error(f"Error calculando niveles: {str(e)}")
            return {}
    
    def find_opportunities(self) -> list:
        """Encuentra oportunidades de trading."""
        try:
            opportunities = []
            
            for pair in self.forex_pairs:
                analysis = self.analyze_forex_pair(pair)
                
                if (analysis['signal'] != 0 and 
                    analysis['confidence'] >= self.min_confidence):
                    opportunities.append(analysis)
            
            # Ordenar por confianza
            opportunities.sort(key=lambda x: x['confidence'], reverse=True)
            
            return opportunities
            
        except Exception as e:
            log.error(f"Error buscando oportunidades: {str(e)}")
            return []
    
    def execute_trade(self, opportunity: dict) -> bool:
        """Ejecuta una operación."""
        try:
            pair = opportunity['pair']
            signal = opportunity['signal']
            
            # Verificar si podemos operar
            if len(self.active_positions) >= self.max_positions:
                log.info(f"⚠️  Máximo de posiciones alcanzado ({self.max_positions})")
                return False
            
            if pair in self.active_positions:
                log.info(f"⚠️  Ya hay posición activa en {pair}")
                return False
            
            # Calcular tamaño de posición
            entry_levels = opportunity.get('entry_levels', {})
            if not entry_levels:
                log.warning(f"⚠️  No hay niveles de entrada para {pair}")
                return False
            
            position_size = self._calculate_position_size(pair, entry_levels)
            
            if position_size <= 0:
                log.warning(f"⚠️  Tamaño de posición inválido para {pair}")
                return False
            
            # Simular ejecución
            side = 'BUY' if signal == 1 else 'SELL'
            
            log.info(f"🎯 Ejecutando: {side} {pair}")
            log.info(f"   💰 Tamaño: {position_size:,.0f} unidades")
            log.info(f"   📊 Entrada: {entry_levels['entry_price']:.5f}")
            log.info(f"   🛡️  SL: {entry_levels['stop_loss_pips']:.1f} pips")
            log.info(f"   🎯 TP: {entry_levels['take_profit_pips']:.1f} pips")
            log.info(f"   📈 R:R: {entry_levels['risk_reward_ratio']:.2f}")
            
            # Registrar posición
            self.active_positions[pair] = {
                'side': side,
                'size': position_size,
                'entry_price': entry_levels['entry_price'],
                'stop_loss': entry_levels['stop_loss'],
                'take_profit': entry_levels['take_profit'],
                'open_time': datetime.now(),
                'confidence': opportunity['confidence']
            }
            
            self.total_trades += 1
            return True
            
        except Exception as e:
            log.error(f"Error ejecutando trade: {str(e)}")
            return False
    
    def _calculate_position_size(self, pair: str, entry_levels: dict) -> float:
        """Calcula tamaño de posición."""
        try:
            # Riesgo monetario
            risk_amount = self.account_balance * self.risk_per_trade
            
            # Riesgo en pips
            risk_pips = entry_levels.get('stop_loss_pips', 20)
            
            # Información del par
            pair_info = self.forex_provider.all_pairs.get(pair, {})
            pip_value = pair_info.get('pip_value', 0.0001)
            
            # Calcular tamaño (simplificado)
            if risk_pips > 0:
                pip_value_per_unit = pip_value
                position_size = risk_amount / (risk_pips * pip_value_per_unit)
                
                # Redondear a miles
                position_size = round(position_size / 1000) * 1000
                
                # Limitar
                position_size = max(1000, min(position_size, 50000))
                
                return position_size
            
            return 0
            
        except Exception as e:
            log.error(f"Error calculando tamaño: {str(e)}")
            return 0
    
    def run_trading_loop(self):
        """Ejecuta el bucle principal de trading."""
        try:
            log.info("🚀 Iniciando trading Forex 24/7...")
            
            cycle = 0
            
            while True:
                try:
                    cycle += 1
                    current_time = datetime.now().strftime('%H:%M:%S')
                    
                    log.info(f"📊 Ciclo {cycle} - {current_time}")
                    
                    # Obtener estado del mercado
                    market_status = self.forex_provider.get_market_status()
                    active_sessions = market_status.get('active_sessions', [])
                    
                    log.info(f"🌍 Sesiones activas: {', '.join(active_sessions) if active_sessions else 'Ninguna'}")
                    
                    # Buscar oportunidades
                    opportunities = self.find_opportunities()
                    
                    if opportunities:
                        log.info(f"🎯 Encontradas {len(opportunities)} oportunidades:")
                        
                        for i, opp in enumerate(opportunities[:3], 1):
                            signal_text = "BUY" if opp['signal'] == 1 else "SELL"
                            log.info(f"   {i}. {opp['pair']}: {signal_text} "
                                   f"(Confianza: {opp['confidence']:.1%}) - {opp['reason']}")
                        
                        # Ejecutar mejor oportunidad si hay espacio
                        if len(self.active_positions) < self.max_positions:
                            best_opportunity = opportunities[0]
                            self.execute_trade(best_opportunity)
                    else:
                        log.info("📊 No hay oportunidades en este momento")
                    
                    # Mostrar resumen
                    self._show_summary()
                    
                    # Esperar 5 minutos
                    log.info("⏳ Esperando 5 minutos para próximo análisis...")
                    time.sleep(300)
                    
                except KeyboardInterrupt:
                    log.info("🛑 Deteniendo bot por solicitud del usuario...")
                    break
                    
                except Exception as e:
                    log.error(f"❌ Error en ciclo de trading: {str(e)}")
                    time.sleep(60)  # Esperar 1 minuto antes de reintentar
                    
        except Exception as e:
            log.error(f"❌ Error crítico en bucle: {str(e)}")
    
    def _show_summary(self):
        """Muestra resumen del bot."""
        try:
            log.info(f"📊 RESUMEN:")
            log.info(f"   💰 Balance: ${self.account_balance:,.2f}")
            log.info(f"   🔄 Posiciones Activas: {len(self.active_positions)}")
            log.info(f"   📈 Total Trades: {self.total_trades}")
            
            if self.active_positions:
                log.info(f"   📊 Pares Activos:")
                for pair, pos in self.active_positions.items():
                    log.info(f"      {pair}: {pos['side']} "
                           f"({pos['confidence']:.1%} confianza)")
                           
        except Exception as e:
            log.error(f"Error mostrando resumen: {str(e)}")


def main():
    """Función principal."""
    try:
        print("🌍 BOT DE TRADING FOREX PROFESIONAL")
        print("=" * 50)
        print(f"🕐 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Crear e inicializar bot
        bot = ForexBotSimple()
        bot.initialize()
        
        # Ejecutar trading
        bot.run_trading_loop()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error crítico: {str(e)}")


if __name__ == "__main__":
    main() 