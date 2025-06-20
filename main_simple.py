#!/usr/bin/env python3
"""
Bot de Trading Automatizado LIT + ML - Versión Simplificada.

Sistema profesional de trading que combina:
- Análisis multi-timeframe (1d, 4h, 1h)
- Estrategia LIT (Liquidity + Inducement Theory)
- Machine Learning
- Conexión real a cuenta demo
"""

import os
import sys
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Configurar path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.config import config
from src.utils.logger import log
from src.data.data_loader import DataLoader
from src.strategies.lit_detector import LITDetector, SignalType
from src.models.predictor import LITMLPredictor
from src.models.feature_engineering import FeatureEngineer

# Importar cuenta demo
from validate_demo_account import SimpleDemoAccount


class SimpleTradingBot:
    """
    Bot de Trading Simplificado con ML.
    
    Características principales:
    - Machine Learning con LIT
    - Conexión real a cuenta demo
    - Gestión de riesgo básica
    """
    
    def __init__(self):
        """Inicializa el bot de trading."""
        log.info("🚀 Inicializando Bot de Trading LIT + ML")
        
        # Componentes principales
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.predictor = LITMLPredictor()
        
        # Cuenta demo
        self.demo_account = SimpleDemoAccount()
        
        # Configuración de trading
        self.symbol = config.trading.symbol
        self.timeframe = config.trading.timeframe
        self.balance_inicial = 2865.05  # Saldo real de la cuenta
        
        # Estado del bot
        self.is_running = False
        self.positions = {}
        self.last_analysis_time = None
        self.analysis_interval = 300  # 5 minutos
        
        # Estadísticas
        self.stats = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'trades_executed': 0,
            'successful_trades': 0,
            'total_pnl': 0.0,
            'start_time': datetime.now()
        }
        
        log.info(f"Símbolo configurado: {self.symbol}")
        log.info(f"Timeframe principal: {self.timeframe}")
        log.info(f"Balance inicial: ${self.balance_inicial:,.2f}")
    
    async def initialize(self) -> bool:
        """Inicializa todos los componentes del bot."""
        try:
            log.info("🔧 Inicializando componentes del bot...")
            
            # 1. Conectar a cuenta demo
            log.info("1️⃣ Conectando a cuenta demo...")
            connection_result = self.demo_account.connect()
            if not connection_result:
                log.error("❌ Error conectando a cuenta demo")
                return False
            
            # Verificar saldo
            account_info = self.demo_account.get_account_info()
            actual_balance = account_info['balance']
            
            log.info(f"✅ Cuenta conectada - Saldo: ${actual_balance:,.2f}")
            
            # 2. Cargar datos iniciales
            log.info("2️⃣ Cargando datos de mercado...")
            initial_data = self.data_loader.load_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                periods=200
            )
            
            if len(initial_data) < 50:
                log.error(f"❌ Datos insuficientes: {len(initial_data)} velas")
                return False
            
            log.info(f"✅ Datos cargados: {len(initial_data)} velas")
            
            # 3. Inicializar predictor ML
            log.info("3️⃣ Inicializando predictor ML...")
            
            try:
                # Preparar features
                data_with_features = self.feature_engineer.create_features(initial_data)
                
                # Entrenar modelo inicial
                self.predictor.train(data_with_features)
                log.info("✅ Modelo entrenado exitosamente")
                
            except Exception as e:
                log.error(f"❌ Error entrenando modelo: {str(e)}")
                log.warning("⚠️  Continuando sin modelo ML entrenado")
            
            log.info("🎯 Inicialización completada exitosamente")
            return True
            
        except Exception as e:
            log.error(f"❌ Error en inicialización: {str(e)}")
            return False
    
    async def run_trading_loop(self):
        """Ejecuta el bucle principal de trading."""
        log.info("🔄 Iniciando bucle de trading en vivo...")
        log.info(f"Intervalo de análisis: {self.analysis_interval} segundos")
        
        self.is_running = True
        
        try:
            while self.is_running:
                try:
                    # Verificar si es momento de analizar
                    now = datetime.now()
                    
                    if (self.last_analysis_time is None or 
                        (now - self.last_analysis_time).seconds >= self.analysis_interval):
                        
                        log.info(f"📊 Ejecutando análisis - {now.strftime('%H:%M:%S')}")
                        
                        # Ejecutar análisis completo
                        await self.execute_trading_analysis()
                        
                        self.last_analysis_time = now
                    
                    # Monitorear posiciones existentes
                    await self.monitor_positions()
                    
                    # Actualizar estadísticas
                    self.update_statistics()
                    
                    # Esperar antes del siguiente ciclo
                    await asyncio.sleep(30)  # Verificar cada 30 segundos
                    
                except KeyboardInterrupt:
                    log.info("🛑 Interrupción detectada - Deteniendo bot...")
                    break
                    
                except Exception as e:
                    log.error(f"❌ Error en bucle de trading: {str(e)}")
                    await asyncio.sleep(60)  # Esperar más tiempo en caso de error
                    
        except Exception as e:
            log.error(f"💥 Error crítico en bucle de trading: {str(e)}")
        finally:
            self.is_running = False
            log.info("🔴 Bucle de trading detenido")
    
    async def execute_trading_analysis(self):
        """Ejecuta análisis completo y toma decisiones de trading."""
        try:
            # 1. Obtener datos actuales
            current_data = self.data_loader.load_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                periods=100
            )
            
            if len(current_data) < 50:
                log.warning("⚠️  Datos insuficientes para análisis")
                return
            
            current_price = current_data['close'].iloc[-1]
            log.info(f"💰 Precio actual {self.symbol}: ${current_price:.5f}")
            
            # 2. Análisis ML
            log.info("🔍 Ejecutando análisis ML...")
            
            # Preparar features
            data_with_features = self.feature_engineer.create_features(current_data)
            
            # Predicción ML
            try:
                ml_prediction = self.predictor.predict(data_with_features)
                primary_signal = ml_prediction['signal']
                signal_confidence = ml_prediction['confidence']
                
                log.info(f"📊 Señal ML: {primary_signal.value}")
                log.info(f"   Confianza: {signal_confidence:.2%}")
                
            except Exception as e:
                log.error(f"Error en predicción ML: {str(e)}")
                primary_signal = SignalType.HOLD
                signal_confidence = 0.0
            
            # Configurar niveles de trading
            risk_level = "medium"
            entry_price = current_price
            
            if primary_signal == SignalType.BUY:
                stop_loss = current_price * 0.98  # 2% stop loss
                take_profit = current_price * 1.03  # 3% take profit
            elif primary_signal == SignalType.SELL:
                stop_loss = current_price * 1.02  # 2% stop loss
                take_profit = current_price * 0.97  # 3% take profit
            else:
                stop_loss = 0
                take_profit = 0
            
            # 3. Actualizar estadísticas
            self.stats['total_signals'] += 1
            if primary_signal == SignalType.BUY:
                self.stats['buy_signals'] += 1
            elif primary_signal == SignalType.SELL:
                self.stats['sell_signals'] += 1
            else:
                self.stats['hold_signals'] += 1
            
            # 4. Evaluar si ejecutar trade
            should_trade = await self.evaluate_trade_decision(
                signal=primary_signal,
                confidence=signal_confidence,
                risk_level=risk_level,
                current_price=current_price
            )
            
            if should_trade:
                # 5. Ejecutar trade
                await self.execute_trade(
                    signal=primary_signal,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=signal_confidence
                )
            else:
                log.info("⏸️  Señal no cumple criterios para trading")
            
        except Exception as e:
            log.error(f"❌ Error en análisis de trading: {str(e)}")
    
    async def evaluate_trade_decision(self, signal: SignalType, confidence: float, 
                                    risk_level: str, current_price: float) -> bool:
        """Evalúa si se debe ejecutar un trade basado en múltiples criterios."""
        try:
            # Criterios de decisión
            criteria_met = 0
            total_criteria = 5
            
            # 1. Confianza mínima
            min_confidence = 0.6  # 60%
            if confidence >= min_confidence:
                criteria_met += 1
                log.info(f"✅ Confianza suficiente: {confidence:.2%} >= {min_confidence:.2%}")
            else:
                log.info(f"❌ Confianza insuficiente: {confidence:.2%} < {min_confidence:.2%}")
            
            # 2. Señal no es HOLD
            if signal != SignalType.HOLD:
                criteria_met += 1
                log.info(f"✅ Señal activa: {signal.value}")
            else:
                log.info("❌ Señal es HOLD")
            
            # 3. Riesgo aceptable
            acceptable_risk_levels = ["low", "medium"]
            if risk_level in acceptable_risk_levels:
                criteria_met += 1
                log.info(f"✅ Riesgo aceptable: {risk_level}")
            else:
                log.info(f"❌ Riesgo muy alto: {risk_level}")
            
            # 4. Balance suficiente
            account_info = self.demo_account.get_account_info()
            available_balance = account_info['balance']
            min_balance = 100.0  # Mínimo $100 para operar
            
            if available_balance >= min_balance:
                criteria_met += 1
                log.info(f"✅ Balance suficiente: ${available_balance:.2f} >= ${min_balance:.2f}")
            else:
                log.info(f"❌ Balance insuficiente: ${available_balance:.2f} < ${min_balance:.2f}")
            
            # 5. No hay demasiadas posiciones abiertas
            current_positions = self.demo_account.get_positions()
            max_positions = 2  # Máximo 2 posiciones simultáneas
            
            if len(current_positions) < max_positions:
                criteria_met += 1
                log.info(f"✅ Posiciones OK: {len(current_positions)} < {max_positions}")
            else:
                log.info(f"❌ Demasiadas posiciones: {len(current_positions)} >= {max_positions}")
            
            # Decisión final
            min_criteria = 4  # Requiere al menos 4 de 5 criterios
            decision = criteria_met >= min_criteria
            
            log.info(f"📊 Criterios cumplidos: {criteria_met}/{total_criteria}")
            log.info(f"🎯 Decisión de trade: {'EJECUTAR' if decision else 'NO EJECUTAR'}")
            
            return decision
            
        except Exception as e:
            log.error(f"❌ Error evaluando decisión: {str(e)}")
            return False
    
    async def execute_trade(self, signal: SignalType, entry_price: float, 
                          stop_loss: float, take_profit: float, confidence: float):
        """Ejecuta un trade en la cuenta demo."""
        try:
            log.info(f"🔄 Ejecutando trade: {signal.value}")
            
            # Calcular tamaño de posición (1% del balance)
            account_info = self.demo_account.get_account_info()
            balance = account_info['balance']
            risk_per_trade = balance * 0.01  # 1% de riesgo
            
            # Calcular tamaño basado en stop loss
            if signal == SignalType.BUY:
                risk_per_share = abs(entry_price - stop_loss) if stop_loss > 0 else entry_price * 0.02
                side = "buy"
            else:
                risk_per_share = abs(stop_loss - entry_price) if stop_loss > 0 else entry_price * 0.02
                side = "sell"
            
            if risk_per_share > 0:
                position_size = min(risk_per_trade / risk_per_share, 1.0)  # Máximo 1 acción
                position_size = max(position_size, 0.01)  # Mínimo 0.01
            else:
                position_size = 0.01  # Tamaño mínimo por defecto
            
            log.info(f"💼 Tamaño calculado: {position_size:.3f}")
            log.info(f"💰 Riesgo por trade: ${risk_per_trade:.2f}")
            
            # Ejecutar orden
            order_result = self.demo_account.place_order(
                symbol=self.symbol,
                side=side,
                size=position_size,
                order_type="market"
            )
            
            if order_result['success']:
                position_id = order_result['position_id']
                execution_price = order_result['execution_price']
                
                log.info(f"✅ Trade ejecutado exitosamente:")
                log.info(f"   Posición ID: {position_id}")
                log.info(f"   Lado: {side.upper()}")
                log.info(f"   Tamaño: {position_size}")
                log.info(f"   Precio: ${execution_price:.5f}")
                log.info(f"   Stop Loss: ${stop_loss:.5f}")
                log.info(f"   Take Profit: ${take_profit:.5f}")
                
                # Guardar información de la posición
                self.positions[position_id] = {
                    'signal': signal,
                    'entry_price': execution_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'size': position_size,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                }
                
                self.stats['trades_executed'] += 1
                
            else:
                log.error(f"❌ Error ejecutando trade: {order_result['error']}")
                
        except Exception as e:
            log.error(f"❌ Error en ejecución de trade: {str(e)}")
    
    async def monitor_positions(self):
        """Monitorea posiciones abiertas y gestiona stop loss/take profit."""
        try:
            current_positions = self.demo_account.get_positions()
            
            if not current_positions:
                return
            
            for position in current_positions:
                position_id = position['id']
                current_price = position['current_price']
                unrealized_pnl = position['unrealized_pnl']
                
                # Verificar si tenemos información adicional de la posición
                if position_id in self.positions:
                    pos_info = self.positions[position_id]
                    stop_loss = pos_info['stop_loss']
                    take_profit = pos_info['take_profit']
                    
                    # Verificar condiciones de cierre
                    should_close = False
                    close_reason = ""
                    
                    if position['side'] == 'buy':
                        if stop_loss > 0 and current_price <= stop_loss:
                            should_close = True
                            close_reason = "Stop Loss"
                        elif take_profit > 0 and current_price >= take_profit:
                            should_close = True
                            close_reason = "Take Profit"
                    else:  # sell
                        if stop_loss > 0 and current_price >= stop_loss:
                            should_close = True
                            close_reason = "Stop Loss"
                        elif take_profit > 0 and current_price <= take_profit:
                            should_close = True
                            close_reason = "Take Profit"
                    
                    if should_close:
                        log.info(f"🔄 Cerrando posición {position_id} - {close_reason}")
                        
                        close_result = self.demo_account.close_position(position_id)
                        
                        if close_result['success']:
                            pnl = close_result['pnl']
                            
                            log.info(f"✅ Posición cerrada: {close_reason}")
                            log.info(f"   PnL: ${pnl:+.2f}")
                            
                            # Actualizar estadísticas
                            self.stats['total_pnl'] += pnl
                            if pnl > 0:
                                self.stats['successful_trades'] += 1
                            
                            # Remover de seguimiento
                            del self.positions[position_id]
                        else:
                            log.error(f"❌ Error cerrando posición: {close_result['error']}")
                
        except Exception as e:
            log.error(f"❌ Error monitoreando posiciones: {str(e)}")
    
    def update_statistics(self):
        """Actualiza y muestra estadísticas del bot."""
        try:
            # Calcular tiempo de ejecución
            runtime = datetime.now() - self.stats['start_time']
            
            # Obtener información de cuenta
            account_info = self.demo_account.get_account_info()
            current_balance = account_info['balance']
            
            # Calcular performance
            total_return = current_balance - self.balance_inicial
            return_pct = (total_return / self.balance_inicial) * 100 if self.balance_inicial > 0 else 0
            
            # Log estadísticas cada 10 análisis
            if self.stats['total_signals'] % 10 == 0 and self.stats['total_signals'] > 0:
                log.info("📊 ESTADÍSTICAS DEL BOT:")
                log.info(f"   ⏱️  Tiempo ejecutándose: {str(runtime).split('.')[0]}")
                log.info(f"   📊 Total señales: {self.stats['total_signals']}")
                log.info(f"   📈 Señales BUY: {self.stats['buy_signals']}")
                log.info(f"   📉 Señales SELL: {self.stats['sell_signals']}")
                log.info(f"   ⏸️  Señales HOLD: {self.stats['hold_signals']}")
                log.info(f"   🔄 Trades ejecutados: {self.stats['trades_executed']}")
                log.info(f"   ✅ Trades exitosos: {self.stats['successful_trades']}")
                log.info(f"   💰 Balance actual: ${current_balance:,.2f}")
                log.info(f"   📊 Retorno total: ${total_return:+.2f} ({return_pct:+.2f}%)")
                log.info(f"   💼 Posiciones activas: {len(self.demo_account.get_positions())}")
                
        except Exception as e:
            log.error(f"❌ Error actualizando estadísticas: {str(e)}")
    
    def stop(self):
        """Detiene el bot de trading."""
        log.info("🛑 Deteniendo bot de trading...")
        self.is_running = False
        
        # Cerrar todas las posiciones abiertas
        try:
            positions = self.demo_account.get_positions()
            for position in positions:
                close_result = self.demo_account.close_position(position['id'])
                if close_result['success']:
                    log.info(f"✅ Posición {position['id']} cerrada al detener bot")
        except Exception as e:
            log.error(f"❌ Error cerrando posiciones: {str(e)}")
        
        # Mostrar resumen final
        self.show_final_summary()
    
    def show_final_summary(self):
        """Muestra resumen final del bot."""
        try:
            account_info = self.demo_account.get_account_info()
            final_balance = account_info['balance']
            total_return = final_balance - self.balance_inicial
            return_pct = (total_return / self.balance_inicial) * 100 if self.balance_inicial > 0 else 0
            
            runtime = datetime.now() - self.stats['start_time']
            
            log.info("🏁 RESUMEN FINAL DEL BOT:")
            log.info("=" * 50)
            log.info(f"⏱️  Tiempo total: {str(runtime).split('.')[0]}")
            log.info(f"💰 Balance inicial: ${self.balance_inicial:,.2f}")
            log.info(f"💰 Balance final: ${final_balance:,.2f}")
            log.info(f"📊 Retorno total: ${total_return:+.2f} ({return_pct:+.2f}%)")
            log.info(f"🔄 Total trades: {self.stats['trades_executed']}")
            log.info(f"✅ Trades exitosos: {self.stats['successful_trades']}")
            
            if self.stats['trades_executed'] > 0:
                success_rate = (self.stats['successful_trades'] / self.stats['trades_executed']) * 100
                log.info(f"📈 Tasa de éxito: {success_rate:.1f}%")
            
            log.info("=" * 50)
            
        except Exception as e:
            log.error(f"❌ Error mostrando resumen: {str(e)}")


async def main():
    """Función principal del bot."""
    log.info("🤖 BOT DE TRADING LIT + ML - VERSIÓN SIMPLIFICADA")
    log.info("=" * 60)
    
    # Crear bot
    bot = SimpleTradingBot()
    
    try:
        # Inicializar
        log.info("🔧 Inicializando bot...")
        initialization_success = await bot.initialize()
        
        if not initialization_success:
            log.error("❌ Error en inicialización - Abortando")
            return
        
        log.info("✅ Bot inicializado correctamente")
        log.info("🚀 Iniciando trading en vivo...")
        
        # Ejecutar bucle de trading
        await bot.run_trading_loop()
        
    except KeyboardInterrupt:
        log.info("🛑 Interrupción por usuario")
    except Exception as e:
        log.error(f"💥 Error crítico: {str(e)}")
    finally:
        # Detener bot
        bot.stop()
        log.info("👋 Bot detenido")


if __name__ == "__main__":
    # Ejecutar bot
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot detenido por usuario")
    except Exception as e:
        print(f"\n💥 Error ejecutando bot: {str(e)}") 