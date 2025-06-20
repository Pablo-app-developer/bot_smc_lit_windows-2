#!/usr/bin/env python3
"""
Bot de Trading Forex Profesional - LIT + ML.

Sistema completo para trading automatizado en Forex con:
- Múltiples pares de divisas
- Estrategia LIT adaptada para Forex
- Machine Learning integrado
- Gestión profesional de riesgo
- Trading 24/7
"""

import sys
import os
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

# Configurar path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.brokers.forex_data_provider import ForexDataProvider
from src.strategies.forex_lit_strategy import ForexLITStrategy
from src.models.predictor import LITMLPredictor
from src.models.feature_engineering import FeatureEngineer
from src.utils.logger import log
from validate_demo_account import DemoAccountConnector


class ForexTradingBot:
    """
    Bot de Trading Forex Profesional.
    
    Características:
    - Trading multi-par simultáneo
    - Análisis 24/7 adaptado a sesiones
    - Gestión avanzada de riesgo
    - Machine Learning integrado
    - Correlaciones entre pares
    """
    
    def __init__(self):
        """Inicializa el bot de trading Forex."""
        
        # Configuración principal
        self.account_balance = 2865.05
        self.risk_per_trade = 0.02  # 2% por operación (más agresivo para Forex)
        self.max_positions = 3      # Máximo 3 posiciones simultáneas
        self.min_confidence = 0.65  # Confianza mínima para operar
        
        # PARES DE DIVISAS A OPERAR
        self.trading_pairs = {
            'majors': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF'],
            'minors': ['EURGBP', 'EURJPY', 'GBPJPY'],
            'selected': ['EURUSD', 'GBPUSD', 'USDJPY', 'EURGBP']  # Pares activos
        }
        
        # Inicializar componentes
        self.forex_provider = ForexDataProvider()
        self.forex_strategy = ForexLITStrategy(lookback_periods=50, liquidity_threshold=0.0001)
        self.feature_engineer = FeatureEngineer()
        self.ml_predictor = LITMLPredictor()
        self.demo_account = DemoAccountConnector()
        
        # Estado del bot
        self.active_positions = {}
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Configuración de análisis
        self.analysis_interval = 300  # 5 minutos
        self.timeframe = '1h'
        self.lookback_periods = 200
        
        log.info("🌍 ForexTradingBot inicializado para trading profesional")
        log.info(f"💰 Balance inicial: ${self.account_balance:,.2f}")
        log.info(f"📊 Pares activos: {', '.join(self.trading_pairs['selected'])}")
    
    async def initialize(self):
        """Inicializa todos los componentes del bot."""
        try:
            log.info("🔧 Inicializando bot Forex...")
            
            # 1. Conectar cuenta demo
            log.info("1️⃣ Conectando a cuenta demo...")
            await self._connect_demo_account()
            
            # 2. Verificar datos de mercado
            log.info("2️⃣ Verificando datos de mercado...")
            await self._verify_market_data()
            
            # 3. Entrenar modelos ML
            log.info("3️⃣ Entrenando modelos ML...")
            await self._train_ml_models()
            
            # 4. Verificar correlaciones
            log.info("4️⃣ Analizando correlaciones...")
            await self._analyze_correlations()
            
            log.info("✅ Bot Forex inicializado correctamente")
            
        except Exception as e:
            log.error(f"❌ Error inicializando bot: {str(e)}")
            raise
    
    async def _connect_demo_account(self):
        """Conecta a la cuenta demo."""
        try:
            self.demo_account.connect()
            balance = self.demo_account.get_balance()
            
            if balance > 0:
                self.account_balance = balance
                log.info(f"✅ Cuenta conectada - Balance: ${balance:,.2f}")
            else:
                log.warning("⚠️  Balance no disponible, usando configuración por defecto")
                
        except Exception as e:
            log.error(f"Error conectando cuenta: {str(e)}")
    
    async def _verify_market_data(self):
        """Verifica disponibilidad de datos para todos los pares."""
        try:
            data_status = {}
            
            for pair in self.trading_pairs['selected']:
                data = self.forex_provider.get_forex_data(pair, self.timeframe, 50)
                data_status[pair] = len(data)
                
                if len(data) >= 50:
                    log.info(f"✅ {pair}: {len(data)} filas disponibles")
                else:
                    log.warning(f"⚠️  {pair}: Solo {len(data)} filas disponibles")
            
            # Filtrar pares con datos insuficientes
            valid_pairs = [pair for pair, count in data_status.items() if count >= 50]
            
            if len(valid_pairs) < len(self.trading_pairs['selected']):
                log.warning(f"Reduciendo pares activos a: {', '.join(valid_pairs)}")
                self.trading_pairs['selected'] = valid_pairs
                
        except Exception as e:
            log.error(f"Error verificando datos: {str(e)}")
    
    async def _train_ml_models(self):
        """Entrena modelos ML para cada par."""
        try:
            for pair in self.trading_pairs['selected']:
                try:
                    # Obtener datos históricos
                    data = self.forex_provider.get_forex_data(pair, self.timeframe, 300)
                    
                    if len(data) < 100:
                        log.warning(f"⚠️  {pair}: Datos insuficientes para ML")
                        continue
                    
                    # Generar features
                    features = self.feature_engineer.create_features(data)
                    
                    if len(features) < 50:
                        log.warning(f"⚠️  {pair}: Features insuficientes")
                        continue
                    
                    # Generar señales LIT para entrenamiento
                    lit_signals = self.forex_strategy.lit_detector.detect_lit_events(data)
                    
                    if len(lit_signals) == len(features):
                        # Entrenar modelo específico para el par
                        self.ml_predictor.train(features, lit_signals)
                        log.info(f"✅ {pair}: Modelo ML entrenado")
                    else:
                        log.warning(f"⚠️  {pair}: Inconsistencia en datos de entrenamiento")
                        
                except Exception as e:
                    log.error(f"Error entrenando modelo {pair}: {str(e)}")
                    
        except Exception as e:
            log.error(f"Error en entrenamiento ML: {str(e)}")
    
    async def _analyze_correlations(self):
        """Analiza correlaciones entre pares."""
        try:
            correlation_matrix = self.forex_provider.get_correlation_matrix(
                self.trading_pairs['selected'], 
                self.timeframe, 
                100
            )
            
            if not correlation_matrix.empty:
                log.info("📊 Matriz de correlaciones calculada")
                
                # Identificar pares altamente correlacionados
                high_corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        pair1 = correlation_matrix.columns[i]
                        pair2 = correlation_matrix.columns[j]
                        corr = correlation_matrix.iloc[i, j]
                        
                        if abs(corr) > 0.8:
                            high_corr_pairs.append((pair1, pair2, corr))
                
                if high_corr_pairs:
                    log.info("⚠️  Pares con alta correlación detectados:")
                    for pair1, pair2, corr in high_corr_pairs:
                        log.info(f"   {pair1} - {pair2}: {corr:.2f}")
                        
        except Exception as e:
            log.error(f"Error analizando correlaciones: {str(e)}")
    
    async def run_trading_loop(self):
        """Ejecuta el bucle principal de trading."""
        try:
            log.info("🚀 Iniciando bucle de trading Forex 24/7...")
            log.info(f"⏱️  Intervalo de análisis: {self.analysis_interval} segundos")
            
            while True:
                try:
                    # Obtener estado del mercado
                    market_status = self.forex_provider.get_market_status()
                    
                    log.info(f"📊 Ejecutando análisis - {datetime.now().strftime('%H:%M:%S')}")
                    log.info(f"🌍 Sesiones activas: {', '.join(market_status.get('active_sessions', ['Ninguna']))}")
                    
                    # Ejecutar análisis para todos los pares
                    await self._execute_forex_analysis(market_status)
                    
                    # Gestionar posiciones existentes
                    await self._manage_existing_positions()
                    
                    # Mostrar resumen
                    await self._show_trading_summary()
                    
                    # Esperar siguiente ciclo
                    await asyncio.sleep(self.analysis_interval)
                    
                except KeyboardInterrupt:
                    log.info("🛑 Deteniendo bot por solicitud del usuario...")
                    break
                    
                except Exception as e:
                    log.error(f"❌ Error en bucle de trading: {str(e)}")
                    await asyncio.sleep(60)  # Esperar 1 minuto antes de reintentar
                    
        except Exception as e:
            log.error(f"❌ Error crítico en bucle de trading: {str(e)}")
    
    async def _execute_forex_analysis(self, market_status: Dict):
        """Ejecuta análisis completo para todos los pares."""
        try:
            # Obtener mejores oportunidades
            opportunities = self.forex_strategy.get_best_opportunities(
                self.trading_pairs['selected'], 
                self.min_confidence
            )
            
            if not opportunities:
                log.info("📊 No hay oportunidades de trading en este momento")
                return
            
            log.info(f"🎯 Encontradas {len(opportunities)} oportunidades:")
            
            for opp in opportunities[:3]:  # Top 3 oportunidades
                pair = opp['pair']
                signal = opp['signal']
                confidence = opp['confidence']
                
                signal_text = "BUY" if signal == 1 else "SELL"
                log.info(f"   📈 {pair}: {signal_text} (Confianza: {confidence:.1%})")
                
                # Verificar si podemos abrir nueva posición
                if await self._can_open_position(pair):
                    await self._execute_trade(opp, market_status)
                    
        except Exception as e:
            log.error(f"Error en análisis Forex: {str(e)}")
    
    async def _can_open_position(self, pair: str) -> bool:
        """Verifica si se puede abrir una nueva posición."""
        try:
            # 1. Verificar máximo de posiciones
            if len(self.active_positions) >= self.max_positions:
                return False
            
            # 2. Verificar si ya hay posición en este par
            if pair in self.active_positions:
                return False
            
            # 3. Verificar balance disponible
            required_margin = self.account_balance * self.risk_per_trade
            if required_margin > self.account_balance * 0.1:  # Máximo 10% de margen
                return False
            
            # 4. Verificar correlaciones (evitar pares altamente correlacionados)
            for active_pair in self.active_positions.keys():
                if await self._are_highly_correlated(pair, active_pair):
                    log.info(f"⚠️  Evitando {pair} por correlación con {active_pair}")
                    return False
            
            return True
            
        except Exception as e:
            log.error(f"Error verificando posición: {str(e)}")
            return False
    
    async def _are_highly_correlated(self, pair1: str, pair2: str) -> bool:
        """Verifica si dos pares están altamente correlacionados."""
        try:
            # Obtener datos recientes de ambos pares
            data1 = self.forex_provider.get_forex_data(pair1, '1d', 30)
            data2 = self.forex_provider.get_forex_data(pair2, '1d', 30)
            
            if len(data1) < 20 or len(data2) < 20:
                return False
            
            # Calcular correlación
            correlation = data1['close'].corr(data2['close'])
            
            # Considerar alta correlación si > 0.8
            return abs(correlation) > 0.8
            
        except Exception as e:
            log.error(f"Error calculando correlación: {str(e)}")
            return False
    
    async def _execute_trade(self, opportunity: Dict, market_status: Dict):
        """Ejecuta una operación de trading."""
        try:
            pair = opportunity['pair']
            signal = opportunity['signal']
            confidence = opportunity['confidence']
            entry_levels = opportunity.get('entry_levels', {})
            
            if not entry_levels:
                log.warning(f"⚠️  No hay niveles de entrada para {pair}")
                return
            
            # Calcular tamaño de posición
            position_size = self._calculate_position_size(pair, entry_levels)
            
            if position_size <= 0:
                log.warning(f"⚠️  Tamaño de posición inválido para {pair}")
                return
            
            # Preparar orden
            order_data = {
                'pair': pair,
                'side': 'BUY' if signal == 1 else 'SELL',
                'size': position_size,
                'entry_price': entry_levels['entry_price'],
                'stop_loss': entry_levels['stop_loss'],
                'take_profit': entry_levels['take_profit'],
                'confidence': confidence,
                'market_session': market_status.get('active_sessions', [])
            }
            
            # Ejecutar orden
            success = await self._place_forex_order(order_data)
            
            if success:
                # Registrar posición activa
                self.active_positions[pair] = {
                    'order_data': order_data,
                    'open_time': datetime.now(),
                    'status': 'OPEN'
                }
                
                self.total_trades += 1
                
                log.info(f"✅ Orden ejecutada: {order_data['side']} {pair}")
                log.info(f"   💰 Tamaño: {position_size}")
                log.info(f"   🎯 Entrada: {entry_levels['entry_price']:.5f}")
                log.info(f"   🛡️  SL: {entry_levels['stop_loss']:.5f}")
                log.info(f"   🎯 TP: {entry_levels['take_profit']:.5f}")
                
        except Exception as e:
            log.error(f"Error ejecutando trade {pair}: {str(e)}")
    
    def _calculate_position_size(self, pair: str, entry_levels: Dict) -> float:
        """Calcula el tamaño de posición basado en riesgo."""
        try:
            # Obtener información del par
            pair_info = self.forex_provider.all_pairs.get(pair, {})
            pip_value = pair_info.get('pip_value', 0.0001)
            
            # Calcular riesgo en pips
            entry_price = entry_levels['entry_price']
            stop_loss = entry_levels['stop_loss']
            risk_pips = abs(entry_price - stop_loss) / pip_value
            
            # Calcular riesgo monetario
            risk_amount = self.account_balance * self.risk_per_trade
            
            # Calcular tamaño de posición
            # Para Forex: 1 lote estándar = 100,000 unidades
            # 1 mini lote = 10,000 unidades
            # 1 micro lote = 1,000 unidades
            
            if risk_pips > 0:
                # Valor por pip para 1 micro lote (1,000 unidades)
                pip_value_per_micro_lot = pip_value * 1000
                
                # Número de micro lotes
                micro_lots = risk_amount / (risk_pips * pip_value_per_micro_lot)
                
                # Convertir a unidades (redondeado a miles)
                position_size = round(micro_lots * 1000, -3)  # Redondear a miles
                
                # Limitar tamaño mínimo y máximo
                position_size = max(1000, min(position_size, 50000))  # Entre 1k y 50k
                
                return position_size
            
            return 0
            
        except Exception as e:
            log.error(f"Error calculando tamaño de posición: {str(e)}")
            return 0
    
    async def _place_forex_order(self, order_data: Dict) -> bool:
        """Coloca una orden Forex en la cuenta demo."""
        try:
            # Simular ejecución de orden
            pair = order_data['pair']
            side = order_data['side']
            size = order_data['size']
            
            # Usar cuenta demo para simular
            if side == 'BUY':
                success = self.demo_account.buy(pair, size)
            else:
                success = self.demo_account.sell(pair, size)
            
            return success
            
        except Exception as e:
            log.error(f"Error colocando orden: {str(e)}")
            return False
    
    async def _manage_existing_positions(self):
        """Gestiona las posiciones existentes."""
        try:
            if not self.active_positions:
                return
            
            positions_to_close = []
            
            for pair, position in self.active_positions.items():
                try:
                    # Obtener precio actual
                    current_prices = self.forex_provider.get_current_forex_price(pair)
                    
                    if not current_prices:
                        continue
                    
                    current_price = current_prices['mid']
                    order_data = position['order_data']
                    
                    # Verificar stop loss y take profit
                    should_close, reason = self._should_close_position(
                        order_data, current_price
                    )
                    
                    if should_close:
                        positions_to_close.append((pair, reason))
                        
                except Exception as e:
                    log.error(f"Error gestionando posición {pair}: {str(e)}")
            
            # Cerrar posiciones marcadas
            for pair, reason in positions_to_close:
                await self._close_position(pair, reason)
                
        except Exception as e:
            log.error(f"Error gestionando posiciones: {str(e)}")
    
    def _should_close_position(self, order_data: Dict, current_price: float) -> Tuple[bool, str]:
        """Determina si una posición debe cerrarse."""
        try:
            side = order_data['side']
            stop_loss = order_data['stop_loss']
            take_profit = order_data['take_profit']
            
            if side == 'BUY':
                if current_price <= stop_loss:
                    return True, "Stop Loss alcanzado"
                elif current_price >= take_profit:
                    return True, "Take Profit alcanzado"
            else:  # SELL
                if current_price >= stop_loss:
                    return True, "Stop Loss alcanzado"
                elif current_price <= take_profit:
                    return True, "Take Profit alcanzado"
            
            # Verificar tiempo máximo de posición (24 horas para Forex)
            open_time = order_data.get('open_time', datetime.now())
            if datetime.now() - open_time > timedelta(hours=24):
                return True, "Tiempo máximo alcanzado"
            
            return False, ""
            
        except Exception as e:
            log.error(f"Error evaluando cierre de posición: {str(e)}")
            return False, "Error"
    
    async def _close_position(self, pair: str, reason: str):
        """Cierra una posición."""
        try:
            if pair not in self.active_positions:
                return
            
            position = self.active_positions[pair]
            order_data = position['order_data']
            
            # Simular cierre de posición
            close_success = self.demo_account.close_position(pair)
            
            if close_success:
                # Calcular PnL (simplificado)
                current_prices = self.forex_provider.get_current_forex_price(pair)
                if current_prices:
                    pnl = self._calculate_pnl(order_data, current_prices['mid'])
                    self.daily_pnl += pnl
                    
                    if pnl > 0:
                        self.winning_trades += 1
                    
                    log.info(f"🔒 Posición cerrada: {pair}")
                    log.info(f"   📊 Razón: {reason}")
                    log.info(f"   💰 PnL: ${pnl:+.2f}")
                
                # Remover de posiciones activas
                del self.active_positions[pair]
                
        except Exception as e:
            log.error(f"Error cerrando posición {pair}: {str(e)}")
    
    def _calculate_pnl(self, order_data: Dict, current_price: float) -> float:
        """Calcula el PnL de una posición."""
        try:
            entry_price = order_data['entry_price']
            size = order_data['size']
            side = order_data['side']
            
            if side == 'BUY':
                pnl = (current_price - entry_price) * size
            else:
                pnl = (entry_price - current_price) * size
            
            return pnl
            
        except Exception as e:
            log.error(f"Error calculando PnL: {str(e)}")
            return 0.0
    
    async def _show_trading_summary(self):
        """Muestra resumen de trading."""
        try:
            if self.total_trades > 0:
                win_rate = (self.winning_trades / self.total_trades) * 100
            else:
                win_rate = 0.0
            
            log.info(f"📊 RESUMEN DE TRADING:")
            log.info(f"   💰 Balance: ${self.account_balance:,.2f}")
            log.info(f"   📈 PnL Diario: ${self.daily_pnl:+.2f}")
            log.info(f"   🎯 Posiciones Activas: {len(self.active_positions)}")
            log.info(f"   📊 Total Trades: {self.total_trades}")
            log.info(f"   🏆 Tasa de Éxito: {win_rate:.1f}%")
            
            if self.active_positions:
                log.info(f"   🔄 Pares Activos: {', '.join(self.active_positions.keys())}")
                
        except Exception as e:
            log.error(f"Error mostrando resumen: {str(e)}")


async def main():
    """Función principal."""
    try:
        print("🌍 BOT DE TRADING FOREX PROFESIONAL - LIT + ML")
        print("=" * 60)
        print(f"🕐 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Crear e inicializar bot
        bot = ForexTradingBot()
        await bot.initialize()
        
        # Ejecutar trading
        await bot.run_trading_loop()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error crítico: {str(e)}")
        log.error(f"Error crítico en main: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main()) 