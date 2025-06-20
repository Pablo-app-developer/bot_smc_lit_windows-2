#!/usr/bin/env python3
"""
Bot de Trading Real con MetaTrader 5 - Conexión REAL.

Basado en mejores prácticas de:
- libro-trading-python-es
- trading-algoritmico-metatrader-5
- TopForex.Trade guide

OPERACIONES REALES con MT5.
"""

import os
import sys
import asyncio
from datetime import datetime
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    print("❌ ERROR: MetaTrader5 no está instalado")
    print("Instalar con: pip install MetaTrader5")
    MT5_AVAILABLE = False
    sys.exit(1)

from src.utils.logger import log
from src.data.mt5_data_loader import MT5DataLoader
from src.strategies.lit_detector import LITDetector, SignalType
from src.brokers.mt5_connector import MT5Connector


class RealTradingBot:
    """Bot de Trading REAL con MetaTrader 5."""
    
    def __init__(self):
        log.info("🚀 Bot de Trading REAL con MT5")
        
        self.mt5_connector = MT5Connector()
        self.data_loader = MT5DataLoader()
        self.lit_detector = LITDetector()
        
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        self.min_confidence = 0.65
        self.max_positions = 2
        self.risk_per_trade = 0.01
        
        self.is_running = False
        self.stats = {
            'trades_executed': 0,
            'start_balance': 0.0,
            'start_time': datetime.now()
        }
    
    async def initialize(self) -> bool:
        """Inicializa conexión REAL a MT5."""
        try:
            log.info("🔌 Conectando a MetaTrader 5 REAL...")
            
            if not self.mt5_connector.connect():
                log.error("❌ No se pudo conectar a MT5")
                log.error("💡 Asegúrate de tener MT5 abierto y cuenta configurada")
                return False
            
            account_info = self.mt5_connector.get_account_info()
            if not account_info:
                log.error("❌ No se pudo obtener info de cuenta")
                return False
            
            self.stats['start_balance'] = account_info['balance']
            
            log.info("✅ CONEXIÓN REAL ESTABLECIDA")
            log.info(f"   📊 Cuenta: {account_info['account_id']}")
            log.info(f"   🏦 Broker: {account_info['company']}")
            log.info(f"   💰 Balance: ${account_info['balance']:,.2f}")
            log.info(f"   🔄 Trading: {'✅' if account_info['trade_allowed'] else '❌'}")
            
            if not account_info['trade_allowed']:
                log.error("❌ TRADING NO PERMITIDO")
                return False
            
            # Verificar símbolos
            available_symbols = []
            for symbol in self.symbols:
                if mt5.symbol_info(symbol) is not None:
                    mt5.symbol_select(symbol, True)
                    available_symbols.append(symbol)
                    log.info(f"   ✅ {symbol}: Disponible")
            
            self.symbols = available_symbols
            return len(available_symbols) > 0
            
        except Exception as e:
            log.error(f"❌ Error inicializando: {str(e)}")
            return False
    
    async def run_trading_loop(self):
        """Bucle principal de trading REAL."""
        log.info("🔄 INICIANDO TRADING EN VIVO - OPERACIONES REALES")
        
        self.is_running = True
        cycle = 0
        
        while self.is_running:
            try:
                cycle += 1
                log.info(f"📊 CICLO {cycle} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Análisis de mercado
                await self.analyze_market()
                
                # Monitorear posiciones
                await self.monitor_positions()
                
                # Esperar
                await asyncio.sleep(60)  # 1 minuto
                
            except KeyboardInterrupt:
                log.info("🛑 Deteniendo bot...")
                break
            except Exception as e:
                log.error(f"❌ Error en ciclo: {str(e)}")
                await asyncio.sleep(30)
    
    async def analyze_market(self):
        """Analiza mercado y ejecuta operaciones REALES."""
        try:
            opportunities = []
            
            for symbol in self.symbols:
                data = self.data_loader.load_data(symbol, "1h", 100)
                
                if len(data) < 50:
                    continue
                
                current_price = data['close'].iloc[-1]
                lit_signal = self.lit_detector.analyze(data)
                
                if (lit_signal.signal != SignalType.HOLD and 
                    lit_signal.confidence >= self.min_confidence):
                    
                    opportunities.append({
                        'symbol': symbol,
                        'signal': lit_signal.signal,
                        'confidence': lit_signal.confidence,
                        'price': current_price
                    })
            
            # Ejecutar oportunidades
            if opportunities:
                current_positions = self.mt5_connector.get_positions()
                available_slots = self.max_positions - len(current_positions)
                
                for opp in opportunities[:available_slots]:
                    await self.execute_real_trade(opp)
            
        except Exception as e:
            log.error(f"❌ Error analizando: {str(e)}")
    
    async def execute_real_trade(self, opportunity: Dict):
        """Ejecuta operación REAL en MT5."""
        try:
            symbol = opportunity['symbol']
            signal = opportunity['signal']
            price = opportunity['price']
            
            log.info(f"💰 OPERACIÓN REAL: {signal.value.upper()} {symbol}")
            
            # Calcular parámetros
            side = "buy" if signal == SignalType.BUY else "sell"
            volume = 0.01  # Lote mínimo para pruebas
            
            # Niveles de riesgo
            atr = price * 0.01
            if side == "buy":
                sl = price - (atr * 1.5)
                tp = price + (atr * 2.0)
            else:
                sl = price + (atr * 1.5)
                tp = price - (atr * 2.0)
            
            log.info(f"   📊 Volumen: {volume} lotes")
            log.info(f"   💰 Precio: {price:.5f}")
            log.info(f"   🛡️  SL: {sl:.5f}")
            log.info(f"   🎯 TP: {tp:.5f}")
            
            # EJECUTAR ORDEN REAL
            result = self.mt5_connector.place_order(
                symbol=symbol,
                side=side,
                volume=volume,
                sl=sl,
                tp=tp
            )
            
            if result.success:
                log.info("✅ ¡OPERACIÓN EJECUTADA EN VIVO!")
                log.info(f"   🎫 Ticket: {result.order_ticket}")
                log.info(f"   💰 Precio ejecución: {result.execution_price:.5f}")
                self.stats['trades_executed'] += 1
            else:
                log.error(f"❌ Error: {result.error_description}")
                
        except Exception as e:
            log.error(f"❌ Error ejecutando: {str(e)}")
    
    async def monitor_positions(self):
        """Monitorea posiciones REALES."""
        try:
            positions = self.mt5_connector.get_positions()
            
            if positions:
                log.info(f"👁️  {len(positions)} posiciones activas:")
                for pos in positions:
                    log.info(f"   📊 {pos.symbol} #{pos.ticket}: {pos.side.upper()} | P&L: ${pos.unrealized_pnl:+.2f}")
                    
        except Exception as e:
            log.error(f"❌ Error monitoreando: {str(e)}")
    
    def stop(self):
        """Detiene bot y muestra resumen."""
        self.is_running = False
        
        try:
            account_info = self.mt5_connector.get_account_info()
            if account_info:
                final_balance = account_info['balance']
                pnl = final_balance - self.stats['start_balance']
                
                log.info("🏁 RESUMEN FINAL:")
                log.info(f"💰 Balance inicial: ${self.stats['start_balance']:,.2f}")
                log.info(f"💰 Balance final: ${final_balance:,.2f}")
                log.info(f"📊 P&L: ${pnl:+.2f}")
                log.info(f"🔄 Trades: {self.stats['trades_executed']}")
            
            self.mt5_connector.disconnect()
            
        except Exception as e:
            log.error(f"❌ Error en resumen: {str(e)}")


async def main():
    """Función principal."""
    print("🤖 BOT DE TRADING REAL CON METATRADER 5")
    print("⚠️  ESTE BOT EJECUTA OPERACIONES REALES")
    print("💡 USA CUENTA DEMO PARA PRUEBAS")
    print("=" * 50)
    
    bot = RealTradingBot()
    
    try:
        if not await bot.initialize():
            print("❌ Error inicializando")
            return
        
        print("✅ Bot inicializado - Iniciando trading...")
        await bot.run_trading_loop()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot detenido por usuario")
    finally:
        bot.stop()


if __name__ == "__main__":
    asyncio.run(main()) 