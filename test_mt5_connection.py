#!/usr/bin/env python3
"""
Script de Prueba para Conexión Real a MetaTrader 5.

Este script verifica que la conexión a MT5 funcione correctamente
y que se puedan ejecutar operaciones reales (demo).
"""

import os
import sys

# Agregar path del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import MetaTrader5 as mt5
    from datetime import datetime
    print("✅ MetaTrader5 importado correctamente")
except ImportError as e:
    print("❌ ERROR: MetaTrader5 no está instalado")
    print("Instalar con: pip install MetaTrader5")
    sys.exit(1)

from src.brokers.mt5_connector import MT5Connector
from src.utils.logger import log


def test_mt5_basic_connection():
    """Prueba conexión básica a MT5."""
    print("\n🔍 PRUEBA 1: Conexión Básica a MT5")
    print("=" * 50)
    
    try:
        # Inicializar MT5
        if not mt5.initialize():
            print("❌ No se pudo inicializar MT5")
            error = mt5.last_error()
            print(f"Error: {error}")
            return False
        
        # Obtener información de cuenta
        account_info = mt5.account_info()
        if account_info is None:
            print("❌ No se pudo obtener información de cuenta")
            return False
        
        print("✅ Conexión exitosa a MT5")
        print(f"   📊 Cuenta: {account_info.login}")
        print(f"   🏦 Broker: {account_info.company}")
        print(f"   💰 Balance: ${account_info.balance:,.2f}")
        print(f"   📈 Equity: ${account_info.equity:,.2f}")
        print(f"   💱 Moneda: {account_info.currency}")
        print(f"   🔄 Trading permitido: {'SÍ' if account_info.trade_allowed else 'NO'}")
        
        # Verificar estado del servidor
        if account_info.trade_allowed:
            print("✅ CUENTA LISTA PARA TRADING")
        else:
            print("⚠️  TRADING NO PERMITIDO - Usar cuenta demo")
        
        mt5.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba básica: {str(e)}")
        return False


def test_mt5_symbols():
    """Prueba disponibilidad de símbolos."""
    print("\n🔍 PRUEBA 2: Verificación de Símbolos")
    print("=" * 50)
    
    try:
        if not mt5.initialize():
            print("❌ No se pudo inicializar MT5")
            return False
        
        symbols_to_test = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF"]
        available_symbols = []
        
        for symbol in symbols_to_test:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is not None:
                # Seleccionar símbolo
                mt5.symbol_select(symbol, True)
                
                # Obtener tick actual
                tick = mt5.symbol_info_tick(symbol)
                
                available_symbols.append(symbol)
                print(f"   ✅ {symbol}: Bid={tick.bid:.5f}, Ask={tick.ask:.5f}, Spread={symbol_info.spread}")
            else:
                print(f"   ❌ {symbol}: No disponible")
        
        print(f"\n📊 Símbolos disponibles: {len(available_symbols)}/{len(symbols_to_test)}")
        
        mt5.shutdown()
        return len(available_symbols) > 0
        
    except Exception as e:
        print(f"❌ Error verificando símbolos: {str(e)}")
        return False


def test_mt5_connector():
    """Prueba el conector personalizado."""
    print("\n🔍 PRUEBA 3: Conector MT5 Personalizado")
    print("=" * 50)
    
    try:
        connector = MT5Connector()
        
        # Conectar
        if not connector.connect():
            print("❌ Fallo conectando con MT5Connector")
            return False
        
        print("✅ MT5Connector conectado exitosamente")
        
        # Obtener información de cuenta
        account_info = connector.get_account_info()
        if account_info:
            print(f"   📊 Cuenta: {account_info['account_id']}")
            print(f"   💰 Balance: ${account_info['balance']:,.2f}")
            print(f"   🏦 Broker: {account_info['company']}")
        
        # Obtener posiciones
        positions = connector.get_positions()
        print(f"   📈 Posiciones activas: {len(positions)}")
        
        for pos in positions:
            print(f"      🔹 {pos.symbol} #{pos.ticket}: {pos.side.upper()} | P&L: ${pos.unrealized_pnl:+.2f}")
        
        # Desconectar
        connector.disconnect()
        print("✅ Desconectado correctamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en MT5Connector: {str(e)}")
        return False


def test_demo_order():
    """Prueba colocación de orden demo (opcional)."""
    print("\n🔍 PRUEBA 4: Orden Demo (OPCIONAL)")
    print("=" * 50)
    print("⚠️  Esta prueba coloca una orden REAL en tu cuenta")
    
    response = input("¿Deseas ejecutar una orden demo? (y/N): ").strip().lower()
    
    if response != 'y':
        print("⏸️  Prueba de orden omitida")
        return True
    
    try:
        connector = MT5Connector()
        
        if not connector.connect():
            print("❌ No se pudo conectar")
            return False
        
        # Obtener información de cuenta
        account_info = connector.get_account_info()
        
        if not account_info['trade_allowed']:
            print("❌ Trading no permitido en esta cuenta")
            return False
        
        print("🚀 Colocando orden demo de 0.01 lotes en EURUSD...")
        
        # Orden mínima
        result = connector.place_order(
            symbol="EURUSD",
            side="buy",
            volume=0.01,  # Lote mínimo
            sl=None,  # Sin SL para prueba
            tp=None   # Sin TP para prueba
        )
        
        if result.success:
            print("✅ ¡ORDEN DEMO EJECUTADA!")
            print(f"   🎫 Ticket: {result.order_ticket}")
            print(f"   💰 Precio: {result.execution_price:.5f}")
            
            # Esperar y cerrar inmediatamente
            import time
            print("⏳ Esperando 5 segundos antes de cerrar...")
            time.sleep(5)
            
            # Cerrar posición
            close_result = connector.close_position(result.order_ticket)
            if close_result.success:
                print("✅ Posición cerrada exitosamente")
            else:
                print(f"❌ Error cerrando: {close_result.error_description}")
                
        else:
            print(f"❌ Error ejecutando orden: {result.error_description}")
        
        connector.disconnect()
        return result.success
        
    except Exception as e:
        print(f"❌ Error en prueba de orden: {str(e)}")
        return False


def main():
    """Función principal de pruebas."""
    print("🤖 PRUEBAS DE CONEXIÓN REAL A METATRADER 5")
    print("=" * 60)
    print("💡 Asegúrate de que MetaTrader 5 esté abierto y configurado")
    print("💡 Para pruebas, usa una cuenta DEMO")
    print("=" * 60)
    
    # Ejecutar pruebas
    tests = [
        ("Conexión Básica", test_mt5_basic_connection),
        ("Símbolos Disponibles", test_mt5_symbols),
        ("Conector Personalizado", test_mt5_connector),
        ("Orden Demo", test_demo_order)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error en {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Resumen
    print("\n📊 RESUMEN DE PRUEBAS")
    print("=" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Resultado: {passed}/{len(results)} pruebas exitosas")
    
    if passed == len(results):
        print("🎉 ¡TODAS LAS PRUEBAS EXITOSAS!")
        print("✅ Tu bot está listo para trading real")
    else:
        print("⚠️  Algunas pruebas fallaron")
        print("💡 Verifica la configuración de MetaTrader 5")


if __name__ == "__main__":
    main() 