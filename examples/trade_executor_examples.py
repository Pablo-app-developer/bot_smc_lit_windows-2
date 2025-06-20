#!/usr/bin/env python3
"""
Ejemplos de uso del Trade Executor.

Este archivo contiene ejemplos prácticos de cómo usar el ejecutor
de operaciones de trading con diferentes configuraciones.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from src.trading.trade_executor import (
    TradeExecutor, TradeSignal, RiskLevel, 
    create_trade_executor, execute_signal_simple
)
from src.utils.logger import log


def ejemplo_basico():
    """Ejemplo básico de uso del ejecutor."""
    print("=" * 60)
    print("EJEMPLO 1: USO BÁSICO DEL TRADE EXECUTOR")
    print("=" * 60)
    
    # Crear ejecutor con configuración por defecto
    executor = create_trade_executor(risk_level="moderate")
    
    try:
        # Conectar
        if not executor.connect():
            print("❌ No se pudo conectar")
            return
        
        # Mostrar información de la cuenta
        account_info = executor.get_account_summary()
        print(f"✅ Conectado a cuenta: {account_info['login']}")
        print(f"💰 Balance: {account_info['balance']:.2f} {account_info['currency']}")
        print(f"📊 Equity: {account_info['equity']:.2f}")
        
        # Crear señal de ejemplo
        signal = TradeSignal(
            symbol="EURUSD",
            signal="buy",
            confidence=0.75,
            price=1.0850,
            probabilities={'buy': 0.75, 'sell': 0.15, 'hold': 0.10}
        )
        
        print(f"\n🎯 Ejecutando señal: {signal.symbol} {signal.signal.upper()}")
        print(f"   Confianza: {signal.confidence:.3f}")
        
        # Ejecutar señal
        order = executor.execute_signal(signal)
        
        if order:
            print(f"✅ Orden ejecutada: Ticket #{order.ticket}")
            print(f"   Precio: {order.fill_price:.5f}")
            print(f"   Volumen: {order.volume}")
            print(f"   SL: {order.stop_loss:.5f}")
            print(f"   TP: {order.take_profit:.5f}")
        else:
            print("❌ No se pudo ejecutar la orden")
        
    finally:
        executor.disconnect()


def ejemplo_multiples_señales():
    """Ejemplo con múltiples señales."""
    print("\n" + "=" * 60)
    print("EJEMPLO 2: MÚLTIPLES SEÑALES")
    print("=" * 60)
    
    # Señales de ejemplo
    señales = [
        {
            'symbol': 'EURUSD',
            'signal': 'buy',
            'confidence': 0.72,
            'price': 1.0850
        },
        {
            'symbol': 'GBPUSD',
            'signal': 'sell',
            'confidence': 0.68,
            'price': 1.2650
        },
        {
            'symbol': 'USDJPY',
            'signal': 'buy',
            'confidence': 0.80,
            'price': 149.50
        },
        {
            'symbol': 'AUDUSD',
            'signal': 'hold',  # Esta no se ejecutará
            'confidence': 0.55,
            'price': 0.6750
        }
    ]
    
    with create_trade_executor("moderate") as executor:
        print(f"✅ Conectado - Balance: {executor.get_account_summary()['balance']:.2f}")
        
        for i, señal_data in enumerate(señales, 1):
            print(f"\n📊 Procesando señal {i}/4: {señal_data['symbol']}")
            
            # Crear señal
            signal = TradeSignal(
                symbol=señal_data['symbol'],
                signal=señal_data['signal'],
                confidence=señal_data['confidence'],
                price=señal_data['price']
            )
            
            # Ejecutar
            order = executor.execute_signal(signal)
            
            if order:
                print(f"   ✅ Ejecutada: Ticket #{order.ticket}")
            else:
                print(f"   ❌ No ejecutada")
            
            # Pausa entre órdenes
            time.sleep(2)
        
        # Mostrar resumen final
        account_info = executor.get_account_summary()
        print(f"\n📈 RESUMEN FINAL:")
        print(f"   Órdenes enviadas: {account_info['orders_sent']}")
        print(f"   Órdenes ejecutadas: {account_info['orders_filled']}")
        print(f"   Órdenes rechazadas: {account_info['orders_rejected']}")
        print(f"   Posiciones abiertas: {account_info['open_positions']}")


def ejemplo_niveles_riesgo():
    """Ejemplo comparando diferentes niveles de riesgo."""
    print("\n" + "=" * 60)
    print("EJEMPLO 3: NIVELES DE RIESGO")
    print("=" * 60)
    
    # Señal de prueba
    test_signal = TradeSignal(
        symbol="EURUSD",
        signal="buy",
        confidence=0.70,
        price=1.0850
    )
    
    risk_levels = ['conservative', 'moderate', 'aggressive']
    
    for risk_level in risk_levels:
        print(f"\n🎯 Probando nivel: {risk_level.upper()}")
        
        with create_trade_executor(risk_level) as executor:
            # Mostrar configuración de riesgo
            risk_config = executor.risk_manager.config
            print(f"   Riesgo por operación: {risk_config['risk_per_trade']*100}%")
            print(f"   Stop Loss: {risk_config['sl_points']} puntos")
            print(f"   Take Profit: {risk_config['tp_points']} puntos")
            print(f"   Confianza mínima: {risk_config['min_confidence']}")
            
            # Verificar si se puede ejecutar
            can_open, reason = executor.risk_manager.can_open_position(
                test_signal, executor.account_info.balance
            )
            
            if can_open:
                print(f"   ✅ Señal ACEPTADA")
                
                # Calcular tamaño de posición
                position_size = executor.risk_manager.calculate_position_size(
                    test_signal.symbol, executor.account_info.balance
                )
                print(f"   📊 Tamaño calculado: {position_size} lotes")
            else:
                print(f"   ❌ Señal RECHAZADA: {reason}")


def ejemplo_gestion_posiciones():
    """Ejemplo de gestión de posiciones."""
    print("\n" + "=" * 60)
    print("EJEMPLO 4: GESTIÓN DE POSICIONES")
    print("=" * 60)
    
    with create_trade_executor("moderate") as executor:
        # Abrir algunas posiciones
        signals = [
            TradeSignal("EURUSD", "buy", 0.75, 1.0850),
            TradeSignal("GBPUSD", "sell", 0.70, 1.2650)
        ]
        
        orders = []
        for signal in signals:
            order = executor.execute_signal(signal)
            if order:
                orders.append(order)
                print(f"✅ Posición abierta: {order.symbol} - Ticket #{order.ticket}")
        
        if orders:
            print(f"\n📊 Posiciones abiertas: {len(orders)}")
            
            # Mostrar posiciones
            positions = executor.get_open_positions()
            for pos in positions:
                print(f"   {pos['symbol']}: {pos['type']} {pos['volume']} lotes")
                print(f"      Precio: {pos['open_price']:.5f}")
                print(f"      P&L: {pos['profit']:.2f}")
            
            # Esperar un poco
            print("\n⏳ Esperando 30 segundos...")
            time.sleep(30)
            
            # Cerrar todas las posiciones
            print("\n🔄 Cerrando todas las posiciones...")
            closed_count = executor.close_all_positions("Ejemplo_Cierre")
            print(f"✅ Cerradas {closed_count} posiciones")


def ejemplo_funcion_simple():
    """Ejemplo usando la función simple."""
    print("\n" + "=" * 60)
    print("EJEMPLO 5: FUNCIÓN SIMPLE")
    print("=" * 60)
    
    # Datos de señal como diccionario
    signal_data = {
        'symbol': 'EURUSD',
        'signal': 'buy',
        'confidence': 0.78,
        'price': 1.0850,
        'probabilities': {'buy': 0.78, 'sell': 0.12, 'hold': 0.10},
        'metadata': {'strategy': 'LIT_ML', 'timeframe': '1h'}
    }
    
    print(f"🎯 Ejecutando señal simple: {signal_data['symbol']} {signal_data['signal'].upper()}")
    print(f"   Confianza: {signal_data['confidence']:.3f}")
    
    # Ejecutar usando función simple
    success = execute_signal_simple(signal_data)
    
    if success:
        print("✅ Señal ejecutada exitosamente")
    else:
        print("❌ Error ejecutando señal")


def ejemplo_monitoreo_cuenta():
    """Ejemplo de monitoreo de cuenta."""
    print("\n" + "=" * 60)
    print("EJEMPLO 6: MONITOREO DE CUENTA")
    print("=" * 60)
    
    with create_trade_executor("moderate") as executor:
        print("📊 INFORMACIÓN DE LA CUENTA:")
        print("-" * 40)
        
        account_info = executor.get_account_summary()
        
        for key, value in account_info.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("\n📈 POSICIONES ABIERTAS:")
        print("-" * 40)
        
        positions = executor.get_open_positions()
        if positions:
            for pos in positions:
                print(f"Símbolo: {pos['symbol']}")
                print(f"  Tipo: {pos['type']}")
                print(f"  Volumen: {pos['volume']} lotes")
                print(f"  Precio apertura: {pos['open_price']:.5f}")
                print(f"  Precio actual: {pos['current_price']:.5f}")
                print(f"  P&L: {pos['profit']:.2f}")
                print(f"  Tiempo: {pos['open_time']}")
                print("-" * 20)
        else:
            print("No hay posiciones abiertas")


def main():
    """Ejecuta todos los ejemplos."""
    print("🤖 EJEMPLOS DEL TRADE EXECUTOR")
    print("=" * 60)
    print("NOTA: Estos ejemplos usan la cuenta demo configurada.")
    print("      Las operaciones NO afectan dinero real.")
    print("=" * 60)
    
    try:
        # Ejecutar ejemplos
        ejemplo_basico()
        ejemplo_multiples_señales()
        ejemplo_niveles_riesgo()
        ejemplo_gestion_posiciones()
        ejemplo_funcion_simple()
        ejemplo_monitoreo_cuenta()
        
        print("\n" + "=" * 60)
        print("✅ TODOS LOS EJEMPLOS COMPLETADOS")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⏹️ Ejemplos interrumpidos por el usuario")
    except Exception as e:
        print(f"\n❌ Error en ejemplos: {str(e)}")
        log.error(f"Error en ejemplos: {str(e)}")


if __name__ == "__main__":
    main() 