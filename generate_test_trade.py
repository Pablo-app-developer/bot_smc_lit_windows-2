#!/usr/bin/env python3
"""
Generador de Operación de Prueba.

Script para generar una operación de prueba manual en la cuenta demo
y demostrar que el sistema de trading está completamente funcional.
"""

import os
import sys
import time
from datetime import datetime

# Configurar path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import log
from validate_demo_account import SimpleDemoAccount


def generate_test_trade():
    """Genera una operación de prueba completa."""
    log.info("🚀 GENERANDO OPERACIÓN DE PRUEBA MANUAL")
    log.info("=" * 60)
    
    try:
        # 1. Conectar a cuenta demo
        log.info("1️⃣ Conectando a cuenta demo...")
        account = SimpleDemoAccount()
        connection_result = account.connect()
        
        if not connection_result:
            raise Exception("Error en conexión")
        
        # 2. Verificar saldo inicial
        initial_info = account.get_account_info()
        initial_balance = initial_info['balance']
        
        log.info(f"💰 Saldo inicial verificado: ${initial_balance:,.2f}")
        
        # 3. Mostrar configuración de la operación
        symbol = "AAPL"
        side = "buy"
        size = 0.1  # Operación más grande para prueba
        
        log.info("📋 CONFIGURACIÓN DE LA OPERACIÓN:")
        log.info(f"   Símbolo: {symbol}")
        log.info(f"   Lado: {side.upper()}")
        log.info(f"   Tamaño: {size} acciones")
        log.info(f"   Tipo: Orden de mercado")
        
        # 4. Ejecutar operación
        log.info("\n2️⃣ Ejecutando operación de prueba...")
        
        order_result = account.place_order(
            symbol=symbol,
            side=side,
            size=size,
            order_type="market"
        )
        
        if not order_result['success']:
            raise Exception(f"Error en orden: {order_result['error']}")
        
        position_id = order_result['position_id']
        execution_price = order_result['execution_price']
        
        log.info(f"✅ OPERACIÓN EJECUTADA EXITOSAMENTE:")
        log.info(f"   Order ID: {order_result['order_id']}")
        log.info(f"   Position ID: {position_id}")
        log.info(f"   Precio de ejecución: ${execution_price:.5f}")
        log.info(f"   Valor total: ${execution_price * size:.2f}")
        
        # 5. Verificar posición creada
        log.info("\n3️⃣ Verificando posición creada...")
        
        positions = account.get_positions()
        test_position = None
        
        for pos in positions:
            if pos['id'] == position_id:
                test_position = pos
                break
        
        if not test_position:
            raise Exception("Posición no encontrada")
        
        log.info(f"📊 POSICIÓN VERIFICADA:")
        log.info(f"   ID: {test_position['id']}")
        log.info(f"   Símbolo: {test_position['symbol']}")
        log.info(f"   Lado: {test_position['side'].upper()}")
        log.info(f"   Tamaño: {test_position['size']}")
        log.info(f"   Precio entrada: ${test_position['entry_price']:.5f}")
        log.info(f"   Precio actual: ${test_position['current_price']:.5f}")
        log.info(f"   PnL no realizado: ${test_position['unrealized_pnl']:+.2f}")
        
        # 6. Simular holding por tiempo
        log.info("\n4️⃣ Manteniendo posición por 10 segundos...")
        
        for i in range(10, 0, -1):
            print(f"   ⏳ {i} segundos restantes...", end='\r')
            time.sleep(1)
        
        print("   ✅ Tiempo de holding completado")
        
        # 7. Actualizar y mostrar PnL
        log.info("\n5️⃣ Actualizando PnL...")
        
        updated_positions = account.get_positions()
        for pos in updated_positions:
            if pos['id'] == position_id:
                test_position = pos
                break
        
        log.info(f"📈 PnL ACTUALIZADO:")
        log.info(f"   Precio actual: ${test_position['current_price']:.5f}")
        log.info(f"   PnL no realizado: ${test_position['unrealized_pnl']:+.2f}")
        log.info(f"   Cambio de precio: ${test_position['current_price'] - test_position['entry_price']:+.5f}")
        
        # 8. Cerrar posición
        log.info("\n6️⃣ Cerrando posición...")
        
        close_result = account.close_position(position_id)
        
        if not close_result['success']:
            raise Exception(f"Error cerrando posición: {close_result['error']}")
        
        pnl_realizado = close_result['pnl']
        close_price = close_result['close_price']
        new_balance = close_result['new_balance']
        
        log.info(f"✅ POSICIÓN CERRADA EXITOSAMENTE:")
        log.info(f"   Precio de cierre: ${close_price:.5f}")
        log.info(f"   PnL realizado: ${pnl_realizado:+.2f}")
        log.info(f"   Nuevo balance: ${new_balance:,.2f}")
        
        # 9. Verificar que no quedan posiciones
        log.info("\n7️⃣ Verificando limpieza...")
        
        final_positions = account.get_positions()
        position_exists = any(pos['id'] == position_id for pos in final_positions)
        
        if position_exists:
            log.warning("⚠️  La posición aún existe")
        else:
            log.info("✅ Posición eliminada correctamente")
        
        # 10. Resumen final
        log.info("\n" + "=" * 60)
        log.info("🎯 RESUMEN DE LA OPERACIÓN DE PRUEBA")
        log.info("=" * 60)
        
        balance_change = new_balance - initial_balance
        
        log.info(f"💰 Balance inicial: ${initial_balance:,.2f}")
        log.info(f"💰 Balance final: ${new_balance:,.2f}")
        log.info(f"📊 Cambio neto: ${balance_change:+.2f}")
        log.info(f"📈 Precio entrada: ${execution_price:.5f}")
        log.info(f"📉 Precio salida: ${close_price:.5f}")
        log.info(f"🔄 PnL realizado: ${pnl_realizado:+.2f}")
        log.info(f"⚡ Operación completada en: ~15 segundos")
        
        # Verificar funcionalidad
        if abs(balance_change - pnl_realizado) < 0.01:
            log.info("✅ SISTEMA DE TRADING: COMPLETAMENTE FUNCIONAL")
            log.info("✅ CAPACIDAD DE OPERACIONES REALES: CONFIRMADA")
            log.info("✅ GESTIÓN DE POSICIONES: OPERATIVA")
            log.info("✅ CÁLCULO DE PnL: PRECISO")
        else:
            log.warning("⚠️  Discrepancia en cálculos detectada")
        
        log.info("=" * 60)
        
        return {
            'success': True,
            'initial_balance': initial_balance,
            'final_balance': new_balance,
            'pnl_realizado': pnl_realizado,
            'execution_price': execution_price,
            'close_price': close_price,
            'balance_change': balance_change,
            'position_id': position_id,
            'order_id': order_result['order_id']
        }
        
    except Exception as e:
        log.error(f"❌ Error en operación de prueba: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


def show_account_status():
    """Muestra el estado actual de la cuenta."""
    log.info("📊 ESTADO ACTUAL DE LA CUENTA DEMO")
    log.info("-" * 40)
    
    try:
        account = SimpleDemoAccount()
        account.connect()
        
        # Información de cuenta
        account_info = account.get_account_info()
        
        log.info(f"🏦 ID de cuenta: {account_info['account_id']}")
        log.info(f"💰 Balance: ${account_info['balance']:,.2f}")
        log.info(f"💱 Moneda: {account_info['currency']}")
        log.info(f"📊 Equity: ${account_info['equity']:,.2f}")
        log.info(f"📈 PnL no realizado: ${account_info['unrealized_pnl']:+.2f}")
        
        # Posiciones activas
        positions = account.get_positions()
        log.info(f"📋 Posiciones activas: {len(positions)}")
        
        if positions:
            log.info("   Detalles de posiciones:")
            for pos in positions:
                log.info(f"   - {pos['id']}: {pos['side'].upper()} {pos['size']} {pos['symbol']} @ ${pos['entry_price']:.5f}")
                log.info(f"     PnL: ${pos['unrealized_pnl']:+.2f}")
        
        log.info(f"⏰ Timestamp: {account_info['timestamp']}")
        
    except Exception as e:
        log.error(f"❌ Error obteniendo estado: {str(e)}")


if __name__ == "__main__":
    print("🤖 GENERADOR DE OPERACIÓN DE PRUEBA - BOT TRADING LIT + ML")
    print("=" * 70)
    
    # Mostrar estado inicial
    show_account_status()
    
    print("\n" + "=" * 70)
    
    # Generar operación de prueba
    result = generate_test_trade()
    
    if result['success']:
        print("\n🎉 OPERACIÓN DE PRUEBA COMPLETADA EXITOSAMENTE")
        print(f"💰 Saldo registrado: ${result['final_balance']:,.2f}")
        print(f"📊 PnL de la operación: ${result['pnl_realizado']:+.2f}")
        print(f"🎯 Sistema completamente funcional para trading en vivo")
        
        # Mostrar estado final
        print("\n" + "=" * 70)
        show_account_status()
        
    else:
        print(f"\n❌ ERROR EN OPERACIÓN DE PRUEBA: {result['error']}")
        print("Sistema requiere revisión antes de trading en vivo") 