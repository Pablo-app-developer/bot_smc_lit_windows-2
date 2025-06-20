"""
Pruebas Unitarias para Conexión de Cuenta Demo.

Valida la conexión real a la cuenta demo, verifica el saldo
y ejecuta operaciones de prueba para confirmar funcionalidad.
"""

import pytest
import time
from datetime import datetime
from typing import Dict, Any

from src.brokers.demo_account_connector import (
    DemoAccountConnector, OrderSide, OrderType, OrderStatus
)
from src.utils.logger import log


class TestDemoAccountConnection:
    """Pruebas de conexión y funcionalidad de cuenta demo."""
    
    @pytest.fixture
    def connector(self):
        """Fixture para crear conector de cuenta demo."""
        return DemoAccountConnector()
    
    def test_account_connection(self, connector):
        """Prueba la conexión a la cuenta demo."""
        log.info("🔍 Probando conexión a cuenta demo...")
        
        # Intentar conectar
        connection_result = connector.connect()
        
        # Verificar conexión exitosa
        assert connection_result == True, "La conexión debe ser exitosa"
        
        log.info("✅ Conexión establecida correctamente")
    
    def test_account_balance_verification(self, connector):
        """Verifica el saldo real de la cuenta demo."""
        log.info("💰 Verificando saldo de cuenta demo...")
        
        # Conectar primero
        connector.connect()
        
        # Obtener información de cuenta
        account_info = connector.get_account_info()
        
        # Verificar que el saldo coincide con el reportado
        expected_balance = 2865.05
        actual_balance = account_info.balance
        
        log.info(f"Saldo esperado: ${expected_balance:,.2f}")
        log.info(f"Saldo actual: ${actual_balance:,.2f}")
        
        # Verificar saldo
        assert actual_balance == expected_balance, f"Saldo debe ser ${expected_balance:,.2f}"
        assert account_info.currency == "USD", "Moneda debe ser USD"
        assert account_info.account_id == "DEMO_001", "ID de cuenta debe ser DEMO_001"
        
        log.info("✅ Saldo verificado correctamente")
        
        return actual_balance
    
    def test_account_info_completeness(self, connector):
        """Verifica que la información de cuenta esté completa."""
        log.info("📊 Verificando información completa de cuenta...")
        
        connector.connect()
        account_info = connector.get_account_info()
        
        # Verificar campos obligatorios
        assert account_info.account_id is not None, "ID de cuenta requerido"
        assert account_info.balance > 0, "Balance debe ser positivo"
        assert account_info.equity >= 0, "Equity debe ser no negativo"
        assert account_info.margin_available >= 0, "Margen disponible debe ser no negativo"
        assert account_info.currency == "USD", "Moneda debe ser USD"
        assert account_info.leverage > 0, "Leverage debe ser positivo"
        assert isinstance(account_info.timestamp, datetime), "Timestamp debe ser datetime"
        
        log.info("✅ Información de cuenta completa y válida")
    
    def test_small_test_trade_execution(self, connector):
        """Ejecuta una operación pequeña de prueba."""
        log.info("🔄 Ejecutando operación de prueba...")
        
        connector.connect()
        
        # Verificar balance inicial
        initial_account = connector.get_account_info()
        initial_balance = initial_account.balance
        
        log.info(f"Balance inicial: ${initial_balance:,.2f}")
        
        # Ejecutar orden pequeña de compra
        order_result = connector.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            size=0.01,  # Tamaño mínimo
            order_type=OrderType.MARKET,
            stop_loss=None,
            take_profit=None
        )
        
        # Verificar que la orden fue exitosa
        assert order_result['success'] == True, "Orden debe ser exitosa"
        assert order_result['order_id'] is not None, "Debe tener ID de orden"
        assert order_result['position_id'] is not None, "Debe crear posición"
        
        order_id = order_result['order_id']
        position_id = order_result['position_id']
        execution_price = order_result['execution_price']
        
        log.info(f"✅ Orden ejecutada: {order_id}")
        log.info(f"   Posición creada: {position_id}")
        log.info(f"   Precio de ejecución: ${execution_price:.5f}")
        
        # Verificar que la posición existe
        positions = connector.get_positions()
        assert len(positions) > 0, "Debe haber al menos una posición"
        
        test_position = None
        for pos in positions:
            if pos.id == position_id:
                test_position = pos
                break
        
        assert test_position is not None, "Posición de prueba debe existir"
        assert test_position.symbol == "AAPL", "Símbolo debe ser AAPL"
        assert test_position.side == "buy", "Lado debe ser buy"
        assert test_position.size == 0.01, "Tamaño debe ser 0.01"
        
        log.info(f"   Posición verificada: {test_position.side} {test_position.size} {test_position.symbol}")
        
        # Esperar un momento para simular holding
        time.sleep(2)
        
        # Cerrar la posición de prueba
        close_result = connector.close_position(position_id)
        
        assert close_result['success'] == True, "Cierre debe ser exitoso"
        
        pnl = close_result['pnl']
        close_price = close_result['close_price']
        new_balance = close_result['new_balance']
        
        log.info(f"✅ Posición cerrada: {position_id}")
        log.info(f"   Precio de cierre: ${close_price:.5f}")
        log.info(f"   PnL realizado: ${pnl:+.2f}")
        log.info(f"   Nuevo balance: ${new_balance:,.2f}")
        
        # Verificar que la posición fue cerrada
        updated_positions = connector.get_positions()
        position_exists = any(pos.id == position_id for pos in updated_positions)
        assert not position_exists, "Posición debe estar cerrada"
        
        # Verificar que el balance se actualizó
        final_account = connector.get_account_info()
        final_balance = final_account.balance
        
        expected_balance = initial_balance + pnl
        assert abs(final_balance - expected_balance) < 0.01, "Balance debe reflejar PnL"
        
        log.info("✅ Operación de prueba completada exitosamente")
        
        return {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'pnl': pnl,
            'execution_price': execution_price,
            'close_price': close_price
        }
    
    def test_multiple_positions_management(self, connector):
        """Prueba manejo de múltiples posiciones."""
        log.info("📈 Probando manejo de múltiples posiciones...")
        
        connector.connect()
        
        initial_positions = len(connector.get_positions())
        
        # Crear múltiples posiciones pequeñas
        positions_created = []
        
        for i in range(2):  # Crear 2 posiciones
            order_result = connector.place_order(
                symbol="AAPL",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                size=0.01,
                order_type=OrderType.MARKET
            )
            
            assert order_result['success'] == True, f"Orden {i+1} debe ser exitosa"
            positions_created.append(order_result['position_id'])
        
        # Verificar que las posiciones existen
        current_positions = connector.get_positions()
        assert len(current_positions) == initial_positions + 2, "Deben existir 2 posiciones nuevas"
        
        log.info(f"✅ {len(positions_created)} posiciones creadas")
        
        # Cerrar todas las posiciones de prueba
        for position_id in positions_created:
            close_result = connector.close_position(position_id)
            assert close_result['success'] == True, f"Cierre de {position_id} debe ser exitoso"
        
        # Verificar que las posiciones fueron cerradas
        final_positions = connector.get_positions()
        assert len(final_positions) == initial_positions, "Posiciones deben estar cerradas"
        
        log.info("✅ Múltiples posiciones manejadas correctamente")
    
    def test_order_validation(self, connector):
        """Prueba validación de órdenes."""
        log.info("🔍 Probando validación de órdenes...")
        
        connector.connect()
        
        # Prueba orden con tamaño muy pequeño
        small_order = connector.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            size=0.001,  # Menor al mínimo
            order_type=OrderType.MARKET
        )
        
        assert small_order['success'] == False, "Orden muy pequeña debe fallar"
        assert "mínimo" in small_order['error'].lower(), "Error debe mencionar tamaño mínimo"
        
        # Prueba orden con tamaño muy grande
        large_order = connector.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            size=100.0,  # Mayor al máximo
            order_type=OrderType.MARKET
        )
        
        assert large_order['success'] == False, "Orden muy grande debe fallar"
        assert "máximo" in large_order['error'].lower(), "Error debe mencionar tamaño máximo"
        
        log.info("✅ Validación de órdenes funcionando correctamente")
    
    def test_trading_summary_generation(self, connector):
        """Prueba generación de resumen de trading."""
        log.info("📊 Probando generación de resumen...")
        
        connector.connect()
        
        # Generar resumen
        summary = connector.get_trading_summary()
        
        # Verificar estructura del resumen
        assert 'account' in summary, "Resumen debe incluir información de cuenta"
        assert 'trading' in summary, "Resumen debe incluir información de trading"
        assert 'timestamp' in summary, "Resumen debe incluir timestamp"
        
        # Verificar campos de cuenta
        account_data = summary['account']
        assert 'balance' in account_data, "Debe incluir balance"
        assert 'equity' in account_data, "Debe incluir equity"
        assert 'margin_used' in account_data, "Debe incluir margen usado"
        assert 'margin_available' in account_data, "Debe incluir margen disponible"
        
        # Verificar campos de trading
        trading_data = summary['trading']
        assert 'total_orders' in trading_data, "Debe incluir total de órdenes"
        assert 'active_positions' in trading_data, "Debe incluir posiciones activas"
        
        log.info("✅ Resumen de trading generado correctamente")
        
        return summary


def run_connection_validation():
    """
    Ejecuta validación completa de conexión.
    
    Returns:
        Dict[str, Any]: Resultados de la validación.
    """
    log.info("🚀 INICIANDO VALIDACIÓN DE CONEXIÓN A CUENTA DEMO")
    log.info("=" * 60)
    
    try:
        # Crear conector
        connector = DemoAccountConnector()
        test_instance = TestDemoAccountConnection()
        
        # Ejecutar pruebas
        results = {}
        
        # 1. Prueba de conexión
        log.info("1️⃣ Probando conexión...")
        test_instance.test_account_connection(connector)
        results['connection'] = True
        
        # 2. Verificación de saldo
        log.info("\n2️⃣ Verificando saldo...")
        balance = test_instance.test_account_balance_verification(connector)
        results['balance'] = balance
        
        # 3. Información completa
        log.info("\n3️⃣ Verificando información completa...")
        test_instance.test_account_info_completeness(connector)
        results['account_info'] = True
        
        # 4. Operación de prueba
        log.info("\n4️⃣ Ejecutando operación de prueba...")
        trade_result = test_instance.test_small_test_trade_execution(connector)
        results['test_trade'] = trade_result
        
        # 5. Múltiples posiciones
        log.info("\n5️⃣ Probando múltiples posiciones...")
        test_instance.test_multiple_positions_management(connector)
        results['multiple_positions'] = True
        
        # 6. Validación de órdenes
        log.info("\n6️⃣ Probando validación...")
        test_instance.test_order_validation(connector)
        results['order_validation'] = True
        
        # 7. Resumen de trading
        log.info("\n7️⃣ Generando resumen...")
        summary = test_instance.test_trading_summary_generation(connector)
        results['trading_summary'] = summary
        
        log.info("\n" + "=" * 60)
        log.info("✅ TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
        log.info("=" * 60)
        
        # Mostrar resumen final
        log.info("📊 RESUMEN DE VALIDACIÓN:")
        log.info(f"   💰 Saldo de cuenta: ${balance:,.2f}")
        log.info(f"   🔄 Operación de prueba: PnL ${trade_result['pnl']:+.2f}")
        log.info(f"   📈 Balance final: ${trade_result['final_balance']:,.2f}")
        log.info(f"   ✅ Conexión: EXITOSA")
        log.info(f"   ✅ Trading: FUNCIONAL")
        
        return {
            'success': True,
            'results': results,
            'summary': {
                'account_balance': balance,
                'test_trade_pnl': trade_result['pnl'],
                'final_balance': trade_result['final_balance'],
                'connection_status': 'CONNECTED',
                'trading_status': 'FUNCTIONAL'
            }
        }
        
    except Exception as e:
        log.error(f"❌ Error en validación: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'results': {},
            'summary': {
                'connection_status': 'FAILED',
                'trading_status': 'NON_FUNCTIONAL'
            }
        }


if __name__ == "__main__":
    # Ejecutar validación directamente
    validation_results = run_connection_validation()
    
    if validation_results['success']:
        print("\n🎉 VALIDACIÓN EXITOSA - CUENTA DEMO LISTA PARA TRADING")
    else:
        print(f"\n❌ VALIDACIÓN FALLIDA: {validation_results['error']}") 