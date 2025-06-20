#!/usr/bin/env python3
"""
Test específico para verificar posiciones MT5.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.brokers.mt5_connector import MT5Connector
from src.utils.logger import log

def test_mt5_positions():
    """Prueba específica para posiciones MT5."""
    log.info("🧪 Iniciando prueba de posiciones MT5...")
    
    # Crear conector
    connector = MT5Connector()
    
    try:
        # Conectar
        log.info("1️⃣ Conectando a MT5...")
        if not connector.connect():
            log.error("❌ Error conectando a MT5")
            return False
        
        log.info("✅ Conectado exitosamente")
        
        # Obtener información de cuenta
        log.info("2️⃣ Obteniendo información de cuenta...")
        account_info = connector.get_account_info()
        
        if account_info:
            log.info(f"✅ Cuenta: {account_info['account_id']}")
            log.info(f"   Balance: ${account_info['balance']:,.2f}")
            log.info(f"   Equity: ${account_info['equity']:,.2f}")
            log.info(f"   Moneda: {account_info['currency']}")
            log.info(f"   Broker: {account_info['company']}")
        else:
            log.error("❌ Error obteniendo información de cuenta")
            return False
        
        # Probar obtener posiciones (sin errores)
        log.info("3️⃣ Obteniendo posiciones...")
        positions = connector.get_positions()
        
        log.info(f"✅ Posiciones obtenidas: {len(positions)}")
        
        if positions:
            for i, pos in enumerate(positions, 1):
                log.info(f"   Posición {i}:")
                log.info(f"     Ticket: {pos.ticket}")
                log.info(f"     Símbolo: {pos.symbol}")
                log.info(f"     Lado: {pos.side}")
                log.info(f"     Volumen: {pos.volume}")
                log.info(f"     Precio apertura: {pos.price_open}")
                log.info(f"     Precio actual: {pos.price_current}")
                log.info(f"     P&L: ${pos.profit:+.2f}")
                log.info(f"     Swap: ${pos.swap:+.2f}")
                log.info(f"     Comisión: ${pos.commission:+.2f}")
                log.info(f"     P&L Total: ${pos.unrealized_pnl:+.2f}")
                if pos.sl > 0:
                    log.info(f"     Stop Loss: {pos.sl}")
                if pos.tp > 0:
                    log.info(f"     Take Profit: {pos.tp}")
                log.info("")
        else:
            log.info("   No hay posiciones abiertas")
        
        # Probar colocar una orden pequeña para generar una posición
        log.info("4️⃣ Probando colocar orden pequeña...")
        
        order_result = connector.place_order(
            symbol="EURUSD",
            side="buy",
            volume=0.01,
            order_type="market"
        )
        
        if order_result.success:
            log.info(f"✅ Orden colocada: Ticket {order_result.order_ticket}")
            log.info(f"   Precio: {order_result.execution_price}")
            
            # Esperar un momento y verificar posiciones nuevamente
            import time
            time.sleep(2)
            
            log.info("5️⃣ Verificando nueva posición...")
            positions_after = connector.get_positions()
            log.info(f"✅ Posiciones después de orden: {len(positions_after)}")
            
            # Cerrar la posición de prueba
            if positions_after:
                for pos in positions_after:
                    if pos.ticket == order_result.order_ticket:
                        log.info(f"6️⃣ Cerrando posición de prueba {pos.ticket}...")
                        close_result = connector.close_position(pos.ticket)
                        
                        if close_result.success:
                            log.info("✅ Posición de prueba cerrada exitosamente")
                        else:
                            log.error(f"❌ Error cerrando posición: {close_result.error_description}")
                        break
        else:
            log.warning(f"⚠️  No se pudo colocar orden de prueba: {order_result.error_description}")
        
        log.info("🎯 Prueba de posiciones completada exitosamente")
        return True
        
    except Exception as e:
        log.error(f"❌ Error en prueba: {str(e)}")
        return False
    
    finally:
        # Desconectar
        connector.disconnect()

if __name__ == "__main__":
    try:
        success = test_mt5_positions()
        if success:
            print("\n✅ TODAS LAS PRUEBAS PASARON")
        else:
            print("\n❌ ALGUNAS PRUEBAS FALLARON")
    except KeyboardInterrupt:
        print("\n🛑 Prueba interrumpida por usuario")
    except Exception as e:
        print(f"\n💥 Error ejecutando pruebas: {str(e)}") 