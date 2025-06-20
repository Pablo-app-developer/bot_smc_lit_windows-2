"""
Conector Real a MetaTrader 5 - Conexión Profesional.

Basado en las mejores prácticas de trading algorítmico y los repositorios:
- trading-algoritmico-metatrader-5
- libro-trading-python-es
- TopForex.Trade guide

Implementa conexión real para operaciones en vivo.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import time
import threading
from dataclasses import dataclass

from src.utils.logger import log


@dataclass
class MT5Position:
    """Posición real de MetaTrader 5."""
    ticket: int
    symbol: str
    type: int  # 0=buy, 1=sell
    volume: float
    price_open: float
    price_current: float
    profit: float
    swap: float
    commission: float
    time_setup: datetime
    sl: float = 0.0
    tp: float = 0.0
    
    @property
    def side(self) -> str:
        return "buy" if self.type == 0 else "sell"
    
    @property
    def unrealized_pnl(self) -> float:
        return self.profit + self.swap + self.commission


@dataclass
class MT5OrderResult:
    """Resultado de orden en MetaTrader 5."""
    success: bool
    order_ticket: int = 0
    error_code: int = 0
    error_description: str = ""
    execution_price: float = 0.0
    execution_time: datetime = None
    volume_executed: float = 0.0


class MT5Connector:
    """
    Conector Profesional a MetaTrader 5.
    
    Características:
    - Conexión real a cuenta demo/live
    - Ejecución de órdenes reales
    - Gestión de posiciones en tiempo real
    - Monitoreo continuo de cuenta
    - Manejo profesional de errores
    """
    
    def __init__(self, account: int = None, password: str = None, server: str = None):
        """
        Inicializa conector MT5.
        
        Args:
            account: Número de cuenta (opcional si ya está configurado)
            password: Contraseña de cuenta
            server: Servidor del broker
        """
        self.account = account
        self.password = password
        self.server = server
        
        # Estado de conexión
        self.is_connected = False
        self.last_heartbeat = None
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        
        # Cache de datos
        self._positions_cache = {}
        self._account_info_cache = {}
        self._last_cache_update = None
        self._cache_ttl = 5  # 5 segundos
        
        # Configuración de trading
        self.default_deviation = 20  # Desviación en puntos
        self.default_timeout = 30000  # Timeout en ms
        
        log.info("MT5Connector inicializado")
    
    def connect(self) -> bool:
        """
        Establece conexión real con MetaTrader 5.
        
        Returns:
            bool: True si la conexión fue exitosa
        """
        try:
            log.info("🔌 Conectando a MetaTrader 5...")
            
            # Intentar inicializar MT5
            if not mt5.initialize():
                log.error(f"❌ Error inicializando MT5: {mt5.last_error()}")
                return False
            
            # Si se proporcionaron credenciales, hacer login
            if self.account and self.password and self.server:
                log.info(f"🔐 Haciendo login a cuenta {self.account} en servidor {self.server}")
                
                if not mt5.login(self.account, self.password, self.server):
                    error = mt5.last_error()
                    log.error(f"❌ Error en login: {error}")
                    mt5.shutdown()
                    return False
            
            # Verificar conexión
            account_info = mt5.account_info()
            if account_info is None:
                log.error("❌ No se pudo obtener información de cuenta")
                mt5.shutdown()
                return False
            
            # Verificar estado de conexión
            if not account_info.trade_allowed:
                log.warning("⚠️  Trading no permitido en esta cuenta")
            
            self.is_connected = True
            self.last_heartbeat = datetime.now()
            
            log.info("✅ Conexión MT5 establecida exitosamente")
            log.info(f"   📊 Cuenta: {account_info.login}")
            log.info(f"   🏦 Broker: {account_info.company}")
            log.info(f"   💰 Balance: ${account_info.balance:,.2f}")
            log.info(f"   📈 Equity: ${account_info.equity:,.2f}")
            log.info(f"   💱 Moneda: {account_info.currency}")
            
            return True
            
        except Exception as e:
            log.error(f"❌ Error crítico conectando a MT5: {str(e)}")
            self.is_connected = False
            return False
    
    def get_account_info(self) -> Dict[str, Any]:
        """Obtiene información real de la cuenta."""
        try:
            if not self.is_connected:
                return {}
            
            account_info = mt5.account_info()
            if account_info is None:
                return {}
            
            return {
                'account_id': account_info.login,
                'balance': float(account_info.balance),
                'equity': float(account_info.equity),
                'margin': float(account_info.margin),
                'free_margin': float(account_info.margin_free),
                'currency': account_info.currency,
                'company': account_info.company,
                'trade_allowed': account_info.trade_allowed,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            log.error(f"Error obteniendo info de cuenta: {str(e)}")
            return {}
    
    def place_order(self, symbol: str, side: str, volume: float,
                   order_type: str = "market", sl: float = None, 
                   tp: float = None) -> MT5OrderResult:
        """Coloca una orden real en MT5."""
        try:
            if not self.is_connected:
                return MT5OrderResult(success=False, error_description="Sin conexión")
            
            log.info(f"📋 Colocando orden real: {side.upper()} {volume} {symbol}")
            
            # Obtener información del símbolo
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return MT5OrderResult(success=False, 
                                    error_description=f"Símbolo {symbol} no encontrado")
            
            # Obtener precios actuales
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return MT5OrderResult(success=False,
                                    error_description=f"No hay precios para {symbol}")
            
            # Configurar orden
            if side.lower() == "buy":
                trade_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            else:
                trade_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            
            # Crear request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": trade_type,
                "price": price,
                "deviation": self.default_deviation,
                "magic": 234000,
                "comment": f"Python Bot - {datetime.now().strftime('%H:%M:%S')}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            if sl:
                request["sl"] = sl
            if tp:
                request["tp"] = tp
            
            # Enviar orden
            result = mt5.order_send(request)
            
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = result.comment if result else "order_send devolvió None"
                log.error(f"❌ Error en orden: {error_msg}")
                return MT5OrderResult(success=False, error_description=error_msg)
            
            log.info(f"✅ Orden ejecutada: Ticket {result.order}, Precio {result.price}")
            
            return MT5OrderResult(
                success=True,
                order_ticket=result.order,
                execution_price=result.price,
                volume_executed=result.volume
            )
            
        except Exception as e:
            log.error(f"❌ Error crítico en orden: {str(e)}")
            return MT5OrderResult(success=False, error_description=str(e))
    
    def get_positions(self) -> List[MT5Position]:
        """Obtiene posiciones reales."""
        try:
            if not self.is_connected:
                return []
            
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            result = []
            for pos in positions:
                # Verificar atributos disponibles para evitar errores
                commission = getattr(pos, 'commission', 0.0)
                swap = getattr(pos, 'swap', 0.0)
                sl = getattr(pos, 'sl', 0.0)
                tp = getattr(pos, 'tp', 0.0)
                
                mt5_pos = MT5Position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    type=pos.type,
                    volume=pos.volume,
                    price_open=pos.price_open,
                    price_current=pos.price_current,
                    profit=pos.profit,
                    swap=swap,
                    commission=commission,
                    time_setup=datetime.fromtimestamp(pos.time),
                    sl=sl,
                    tp=tp
                )
                result.append(mt5_pos)
            
            return result
            
        except Exception as e:
            log.error(f"Error obteniendo posiciones: {str(e)}")
            return []
    
    def close_position(self, ticket: int) -> MT5OrderResult:
        """Cierra una posición real."""
        try:
            if not self.is_connected:
                return MT5OrderResult(success=False, error_description="Sin conexión")
            
            # Obtener posición
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return MT5OrderResult(success=False, 
                                    error_description=f"Posición {ticket} no encontrada")
            
            position = position[0]
            
            # Obtener precio actual
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                return MT5OrderResult(success=False,
                                    error_description="No hay precios")
            
            # Configurar orden de cierre
            if position.type == mt5.ORDER_TYPE_BUY:
                trade_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                trade_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": trade_type,
                "position": ticket,
                "price": price,
                "deviation": self.default_deviation,
                "magic": 234000,
                "comment": f"Close - {datetime.now().strftime('%H:%M:%S')}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Enviar orden de cierre
            result = mt5.order_send(request)
            
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = result.comment if result else "Error cerrando"
                return MT5OrderResult(success=False, error_description=error_msg)
            
            log.info(f"✅ Posición {ticket} cerrada. P&L: {position.profit:.2f}")
            
            return MT5OrderResult(
                success=True,
                order_ticket=result.order,
                execution_price=result.price
            )
            
        except Exception as e:
            log.error(f"❌ Error cerrando posición: {str(e)}")
            return MT5OrderResult(success=False, error_description=str(e))
    
    def disconnect(self):
        """Desconecta de MT5."""
        if self.is_connected:
            mt5.shutdown()
            self.is_connected = False
            log.info("🔌 Desconectado de MT5")


# Función de conveniencia
def create_mt5_connector(**kwargs) -> MT5Connector:
    """Crea conector MT5."""
    return MT5Connector(**kwargs) 