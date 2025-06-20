#!/usr/bin/env python3
"""
Trade Executor - Ejecutor de Operaciones de Trading Real.

Este módulo ejecuta operaciones de trading reales en MetaTrader 5
basadas en señales del predictor LIT + ML, con gestión avanzada
de riesgos y logging completo.
"""

import os
import sys
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 no disponible")

from src.utils.logger import log


class OrderType(Enum):
    """Tipos de órdenes de trading."""
    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"


class OrderStatus(Enum):
    """Estados de las órdenes."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"


class RiskLevel(Enum):
    """Niveles de riesgo."""
    CONSERVATIVE = "CONSERVATIVE"  # 1% por operación
    MODERATE = "MODERATE"         # 2% por operación
    AGGRESSIVE = "AGGRESSIVE"     # 3% por operación


class TradeSignal:
    """
    Clase para representar una señal de trading.
    """
    
    def __init__(self, 
                 symbol: str,
                 signal: str,  # 'buy', 'sell', 'hold'
                 confidence: float,
                 price: float,
                 timestamp: datetime = None,
                 probabilities: Dict[str, float] = None,
                 metadata: Dict[str, Any] = None):
        """
        Inicializa una señal de trading.
        
        Args:
            symbol: Símbolo del instrumento
            signal: Tipo de señal ('buy', 'sell', 'hold')
            confidence: Nivel de confianza [0-1]
            price: Precio de referencia
            timestamp: Momento de la señal
            probabilities: Probabilidades por clase
            metadata: Metadatos adicionales
        """
        self.symbol = symbol
        self.signal = signal.lower()
        self.confidence = confidence
        self.price = price
        self.timestamp = timestamp or datetime.now()
        self.probabilities = probabilities or {}
        self.metadata = metadata or {}
        
        # Validar señal
        if self.signal not in ['buy', 'sell', 'hold']:
            raise ValueError(f"Señal inválida: {signal}")
        
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confianza inválida: {confidence}")


class TradeOrder:
    """
    Clase para representar una orden de trading.
    """
    
    def __init__(self,
                 symbol: str,
                 order_type: OrderType,
                 volume: float,
                 price: float = None,
                 stop_loss: float = None,
                 take_profit: float = None,
                 comment: str = "",
                 magic: int = 12345):
        """
        Inicializa una orden de trading.
        
        Args:
            symbol: Símbolo del instrumento
            order_type: Tipo de orden
            volume: Volumen en lotes
            price: Precio de la orden (para órdenes pendientes)
            stop_loss: Nivel de Stop Loss
            take_profit: Nivel de Take Profit
            comment: Comentario de la orden
            magic: Número mágico para identificación
        """
        self.symbol = symbol
        self.order_type = order_type
        self.volume = volume
        self.price = price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.comment = comment
        self.magic = magic
        
        # Estado de la orden
        self.status = OrderStatus.PENDING
        self.ticket = None
        self.fill_price = None
        self.fill_time = None
        self.error_code = None
        self.error_message = None
        
        # Timestamps
        self.created_at = datetime.now()
        self.updated_at = datetime.now()


class RiskManager:
    """
    Gestor de riesgos para operaciones de trading.
    """
    
    def __init__(self,
                 risk_level: RiskLevel = RiskLevel.MODERATE,
                 max_risk_per_trade: float = 0.02,
                 max_daily_risk: float = 0.10,
                 max_open_positions: int = 5,
                 max_correlation: float = 0.7):
        """
        Inicializa el gestor de riesgos.
        
        Args:
            risk_level: Nivel de riesgo predefinido
            max_risk_per_trade: Riesgo máximo por operación
            max_daily_risk: Riesgo máximo diario
            max_open_positions: Máximo de posiciones abiertas
            max_correlation: Correlación máxima entre posiciones
        """
        self.risk_level = risk_level
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_risk = max_daily_risk
        self.max_open_positions = max_open_positions
        self.max_correlation = max_correlation
        
        # Configuración por nivel de riesgo
        risk_configs = {
            RiskLevel.CONSERVATIVE: {
                'risk_per_trade': 0.01,
                'sl_points': 30,
                'tp_points': 60,
                'min_confidence': 0.75
            },
            RiskLevel.MODERATE: {
                'risk_per_trade': 0.02,
                'sl_points': 50,
                'tp_points': 100,
                'min_confidence': 0.65
            },
            RiskLevel.AGGRESSIVE: {
                'risk_per_trade': 0.03,
                'sl_points': 80,
                'tp_points': 160,
                'min_confidence': 0.55
            }
        }
        
        self.config = risk_configs[risk_level]
        
        # Estadísticas de riesgo
        self.daily_risk_used = 0.0
        self.open_positions_count = 0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        
        log.info(f"RiskManager inicializado - Nivel: {risk_level.value}")
        log.info(f"  Riesgo por operación: {self.config['risk_per_trade']*100}%")
        log.info(f"  SL: {self.config['sl_points']} puntos")
        log.info(f"  TP: {self.config['tp_points']} puntos")
    
    def reset_daily_stats(self):
        """Reinicia estadísticas diarias."""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_risk_used = 0.0
            self.daily_trades = 0
            self.last_reset_date = current_date
            log.info("Estadísticas diarias reiniciadas")
    
    def can_open_position(self, signal: TradeSignal, account_balance: float) -> Tuple[bool, str]:
        """
        Verifica si se puede abrir una posición.
        
        Args:
            signal: Señal de trading
            account_balance: Balance de la cuenta
            
        Returns:
            Tuple[bool, str]: (Puede abrir, Razón si no puede)
        """
        self.reset_daily_stats()
        
        # Verificar confianza mínima
        if signal.confidence < self.config['min_confidence']:
            return False, f"Confianza insuficiente: {signal.confidence:.3f} < {self.config['min_confidence']:.3f}"
        
        # Verificar señal válida
        if signal.signal == 'hold':
            return False, "Señal HOLD - No ejecutar operación"
        
        # Verificar máximo de posiciones abiertas
        if self.open_positions_count >= self.max_open_positions:
            return False, f"Máximo de posiciones abiertas alcanzado: {self.open_positions_count}"
        
        # Verificar riesgo diario
        trade_risk = self.config['risk_per_trade']
        if self.daily_risk_used + trade_risk > self.max_daily_risk:
            return False, f"Riesgo diario excedido: {(self.daily_risk_used + trade_risk)*100:.1f}% > {self.max_daily_risk*100:.1f}%"
        
        # Verificar balance mínimo
        min_balance = 1000  # Balance mínimo requerido
        if account_balance < min_balance:
            return False, f"Balance insuficiente: {account_balance} < {min_balance}"
        
        return True, "OK"
    
    def calculate_position_size(self, 
                              symbol: str, 
                              account_balance: float,
                              stop_loss_points: int = None) -> float:
        """
        Calcula el tamaño de posición basado en gestión de riesgos.
        
        Args:
            symbol: Símbolo del instrumento
            account_balance: Balance de la cuenta
            stop_loss_points: Puntos de Stop Loss
            
        Returns:
            float: Tamaño de posición en lotes
        """
        try:
            # Obtener información del símbolo
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                log.error(f"No se pudo obtener información de {symbol}")
                return 0.01  # Lote mínimo por defecto
            
            # Riesgo por operación
            risk_amount = account_balance * self.config['risk_per_trade']
            
            # Puntos de Stop Loss
            sl_points = stop_loss_points or self.config['sl_points']
            
            # Valor por punto
            point_value = symbol_info.trade_tick_value
            if point_value == 0:
                point_value = symbol_info.trade_contract_size * symbol_info.point
            
            # Calcular tamaño de posición
            if point_value > 0:
                position_size = risk_amount / (sl_points * point_value)
            else:
                position_size = 0.01
            
            # Ajustar a límites del símbolo
            position_size = max(symbol_info.volume_min, position_size)
            position_size = min(symbol_info.volume_max, position_size)
            
            # Redondear al step del símbolo
            volume_step = symbol_info.volume_step
            position_size = round(position_size / volume_step) * volume_step
            
            log.info(f"Tamaño calculado para {symbol}: {position_size} lotes")
            log.info(f"  Riesgo: {risk_amount:.2f} ({self.config['risk_per_trade']*100}%)")
            log.info(f"  SL: {sl_points} puntos")
            
            return position_size
            
        except Exception as e:
            log.error(f"Error calculando tamaño de posición: {str(e)}")
            return 0.01
    
    def get_sl_tp_levels(self, 
                        symbol: str, 
                        order_type: OrderType, 
                        entry_price: float) -> Tuple[float, float]:
        """
        Calcula niveles de Stop Loss y Take Profit.
        
        Args:
            symbol: Símbolo del instrumento
            order_type: Tipo de orden
            entry_price: Precio de entrada
            
        Returns:
            Tuple[float, float]: (Stop Loss, Take Profit)
        """
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None, None
            
            point = symbol_info.point
            sl_points = self.config['sl_points']
            tp_points = self.config['tp_points']
            
            if order_type in [OrderType.BUY]:
                stop_loss = entry_price - (sl_points * point)
                take_profit = entry_price + (tp_points * point)
            elif order_type in [OrderType.SELL]:
                stop_loss = entry_price + (sl_points * point)
                take_profit = entry_price - (tp_points * point)
            else:
                return None, None
            
            return stop_loss, take_profit
            
        except Exception as e:
            log.error(f"Error calculando SL/TP: {str(e)}")
            return None, None
    
    def update_position_opened(self, risk_amount: float):
        """Actualiza estadísticas cuando se abre una posición."""
        self.daily_risk_used += self.config['risk_per_trade']
        self.open_positions_count += 1
        self.daily_trades += 1
        
        log.info(f"Posición abierta - Riesgo diario usado: {self.daily_risk_used*100:.1f}%")
    
    def update_position_closed(self):
        """Actualiza estadísticas cuando se cierra una posición."""
        self.open_positions_count = max(0, self.open_positions_count - 1)
        log.info(f"Posición cerrada - Posiciones abiertas: {self.open_positions_count}")


class TradeExecutor:
    """
    Ejecutor principal de operaciones de trading.
    """
    
    def __init__(self,
                 login: int = 5036791117,
                 password: str = "BtUvF-X8",
                 server: str = "MetaQuotes-Demo",
                 risk_level: RiskLevel = RiskLevel.MODERATE,
                 magic_number: int = 12345):
        """
        Inicializa el ejecutor de trading.
        
        Args:
            login: Login de MT5
            password: Password de MT5
            server: Servidor de MT5
            risk_level: Nivel de riesgo
            magic_number: Número mágico para identificación
        """
        self.login = login
        self.password = password
        self.server = server
        self.magic_number = magic_number
        
        # Componentes
        self.risk_manager = RiskManager(risk_level)
        
        # Estado de conexión
        self.connected = False
        self.account_info = None
        
        # Órdenes y posiciones
        self.active_orders: Dict[int, TradeOrder] = {}
        self.order_history: List[TradeOrder] = []
        
        # Estadísticas
        self.stats = {
            'orders_sent': 0,
            'orders_filled': 0,
            'orders_rejected': 0,
            'total_profit': 0.0,
            'start_time': datetime.now()
        }
        
        # Configuración
        self.max_slippage = 20  # Puntos de slippage máximo
        self.order_timeout = 30  # Timeout para órdenes en segundos
        
        log.info(f"TradeExecutor inicializado")
        log.info(f"  Login: {login}")
        log.info(f"  Servidor: {server}")
        log.info(f"  Nivel de riesgo: {risk_level.value}")
        log.info(f"  Magic Number: {magic_number}")
    
    def connect(self) -> bool:
        """
        Establece conexión con MetaTrader 5.
        
        Returns:
            bool: True si se conectó exitosamente
        """
        if not MT5_AVAILABLE:
            log.error("MetaTrader5 no está disponible")
            return False
        
        try:
            log.info("🔄 Conectando a MetaTrader 5...")
            
            # Inicializar MT5
            if not mt5.initialize():
                error = mt5.last_error()
                log.error(f"Error inicializando MT5: {error}")
                return False
            
            # Conectar con credenciales
            if not mt5.login(self.login, password=self.password, server=self.server):
                error = mt5.last_error()
                log.error(f"Error login MT5: {error}")
                mt5.shutdown()
                return False
            
            # Obtener información de la cuenta
            self.account_info = mt5.account_info()
            if self.account_info is None:
                log.error("No se pudo obtener información de la cuenta")
                return False
            
            self.connected = True
            
            log.info("✅ Conectado a MT5 exitosamente")
            log.info(f"  Cuenta: {self.account_info.login}")
            log.info(f"  Servidor: {self.account_info.server}")
            log.info(f"  Balance: {self.account_info.balance:.2f} {self.account_info.currency}")
            log.info(f"  Equity: {self.account_info.equity:.2f} {self.account_info.currency}")
            log.info(f"  Margen libre: {self.account_info.margin_free:.2f} {self.account_info.currency}")
            
            return True
            
        except Exception as e:
            log.error(f"Error conectando a MT5: {str(e)}")
            return False
    
    def disconnect(self):
        """Desconecta de MetaTrader 5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            log.info("✅ Desconectado de MT5")
    
    def execute_signal(self, signal: TradeSignal) -> Optional[TradeOrder]:
        """
        Ejecuta una señal de trading.
        
        Args:
            signal: Señal de trading a ejecutar
            
        Returns:
            Optional[TradeOrder]: Orden ejecutada o None si no se pudo ejecutar
        """
        if not self.connected:
            log.error("No hay conexión a MT5")
            return None
        
        try:
            log.info(f"🎯 Ejecutando señal: {signal.symbol} {signal.signal.upper()}")
            log.info(f"  Confianza: {signal.confidence:.3f}")
            log.info(f"  Precio: {signal.price:.5f}")
            
            # Actualizar información de la cuenta
            self.account_info = mt5.account_info()
            if self.account_info is None:
                log.error("No se pudo actualizar información de la cuenta")
                return None
            
            # Verificar si se puede abrir la posición
            can_open, reason = self.risk_manager.can_open_position(
                signal, self.account_info.balance
            )
            
            if not can_open:
                log.warning(f"❌ No se puede abrir posición: {reason}")
                return None
            
            # Determinar tipo de orden
            if signal.signal == 'buy':
                order_type = OrderType.BUY
            elif signal.signal == 'sell':
                order_type = OrderType.SELL
            else:
                log.warning(f"Señal no ejecutable: {signal.signal}")
                return None
            
            # Calcular tamaño de posición
            volume = self.risk_manager.calculate_position_size(
                signal.symbol, self.account_info.balance
            )
            
            if volume <= 0:
                log.error("Tamaño de posición inválido")
                return None
            
            # Obtener precio actual
            tick = mt5.symbol_info_tick(signal.symbol)
            if tick is None:
                log.error(f"No se pudo obtener tick para {signal.symbol}")
                return None
            
            # Precio de ejecución
            if order_type == OrderType.BUY:
                execution_price = tick.ask
            else:
                execution_price = tick.bid
            
            # Calcular Stop Loss y Take Profit
            stop_loss, take_profit = self.risk_manager.get_sl_tp_levels(
                signal.symbol, order_type, execution_price
            )
            
            # Crear orden
            order = TradeOrder(
                symbol=signal.symbol,
                order_type=order_type,
                volume=volume,
                price=execution_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=f"LIT_ML_{signal.signal.upper()}_{signal.confidence:.2f}",
                magic=self.magic_number
            )
            
            # Ejecutar orden
            success = self._send_order(order)
            
            if success:
                # Actualizar estadísticas de riesgo
                self.risk_manager.update_position_opened(
                    self.account_info.balance * self.risk_manager.config['risk_per_trade']
                )
                
                log.info(f"✅ Orden ejecutada exitosamente: {order.ticket}")
                return order
            else:
                log.error(f"❌ Error ejecutando orden: {order.error_message}")
                return None
                
        except Exception as e:
            log.error(f"Error ejecutando señal: {str(e)}")
            return None
    
    def _send_order(self, order: TradeOrder) -> bool:
        """
        Envía una orden a MetaTrader 5.
        
        Args:
            order: Orden a enviar
            
        Returns:
            bool: True si se envió exitosamente
        """
        try:
            # Mapear tipo de orden
            mt5_order_types = {
                OrderType.BUY: mt5.ORDER_TYPE_BUY,
                OrderType.SELL: mt5.ORDER_TYPE_SELL,
                OrderType.BUY_LIMIT: mt5.ORDER_TYPE_BUY_LIMIT,
                OrderType.SELL_LIMIT: mt5.ORDER_TYPE_SELL_LIMIT,
                OrderType.BUY_STOP: mt5.ORDER_TYPE_BUY_STOP,
                OrderType.SELL_STOP: mt5.ORDER_TYPE_SELL_STOP
            }
            
            # Crear request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": order.symbol,
                "volume": order.volume,
                "type": mt5_order_types[order.order_type],
                "deviation": self.max_slippage,
                "magic": order.magic,
                "comment": order.comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Agregar precio si es necesario
            if order.price is not None:
                request["price"] = order.price
            
            # Agregar Stop Loss y Take Profit
            if order.stop_loss is not None:
                request["sl"] = order.stop_loss
            
            if order.take_profit is not None:
                request["tp"] = order.take_profit
            
            log.info(f"📤 Enviando orden: {order.symbol} {order.order_type.value} {order.volume}")
            log.info(f"  Precio: {order.price:.5f}")
            log.info(f"  SL: {order.stop_loss:.5f}")
            log.info(f"  TP: {order.take_profit:.5f}")
            
            # Enviar orden
            result = mt5.order_send(request)
            
            # Procesar resultado
            if result is None:
                order.status = OrderStatus.REJECTED
                order.error_code = -1
                order.error_message = "No se recibió respuesta de MT5"
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                order.status = OrderStatus.REJECTED
                order.error_code = result.retcode
                order.error_message = f"Error MT5: {result.retcode}"
                
                log.error(f"❌ Orden rechazada: {result.retcode}")
                if hasattr(result, 'comment'):
                    log.error(f"   Comentario: {result.comment}")
                
                self.stats['orders_rejected'] += 1
                return False
            
            # Orden exitosa
            order.status = OrderStatus.FILLED
            order.ticket = result.order
            order.fill_price = result.price
            order.fill_time = datetime.now()
            order.updated_at = datetime.now()
            
            # Guardar orden
            self.active_orders[order.ticket] = order
            self.order_history.append(order)
            
            # Actualizar estadísticas
            self.stats['orders_sent'] += 1
            self.stats['orders_filled'] += 1
            
            log.info(f"✅ Orden ejecutada: Ticket #{order.ticket}")
            log.info(f"  Precio de ejecución: {order.fill_price:.5f}")
            log.info(f"  Volumen: {order.volume}")
            
            return True
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            log.error(f"Error enviando orden: {str(e)}")
            return False
    
    def close_position(self, ticket: int, reason: str = "Manual") -> bool:
        """
        Cierra una posición específica.
        
        Args:
            ticket: Ticket de la posición a cerrar
            reason: Razón del cierre
            
        Returns:
            bool: True si se cerró exitosamente
        """
        try:
            # Obtener información de la posición
            position = mt5.positions_get(ticket=ticket)
            if not position:
                log.warning(f"Posición {ticket} no encontrada")
                return False
            
            position = position[0]
            
            # Determinar tipo de orden de cierre
            if position.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(position.symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(position.symbol).ask
            
            # Crear request de cierre
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": self.max_slippage,
                "magic": self.magic_number,
                "comment": f"Close_{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            log.info(f"🔄 Cerrando posición: Ticket #{ticket}")
            log.info(f"  Símbolo: {position.symbol}")
            log.info(f"  Volumen: {position.volume}")
            log.info(f"  Precio: {price:.5f}")
            
            # Enviar orden de cierre
            result = mt5.order_send(request)
            
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error_code = result.retcode if result else -1
                log.error(f"❌ Error cerrando posición: {error_code}")
                return False
            
            # Actualizar estadísticas
            self.risk_manager.update_position_closed()
            
            # Remover de órdenes activas
            if ticket in self.active_orders:
                del self.active_orders[ticket]
            
            log.info(f"✅ Posición cerrada: Ticket #{ticket}")
            return True
            
        except Exception as e:
            log.error(f"Error cerrando posición {ticket}: {str(e)}")
            return False
    
    def close_all_positions(self, reason: str = "CloseAll") -> int:
        """
        Cierra todas las posiciones abiertas.
        
        Args:
            reason: Razón del cierre
            
        Returns:
            int: Número de posiciones cerradas
        """
        try:
            positions = mt5.positions_get()
            if not positions:
                log.info("No hay posiciones abiertas para cerrar")
                return 0
            
            closed_count = 0
            
            for position in positions:
                if position.magic == self.magic_number:
                    if self.close_position(position.ticket, reason):
                        closed_count += 1
                        time.sleep(0.1)  # Pequeña pausa entre cierres
            
            log.info(f"✅ Cerradas {closed_count} posiciones")
            return closed_count
            
        except Exception as e:
            log.error(f"Error cerrando todas las posiciones: {str(e)}")
            return 0
    
    def get_account_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen de la cuenta.
        
        Returns:
            Dict[str, Any]: Información de la cuenta
        """
        if not self.connected:
            return {}
        
        try:
            # Actualizar información
            self.account_info = mt5.account_info()
            if self.account_info is None:
                return {}
            
            # Obtener posiciones
            positions = mt5.positions_get()
            open_positions = len([p for p in positions if p.magic == self.magic_number]) if positions else 0
            
            # Calcular P&L no realizado
            unrealized_pnl = sum([p.profit for p in positions if p.magic == self.magic_number]) if positions else 0
            
            return {
                'login': self.account_info.login,
                'server': self.account_info.server,
                'balance': self.account_info.balance,
                'equity': self.account_info.equity,
                'margin': self.account_info.margin,
                'free_margin': self.account_info.margin_free,
                'margin_level': self.account_info.margin_level,
                'currency': self.account_info.currency,
                'open_positions': open_positions,
                'unrealized_pnl': unrealized_pnl,
                'daily_risk_used': self.risk_manager.daily_risk_used,
                'daily_trades': self.risk_manager.daily_trades,
                'orders_sent': self.stats['orders_sent'],
                'orders_filled': self.stats['orders_filled'],
                'orders_rejected': self.stats['orders_rejected']
            }
            
        except Exception as e:
            log.error(f"Error obteniendo resumen de cuenta: {str(e)}")
            return {}
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Obtiene lista de posiciones abiertas.
        
        Returns:
            List[Dict[str, Any]]: Lista de posiciones
        """
        try:
            positions = mt5.positions_get()
            if not positions:
                return []
            
            result = []
            for position in positions:
                if position.magic == self.magic_number:
                    result.append({
                        'ticket': position.ticket,
                        'symbol': position.symbol,
                        'type': 'BUY' if position.type == mt5.POSITION_TYPE_BUY else 'SELL',
                        'volume': position.volume,
                        'open_price': position.price_open,
                        'current_price': position.price_current,
                        'sl': position.sl,
                        'tp': position.tp,
                        'profit': position.profit,
                        'open_time': datetime.fromtimestamp(position.time),
                        'comment': position.comment
                    })
            
            return result
            
        except Exception as e:
            log.error(f"Error obteniendo posiciones: {str(e)}")
            return []
    
    def emergency_stop(self) -> bool:
        """
        Parada de emergencia: cierra todas las posiciones.
        
        Returns:
            bool: True si se ejecutó correctamente
        """
        log.warning("🚨 PARADA DE EMERGENCIA ACTIVADA")
        
        try:
            # Cerrar todas las posiciones
            closed = self.close_all_positions("EMERGENCY_STOP")
            
            # Cancelar órdenes pendientes
            orders = mt5.orders_get()
            cancelled = 0
            
            if orders:
                for order in orders:
                    if order.magic == self.magic_number:
                        request = {
                            "action": mt5.TRADE_ACTION_REMOVE,
                            "order": order.ticket,
                        }
                        
                        result = mt5.order_send(request)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            cancelled += 1
            
            log.warning(f"🚨 Parada de emergencia completada:")
            log.warning(f"  Posiciones cerradas: {closed}")
            log.warning(f"  Órdenes canceladas: {cancelled}")
            
            return True
            
        except Exception as e:
            log.error(f"Error en parada de emergencia: {str(e)}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Funciones de utilidad

def create_trade_executor(risk_level: str = "moderate") -> TradeExecutor:
    """
    Crea un ejecutor de trading con configuración predeterminada.
    
    Args:
        risk_level: Nivel de riesgo ('conservative', 'moderate', 'aggressive')
        
    Returns:
        TradeExecutor: Instancia configurada
    """
    risk_levels = {
        'conservative': RiskLevel.CONSERVATIVE,
        'moderate': RiskLevel.MODERATE,
        'aggressive': RiskLevel.AGGRESSIVE
    }
    
    risk = risk_levels.get(risk_level.lower(), RiskLevel.MODERATE)
    
    return TradeExecutor(
        login=5036791117,
        password="BtUvF-X8",
        server="MetaQuotes-Demo",
        risk_level=risk
    )


def execute_signal_simple(signal_dict: Dict[str, Any]) -> bool:
    """
    Función simple para ejecutar una señal.
    
    Args:
        signal_dict: Diccionario con datos de la señal
        
    Returns:
        bool: True si se ejecutó exitosamente
    """
    try:
        # Crear señal
        signal = TradeSignal(
            symbol=signal_dict['symbol'],
            signal=signal_dict['signal'],
            confidence=signal_dict['confidence'],
            price=signal_dict['price'],
            probabilities=signal_dict.get('probabilities', {}),
            metadata=signal_dict.get('metadata', {})
        )
        
        # Ejecutar con executor
        with create_trade_executor() as executor:
            order = executor.execute_signal(signal)
            return order is not None
            
    except Exception as e:
        log.error(f"Error ejecutando señal simple: {str(e)}")
        return False 