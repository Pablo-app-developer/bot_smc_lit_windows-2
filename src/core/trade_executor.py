"""
Módulo de ejecución de trades y control de riesgos.

Este módulo maneja la ejecución de órdenes de trading, control de riesgos,
gestión de posiciones y cálculo de métricas de rendimiento.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from src.core.config import config
from src.utils.logger import log, trading_logger
from src.utils.helpers import calculate_position_size, get_current_timestamp


class OrderType(Enum):
    """Tipos de órdenes."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Estados de órdenes."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionStatus(Enum):
    """Estados de posiciones."""
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class Order:
    """Representa una orden de trading."""
    
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_type: OrderType
    status: OrderStatus
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    fill_price: Optional[float] = None
    fill_timestamp: Optional[datetime] = None


@dataclass
class Position:
    """Representa una posición de trading."""
    
    id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    entry_timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    pnl: float = 0.0
    strategy: str = "LIT_ML"


class TradeExecutor:
    """
    Ejecutor de trades con control de riesgos.
    
    Maneja la ejecución de órdenes, gestión de posiciones,
    control de riesgos y cálculo de métricas de rendimiento.
    """
    
    def __init__(self):
        """Inicializa el ejecutor de trades."""
        self.balance = config.trading.balance_inicial
        self.initial_balance = config.trading.balance_inicial
        self.risk_per_trade = config.trading.risk_per_trade
        self.max_positions = config.trading.max_positions
        self.leverage = config.trading.leverage
        
        # Almacenamiento de órdenes y posiciones
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
        # Métricas de rendimiento
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = self.balance
        
        log.info(f"TradeExecutor inicializado - Balance: {self.balance}")
    
    def execute_signal(self, 
                      signal: Dict[str, Any], 
                      current_price: float,
                      symbol: str = None) -> Optional[str]:
        """
        Ejecuta una señal de trading.
        
        Args:
            signal: Diccionario con la señal de trading.
            current_price: Precio actual del mercado.
            symbol: Símbolo del instrumento.
            
        Returns:
            Optional[str]: ID de la orden ejecutada o None si no se ejecuta.
        """
        symbol = symbol or config.trading.symbol
        
        # Validar señal
        if not self._validate_signal(signal):
            return None
        
        # Verificar control de riesgos
        if not self._check_risk_management(signal, symbol):
            return None
        
        # Ejecutar según el tipo de señal
        if signal['signal'] == 'buy':
            return self._execute_buy_order(signal, current_price, symbol)
        elif signal['signal'] == 'sell':
            return self._execute_sell_order(signal, current_price, symbol)
        else:
            # Hold - verificar posiciones existentes
            self._check_existing_positions(current_price)
            return None
    
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valida una señal antes de ejecutarla.
        
        Args:
            signal: Señal a validar.
            
        Returns:
            bool: True si la señal es válida.
        """
        required_fields = ['signal', 'confidence', 'entry_price']
        
        for field in required_fields:
            if field not in signal:
                log.warning(f"Señal inválida: falta campo '{field}'")
                return False
        
        if signal['confidence'] < 0.3:
            log.info("Señal rechazada: confianza muy baja")
            return False
        
        return True
    
    def _check_risk_management(self, signal: Dict[str, Any], symbol: str) -> bool:
        """
        Verifica las reglas de gestión de riesgos.
        
        Args:
            signal: Señal a evaluar.
            symbol: Símbolo del instrumento.
            
        Returns:
            bool: True si pasa los controles de riesgo.
        """
        # Verificar número máximo de posiciones
        open_positions = [p for p in self.positions.values() if p.status == PositionStatus.OPEN]
        if len(open_positions) >= self.max_positions:
            log.info(f"Máximo de posiciones alcanzado: {len(open_positions)}/{self.max_positions}")
            return False
        
        # Verificar balance mínimo
        min_balance = self.initial_balance * 0.2  # 20% del balance inicial
        if self.balance < min_balance:
            log.warning(f"Balance muy bajo: {self.balance:.2f} < {min_balance:.2f}")
            return False
        
        # Verificar riesgo por trade
        entry_price = signal['entry_price']
        stop_loss = signal.get('stop_loss')
        
        if stop_loss:
            position_size = calculate_position_size(
                self.balance, self.risk_per_trade, entry_price, stop_loss
            )
            
            if position_size == 0:
                log.warning("Tamaño de posición calculado es 0")
                return False
        
        return True
    
    def _execute_buy_order(self, 
                          signal: Dict[str, Any], 
                          current_price: float, 
                          symbol: str) -> Optional[str]:
        """
        Ejecuta una orden de compra.
        
        Args:
            signal: Señal de compra.
            current_price: Precio actual.
            symbol: Símbolo del instrumento.
            
        Returns:
            Optional[str]: ID de la orden.
        """
        entry_price = signal['entry_price']
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        
        # Calcular tamaño de posición
        position_size = calculate_position_size(
            self.balance, self.risk_per_trade, entry_price, stop_loss
        ) if stop_loss else self.balance * self.risk_per_trade / entry_price
        
        # Crear orden
        order_id = self._generate_order_id()
        order = Order(
            id=order_id,
            symbol=symbol,
            side='buy',
            quantity=position_size,
            price=current_price,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            timestamp=get_current_timestamp(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Simular ejecución inmediata (en producción sería asíncrona)
        order.status = OrderStatus.FILLED
        order.fill_price = current_price
        order.fill_timestamp = get_current_timestamp()
        
        self.orders[order_id] = order
        
        # Crear posición
        position_id = self._generate_position_id()
        position = Position(
            id=position_id,
            symbol=symbol,
            side='buy',
            quantity=position_size,
            entry_price=current_price,
            entry_timestamp=get_current_timestamp(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[position_id] = position
        
        # Actualizar balance
        cost = position_size * current_price / self.leverage
        self.balance -= cost
        
        # Log del trade
        trading_logger.log_trade(
            action="BUY",
            symbol=symbol,
            price=current_price,
            quantity=position_size,
            signal_type="LIT_ML"
        )
        
        log.info(f"Orden de compra ejecutada: {symbol} @ {current_price:.5f} | Cantidad: {position_size:.4f}")
        
        return order_id
    
    def _execute_sell_order(self, 
                           signal: Dict[str, Any], 
                           current_price: float, 
                           symbol: str) -> Optional[str]:
        """
        Ejecuta una orden de venta.
        
        Args:
            signal: Señal de venta.
            current_price: Precio actual.
            symbol: Símbolo del instrumento.
            
        Returns:
            Optional[str]: ID de la orden.
        """
        entry_price = signal['entry_price']
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        
        # Calcular tamaño de posición
        position_size = calculate_position_size(
            self.balance, self.risk_per_trade, entry_price, stop_loss
        ) if stop_loss else self.balance * self.risk_per_trade / entry_price
        
        # Crear orden
        order_id = self._generate_order_id()
        order = Order(
            id=order_id,
            symbol=symbol,
            side='sell',
            quantity=position_size,
            price=current_price,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            timestamp=get_current_timestamp(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Simular ejecución inmediata
        order.status = OrderStatus.FILLED
        order.fill_price = current_price
        order.fill_timestamp = get_current_timestamp()
        
        self.orders[order_id] = order
        
        # Crear posición
        position_id = self._generate_position_id()
        position = Position(
            id=position_id,
            symbol=symbol,
            side='sell',
            quantity=position_size,
            entry_price=current_price,
            entry_timestamp=get_current_timestamp(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[position_id] = position
        
        # Actualizar balance
        cost = position_size * current_price / self.leverage
        self.balance -= cost
        
        # Log del trade
        trading_logger.log_trade(
            action="SELL",
            symbol=symbol,
            price=current_price,
            quantity=position_size,
            signal_type="LIT_ML"
        )
        
        log.info(f"Orden de venta ejecutada: {symbol} @ {current_price:.5f} | Cantidad: {position_size:.4f}")
        
        return order_id
    
    def _check_existing_positions(self, current_price: float) -> None:
        """
        Verifica posiciones existentes para stop loss y take profit.
        
        Args:
            current_price: Precio actual del mercado.
        """
        open_positions = [p for p in self.positions.values() if p.status == PositionStatus.OPEN]
        
        for position in open_positions:
            should_close, reason = self._should_close_position(position, current_price)
            
            if should_close:
                self._close_position(position, current_price, reason)
    
    def _should_close_position(self, position: Position, current_price: float) -> Tuple[bool, str]:
        """
        Determina si una posición debe cerrarse.
        
        Args:
            position: Posición a evaluar.
            current_price: Precio actual.
            
        Returns:
            Tuple[bool, str]: (Debe cerrar, Razón).
        """
        if position.side == 'buy':
            # Posición larga
            if position.stop_loss and current_price <= position.stop_loss:
                return True, "stop_loss"
            if position.take_profit and current_price >= position.take_profit:
                return True, "take_profit"
        else:
            # Posición corta
            if position.stop_loss and current_price >= position.stop_loss:
                return True, "stop_loss"
            if position.take_profit and current_price <= position.take_profit:
                return True, "take_profit"
        
        return False, ""
    
    def _close_position(self, position: Position, exit_price: float, reason: str) -> None:
        """
        Cierra una posición.
        
        Args:
            position: Posición a cerrar.
            exit_price: Precio de salida.
            reason: Razón del cierre.
        """
        # Calcular PnL
        if position.side == 'buy':
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity
        
        # Actualizar posición
        position.status = PositionStatus.CLOSED
        position.exit_price = exit_price
        position.exit_timestamp = get_current_timestamp()
        position.pnl = pnl
        
        # Actualizar balance
        position_value = position.quantity * exit_price / self.leverage
        self.balance += position_value + pnl
        
        # Actualizar métricas
        self._update_metrics(position)
        
        # Agregar al historial
        trade_record = {
            'position_id': position.id,
            'symbol': position.symbol,
            'side': position.side,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'entry_time': position.entry_timestamp,
            'exit_time': position.exit_timestamp,
            'pnl': pnl,
            'reason': reason,
            'strategy': position.strategy
        }
        
        self.trade_history.append(trade_record)
        
        # Log del cierre
        trading_logger.log_trade(
            action="CLOSE",
            symbol=position.symbol,
            price=exit_price,
            quantity=position.quantity,
            signal_type=f"LIT_ML_{reason.upper()}"
        )
        
        log.info(f"Posición cerrada: {position.symbol} | PnL: {pnl:.2f} | Razón: {reason}")
    
    def _update_metrics(self, position: Position) -> None:
        """
        Actualiza las métricas de rendimiento.
        
        Args:
            position: Posición cerrada.
        """
        self.total_trades += 1
        self.total_pnl += position.pnl
        
        if position.pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Actualizar peak y drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def _generate_order_id(self) -> str:
        """Genera un ID único para órdenes."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"ORD_{timestamp}"
    
    def _generate_position_id(self) -> str:
        """Genera un ID único para posiciones."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"POS_{timestamp}"
    
    def get_open_positions(self) -> List[Position]:
        """
        Obtiene las posiciones abiertas.
        
        Returns:
            List[Position]: Lista de posiciones abiertas.
        """
        return [p for p in self.positions.values() if p.status == PositionStatus.OPEN]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas de rendimiento.
        
        Returns:
            Dict[str, Any]: Métricas de rendimiento.
        """
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'return_percentage': 0.0,
                'max_drawdown': 0.0,
                'balance': self.balance,
                'open_positions': len(self.get_open_positions())
            }
        
        win_rate = self.winning_trades / self.total_trades * 100
        return_percentage = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'return_percentage': return_percentage,
            'max_drawdown': self.max_drawdown * 100,
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'open_positions': len(self.get_open_positions()),
            'peak_balance': self.peak_balance
        }
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """
        Retorna el historial de trades como DataFrame.
        
        Returns:
            pd.DataFrame: Historial de trades.
        """
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)
    
    def close_all_positions(self, current_price: float, reason: str = "manual") -> int:
        """
        Cierra todas las posiciones abiertas.
        
        Args:
            current_price: Precio actual del mercado.
            reason: Razón del cierre.
            
        Returns:
            int: Número de posiciones cerradas.
        """
        open_positions = self.get_open_positions()
        
        for position in open_positions:
            self._close_position(position, current_price, reason)
        
        log.info(f"Cerradas {len(open_positions)} posiciones por: {reason}")
        
        return len(open_positions)
    
    def reset(self) -> None:
        """Reinicia el ejecutor a su estado inicial."""
        self.balance = self.initial_balance
        self.orders.clear()
        self.positions.clear()
        self.trade_history.clear()
        
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = self.balance
        
        log.info("TradeExecutor reiniciado")