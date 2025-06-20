#!/usr/bin/env python3
"""
Pruebas unitarias para el Trade Executor.

Valida que el trade_executor responde a señales sin lanzar excepciones
y que todas las funcionalidades de trading funcionan correctamente.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from src.trading.trade_executor import (
    TradeExecutor, TradeSignal, TradeOrder, RiskManager,
    OrderType, OrderStatus, RiskLevel,
    create_trade_executor, execute_signal_simple
)


class TestTradeSignal:
    """Pruebas para la clase TradeSignal."""
    
    def test_trade_signal_creation(self):
        """Prueba creación básica de señal de trading."""
        signal = TradeSignal(
            symbol="EURUSD",
            signal="buy",
            confidence=0.75,
            price=1.0850
        )
        
        assert signal.symbol == "EURUSD"
        assert signal.signal == "buy"
        assert signal.confidence == 0.75
        assert signal.price == 1.0850
        assert isinstance(signal.timestamp, datetime)
    
    def test_trade_signal_with_metadata(self):
        """Prueba creación de señal con metadatos."""
        probabilities = {'buy': 0.75, 'sell': 0.15, 'hold': 0.10}
        metadata = {'strategy': 'LIT_ML', 'timeframe': '1h'}
        
        signal = TradeSignal(
            symbol="GBPUSD",
            signal="sell",
            confidence=0.68,
            price=1.2650,
            probabilities=probabilities,
            metadata=metadata
        )
        
        assert signal.probabilities == probabilities
        assert signal.metadata == metadata
    
    def test_trade_signal_validation(self):
        """Prueba validación de señales inválidas."""
        # Señal inválida
        with pytest.raises(ValueError):
            TradeSignal("EURUSD", "invalid_signal", 0.75, 1.0850)
        
        # Confianza inválida
        with pytest.raises(ValueError):
            TradeSignal("EURUSD", "buy", 1.5, 1.0850)  # > 1
        
        with pytest.raises(ValueError):
            TradeSignal("EURUSD", "buy", -0.1, 1.0850)  # < 0
    
    def test_trade_signal_case_insensitive(self):
        """Prueba que las señales son case-insensitive."""
        signal = TradeSignal("EURUSD", "BUY", 0.75, 1.0850)
        assert signal.signal == "buy"
        
        signal = TradeSignal("EURUSD", "SELL", 0.75, 1.0850)
        assert signal.signal == "sell"


class TestTradeOrder:
    """Pruebas para la clase TradeOrder."""
    
    def test_trade_order_creation(self):
        """Prueba creación básica de orden."""
        order = TradeOrder(
            symbol="EURUSD",
            order_type=OrderType.BUY,
            volume=0.1,
            price=1.0850,
            stop_loss=1.0800,
            take_profit=1.0900
        )
        
        assert order.symbol == "EURUSD"
        assert order.order_type == OrderType.BUY
        assert order.volume == 0.1
        assert order.price == 1.0850
        assert order.stop_loss == 1.0800
        assert order.take_profit == 1.0900
        assert order.status == OrderStatus.PENDING
    
    def test_trade_order_timestamps(self):
        """Prueba que los timestamps se crean correctamente."""
        order = TradeOrder("EURUSD", OrderType.BUY, 0.1)
        
        assert isinstance(order.created_at, datetime)
        assert isinstance(order.updated_at, datetime)
        assert order.created_at <= order.updated_at


class TestRiskManager:
    """Pruebas para el gestor de riesgos."""
    
    @pytest.fixture
    def risk_manager(self):
        """Fixture que crea un gestor de riesgos."""
        return RiskManager(RiskLevel.MODERATE)
    
    @pytest.fixture
    def sample_signal(self):
        """Fixture que crea una señal de muestra."""
        return TradeSignal("EURUSD", "buy", 0.75, 1.0850)
    
    def test_risk_manager_initialization(self, risk_manager):
        """Prueba inicialización del gestor de riesgos."""
        assert risk_manager.risk_level == RiskLevel.MODERATE
        assert risk_manager.config['risk_per_trade'] == 0.02
        assert risk_manager.config['sl_points'] == 50
        assert risk_manager.config['tp_points'] == 100
        assert risk_manager.config['min_confidence'] == 0.65
    
    def test_risk_levels_configuration(self):
        """Prueba configuración de diferentes niveles de riesgo."""
        # Conservative
        conservative = RiskManager(RiskLevel.CONSERVATIVE)
        assert conservative.config['risk_per_trade'] == 0.01
        assert conservative.config['sl_points'] == 30
        assert conservative.config['min_confidence'] == 0.75
        
        # Aggressive
        aggressive = RiskManager(RiskLevel.AGGRESSIVE)
        assert aggressive.config['risk_per_trade'] == 0.03
        assert aggressive.config['sl_points'] == 80
        assert aggressive.config['min_confidence'] == 0.55
    
    def test_can_open_position_valid(self, risk_manager, sample_signal):
        """Prueba validación de posición válida."""
        can_open, reason = risk_manager.can_open_position(sample_signal, 10000)
        
        assert can_open is True
        assert reason == "OK"
    
    def test_can_open_position_low_confidence(self, risk_manager):
        """Prueba rechazo por confianza baja."""
        low_confidence_signal = TradeSignal("EURUSD", "buy", 0.50, 1.0850)
        can_open, reason = risk_manager.can_open_position(low_confidence_signal, 10000)
        
        assert can_open is False
        assert "Confianza insuficiente" in reason
    
    def test_can_open_position_hold_signal(self, risk_manager):
        """Prueba rechazo de señal HOLD."""
        hold_signal = TradeSignal("EURUSD", "hold", 0.80, 1.0850)
        can_open, reason = risk_manager.can_open_position(hold_signal, 10000)
        
        assert can_open is False
        assert "HOLD" in reason
    
    def test_can_open_position_max_positions(self, risk_manager, sample_signal):
        """Prueba límite de posiciones abiertas."""
        # Simular máximo de posiciones alcanzado
        risk_manager.open_positions_count = risk_manager.max_open_positions
        
        can_open, reason = risk_manager.can_open_position(sample_signal, 10000)
        
        assert can_open is False
        assert "Máximo de posiciones" in reason
    
    @patch('src.trading.trade_executor.mt5')
    def test_calculate_position_size(self, mock_mt5, risk_manager):
        """Prueba cálculo de tamaño de posición."""
        # Mock de información del símbolo
        mock_symbol_info = Mock()
        mock_symbol_info.trade_tick_value = 1.0
        mock_symbol_info.point = 0.00001
        mock_symbol_info.volume_min = 0.01
        mock_symbol_info.volume_max = 100.0
        mock_symbol_info.volume_step = 0.01
        mock_symbol_info.trade_contract_size = 100000
        
        mock_mt5.symbol_info.return_value = mock_symbol_info
        
        position_size = risk_manager.calculate_position_size("EURUSD", 10000)
        
        assert isinstance(position_size, float)
        assert position_size >= mock_symbol_info.volume_min
        assert position_size <= mock_symbol_info.volume_max
    
    @patch('src.trading.trade_executor.mt5')
    def test_get_sl_tp_levels(self, mock_mt5, risk_manager):
        """Prueba cálculo de niveles SL/TP."""
        # Mock de información del símbolo
        mock_symbol_info = Mock()
        mock_symbol_info.point = 0.00001
        
        mock_mt5.symbol_info.return_value = mock_symbol_info
        
        entry_price = 1.0850
        sl, tp = risk_manager.get_sl_tp_levels("EURUSD", OrderType.BUY, entry_price)
        
        assert sl is not None
        assert tp is not None
        assert sl < entry_price  # SL debe estar por debajo para BUY
        assert tp > entry_price  # TP debe estar por encima para BUY
    
    def test_position_tracking(self, risk_manager):
        """Prueba seguimiento de posiciones."""
        initial_count = risk_manager.open_positions_count
        
        # Simular apertura de posición
        risk_manager.update_position_opened(100)
        assert risk_manager.open_positions_count == initial_count + 1
        
        # Simular cierre de posición
        risk_manager.update_position_closed()
        assert risk_manager.open_positions_count == initial_count


class TestTradeExecutor:
    """Pruebas para el ejecutor de trading."""
    
    @pytest.fixture
    def mock_executor(self):
        """Fixture que crea un ejecutor mock."""
        with patch('src.trading.trade_executor.mt5') as mock_mt5:
            # Configurar mocks básicos
            mock_mt5.initialize.return_value = True
            mock_mt5.login.return_value = True
            
            # Mock de información de cuenta
            mock_account = Mock()
            mock_account.login = 12345
            mock_account.server = "Test-Server"
            mock_account.balance = 10000.0
            mock_account.equity = 10000.0
            mock_account.margin = 0.0
            mock_account.margin_free = 10000.0
            mock_account.currency = "USD"
            
            mock_mt5.account_info.return_value = mock_account
            
            executor = TradeExecutor(
                login=12345,
                password="test",
                server="Test-Server",
                risk_level=RiskLevel.MODERATE
            )
            
            return executor, mock_mt5
    
    def test_executor_initialization(self):
        """Prueba inicialización del ejecutor."""
        executor = TradeExecutor(
            login=12345,
            password="test",
            server="Test-Server"
        )
        
        assert executor.login == 12345
        assert executor.password == "test"
        assert executor.server == "Test-Server"
        assert executor.connected is False
        assert isinstance(executor.risk_manager, RiskManager)
    
    def test_executor_connect_success(self, mock_executor):
        """Prueba conexión exitosa."""
        executor, mock_mt5 = mock_executor
        
        success = executor.connect()
        
        assert success is True
        assert executor.connected is True
        assert mock_mt5.initialize.called
        assert mock_mt5.login.called
    
    def test_executor_connect_failure(self):
        """Prueba fallo de conexión."""
        with patch('src.trading.trade_executor.mt5') as mock_mt5:
            mock_mt5.initialize.return_value = False
            
            executor = TradeExecutor(12345, "test", "Test-Server")
            success = executor.connect()
            
            assert success is False
            assert executor.connected is False
    
    def test_executor_disconnect(self, mock_executor):
        """Prueba desconexión."""
        executor, mock_mt5 = mock_executor
        
        executor.connected = True
        executor.disconnect()
        
        assert executor.connected is False
        assert mock_mt5.shutdown.called
    
    def test_execute_signal_without_connection(self):
        """Prueba ejecución de señal sin conexión."""
        executor = TradeExecutor(12345, "test", "Test-Server")
        signal = TradeSignal("EURUSD", "buy", 0.75, 1.0850)
        
        result = executor.execute_signal(signal)
        
        assert result is None
    
    def test_execute_signal_hold(self, mock_executor):
        """Prueba ejecución de señal HOLD."""
        executor, mock_mt5 = mock_executor
        executor.connected = True
        
        hold_signal = TradeSignal("EURUSD", "hold", 0.80, 1.0850)
        result = executor.execute_signal(hold_signal)
        
        assert result is None
    
    def test_execute_signal_low_confidence(self, mock_executor):
        """Prueba ejecución de señal con confianza baja."""
        executor, mock_mt5 = mock_executor
        executor.connected = True
        
        low_conf_signal = TradeSignal("EURUSD", "buy", 0.50, 1.0850)
        result = executor.execute_signal(low_conf_signal)
        
        assert result is None
    
    @patch('src.trading.trade_executor.mt5')
    def test_execute_signal_success_mock(self, mock_mt5):
        """Prueba ejecución exitosa de señal con mocks completos."""
        # Configurar mocks
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True
        
        # Mock de cuenta
        mock_account = Mock()
        mock_account.balance = 10000.0
        mock_mt5.account_info.return_value = mock_account
        
        # Mock de símbolo
        mock_symbol_info = Mock()
        mock_symbol_info.trade_tick_value = 1.0
        mock_symbol_info.point = 0.00001
        mock_symbol_info.volume_min = 0.01
        mock_symbol_info.volume_max = 100.0
        mock_symbol_info.volume_step = 0.01
        mock_symbol_info.trade_contract_size = 100000
        mock_mt5.symbol_info.return_value = mock_symbol_info
        
        # Mock de tick
        mock_tick = Mock()
        mock_tick.ask = 1.0855
        mock_tick.bid = 1.0850
        mock_mt5.symbol_info_tick.return_value = mock_tick
        
        # Mock de resultado de orden
        mock_result = Mock()
        mock_result.retcode = 10009  # TRADE_RETCODE_DONE
        mock_result.order = 12345
        mock_result.price = 1.0855
        mock_mt5.order_send.return_value = mock_result
        mock_mt5.TRADE_RETCODE_DONE = 10009
        
        # Crear ejecutor y conectar
        executor = TradeExecutor(12345, "test", "Test-Server")
        executor.connect()
        
        # Ejecutar señal
        signal = TradeSignal("EURUSD", "buy", 0.75, 1.0850)
        result = executor.execute_signal(signal)
        
        # Verificar resultado
        assert result is not None
        assert isinstance(result, TradeOrder)
        assert result.status == OrderStatus.FILLED
        assert result.ticket == 12345
    
    def test_get_account_summary(self, mock_executor):
        """Prueba obtención de resumen de cuenta."""
        executor, mock_mt5 = mock_executor
        executor.connected = True
        
        # Mock de posiciones
        mock_mt5.positions_get.return_value = []
        
        summary = executor.get_account_summary()
        
        assert isinstance(summary, dict)
        assert 'login' in summary
        assert 'balance' in summary
        assert 'equity' in summary
        assert 'open_positions' in summary
    
    def test_get_account_summary_disconnected(self):
        """Prueba resumen de cuenta sin conexión."""
        executor = TradeExecutor(12345, "test", "Test-Server")
        
        summary = executor.get_account_summary()
        
        assert summary == {}
    
    def test_context_manager(self, mock_executor):
        """Prueba uso como context manager."""
        executor, mock_mt5 = mock_executor
        
        with executor as ctx_executor:
            assert ctx_executor is executor
            assert executor.connected is True
        
        assert executor.connected is False


class TestUtilityFunctions:
    """Pruebas para funciones de utilidad."""
    
    @patch('src.trading.trade_executor.TradeExecutor')
    def test_create_trade_executor(self, mock_executor_class):
        """Prueba creación de ejecutor con función de utilidad."""
        mock_instance = Mock()
        mock_executor_class.return_value = mock_instance
        
        executor = create_trade_executor("moderate")
        
        assert executor is mock_instance
        mock_executor_class.assert_called_once()
    
    @patch('src.trading.trade_executor.create_trade_executor')
    def test_execute_signal_simple(self, mock_create_executor):
        """Prueba función de ejecución simple."""
        # Mock del ejecutor
        mock_executor = Mock()
        mock_executor.__enter__ = Mock(return_value=mock_executor)
        mock_executor.__exit__ = Mock(return_value=None)
        mock_executor.execute_signal.return_value = Mock()  # Orden exitosa
        
        mock_create_executor.return_value = mock_executor
        
        signal_data = {
            'symbol': 'EURUSD',
            'signal': 'buy',
            'confidence': 0.75,
            'price': 1.0850
        }
        
        result = execute_signal_simple(signal_data)
        
        assert result is True
        mock_executor.execute_signal.assert_called_once()
    
    @patch('src.trading.trade_executor.create_trade_executor')
    def test_execute_signal_simple_failure(self, mock_create_executor):
        """Prueba función de ejecución simple con fallo."""
        # Mock del ejecutor que falla
        mock_executor = Mock()
        mock_executor.__enter__ = Mock(return_value=mock_executor)
        mock_executor.__exit__ = Mock(return_value=None)
        mock_executor.execute_signal.return_value = None  # Orden fallida
        
        mock_create_executor.return_value = mock_executor
        
        signal_data = {
            'symbol': 'EURUSD',
            'signal': 'buy',
            'confidence': 0.75,
            'price': 1.0850
        }
        
        result = execute_signal_simple(signal_data)
        
        assert result is False


class TestErrorHandling:
    """Pruebas para manejo de errores."""
    
    def test_invalid_signal_data(self):
        """Prueba manejo de datos de señal inválidos."""
        with pytest.raises(ValueError):
            TradeSignal("", "buy", 0.75, 1.0850)  # Símbolo vacío
    
    @patch('src.trading.trade_executor.mt5')
    def test_mt5_not_available(self, mock_mt5):
        """Prueba manejo cuando MT5 no está disponible."""
        mock_mt5.initialize.return_value = False
        
        executor = TradeExecutor(12345, "test", "Test-Server")
        success = executor.connect()
        
        assert success is False
    
    def test_execute_signal_exception_handling(self, mock_executor):
        """Prueba manejo de excepciones en ejecución de señales."""
        executor, mock_mt5 = mock_executor
        executor.connected = True
        
        # Simular excepción en account_info
        mock_mt5.account_info.side_effect = Exception("Test exception")
        
        signal = TradeSignal("EURUSD", "buy", 0.75, 1.0850)
        result = executor.execute_signal(signal)
        
        # Debe manejar la excepción graciosamente
        assert result is None
    
    def test_risk_manager_with_invalid_data(self):
        """Prueba gestor de riesgos con datos inválidos."""
        risk_manager = RiskManager(RiskLevel.MODERATE)
        
        # Señal con datos inválidos
        invalid_signal = TradeSignal("EURUSD", "buy", 0.75, 1.0850)
        
        # No debe lanzar excepciones
        can_open, reason = risk_manager.can_open_position(invalid_signal, 0)  # Balance 0
        
        assert isinstance(can_open, bool)
        assert isinstance(reason, str)


class TestIntegration:
    """Pruebas de integración."""
    
    @patch('src.trading.trade_executor.mt5')
    def test_full_signal_execution_flow(self, mock_mt5):
        """Prueba flujo completo de ejecución de señal."""
        # Configurar todos los mocks necesarios
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True
        
        mock_account = Mock()
        mock_account.balance = 10000.0
        mock_mt5.account_info.return_value = mock_account
        
        mock_symbol_info = Mock()
        mock_symbol_info.trade_tick_value = 1.0
        mock_symbol_info.point = 0.00001
        mock_symbol_info.volume_min = 0.01
        mock_symbol_info.volume_max = 100.0
        mock_symbol_info.volume_step = 0.01
        mock_symbol_info.trade_contract_size = 100000
        mock_mt5.symbol_info.return_value = mock_symbol_info
        
        mock_tick = Mock()
        mock_tick.ask = 1.0855
        mock_tick.bid = 1.0850
        mock_mt5.symbol_info_tick.return_value = mock_tick
        
        mock_result = Mock()
        mock_result.retcode = 10009
        mock_result.order = 12345
        mock_result.price = 1.0855
        mock_mt5.order_send.return_value = mock_result
        mock_mt5.TRADE_RETCODE_DONE = 10009
        
        # Crear y usar ejecutor
        with create_trade_executor("moderate") as executor:
            signal = TradeSignal("EURUSD", "buy", 0.75, 1.0850)
            result = executor.execute_signal(signal)
            
            # Verificar que el flujo completo funciona
            assert result is not None
            assert result.status == OrderStatus.FILLED
    
    def test_multiple_signals_processing(self, mock_executor):
        """Prueba procesamiento de múltiples señales."""
        executor, mock_mt5 = mock_executor
        executor.connected = True
        
        signals = [
            TradeSignal("EURUSD", "buy", 0.75, 1.0850),
            TradeSignal("GBPUSD", "sell", 0.70, 1.2650),
            TradeSignal("USDJPY", "hold", 0.60, 149.50)  # Esta no se ejecutará
        ]
        
        results = []
        for signal in signals:
            result = executor.execute_signal(signal)
            results.append(result)
        
        # Verificar que se procesaron todas las señales
        assert len(results) == 3
        # Las dos primeras deberían ser None (sin mocks completos)
        # La tercera debería ser None (señal HOLD)
        assert all(result is None for result in results)


if __name__ == "__main__":
    # Ejecutar pruebas si se ejecuta directamente
    pytest.main([__file__, "-v"]) 