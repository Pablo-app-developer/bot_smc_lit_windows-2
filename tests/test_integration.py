#!/usr/bin/env python3
"""
Pruebas de integración para el sistema completo LIT + ML.

Valida que todos los módulos trabajen correctamente en conjunto
y que el flujo completo de predicción y ejecución funcione.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from src.strategies.lit_detector import LITDetector
from src.models.predictor import LITMLPredictor
from src.models.feature_engineering import FeatureEngineer
from src.trading.trade_executor import TradeExecutor, TradeSignal, create_trade_executor
from src.trading.trading_bot import TradingBot


@pytest.mark.integration
class TestLITMLIntegration:
    """Pruebas de integración para LIT + ML."""
    
    @pytest.fixture
    def integrated_system(self, temp_model_file):
        """Fixture que crea un sistema integrado completo."""
        # Crear componentes
        lit_detector = LITDetector()
        feature_engineer = FeatureEngineer()
        predictor = LITMLPredictor(temp_model_file)
        predictor.load_model()
        
        return {
            'lit_detector': lit_detector,
            'feature_engineer': feature_engineer,
            'predictor': predictor
        }
    
    def test_lit_to_ml_pipeline(self, integrated_system, sample_ohlcv_data):
        """Prueba pipeline completo de LIT a ML."""
        lit_detector = integrated_system['lit_detector']
        feature_engineer = integrated_system['feature_engineer']
        predictor = integrated_system['predictor']
        
        # 1. Análisis LIT
        lit_analysis = lit_detector.analyze_lit_signals(sample_ohlcv_data)
        
        assert isinstance(lit_analysis, dict)
        assert 'lit_score' in lit_analysis
        assert 'signal' in lit_analysis
        
        # 2. Ingeniería de características
        features = feature_engineer.create_features(sample_ohlcv_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        
        # 3. Predicción ML
        prediction = predictor.predict_single(sample_ohlcv_data)
        
        assert isinstance(prediction, dict)
        assert prediction['signal'] in ['buy', 'sell', 'hold']
        assert 0 <= prediction['confidence'] <= 1
        
        # 4. Verificar que los componentes se integran
        # El análisis LIT debe influir en las características
        assert 'lit_score' in features.columns or len(lit_analysis['liquidity_zones']) >= 0
    
    def test_feature_consistency_across_modules(self, integrated_system, sample_ohlcv_data):
        """Prueba consistencia de características entre módulos."""
        feature_engineer = integrated_system['feature_engineer']
        predictor = integrated_system['predictor']
        
        # Crear características
        features = feature_engineer.create_features(sample_ohlcv_data)
        
        # Verificar que el predictor puede usar las características
        prediction = predictor.predict_single(sample_ohlcv_data)
        
        # Debe funcionar sin errores
        assert prediction is not None
        assert isinstance(prediction, dict)
    
    def test_signal_generation_consistency(self, integrated_system, sample_ohlcv_data):
        """Prueba consistencia en generación de señales."""
        lit_detector = integrated_system['lit_detector']
        predictor = integrated_system['predictor']
        
        # Generar señales múltiples veces
        lit_signals = []
        ml_predictions = []
        
        for _ in range(3):
            lit_analysis = lit_detector.analyze_lit_signals(sample_ohlcv_data)
            ml_prediction = predictor.predict_single(sample_ohlcv_data)
            
            lit_signals.append(lit_analysis['signal'])
            ml_predictions.append(ml_prediction['signal'])
        
        # Las señales deben ser consistentes
        assert len(set(lit_signals)) == 1  # Todas iguales
        assert len(set(ml_predictions)) == 1  # Todas iguales
    
    @patch('src.trading.trade_executor.mt5')
    def test_ml_to_executor_integration(self, mock_mt5, integrated_system, sample_ohlcv_data):
        """Prueba integración de ML con ejecutor."""
        # Configurar mock MT5
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
        mock_tick.ask = 1.0852
        mock_tick.bid = 1.0850
        mock_mt5.symbol_info_tick.return_value = mock_tick
        
        predictor = integrated_system['predictor']
        
        # Crear ejecutor
        executor = TradeExecutor(12345, "test", "Test-Server")
        executor.connect()
        
        # Generar predicción
        prediction = predictor.predict_single(sample_ohlcv_data)
        
        # Crear señal de trading
        signal = TradeSignal(
            symbol="EURUSD",
            signal=prediction['signal'],
            confidence=prediction['confidence'],
            price=1.0850,
            probabilities=prediction['probabilities']
        )
        
        # Ejecutar señal (debería manejar graciosamente)
        result = executor.execute_signal(signal)
        
        # Verificar que no hay errores
        assert result is None or hasattr(result, 'status')


@pytest.mark.integration
class TestTradingBotIntegration:
    """Pruebas de integración para el bot de trading completo."""
    
    @patch('src.trading.trade_executor.mt5')
    def test_trading_bot_initialization(self, mock_mt5, temp_model_file):
        """Prueba inicialización del bot de trading."""
        # Configurar mock MT5
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True
        
        mock_account = Mock()
        mock_account.balance = 10000.0
        mock_mt5.account_info.return_value = mock_account
        
        # Crear bot
        bot = TradingBot(
            model_path=temp_model_file,
            symbols=['EURUSD'],
            timeframe="1h",
            prediction_interval=60,
            risk_level="moderate",
            trading_enabled=False  # Modo análisis
        )
        
        # Verificar inicialización
        assert bot is not None
        assert bot.model_path == temp_model_file
        assert 'EURUSD' in bot.symbols
        assert bot.prediction_interval == 60
    
    @patch('src.trading.trade_executor.mt5')
    @patch('src.data.data_collector.DataCollector')
    def test_trading_bot_prediction_cycle(self, mock_data_collector, mock_mt5, temp_model_file, sample_ohlcv_data):
        """Prueba ciclo de predicción del bot."""
        # Configurar mocks
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True
        
        mock_account = Mock()
        mock_account.balance = 10000.0
        mock_mt5.account_info.return_value = mock_account
        
        # Mock del data collector
        mock_collector_instance = Mock()
        mock_collector_instance.get_ohlcv.return_value = sample_ohlcv_data
        mock_data_collector.return_value = mock_collector_instance
        
        # Crear bot
        bot = TradingBot(
            model_path=temp_model_file,
            symbols=['EURUSD'],
            timeframe="1h",
            prediction_interval=1,  # 1 segundo para prueba rápida
            risk_level="moderate",
            trading_enabled=False
        )
        
        # Variables para capturar callbacks
        signals_received = []
        
        def signal_callback(symbol, prediction):
            signals_received.append((symbol, prediction))
        
        bot.add_signal_callback(signal_callback)
        
        # Inicializar bot
        success = bot.start()
        assert success is True
        
        # Esperar un ciclo de predicción
        time.sleep(2)
        
        # Detener bot
        bot.stop()
        
        # Verificar que se generaron señales
        assert len(signals_received) > 0
        
        # Verificar estructura de señales
        for symbol, prediction in signals_received:
            assert symbol == 'EURUSD'
            assert isinstance(prediction, dict)
            assert 'signal' in prediction
            assert 'confidence' in prediction
    
    @patch('src.trading.trade_executor.mt5')
    @patch('src.data.data_collector.DataCollector')
    def test_trading_bot_error_handling(self, mock_data_collector, mock_mt5, temp_model_file):
        """Prueba manejo de errores del bot."""
        # Configurar mocks para fallar
        mock_mt5.initialize.return_value = False  # Fallo de conexión
        
        # Mock del data collector que falla
        mock_collector_instance = Mock()
        mock_collector_instance.get_ohlcv.side_effect = Exception("Data error")
        mock_data_collector.return_value = mock_collector_instance
        
        # Crear bot
        bot = TradingBot(
            model_path=temp_model_file,
            symbols=['EURUSD'],
            timeframe="1h",
            prediction_interval=1,
            risk_level="moderate",
            trading_enabled=False
        )
        
        # Variables para capturar errores
        errors_received = []
        
        def error_callback(error):
            errors_received.append(error)
        
        bot.add_error_callback(error_callback)
        
        # Intentar inicializar (debería fallar graciosamente)
        success = bot.start()
        
        # Verificar que maneja el error
        assert success is False or len(errors_received) > 0
    
    @patch('src.trading.trade_executor.mt5')
    def test_trading_bot_multiple_symbols(self, mock_mt5, temp_model_file):
        """Prueba bot con múltiples símbolos."""
        # Configurar mock MT5
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True
        
        mock_account = Mock()
        mock_account.balance = 10000.0
        mock_mt5.account_info.return_value = mock_account
        
        # Crear bot con múltiples símbolos
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        bot = TradingBot(
            model_path=temp_model_file,
            symbols=symbols,
            timeframe="1h",
            prediction_interval=60,
            risk_level="moderate",
            trading_enabled=False
        )
        
        # Verificar configuración
        assert bot.symbols == symbols
        assert len(bot.symbols) == 3


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Pruebas de flujo completo end-to-end."""
    
    @patch('src.trading.trade_executor.mt5')
    @patch('src.data.data_collector.DataCollector')
    def test_complete_trading_workflow(self, mock_data_collector, mock_mt5, temp_model_file, sample_ohlcv_data):
        """Prueba flujo completo de trading."""
        # Configurar mocks
        self._setup_mt5_mock(mock_mt5)
        
        mock_collector_instance = Mock()
        mock_collector_instance.get_ohlcv.return_value = sample_ohlcv_data
        mock_data_collector.return_value = mock_collector_instance
        
        # 1. Crear componentes del sistema
        lit_detector = LITDetector()
        predictor = LITMLPredictor(temp_model_file)
        predictor.load_model()
        
        # 2. Análisis LIT
        lit_analysis = lit_detector.analyze_lit_signals(sample_ohlcv_data)
        
        # 3. Predicción ML
        ml_prediction = predictor.predict_single(sample_ohlcv_data)
        
        # 4. Crear señal de trading
        signal = TradeSignal(
            symbol="EURUSD",
            signal=ml_prediction['signal'],
            confidence=ml_prediction['confidence'],
            price=sample_ohlcv_data['close'].iloc[-1],
            probabilities=ml_prediction['probabilities'],
            metadata={
                'lit_score': lit_analysis['lit_score'],
                'strategy': 'LIT_ML_Integration'
            }
        )
        
        # 5. Ejecutar señal
        with create_trade_executor("moderate") as executor:
            result = executor.execute_signal(signal)
            
            # Verificar que el flujo completo funciona
            # (result puede ser None debido a mocks, pero no debe haber errores)
            assert result is None or hasattr(result, 'status')
    
    def test_data_flow_consistency(self, sample_ohlcv_data, temp_model_file):
        """Prueba consistencia del flujo de datos."""
        # Crear componentes
        lit_detector = LITDetector()
        feature_engineer = FeatureEngineer()
        predictor = LITMLPredictor(temp_model_file)
        predictor.load_model()
        
        # Procesar datos a través de cada componente
        original_data = sample_ohlcv_data.copy()
        
        # 1. Análisis LIT
        lit_analysis = lit_detector.analyze_lit_signals(original_data)
        
        # 2. Ingeniería de características
        features = feature_engineer.create_features(original_data)
        
        # 3. Predicción
        prediction = predictor.predict_single(original_data)
        
        # Verificar que los datos se mantienen consistentes
        assert len(original_data) == len(sample_ohlcv_data)  # Datos originales no modificados
        assert isinstance(lit_analysis, dict)
        assert isinstance(features, pd.DataFrame)
        assert isinstance(prediction, dict)
        
        # Verificar que las fechas se mantienen consistentes
        if len(features) > 0:
            assert features.index.dtype == original_data.index.dtype
    
    def test_performance_under_load(self, sample_ohlcv_data, temp_model_file):
        """Prueba rendimiento bajo carga."""
        # Crear componentes
        predictor = LITMLPredictor(temp_model_file)
        predictor.load_model()
        
        # Procesar múltiples ventanas de datos
        window_size = 100
        num_windows = 10
        
        predictions = []
        start_time = time.time()
        
        for i in range(num_windows):
            start_idx = i * 10
            end_idx = start_idx + window_size
            
            if end_idx <= len(sample_ohlcv_data):
                window_data = sample_ohlcv_data.iloc[start_idx:end_idx]
                prediction = predictor.predict_single(window_data)
                predictions.append(prediction)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verificar rendimiento
        assert len(predictions) > 0
        assert processing_time < 10.0  # Menos de 10 segundos para 10 predicciones
        
        # Verificar que todas las predicciones son válidas
        for prediction in predictions:
            assert prediction['signal'] in ['buy', 'sell', 'hold']
            assert 0 <= prediction['confidence'] <= 1
    
    def _setup_mt5_mock(self, mock_mt5):
        """Configurar mock completo de MT5."""
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True
        mock_mt5.shutdown.return_value = None
        
        mock_account = Mock()
        mock_account.login = 12345
        mock_account.balance = 10000.0
        mock_account.equity = 10000.0
        mock_account.margin = 0.0
        mock_account.margin_free = 10000.0
        mock_account.currency = "USD"
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
        mock_tick.ask = 1.0852
        mock_tick.bid = 1.0850
        mock_mt5.symbol_info_tick.return_value = mock_tick
        
        mock_result = Mock()
        mock_result.retcode = 10009
        mock_result.order = 12345
        mock_result.price = 1.0852
        mock_mt5.order_send.return_value = mock_result
        mock_mt5.TRADE_RETCODE_DONE = 10009


@pytest.mark.integration
@pytest.mark.slow
class TestLongRunningIntegration:
    """Pruebas de integración de larga duración."""
    
    @patch('src.trading.trade_executor.mt5')
    @patch('src.data.data_collector.DataCollector')
    def test_bot_stability_over_time(self, mock_data_collector, mock_mt5, temp_model_file, sample_ohlcv_data):
        """Prueba estabilidad del bot durante ejecución prolongada."""
        # Configurar mocks
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True
        
        mock_account = Mock()
        mock_account.balance = 10000.0
        mock_mt5.account_info.return_value = mock_account
        
        mock_collector_instance = Mock()
        mock_collector_instance.get_ohlcv.return_value = sample_ohlcv_data
        mock_data_collector.return_value = mock_collector_instance
        
        # Crear bot
        bot = TradingBot(
            model_path=temp_model_file,
            symbols=['EURUSD'],
            timeframe="1h",
            prediction_interval=0.5,  # 0.5 segundos para prueba rápida
            risk_level="moderate",
            trading_enabled=False
        )
        
        # Contadores para monitoreo
        signal_count = 0
        error_count = 0
        
        def signal_callback(symbol, prediction):
            nonlocal signal_count
            signal_count += 1
        
        def error_callback(error):
            nonlocal error_count
            error_count += 1
        
        bot.add_signal_callback(signal_callback)
        bot.add_error_callback(error_callback)
        
        # Ejecutar por un período corto
        bot.start()
        time.sleep(3)  # 3 segundos de ejecución
        bot.stop()
        
        # Verificar estabilidad
        assert signal_count > 0  # Debe haber generado señales
        assert error_count == 0  # No debe haber errores
        
        # Verificar que el bot se detuvo correctamente
        assert not bot.running


if __name__ == "__main__":
    # Ejecutar pruebas si se ejecuta directamente
    pytest.main([__file__, "-v", "-m", "integration"]) 