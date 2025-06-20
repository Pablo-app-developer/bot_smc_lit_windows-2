#!/usr/bin/env python3
"""
Configuración de pytest y fixtures compartidos.

Este archivo contiene fixtures y configuraciones que pueden ser
utilizadas por todas las pruebas del proyecto.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Fixture que retorna la ruta raíz del proyecto."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_ohlcv_data():
    """Fixture que crea datos OHLCV de muestra para todas las pruebas."""
    np.random.seed(42)  # Para reproducibilidad
    
    dates = pd.date_range('2024-01-01', periods=500, freq='1H')
    data = []
    base_price = 1.0800
    
    for i in range(500):
        # Simular movimiento de precios realista
        if i == 0:
            price = base_price
        else:
            # Movimiento browniano con tendencia
            trend = 0.00001 * np.sin(i / 50)  # Tendencia sinusoidal lenta
            volatility = 0.0002 * (1 + 0.5 * np.sin(i / 20))  # Volatilidad variable
            price = data[i-1]['close'] + np.random.normal(trend, volatility)
        
        # Crear OHLC realista
        daily_range = np.random.uniform(0.0001, 0.0005)
        high = price + np.random.uniform(0, daily_range)
        low = price - np.random.uniform(0, daily_range)
        open_price = data[i-1]['close'] if i > 0 else price
        
        # Asegurar que OHLC sea consistente
        high = max(high, open_price, price)
        low = min(low, open_price, price)
        
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': dates[i],
            'open': round(open_price, 5),
            'high': round(high, 5),
            'low': round(low, 5),
            'close': round(price, 5),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


@pytest.fixture
def small_ohlcv_data():
    """Fixture que crea un dataset pequeño para pruebas rápidas."""
    dates = pd.date_range('2024-01-01', periods=50, freq='1H')
    data = []
    
    for i in range(50):
        price = 1.0800 + (i * 0.0001) + np.random.normal(0, 0.00005)
        
        data.append({
            'timestamp': dates[i],
            'open': round(price - 0.00002, 5),
            'high': round(price + 0.00003, 5),
            'low': round(price - 0.00003, 5),
            'close': round(price, 5),
            'volume': np.random.randint(1000, 3000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


@pytest.fixture
def liquidity_pattern_data():
    """Fixture que crea datos con patrones de liquidez específicos."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    data = []
    
    # Crear patrón de equal highs seguido de barrido
    resistance_level = 1.0850
    
    for i in range(100):
        if 20 <= i <= 30:  # Formación de equal highs
            high = resistance_level + np.random.normal(0, 0.00002)
            low = high - 0.0015
            close = high - 0.0005
            open_price = low + 0.0003
        elif i == 35:  # Barrido de liquidez
            high = resistance_level + 0.0020  # Rompe la resistencia
            low = resistance_level - 0.0005
            close = resistance_level + 0.0005
            open_price = resistance_level - 0.0002
        elif i > 35:  # Después del barrido
            high = resistance_level - 0.0010 + np.random.uniform(0, 0.0005)
            low = high - 0.0012
            close = high - 0.0003
            open_price = low + 0.0002
        else:  # Antes de la formación
            high = resistance_level - 0.0030 + np.random.uniform(0, 0.0020)
            low = high - 0.0015
            close = high - 0.0005
            open_price = low + 0.0003
        
        data.append({
            'timestamp': dates[i],
            'open': round(open_price, 5),
            'high': round(high, 5),
            'low': round(low, 5),
            'close': round(close, 5),
            'volume': 2000 if i == 35 else np.random.randint(1000, 1500)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


@pytest.fixture
def trending_data():
    """Fixture que crea datos con tendencia clara."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    data = []
    
    # Tendencia alcista clara
    for i in range(100):
        base_price = 1.0800 + (i * 0.0002)  # Tendencia alcista
        noise = np.random.normal(0, 0.00005)
        price = base_price + noise
        
        data.append({
            'timestamp': dates[i],
            'open': round(price - 0.00002, 5),
            'high': round(price + 0.00003, 5),
            'low': round(price - 0.00003, 5),
            'close': round(price, 5),
            'volume': np.random.randint(1000, 2000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


@pytest.fixture
def mock_mt5():
    """Fixture que crea un mock completo de MetaTrader5."""
    mock = Mock()
    
    # Configurar métodos básicos
    mock.initialize.return_value = True
    mock.login.return_value = True
    mock.shutdown.return_value = None
    
    # Mock de información de cuenta
    mock_account = Mock()
    mock_account.login = 12345
    mock_account.server = "Test-Server"
    mock_account.balance = 10000.0
    mock_account.equity = 10000.0
    mock_account.margin = 0.0
    mock_account.margin_free = 10000.0
    mock_account.currency = "USD"
    mock.account_info.return_value = mock_account
    
    # Mock de información de símbolo
    mock_symbol_info = Mock()
    mock_symbol_info.trade_tick_value = 1.0
    mock_symbol_info.point = 0.00001
    mock_symbol_info.volume_min = 0.01
    mock_symbol_info.volume_max = 100.0
    mock_symbol_info.volume_step = 0.01
    mock_symbol_info.trade_contract_size = 100000
    mock_symbol_info.spread = 2
    mock.symbol_info.return_value = mock_symbol_info
    
    # Mock de tick
    mock_tick = Mock()
    mock_tick.ask = 1.0852
    mock_tick.bid = 1.0850
    mock_tick.time = 1640995200  # Timestamp
    mock.symbol_info_tick.return_value = mock_tick
    
    # Mock de resultado de orden
    mock_result = Mock()
    mock_result.retcode = 10009  # TRADE_RETCODE_DONE
    mock_result.order = 12345
    mock_result.price = 1.0852
    mock_result.volume = 0.1
    mock.order_send.return_value = mock_result
    
    # Constantes
    mock.TRADE_RETCODE_DONE = 10009
    mock.ORDER_TYPE_BUY = 0
    mock.ORDER_TYPE_SELL = 1
    mock.TRADE_ACTION_DEAL = 1
    mock.ORDER_FILLING_IOC = 1
    
    # Mock de posiciones
    mock.positions_get.return_value = []
    
    return mock


@pytest.fixture
def temp_model_file():
    """Fixture que crea un archivo de modelo temporal."""
    import joblib
    
    # Crear datos de modelo mock
    mock_model_data = {
        'model': Mock(),
        'scaler': Mock(),
        'feature_columns': ['rsi', 'macd', 'bb_upper', 'atr', 'volume_sma'],
        'model_info': {
            'model_type': 'XGBoost',
            'version': '1.0',
            'features_count': 5,
            'training_date': '2024-01-01',
            'accuracy': 0.75
        }
    }
    
    # Configurar mocks
    mock_model_data['model'].predict_proba = Mock(return_value=np.array([[0.2, 0.6, 0.2]]))
    mock_model_data['model'].predict = Mock(return_value=np.array([1]))
    mock_model_data['model'].feature_importances_ = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    mock_model_data['scaler'].transform = Mock(return_value=np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]))
    
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        joblib.dump(mock_model_data, f.name)
        yield f.name
    
    # Limpiar archivo temporal
    try:
        os.unlink(f.name)
    except FileNotFoundError:
        pass


@pytest.fixture
def sample_signals():
    """Fixture que crea señales de trading de muestra."""
    from src.trading.trade_executor import TradeSignal
    
    signals = [
        TradeSignal(
            symbol="EURUSD",
            signal="buy",
            confidence=0.75,
            price=1.0850,
            probabilities={'buy': 0.75, 'sell': 0.15, 'hold': 0.10},
            metadata={'strategy': 'LIT_ML', 'timeframe': '1h'}
        ),
        TradeSignal(
            symbol="GBPUSD",
            signal="sell",
            confidence=0.68,
            price=1.2650,
            probabilities={'buy': 0.12, 'sell': 0.68, 'hold': 0.20},
            metadata={'strategy': 'LIT_ML', 'timeframe': '1h'}
        ),
        TradeSignal(
            symbol="USDJPY",
            signal="hold",
            confidence=0.55,
            price=149.50,
            probabilities={'buy': 0.25, 'sell': 0.20, 'hold': 0.55},
            metadata={'strategy': 'LIT_ML', 'timeframe': '1h'}
        )
    ]
    
    return signals


@pytest.fixture(autouse=True)
def setup_logging():
    """Fixture que configura logging para las pruebas."""
    import logging
    
    # Configurar logging para pruebas
    logging.basicConfig(
        level=logging.WARNING,  # Solo mostrar warnings y errores en pruebas
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Silenciar logs de bibliotecas externas
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


@pytest.fixture
def suppress_warnings():
    """Fixture para suprimir warnings específicos durante las pruebas."""
    import warnings
    
    # Suprimir warnings comunes en pruebas
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    yield
    
    # Restaurar warnings
    warnings.resetwarnings()


# Configuración de pytest
def pytest_configure(config):
    """Configuración global de pytest."""
    # Agregar marcadores personalizados
    config.addinivalue_line(
        "markers", "slow: marca las pruebas como lentas"
    )
    config.addinivalue_line(
        "markers", "integration: marca las pruebas de integración"
    )
    config.addinivalue_line(
        "markers", "unit: marca las pruebas unitarias"
    )


def pytest_collection_modifyitems(config, items):
    """Modificar la colección de pruebas."""
    # Agregar marcador 'unit' a todas las pruebas por defecto
    for item in items:
        if not any(marker.name in ['slow', 'integration'] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# Hooks para reportes
def pytest_runtest_setup(item):
    """Hook que se ejecuta antes de cada prueba."""
    # Limpiar cualquier estado global si es necesario
    pass


def pytest_runtest_teardown(item, nextitem):
    """Hook que se ejecuta después de cada prueba."""
    # Limpiar recursos si es necesario
    pass 