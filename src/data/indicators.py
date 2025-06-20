"""
Módulo de indicadores técnicos usando la librería 'ta'.

Este módulo calcula indicadores técnicos estándar para análisis de trading
usando la librería 'ta' que es más estable y confiable que ta-lib.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any

# Usar librería ta en lugar de ta-lib (más estable)
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

from src.utils.logger import log


def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos los indicadores técnicos disponibles.
    
    Args:
        data: DataFrame con datos OHLCV.
        
    Returns:
        pd.DataFrame: DataFrame con indicadores añadidos.
    """
    if not TA_AVAILABLE:
        log.warning("Librería 'ta' no disponible. Usando indicadores básicos.")
        return calculate_basic_indicators(data)
    
    df = data.copy()
    
    try:
        # Indicadores de tendencia
        df = add_trend_indicators(df)
        
        # Indicadores de momentum
        df = add_momentum_indicators(df)
        
        # Indicadores de volatilidad
        df = add_volatility_indicators(df)
        
        # Indicadores de volumen
        df = add_volume_indicators(df)
        
        # Indicadores personalizados
        df = add_custom_indicators(df)
        
        indicator_count = len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
        log.info(f"Indicadores técnicos calculados: {indicator_count} indicadores")
        
        return df
        
    except Exception as e:
        log.error(f"Error calculando indicadores: {str(e)}")
        return calculate_basic_indicators(data)


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Añade indicadores de tendencia usando la librería ta."""
    
    # Medias móviles simples
    df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
    
    # Medias móviles exponenciales
    df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    
    # ADX (Average Directional Index)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['adx_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)
    df['adx_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)
    
    # Parabolic SAR
    df['psar'] = ta.trend.psar_up(df['high'], df['low'], df['close'])
    
    return df


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Añade indicadores de momentum usando la librería ta."""
    
    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    
    # Stochastic
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
    
    # Williams %R
    df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
    
    # Rate of Change
    df['roc'] = ta.momentum.roc(df['close'], window=10)
    
    # Momentum
    df['momentum'] = ta.momentum.tsi(df['close'])
    
    return df


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Añade indicadores de volatilidad usando la librería ta."""
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = bb.bollinger_wband()
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Average True Range
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    
    # Keltner Channel
    kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
    df['kc_upper'] = kc.keltner_channel_hband()
    df['kc_middle'] = kc.keltner_channel_mband()
    df['kc_lower'] = kc.keltner_channel_lband()
    
    return df


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Añade indicadores de volumen usando la librería ta."""
    
    # On Balance Volume
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    # Volume SMA
    df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Accumulation/Distribution Line
    df['ad_line'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
    
    # Chaikin Money Flow
    df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])
    
    # Volume Price Trend
    df['vpt'] = ta.volume.volume_price_trend(df['close'], df['volume'])
    
    return df


def add_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Añade indicadores personalizados."""
    
    # Retornos
    df['returns'] = df['close'].pct_change()
    df['returns_5'] = df['close'].pct_change(5)
    df['returns_10'] = df['close'].pct_change(10)
    
    # Volatilidad realizada
    df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
    
    # Rangos
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Gaps
    df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Momentum personalizado
    df['price_momentum'] = df['close'] / df['close'].shift(10) - 1
    
    return df


def calculate_basic_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores básicos sin dependencias externas.
    
    Args:
        data: DataFrame con datos OHLCV.
        
    Returns:
        pd.DataFrame: DataFrame con indicadores básicos.
    """
    df = data.copy()
    
    # Medias móviles simples
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # RSI básico
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands básicas
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # ATR básico
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = true_range.rolling(14).mean()
    
    # Retornos
    df['returns'] = df['close'].pct_change()
    
    log.info("Indicadores básicos calculados (sin librería ta)")
    
    return df


def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """
    Calcula niveles de Fibonacci.
    
    Args:
        high: Precio máximo.
        low: Precio mínimo.
        
    Returns:
        Dict[str, float]: Niveles de Fibonacci.
    """
    diff = high - low
    
    return {
        '0.0': high,
        '23.6': high - 0.236 * diff,
        '38.2': high - 0.382 * diff,
        '50.0': high - 0.5 * diff,
        '61.8': high - 0.618 * diff,
        '78.6': high - 0.786 * diff,
        '100.0': low
    }


def calculate_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """
    Calcula puntos pivot.
    
    Args:
        high: Precio máximo del período anterior.
        low: Precio mínimo del período anterior.
        close: Precio de cierre del período anterior.
        
    Returns:
        Dict[str, float]: Puntos pivot.
    """
    pivot = (high + low + close) / 3
    
    return {
        'pivot': pivot,
        'r1': 2 * pivot - low,
        'r2': pivot + (high - low),
        'r3': high + 2 * (pivot - low),
        's1': 2 * pivot - high,
        's2': pivot - (high - low),
        's3': low - 2 * (high - pivot)
    }


def get_indicator_summary(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Genera un resumen de los indicadores.
    
    Args:
        data: DataFrame con indicadores.
        
    Returns:
        Dict[str, Any]: Resumen de indicadores.
    """
    if len(data) == 0:
        return {}
    
    latest = data.iloc[-1]
    
    summary = {
        'trend': {
            'sma_signal': 'bullish' if latest.get('close', 0) > latest.get('sma_20', 0) else 'bearish',
            'macd_signal': 'bullish' if latest.get('macd', 0) > latest.get('macd_signal', 0) else 'bearish',
            'adx_strength': 'strong' if latest.get('adx', 0) > 25 else 'weak'
        },
        'momentum': {
            'rsi_level': 'overbought' if latest.get('rsi', 50) > 70 else 'oversold' if latest.get('rsi', 50) < 30 else 'neutral',
            'stoch_level': 'overbought' if latest.get('stoch_k', 50) > 80 else 'oversold' if latest.get('stoch_k', 50) < 20 else 'neutral'
        },
        'volatility': {
            'bb_position': latest.get('bb_position', 0.5),
            'atr_level': latest.get('atr', 0)
        },
        'volume': {
            'volume_trend': 'high' if latest.get('volume_ratio', 1) > 1.5 else 'low' if latest.get('volume_ratio', 1) < 0.5 else 'normal',
            'obv_trend': 'bullish' if latest.get('obv', 0) > data['obv'].iloc[-10:].mean() else 'bearish'
        }
    }
    
    return summary