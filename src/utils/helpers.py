"""
Módulo de funciones auxiliares y utilidades comunes.

Este módulo contiene funciones helper utilizadas en diferentes
partes del bot de trading LIT.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime, timezone
import hashlib
import json


def normalize_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los datos de precios OHLCV.
    
    Args:
        df: DataFrame con datos OHLCV.
        
    Returns:
        pd.DataFrame: DataFrame normalizado.
    """
    df = df.copy()
    
    # Asegurar que las columnas están en el formato correcto
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Normalizar nombres de columnas
    column_mapping = {
        'Open': 'open', 'High': 'high', 'Low': 'low', 
        'Close': 'close', 'Volume': 'volume',
        'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 
        'CLOSE': 'close', 'VOLUME': 'volume'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Verificar que todas las columnas requeridas estén presentes
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columnas faltantes en los datos: {missing_columns}")
    
    # Convertir a float y manejar valores nulos
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Eliminar filas con valores nulos
    df = df.dropna(subset=required_columns)
    
    # Ordenar por índice (tiempo)
    df = df.sort_index()
    
    return df


def calculate_position_size(balance: float, 
                          risk_per_trade: float, 
                          entry_price: float, 
                          stop_loss: float) -> float:
    """
    Calcula el tamaño de posición basado en el riesgo.
    
    Args:
        balance: Balance disponible.
        risk_per_trade: Porcentaje de riesgo por trade (0.01 = 1%).
        entry_price: Precio de entrada.
        stop_loss: Precio de stop loss.
        
    Returns:
        float: Tamaño de posición calculado.
    """
    if entry_price <= 0 or stop_loss <= 0:
        return 0.0
    
    risk_amount = balance * risk_per_trade
    price_diff = abs(entry_price - stop_loss)
    
    if price_diff == 0:
        return 0.0
    
    position_size = risk_amount / price_diff
    return round(position_size, 4)


def detect_price_levels(highs: pd.Series, 
                       lows: pd.Series, 
                       tolerance: float = 0.001) -> Dict[str, List[float]]:
    """
    Detecta niveles de soporte y resistencia.
    
    Args:
        highs: Serie de precios máximos.
        lows: Serie de precios mínimos.
        tolerance: Tolerancia para considerar niveles iguales.
        
    Returns:
        Dict[str, List[float]]: Diccionario con niveles de soporte y resistencia.
    """
    # Detectar máximos locales (resistencias)
    resistance_levels = []
    for i in range(2, len(highs) - 2):
        if (highs.iloc[i] > highs.iloc[i-1] and 
            highs.iloc[i] > highs.iloc[i+1] and
            highs.iloc[i] > highs.iloc[i-2] and 
            highs.iloc[i] > highs.iloc[i+2]):
            resistance_levels.append(highs.iloc[i])
    
    # Detectar mínimos locales (soportes)
    support_levels = []
    for i in range(2, len(lows) - 2):
        if (lows.iloc[i] < lows.iloc[i-1] and 
            lows.iloc[i] < lows.iloc[i+1] and
            lows.iloc[i] < lows.iloc[i-2] and 
            lows.iloc[i] < lows.iloc[i+2]):
            support_levels.append(lows.iloc[i])
    
    # Agrupar niveles similares
    resistance_levels = _group_similar_levels(resistance_levels, tolerance)
    support_levels = _group_similar_levels(support_levels, tolerance)
    
    return {
        'resistance': resistance_levels,
        'support': support_levels
    }


def _group_similar_levels(levels: List[float], tolerance: float) -> List[float]:
    """
    Agrupa niveles de precios similares.
    
    Args:
        levels: Lista de niveles de precios.
        tolerance: Tolerancia para agrupar niveles.
        
    Returns:
        List[float]: Lista de niveles agrupados.
    """
    if not levels:
        return []
    
    levels = sorted(levels)
    grouped_levels = [levels[0]]
    
    for level in levels[1:]:
        # Si el nivel está dentro de la tolerancia del último nivel agrupado
        if abs(level - grouped_levels[-1]) / grouped_levels[-1] <= tolerance:
            # Actualizar con el promedio
            grouped_levels[-1] = (grouped_levels[-1] + level) / 2
        else:
            grouped_levels.append(level)
    
    return grouped_levels


def calculate_atr(high: pd.Series, 
                  low: pd.Series, 
                  close: pd.Series, 
                  period: int = 14) -> pd.Series:
    """
    Calcula el Average True Range (ATR).
    
    Args:
        high: Serie de precios máximos.
        low: Serie de precios mínimos.
        close: Serie de precios de cierre.
        period: Período para el cálculo del ATR.
        
    Returns:
        pd.Series: Serie con valores de ATR.
    """
    # Calcular True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calcular ATR como media móvil del True Range
    atr = tr.rolling(window=period).mean()
    
    return atr


def is_market_hours(timestamp: datetime, 
                   market_open: str = "09:00", 
                   market_close: str = "17:00") -> bool:
    """
    Verifica si un timestamp está dentro del horario de mercado.
    
    Args:
        timestamp: Timestamp a verificar.
        market_open: Hora de apertura del mercado (formato HH:MM).
        market_close: Hora de cierre del mercado (formato HH:MM).
        
    Returns:
        bool: True si está dentro del horario de mercado.
    """
    time_str = timestamp.strftime("%H:%M")
    return market_open <= time_str <= market_close


def create_data_hash(data: pd.DataFrame) -> str:
    """
    Crea un hash único para un DataFrame.
    
    Args:
        data: DataFrame para crear el hash.
        
    Returns:
        str: Hash MD5 del DataFrame.
    """
    data_str = data.to_string()
    return hashlib.md5(data_str.encode()).hexdigest()


def validate_ohlcv_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Valida que los datos OHLCV sean correctos.
    
    Args:
        df: DataFrame con datos OHLCV.
        
    Returns:
        Tuple[bool, List[str]]: (Es válido, Lista de errores encontrados).
    """
    errors = []
    
    # Verificar columnas requeridas
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        errors.append(f"Columnas faltantes: {missing_columns}")
        return False, errors
    
    # Verificar que high >= low
    invalid_hl = df[df['high'] < df['low']]
    if not invalid_hl.empty:
        errors.append(f"High < Low en {len(invalid_hl)} filas")
    
    # Verificar que open y close estén entre high y low
    invalid_oc = df[(df['open'] > df['high']) | (df['open'] < df['low']) |
                    (df['close'] > df['high']) | (df['close'] < df['low'])]
    if not invalid_oc.empty:
        errors.append(f"Open/Close fuera del rango High/Low en {len(invalid_oc)} filas")
    
    # Verificar valores negativos en precios
    negative_prices = df[(df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)]
    if not negative_prices.empty:
        errors.append(f"Precios negativos o cero en {len(negative_prices)} filas")
    
    # Verificar volumen negativo
    negative_volume = df[df['volume'] < 0]
    if not negative_volume.empty:
        errors.append(f"Volumen negativo en {len(negative_volume)} filas")
    
    return len(errors) == 0, errors


def format_number(number: float, decimals: int = 2) -> str:
    """
    Formatea un número para mostrar en logs o reportes.
    
    Args:
        number: Número a formatear.
        decimals: Número de decimales.
        
    Returns:
        str: Número formateado.
    """
    if abs(number) >= 1e6:
        return f"{number/1e6:.{decimals}f}M"
    elif abs(number) >= 1e3:
        return f"{number/1e3:.{decimals}f}K"
    else:
        return f"{number:.{decimals}f}"


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calcula los retornos porcentuales de una serie de precios.
    
    Args:
        prices: Serie de precios.
        
    Returns:
        pd.Series: Serie de retornos porcentuales.
    """
    return prices.pct_change().dropna()


def get_current_timestamp() -> datetime:
    """
    Obtiene el timestamp actual en UTC.
    
    Returns:
        datetime: Timestamp actual en UTC.
    """
    return datetime.now(timezone.utc)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    División segura que maneja división por cero.
    
    Args:
        numerator: Numerador.
        denominator: Denominador.
        default: Valor por defecto si denominador es cero.
        
    Returns:
        float: Resultado de la división o valor por defecto.
    """
    return numerator / denominator if denominator != 0 else default 