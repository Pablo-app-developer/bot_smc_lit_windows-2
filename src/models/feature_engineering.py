"""
Módulo de ingeniería de características para el modelo ML.

Este módulo genera características (features) para el modelo de Machine Learning
combinando indicadores técnicos, señales LIT y estadísticas de velas.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from src.data.indicators import calculate_all_indicators
from src.strategies.lit_detector import LITDetector, SignalType
from src.core.config import config
from src.utils.logger import log


class FeatureEngineer:
    """
    Ingeniero de características para el modelo ML.
    
    Genera características combinando:
    - Indicadores técnicos
    - Señales LIT
    - Estadísticas de velas
    - Patrones de precio
    """
    
    def __init__(self):
        """Inicializa el ingeniero de características."""
        self.feature_lookback = config.ml.feature_lookback
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.lit_detector = LITDetector()
        
        # Cache de características calculadas
        self._feature_cache = {}
        
        log.info("FeatureEngineer inicializado")
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Crea todas las características para el modelo ML.
        
        Args:
            data: DataFrame con datos OHLCV preprocesados.
            
        Returns:
            pd.DataFrame: DataFrame con características generadas.
        """
        if len(data) < self.feature_lookback:
            raise ValueError(f"Necesita al menos {self.feature_lookback} períodos de datos")
        
        df = data.copy()
        
        # Calcular indicadores técnicos
        df = calculate_all_indicators(df)
        
        # Generar características de velas
        df = self._create_candle_features(df)
        
        # Generar características de precio
        df = self._create_price_features(df)
        
        # Generar características de volumen
        df = self._create_volume_features(df)
        
        # Generar características de volatilidad
        df = self._create_volatility_features(df)
        
        # Generar características LIT
        df = self._create_lit_features(df)
        
        # Generar características de patrones
        df = self._create_pattern_features(df)
        
        # Generar características de momentum
        df = self._create_momentum_features(df)
        
        # Generar características de estadísticas móviles
        df = self._create_rolling_features(df)
        
        log.info(f"Características creadas: {len(df.columns)} columnas")
        
        return df
    
    def _create_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características basadas en estadísticas de velas.
        
        Args:
            df: DataFrame con datos OHLCV.
            
        Returns:
            pd.DataFrame: DataFrame con características de velas añadidas.
        """
        # Características básicas de velas (ya calculadas en data_loader)
        if 'body_size' not in df.columns:
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
            df['total_range'] = df['high'] - df['low']
        
        # Ratios de velas
        df['body_ratio'] = df['body_size'] / df['total_range']
        df['upper_wick_ratio'] = df['upper_wick'] / df['total_range']
        df['lower_wick_ratio'] = df['lower_wick'] / df['total_range']
        
        # Características de sombras
        df['shadow_ratio'] = (df['upper_wick'] + df['lower_wick']) / df['total_range']
        df['wick_imbalance'] = (df['upper_wick'] - df['lower_wick']) / df['total_range']
        
        # Posición del cuerpo en el rango
        df['body_position'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['total_range']
        
        # Características de tipo de vela
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['is_bearish'] = (df['close'] < df['open']).astype(int)
        df['is_doji'] = (df['body_ratio'] < 0.1).astype(int)
        df['is_hammer'] = ((df['lower_wick_ratio'] > 0.6) & (df['body_ratio'] < 0.3)).astype(int)
        df['is_shooting_star'] = ((df['upper_wick_ratio'] > 0.6) & (df['body_ratio'] < 0.3)).astype(int)
        
        return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características basadas en acción del precio.
        
        Args:
            df: DataFrame con datos.
            
        Returns:
            pd.DataFrame: DataFrame con características de precio añadidas.
        """
        # Retornos
        df['returns_1'] = df['close'].pct_change()
        df['returns_2'] = df['close'].pct_change(2)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        
        # Logaritmos de retornos
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Posición relativa en el rango
        df['hl_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Gaps
        df['gap_up'] = ((df['low'] > df['high'].shift(1))).astype(int)
        df['gap_down'] = ((df['high'] < df['low'].shift(1))).astype(int)
        df['gap_size'] = np.where(df['gap_up'], df['low'] - df['high'].shift(1),
                                 np.where(df['gap_down'], df['low'].shift(1) - df['high'], 0))
        
        # Niveles de Fibonacci
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        fib_range = high_20 - low_20
        
        df['fib_23.6'] = low_20 + 0.236 * fib_range
        df['fib_38.2'] = low_20 + 0.382 * fib_range
        df['fib_50.0'] = low_20 + 0.500 * fib_range
        df['fib_61.8'] = low_20 + 0.618 * fib_range
        
        # Distancia a niveles Fibonacci
        df['dist_fib_23.6'] = (df['close'] - df['fib_23.6']) / df['close']
        df['dist_fib_38.2'] = (df['close'] - df['fib_38.2']) / df['close']
        df['dist_fib_50.0'] = (df['close'] - df['fib_50.0']) / df['close']
        df['dist_fib_61.8'] = (df['close'] - df['fib_61.8']) / df['close']
        
        return df
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características basadas en volumen.
        
        Args:
            df: DataFrame con datos.
            
        Returns:
            pd.DataFrame: DataFrame con características de volumen añadidas.
        """
        # Ratios de volumen
        df['volume_ratio_sma'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_ratio_ema'] = df['volume'] / df['volume'].ewm(span=20).mean()
        
        # Volumen relativo
        df['volume_percentile'] = df['volume'].rolling(50).rank() / 50
        
        # Volumen ponderado por precio
        df['vwap_deviation'] = (df['close'] - ((df['high'] + df['low'] + df['close']) / 3 * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()) / df['close']
        
        # Indicadores de volumen
        if 'obv' in df.columns:
            df['obv_slope'] = (df['obv'] - df['obv'].shift(5)) / 5
            df['obv_momentum'] = df['obv'].diff(5)
        
        # Volumen en breakouts
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        df['breakout_volume'] = np.where(
            (df['high'] > high_20.shift(1)) | (df['low'] < low_20.shift(1)),
            df['volume'], 0
        )
        
        return df
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características basadas en volatilidad.
        
        Args:
            df: DataFrame con datos.
            
        Returns:
            pd.DataFrame: DataFrame con características de volatilidad añadidas.
        """
        # Volatilidad histórica
        df['volatility_10'] = df['returns_1'].rolling(10).std()
        df['volatility_20'] = df['returns_1'].rolling(20).std()
        df['volatility_50'] = df['returns_1'].rolling(50).std()
        
        # Ratio de volatilidad
        df['volatility_ratio'] = df['volatility_10'] / df['volatility_50']
        
        # Rango verdadero normalizado
        if 'atr' in df.columns:
            df['atr_normalized'] = df['atr'] / df['close']
            df['atr_percentile'] = df['atr'].rolling(50).rank() / 50
        
        # Régimen de volatilidad
        df['vol_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(50).mean()).astype(int)
        
        # Volatilidad de gaps
        df['gap_volatility'] = abs(df['gap_size']).rolling(10).mean()
        
        return df
    
    def _create_lit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características basadas en señales LIT.
        
        Args:
            df: DataFrame con datos.
            
        Returns:
            pd.DataFrame: DataFrame con características LIT añadidas.
        """
        # Inicializar características LIT
        df['lit_signal'] = 0  # 0=hold, 1=buy, -1=sell
        df['lit_confidence'] = 0.0
        df['lit_events_count'] = 0
        df['lit_bullish_score'] = 0.0
        df['lit_bearish_score'] = 0.0
        
        # Calcular señales LIT para cada punto (solo para datos suficientes)
        for i in range(self.feature_lookback, len(df)):
            try:
                window_data = df.iloc[:i+1]
                lit_signal = self.lit_detector.analyze(window_data)
                
                # Mapear señal a número
                if lit_signal.signal == SignalType.BUY:
                    df.iloc[i, df.columns.get_loc('lit_signal')] = 1
                elif lit_signal.signal == SignalType.SELL:
                    df.iloc[i, df.columns.get_loc('lit_signal')] = -1
                
                df.iloc[i, df.columns.get_loc('lit_confidence')] = lit_signal.confidence
                df.iloc[i, df.columns.get_loc('lit_events_count')] = len(lit_signal.events) if lit_signal.events else 0
                
                if lit_signal.context:
                    df.iloc[i, df.columns.get_loc('lit_bullish_score')] = lit_signal.context.get('bullish_score', 0)
                    df.iloc[i, df.columns.get_loc('lit_bearish_score')] = lit_signal.context.get('bearish_score', 0)
                    
            except Exception as e:
                log.warning(f"Error calculando señal LIT en índice {i}: {str(e)}")
                continue
        
        # Características derivadas de LIT
        df['lit_signal_change'] = df['lit_signal'].diff()
        df['lit_signal_strength'] = df['lit_signal'] * df['lit_confidence']
        df['lit_score_ratio'] = np.where(
            df['lit_bearish_score'] != 0,
            df['lit_bullish_score'] / df['lit_bearish_score'],
            df['lit_bullish_score']
        )
        
        return df
    
    def _create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características basadas en patrones de velas.
        
        Args:
            df: DataFrame con datos.
            
        Returns:
            pd.DataFrame: DataFrame con características de patrones añadidas.
        """
        # Patrones de reversión
        df['engulfing_bullish'] = (
            (df['is_bearish'].shift(1) == 1) & 
            (df['is_bullish'] == 1) & 
            (df['open'] < df['close'].shift(1)) & 
            (df['close'] > df['open'].shift(1))
        ).astype(int)
        
        df['engulfing_bearish'] = (
            (df['is_bullish'].shift(1) == 1) & 
            (df['is_bearish'] == 1) & 
            (df['open'] > df['close'].shift(1)) & 
            (df['close'] < df['open'].shift(1))
        ).astype(int)
        
        # Patrones de continuación
        df['three_white_soldiers'] = (
            (df['is_bullish'] == 1) & 
            (df['is_bullish'].shift(1) == 1) & 
            (df['is_bullish'].shift(2) == 1) &
            (df['close'] > df['close'].shift(1)) &
            (df['close'].shift(1) > df['close'].shift(2))
        ).astype(int)
        
        df['three_black_crows'] = (
            (df['is_bearish'] == 1) & 
            (df['is_bearish'].shift(1) == 1) & 
            (df['is_bearish'].shift(2) == 1) &
            (df['close'] < df['close'].shift(1)) &
            (df['close'].shift(1) < df['close'].shift(2))
        ).astype(int)
        
        # Inside/Outside bars
        df['inside_bar'] = (
            (df['high'] < df['high'].shift(1)) & 
            (df['low'] > df['low'].shift(1))
        ).astype(int)
        
        df['outside_bar'] = (
            (df['high'] > df['high'].shift(1)) & 
            (df['low'] < df['low'].shift(1))
        ).astype(int)
        
        return df
    
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características basadas en momentum.
        
        Args:
            df: DataFrame con datos.
            
        Returns:
            pd.DataFrame: DataFrame con características de momentum añadidas.
        """
        # Momentum de diferentes períodos
        if 'momentum' in df.columns:
            df['momentum_5'] = df['close'] - df['close'].shift(5)
            df['momentum_10'] = df['close'] - df['close'].shift(10)
            df['momentum_20'] = df['close'] - df['close'].shift(20)
            
            # Aceleración del momentum
            df['momentum_acceleration'] = df['momentum'].diff()
        
        # Rate of Change de diferentes períodos
        if 'roc' in df.columns:
            df['roc_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
            df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        
        # Momentum del RSI
        if 'rsi' in df.columns:
            df['rsi_momentum'] = df['rsi'].diff()
            df['rsi_acceleration'] = df['rsi_momentum'].diff()
        
        # Momentum del MACD
        if 'macd' in df.columns:
            df['macd_momentum'] = df['macd'].diff()
            df['macd_velocity'] = df['macd_momentum'].diff()
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características de estadísticas móviles.
        
        Args:
            df: DataFrame con datos.
            
        Returns:
            pd.DataFrame: DataFrame con características de estadísticas móviles añadidas.
        """
        # Estadísticas móviles de precios
        for period in [5, 10, 20]:
            df[f'close_mean_{period}'] = df['close'].rolling(period).mean()
            df[f'close_std_{period}'] = df['close'].rolling(period).std()
            df[f'close_skew_{period}'] = df['close'].rolling(period).skew()
            df[f'close_kurt_{period}'] = df['close'].rolling(period).kurt()
            
            # Posición relativa
            df[f'close_percentile_{period}'] = df['close'].rolling(period).rank() / period
            
            # Distancia a la media
            df[f'close_zscore_{period}'] = (df['close'] - df[f'close_mean_{period}']) / df[f'close_std_{period}']
        
        # Características de rangos móviles
        df['range_expansion'] = df['total_range'] / df['total_range'].rolling(20).mean()
        df['range_contraction'] = df['total_range'].rolling(5).mean() / df['total_range'].rolling(20).mean()
        
        return df
    
    def prepare_ml_dataset(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara el dataset para Machine Learning.
        
        Args:
            df: DataFrame con características.
            target_column: Nombre de la columna objetivo.
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: (Features, Target).
        """
        # Eliminar filas con valores NaN
        df_clean = df.dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No hay datos válidos después de limpiar NaN")
        
        # Crear target si no se proporciona
        if target_column is None:
            target = self._create_target_variable(df_clean)
        else:
            target = df_clean[target_column]
        
        # Seleccionar características para ML
        feature_columns = self._select_features(df_clean)
        features = df_clean[feature_columns]
        
        # Validar que features y target tengan la misma longitud
        min_length = min(len(features), len(target))
        features = features.iloc[:min_length]
        target = target.iloc[:min_length]
        
        log.info(f"Dataset preparado: {len(features)} muestras, {len(feature_columns)} características")
        
        return features, target
    
    def _create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """
        Crea la variable objetivo para el modelo.
        
        Args:
            df: DataFrame con datos.
            
        Returns:
            pd.Series: Variable objetivo.
        """
        # Crear target basado en retornos futuros
        future_periods = 5
        future_returns = df['close'].shift(-future_periods) / df['close'] - 1
        
        # Clasificación en 3 clases: 0 (sell), 1 (hold), 2 (buy)
        buy_threshold = 0.01  # 1% de ganancia
        sell_threshold = -0.01  # 1% de pérdida
        
        target = pd.Series(1, index=df.index)  # Por defecto hold (1)
        target[future_returns > buy_threshold] = 2  # buy (2)
        target[future_returns < sell_threshold] = 0  # sell (0)
        
        return target
    
    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Selecciona las características más relevantes para el modelo.
        
        Args:
            df: DataFrame con todas las características.
            
        Returns:
            List[str]: Lista de nombres de características seleccionadas.
        """
        # Excluir columnas que no son características
        exclude_columns = [
            'open', 'high', 'low', 'close', 'volume',  # OHLCV originales
            'timestamp', 'symbol', 'datetime'  # Metadata
        ]
        
        # Seleccionar todas las características numéricas
        feature_columns = []
        for col in df.columns:
            if col not in exclude_columns and pd.api.types.is_numeric_dtype(df[col]):
                feature_columns.append(col)
        
        # Eliminar características con varianza muy baja
        low_variance_cols = []
        for col in feature_columns:
            if df[col].var() < 1e-10:
                low_variance_cols.append(col)
        
        feature_columns = [col for col in feature_columns if col not in low_variance_cols]
        
        log.info(f"Características seleccionadas: {len(feature_columns)}")
        
        return feature_columns
    
    def get_feature_importance_names(self) -> List[str]:
        """
        Retorna nombres de características para interpretación de importancia.
        
        Returns:
            List[str]: Lista de nombres de características.
        """
        return [
            # Indicadores técnicos
            'rsi', 'macd', 'bb_position', 'stoch_k', 'adx',
            # Características LIT
            'lit_signal', 'lit_confidence', 'lit_score_ratio',
            # Características de velas
            'body_ratio', 'wick_imbalance', 'is_hammer',
            # Características de precio
            'returns_5', 'close_zscore_20', 'fib_distance',
            # Características de volumen
            'volume_ratio_sma', 'obv_momentum',
            # Características de volatilidad
            'volatility_ratio', 'atr_percentile'
        ]