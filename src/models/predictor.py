#!/usr/bin/env python3
"""
Módulo Predictor LIT + ML - Predicciones en Tiempo Real.

Este módulo carga el modelo entrenado y realiza predicciones en tiempo real
o para backtesting, integrándose con el flujo general del bot de trading.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.data_loader import DataLoader
from src.models.feature_engineering import FeatureEngineer
from src.strategies.lit_detector import LITDetector, SignalType
from src.utils.logger import log
from src.core.config import config


class LITMLPredictor:
    """
    Predictor profesional LIT + ML para señales de trading.
    
    Carga el modelo entrenado y realiza predicciones en tiempo real
    o para backtesting, manteniendo consistencia con el entrenamiento.
    """
    
    def __init__(self, model_path: str = "models/lit_ml_model.pkl"):
        """
        Inicializa el predictor.
        
        Args:
            model_path: Ruta al modelo entrenado.
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.model_config = {}
        self.training_config = {}
        self.feature_engineer = None
        self.lit_detector = None
        
        # Componentes para datos
        self.data_loader = DataLoader()
        
        # Estado del predictor
        self.is_loaded = False
        self.last_prediction = None
        self.prediction_history = []
        
        # Cache para optimización
        self._feature_cache = {}
        self._data_cache = {}
        
        log.info(f"LITMLPredictor inicializado con modelo: {model_path}")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Carga el modelo entrenado desde archivo.
        
        Args:
            model_path: Ruta al modelo (opcional, usa el configurado).
            
        Returns:
            bool: True si se cargó exitosamente.
        """
        model_path = model_path or self.model_path
        
        if not os.path.exists(model_path):
            log.error(f"Modelo no encontrado: {model_path}")
            return False
        
        try:
            log.info(f"Cargando modelo desde: {model_path}")
            
            # Cargar datos del modelo
            model_data = joblib.load(model_path)
            
            # Extraer componentes
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_config = model_data.get('model_config', {})
            self.training_config = model_data.get('training_config', {})
            
            # Componentes de ingeniería de características
            self.feature_engineer = model_data.get('feature_engineer', FeatureEngineer())
            self.lit_detector = model_data.get('lit_detector', LITDetector())
            
            # Información del modelo
            timestamp = model_data.get('timestamp', 'Desconocido')
            
            self.is_loaded = True
            
            log.info(f"✅ Modelo cargado exitosamente")
            log.info(f"   Timestamp: {timestamp}")
            log.info(f"   Características: {len(self.feature_names)}")
            log.info(f"   Tipo de modelo: {type(self.model).__name__}")
            
            return True
            
        except Exception as e:
            log.error(f"Error cargando modelo: {str(e)}")
            return False
    
    def predict_single(self, data: pd.DataFrame, 
                      return_probabilities: bool = True,
                      return_confidence: bool = True) -> Dict[str, Any]:
        """
        Realiza predicción para una sola muestra (última vela).
        
        Args:
            data: DataFrame con datos OHLCV.
            return_probabilities: Si retornar probabilidades.
            return_confidence: Si retornar nivel de confianza.
            
        Returns:
            Dict[str, Any]: Resultado de la predicción.
        """
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Ejecutar load_model() primero.")
        
        if len(data) < 50:  # Mínimo para características
            log.warning(f"Datos insuficientes para predicción: {len(data)} velas")
            return self._get_default_prediction()
        
        try:
            # Generar características
            features_data = self._create_features_optimized(data)
            
            # Preparar para predicción
            X = self._prepare_prediction_data(features_data)
            
            if X is None or len(X) == 0:
                log.warning("No se pudieron generar características válidas")
                return self._get_default_prediction()
            
            # Realizar predicción
            prediction = self.model.predict(X)[-1]  # Última predicción
            
            result = {
                'signal': self._map_prediction_to_signal(prediction),
                'prediction_raw': int(prediction),
                'timestamp': datetime.now().isoformat(),
                'data_points': len(data)
            }
            
            # Agregar probabilidades si se solicita
            if return_probabilities:
                probabilities = self.model.predict_proba(X)[-1]
                result['probabilities'] = {
                    'sell': float(probabilities[0]),
                    'hold': float(probabilities[1]),
                    'buy': float(probabilities[2])
                }
                
                # Confianza como probabilidad máxima
                if return_confidence:
                    result['confidence'] = float(max(probabilities))
            
            # Guardar en historial
            self.last_prediction = result
            self.prediction_history.append(result)
            
            # Mantener solo las últimas 100 predicciones
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
            log.info(f"Predicción: {result['signal']} (confianza: {result.get('confidence', 0):.3f})")
            
            return result
            
        except Exception as e:
            log.error(f"Error en predicción: {str(e)}")
            return self._get_default_prediction()
    
    def predict_batch(self, data: pd.DataFrame, 
                     window_size: int = 50) -> List[Dict[str, Any]]:
        """
        Realiza predicciones en lote para backtesting.
        
        Args:
            data: DataFrame con datos OHLCV.
            window_size: Tamaño de ventana para predicciones.
            
        Returns:
            List[Dict[str, Any]]: Lista de predicciones.
        """
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Ejecutar load_model() primero.")
        
        if len(data) < window_size:
            log.warning(f"Datos insuficientes para batch: {len(data)} < {window_size}")
            return []
        
        predictions = []
        
        log.info(f"Iniciando predicción en lote: {len(data)} velas, ventana: {window_size}")
        
        # Procesar en ventanas deslizantes
        for i in range(window_size, len(data)):
            try:
                # Ventana de datos
                window_data = data.iloc[:i+1]
                
                # Predicción para esta ventana
                prediction = self.predict_single(
                    window_data, 
                    return_probabilities=True,
                    return_confidence=True
                )
                
                # Agregar información de índice
                prediction['index'] = i
                prediction['timestamp'] = data.index[i].isoformat() if hasattr(data.index[i], 'isoformat') else str(data.index[i])
                
                predictions.append(prediction)
                
                # Log progreso cada 50 predicciones
                if (i - window_size + 1) % 50 == 0:
                    progress = ((i - window_size + 1) / (len(data) - window_size)) * 100
                    log.info(f"Progreso batch: {progress:.1f}%")
                
            except Exception as e:
                log.warning(f"Error en predicción índice {i}: {str(e)}")
                continue
        
        log.info(f"Predicción en lote completada: {len(predictions)} predicciones")
        
        return predictions
    
    def predict_realtime(self, symbol: str, timeframe: str = "1h", 
                        periods: int = 100) -> Dict[str, Any]:
        """
        Realiza predicción en tiempo real cargando datos frescos.
        
        Args:
            symbol: Símbolo a analizar.
            timeframe: Marco temporal.
            periods: Número de períodos a cargar.
            
        Returns:
            Dict[str, Any]: Resultado de la predicción.
        """
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Ejecutar load_model() primero.")
        
        try:
            log.info(f"Predicción en tiempo real: {symbol} {timeframe}")
            
            # Cargar datos frescos
            data = self.data_loader.load_data(
                symbol=symbol,
                timeframe=timeframe,
                periods=periods
            )
            
            if data.empty:
                log.error(f"No se pudieron cargar datos para {symbol}")
                return self._get_default_prediction()
            
            # Realizar predicción
            prediction = self.predict_single(data)
            
            # Agregar información del símbolo
            prediction['symbol'] = symbol
            prediction['timeframe'] = timeframe
            prediction['last_price'] = float(data['close'].iloc[-1])
            prediction['last_timestamp'] = data.index[-1].isoformat() if hasattr(data.index[-1], 'isoformat') else str(data.index[-1])
            
            return prediction
            
        except Exception as e:
            log.error(f"Error en predicción tiempo real: {str(e)}")
            return self._get_default_prediction()
    
    def _create_features_optimized(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características optimizadas para predicción.
        
        Args:
            data: DataFrame con datos OHLCV.
            
        Returns:
            pd.DataFrame: DataFrame con características.
        """
        # Usar cache si los datos no han cambiado
        data_hash = hash(str(data.tail(10).values.tobytes()))
        if data_hash in self._feature_cache:
            return self._feature_cache[data_hash]
        
        try:
            # 1. Indicadores técnicos
            df = TechnicalIndicators.calculate_all_indicators(data.copy())
            
            # 2. Características básicas
            df = self.feature_engineer.create_features(df)
            
            # 3. Señales LIT optimizadas (solo últimas velas)
            df = self._create_lit_features_fast(df)
            
            # 4. Características de interacción básicas
            df = self._create_basic_interactions(df)
            
            # Guardar en cache
            self._feature_cache[data_hash] = df
            
            # Limpiar cache si es muy grande
            if len(self._feature_cache) > 10:
                oldest_key = next(iter(self._feature_cache))
                del self._feature_cache[oldest_key]
            
            return df
            
        except Exception as e:
            log.error(f"Error creando características: {str(e)}")
            return data.copy()
    
    def _create_lit_features_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características LIT optimizadas para predicción rápida.
        
        Args:
            df: DataFrame con datos.
            
        Returns:
            pd.DataFrame: DataFrame con características LIT.
        """
        # Inicializar columnas LIT
        df['lit_signal'] = 0
        df['lit_confidence'] = 0.0
        df['lit_events_count'] = 0
        df['lit_bullish_score'] = 0.0
        df['lit_bearish_score'] = 0.0
        
        try:
            # Solo analizar las últimas velas para eficiencia
            if len(df) > 50:
                lit_signal = self.lit_detector.analyze(df)
                
                # Aplicar a las últimas velas
                last_idx = len(df) - 1
                
                if lit_signal.signal == SignalType.BUY:
                    df.iloc[last_idx, df.columns.get_loc('lit_signal')] = 1
                elif lit_signal.signal == SignalType.SELL:
                    df.iloc[last_idx, df.columns.get_loc('lit_signal')] = -1
                
                df.iloc[last_idx, df.columns.get_loc('lit_confidence')] = lit_signal.confidence
                df.iloc[last_idx, df.columns.get_loc('lit_events_count')] = len(lit_signal.events)
                
                if lit_signal.context:
                    df.iloc[last_idx, df.columns.get_loc('lit_bullish_score')] = lit_signal.context.get('bullish_score', 0)
                    df.iloc[last_idx, df.columns.get_loc('lit_bearish_score')] = lit_signal.context.get('bearish_score', 0)
        
        except Exception as e:
            log.warning(f"Error en características LIT rápidas: {str(e)}")
        
        # Características derivadas básicas
        df['lit_signal_momentum'] = df['lit_signal'].rolling(5).mean()
        df['lit_score_ratio'] = np.where(
            df['lit_bearish_score'] != 0,
            df['lit_bullish_score'] / df['lit_bearish_score'],
            df['lit_bullish_score']
        )
        
        return df
    
    def _create_basic_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea características de interacción básicas.
        
        Args:
            df: DataFrame con datos.
            
        Returns:
            pd.DataFrame: DataFrame con interacciones.
        """
        try:
            # Interacciones RSI + LIT
            if 'rsi' in df.columns:
                df['rsi_lit_signal'] = df['rsi'] * df['lit_signal']
                df['rsi_overbought_lit_sell'] = ((df['rsi'] > 70) & (df['lit_signal'] == -1)).astype(int)
                df['rsi_oversold_lit_buy'] = ((df['rsi'] < 30) & (df['lit_signal'] == 1)).astype(int)
            
            # Interacciones MACD + LIT
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                df['macd_lit_alignment'] = np.sign(df['macd'] - df['macd_signal']) * df['lit_signal']
                df['macd_bullish_lit_buy'] = ((df['macd'] > df['macd_signal']) & (df['lit_signal'] == 1)).astype(int)
            
            # Interacciones Bollinger + LIT
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                df['bb_upper_break_lit'] = ((df['close'] > df['bb_upper']) & (df['lit_signal'] == 1)).astype(int)
                df['bb_lower_break_lit'] = ((df['close'] < df['bb_lower']) & (df['lit_signal'] == -1)).astype(int)
            
        except Exception as e:
            log.warning(f"Error en interacciones básicas: {str(e)}")
        
        return df
    
    def _prepare_prediction_data(self, features_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepara los datos para predicción.
        
        Args:
            features_data: DataFrame con características.
            
        Returns:
            Optional[pd.DataFrame]: Datos preparados o None si hay error.
        """
        try:
            # Seleccionar solo las características del modelo
            available_features = [f for f in self.feature_names if f in features_data.columns]
            
            if len(available_features) < len(self.feature_names) * 0.8:  # Al menos 80% de características
                log.warning(f"Características faltantes: {len(available_features)}/{len(self.feature_names)}")
                missing = set(self.feature_names) - set(available_features)
                log.warning(f"Faltantes: {list(missing)[:10]}...")  # Mostrar solo las primeras 10
            
            # Usar características disponibles
            X = features_data[available_features].fillna(0)
            
            # Agregar características faltantes con valor 0
            for feature in self.feature_names:
                if feature not in X.columns:
                    X[feature] = 0
            
            # Reordenar columnas según el orden del entrenamiento
            X = X[self.feature_names]
            
            # Normalizar con el scaler del entrenamiento
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
            
            return X_scaled
            
        except Exception as e:
            log.error(f"Error preparando datos para predicción: {str(e)}")
            return None
    
    def _map_prediction_to_signal(self, prediction: int) -> str:
        """
        Mapea predicción numérica a señal de trading.
        
        Args:
            prediction: Predicción numérica (0, 1, 2).
            
        Returns:
            str: Señal de trading.
        """
        signal_map = {0: 'sell', 1: 'hold', 2: 'buy'}
        return signal_map.get(prediction, 'hold')
    
    def _get_default_prediction(self) -> Dict[str, Any]:
        """
        Retorna predicción por defecto en caso de error.
        
        Returns:
            Dict[str, Any]: Predicción por defecto.
        """
        return {
            'signal': 'hold',
            'prediction_raw': 1,
            'confidence': 0.33,
            'probabilities': {'sell': 0.33, 'hold': 0.34, 'buy': 0.33},
            'timestamp': datetime.now().isoformat(),
            'error': True
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información del modelo cargado.
        
        Returns:
            Dict[str, Any]: Información del modelo.
        """
        if not self.is_loaded:
            return {'status': 'not_loaded'}
        
        return {
            'status': 'loaded',
            'model_path': self.model_path,
            'model_type': type(self.model).__name__,
            'features_count': len(self.feature_names),
            'model_config': self.model_config,
            'training_config': self.training_config,
            'predictions_made': len(self.prediction_history),
            'last_prediction': self.last_prediction
        }
    
    def get_prediction_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtiene historial de predicciones.
        
        Args:
            limit: Número máximo de predicciones a retornar.
            
        Returns:
            List[Dict[str, Any]]: Historial de predicciones.
        """
        return self.prediction_history[-limit:] if self.prediction_history else []
    
    def clear_cache(self):
        """Limpia el cache de características y datos."""
        self._feature_cache.clear()
        self._data_cache.clear()
        log.info("Cache limpiado")
    
    def validate_prediction(self, prediction: Dict[str, Any]) -> bool:
        """
        Valida que una predicción sea válida.
        
        Args:
            prediction: Diccionario de predicción.
            
        Returns:
            bool: True si es válida.
        """
        required_keys = ['signal', 'prediction_raw', 'timestamp']
        
        if not all(key in prediction for key in required_keys):
            return False
        
        if prediction['signal'] not in ['buy', 'sell', 'hold']:
            return False
        
        if prediction['prediction_raw'] not in [0, 1, 2]:
            return False
        
        return True


class RealtimePredictor:
    """
    Predictor en tiempo real con integración MT5 y gestión de estado.
    
    Maneja predicciones continuas y se integra con el flujo del bot.
    """
    
    def __init__(self, model_path: str = "models/lit_ml_model.pkl",
                 update_interval: int = 60):
        """
        Inicializa el predictor en tiempo real.
        
        Args:
            model_path: Ruta al modelo entrenado.
            update_interval: Intervalo de actualización en segundos.
        """
        self.predictor = LITMLPredictor(model_path)
        self.update_interval = update_interval
        self.is_running = False
        self.last_update = None
        
        # Configuración de símbolos
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        self.timeframe = '1h'
        
        # Estado de predicciones
        self.current_predictions = {}
        self.prediction_callbacks = []
        
        log.info(f"RealtimePredictor inicializado - Intervalo: {update_interval}s")
    
    def start(self) -> bool:
        """
        Inicia el predictor en tiempo real.
        
        Returns:
            bool: True si se inició exitosamente.
        """
        if not self.predictor.load_model():
            log.error("No se pudo cargar el modelo")
            return False
        
        self.is_running = True
        log.info("Predictor en tiempo real iniciado")
        
        return True
    
    def stop(self):
        """Detiene el predictor en tiempo real."""
        self.is_running = False
        log.info("Predictor en tiempo real detenido")
    
    def update_predictions(self) -> Dict[str, Dict[str, Any]]:
        """
        Actualiza predicciones para todos los símbolos.
        
        Returns:
            Dict[str, Dict[str, Any]]: Predicciones por símbolo.
        """
        if not self.is_running:
            return {}
        
        predictions = {}
        
        for symbol in self.symbols:
            try:
                prediction = self.predictor.predict_realtime(
                    symbol=symbol,
                    timeframe=self.timeframe
                )
                
                predictions[symbol] = prediction
                
                # Ejecutar callbacks si hay cambio de señal
                if symbol in self.current_predictions:
                    old_signal = self.current_predictions[symbol].get('signal')
                    new_signal = prediction.get('signal')
                    
                    if old_signal != new_signal:
                        self._execute_callbacks(symbol, prediction, old_signal)
                
            except Exception as e:
                log.error(f"Error actualizando predicción para {symbol}: {str(e)}")
                continue
        
        self.current_predictions = predictions
        self.last_update = datetime.now()
        
        return predictions
    
    def get_current_predictions(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene las predicciones actuales.
        
        Returns:
            Dict[str, Dict[str, Any]]: Predicciones actuales.
        """
        return self.current_predictions.copy()
    
    def add_prediction_callback(self, callback):
        """
        Agrega callback para cambios de predicción.
        
        Args:
            callback: Función callback(symbol, prediction, old_signal).
        """
        self.prediction_callbacks.append(callback)
    
    def _execute_callbacks(self, symbol: str, prediction: Dict[str, Any], old_signal: str):
        """
        Ejecuta callbacks para cambios de señal.
        
        Args:
            symbol: Símbolo que cambió.
            prediction: Nueva predicción.
            old_signal: Señal anterior.
        """
        for callback in self.prediction_callbacks:
            try:
                callback(symbol, prediction, old_signal)
            except Exception as e:
                log.error(f"Error en callback: {str(e)}")


# Funciones de utilidad para integración

def load_and_predict(model_path: str, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Función de utilidad para cargar modelo y predecir rápidamente.
    
    Args:
        model_path: Ruta al modelo.
        data: Datos para predicción.
        
    Returns:
        Dict[str, Any]: Resultado de predicción.
    """
    predictor = LITMLPredictor(model_path)
    
    if not predictor.load_model():
        return predictor._get_default_prediction()
    
    return predictor.predict_single(data)


def batch_predict_for_backtesting(model_path: str, data: pd.DataFrame, 
                                 window_size: int = 50) -> pd.DataFrame:
    """
    Función de utilidad para backtesting con predicciones en lote.
    
    Args:
        model_path: Ruta al modelo.
        data: Datos históricos.
        window_size: Tamaño de ventana.
        
    Returns:
        pd.DataFrame: DataFrame con predicciones.
    """
    predictor = LITMLPredictor(model_path)
    
    if not predictor.load_model():
        log.error("No se pudo cargar el modelo para backtesting")
        return pd.DataFrame()
    
    predictions = predictor.predict_batch(data, window_size)
    
    if not predictions:
        return pd.DataFrame()
    
    # Convertir a DataFrame
    df_predictions = pd.DataFrame(predictions)
    
    # Agregar información de precios
    if 'index' in df_predictions.columns:
        df_predictions['price'] = [data.iloc[idx]['close'] for idx in df_predictions['index']]
        df_predictions['timestamp'] = [data.index[idx] for idx in df_predictions['index']]
    
    return df_predictions


def create_prediction_summary(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Crea resumen de predicciones para análisis.
    
    Args:
        predictions: Lista de predicciones.
        
    Returns:
        Dict[str, Any]: Resumen estadístico.
    """
    if not predictions:
        return {}
    
    signals = [p['signal'] for p in predictions]
    confidences = [p.get('confidence', 0) for p in predictions]
    
    summary = {
        'total_predictions': len(predictions),
        'signal_distribution': {
            'buy': signals.count('buy'),
            'sell': signals.count('sell'),
            'hold': signals.count('hold')
        },
        'signal_percentages': {
            'buy': (signals.count('buy') / len(signals)) * 100,
            'sell': (signals.count('sell') / len(signals)) * 100,
            'hold': (signals.count('hold') / len(signals)) * 100
        },
        'confidence_stats': {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        },
        'high_confidence_predictions': len([c for c in confidences if c > 0.7]),
        'low_confidence_predictions': len([c for c in confidences if c < 0.4])
    }
    
    return summary