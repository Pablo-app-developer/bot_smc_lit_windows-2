#!/usr/bin/env python3
"""
Script de Entrenamiento del Modelo LIT + ML - Versión Profesional.

Este script entrena un modelo XGBoost que combina:
- Indicadores técnicos tradicionales
- Señales LIT (Liquidity + Inducement Theory)
- Features de velas y patrones
- Análisis de volumen y momentum

Uso:
    python scripts/train_model.py --symbol EURUSD=X --timeframe 1h
    python scripts/train_model.py --target-method hybrid --output models/my_model.pkl
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.models.feature_engineering import FeatureEngineer
from src.strategies.lit_detector import LITDetector, SignalType
from src.data.indicators import TechnicalIndicators
from src.utils.logger import log


class LITMLTrainer:
    """Entrenador profesional del modelo LIT + ML."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Inicializa el entrenador."""
        self.feature_engineer = FeatureEngineer()
        self.lit_detector = LITDetector()
        self.data_loader = DataLoader()
        
        # Configuración del modelo
        self.model_config = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'verbosity': 0
        }
        
        # Configuración de entrenamiento
        self.training_config = {
            'test_size': 0.2,
            'cv_folds': 3,
            'min_samples': 200,  # Reducido para pruebas
            'max_features': 30   # Reducido para pruebas
        }
        
        # Cargar configuración personalizada
        if config_path and os.path.exists(config_path):
            self._load_custom_config(config_path)
        
        # Inicializar componentes
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.training_metrics = {}
        
        log.info("LITMLTrainer inicializado")
    
    def _load_custom_config(self, config_path: str):
        """Carga configuración personalizada."""
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            
            if 'model' in custom_config:
                self.model_config.update(custom_config['model'])
            if 'training' in custom_config:
                self.training_config.update(custom_config['training'])
            
            log.info(f"Configuración cargada desde {config_path}")
        except Exception as e:
            log.warning(f"Error cargando configuración: {e}")
    
    def load_and_prepare_data(self, symbol: str = "EURUSD=X", timeframe: str = "1h") -> pd.DataFrame:
        """Carga y prepara los datos."""
        log.info(f"Cargando datos: {symbol} {timeframe}")
        
        # Calcular períodos para 2 años de datos
        periods_map = {
            '1h': 24 * 365 * 2,    # 2 años de datos horarios
            '4h': 6 * 365 * 2,     # 2 años de datos de 4h
            '1d': 365 * 2,         # 2 años de datos diarios
            '1m': 60 * 24 * 30,    # 1 mes de datos de 1m (limitado)
            '5m': 12 * 24 * 90,    # 3 meses de datos de 5m
            '15m': 4 * 24 * 180,   # 6 meses de datos de 15m
            '30m': 2 * 24 * 365    # 1 año de datos de 30m
        }
        
        periods = periods_map.get(timeframe, 1000)
        
        # Cargar datos usando el método público
        data = self.data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            periods=periods
        )
        
        if data.empty:
            raise ValueError("No se pudieron cargar datos")
        
        log.info(f"Datos cargados: {len(data)} velas")
        
        if len(data) < self.training_config['min_samples']:
            raise ValueError(f"Datos insuficientes: {len(data)}")
        
        return data
    
    def create_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crea características mejoradas combinando indicadores técnicos y LIT."""
        log.info("Creando características mejoradas...")
        
        # 1. Indicadores técnicos básicos
        df = TechnicalIndicators.calculate_all_indicators(data.copy())
        log.info("✅ Indicadores técnicos calculados")
        
        # 2. Características básicas de ingeniería
        df = self.feature_engineer.create_features(df)
        log.info("✅ Características básicas creadas")
        
        # 3. Características LIT mejoradas (procesamiento optimizado)
        df = self._create_lit_features_optimized(df)
        log.info("✅ Señales LIT creadas")
        
        # 4. Características de interacción
        df = self._create_interaction_features(df)
        log.info("✅ Características de interacción creadas")
        
        log.info(f"Total características: {len(df.columns)}")
        return df
    
    def _create_lit_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características LIT con procesamiento optimizado."""
        # Inicializar columnas LIT
        df['lit_signal'] = 0
        df['lit_confidence'] = 0.0
        df['lit_events_count'] = 0
        df['lit_bullish_score'] = 0.0
        df['lit_bearish_score'] = 0.0
        
        # Procesar cada 10 velas para eficiencia
        lookback = 50
        for i in range(lookback, len(df), 10):
            try:
                window_data = df.iloc[:i+1]
                lit_signal = self.lit_detector.analyze(window_data)
                
                # Mapear señal
                signal_value = 0
                if lit_signal.signal == SignalType.BUY:
                    signal_value = 1
                elif lit_signal.signal == SignalType.SELL:
                    signal_value = -1
                
                # Aplicar a las próximas 10 velas
                end_idx = min(i + 10, len(df))
                df.iloc[i:end_idx, df.columns.get_loc('lit_signal')] = signal_value
                df.iloc[i:end_idx, df.columns.get_loc('lit_confidence')] = lit_signal.confidence
                df.iloc[i:end_idx, df.columns.get_loc('lit_events_count')] = len(lit_signal.events)
                
                if lit_signal.context:
                    bullish_score = lit_signal.context.get('bullish_score', 0)
                    bearish_score = lit_signal.context.get('bearish_score', 0)
                    df.iloc[i:end_idx, df.columns.get_loc('lit_bullish_score')] = bullish_score
                    df.iloc[i:end_idx, df.columns.get_loc('lit_bearish_score')] = bearish_score
                
            except Exception as e:
                log.warning(f"Error en LIT análisis índice {i}: {str(e)}")
                continue
        
        # Características derivadas
        df['lit_signal_momentum'] = df['lit_signal'].rolling(5).mean()
        df['lit_confidence_trend'] = df['lit_confidence'].diff()
        df['lit_score_ratio'] = np.where(
            df['lit_bearish_score'] != 0,
            df['lit_bullish_score'] / df['lit_bearish_score'],
            df['lit_bullish_score']
        )
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características de interacción entre indicadores y LIT."""
        # Interacciones RSI + LIT
        if 'rsi' in df.columns:
            df['rsi_lit_signal'] = df['rsi'] * df['lit_signal']
            df['rsi_overbought_lit_sell'] = ((df['rsi'] > 70) & (df['lit_signal'] == -1)).astype(int)
            df['rsi_oversold_lit_buy'] = ((df['rsi'] < 30) & (df['lit_signal'] == 1)).astype(int)
        
        # Interacciones MACD + LIT
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_lit_alignment'] = np.sign(df['macd'] - df['macd_signal']) * df['lit_signal']
            df['macd_bullish_lit_buy'] = ((df['macd'] > df['macd_signal']) & (df['lit_signal'] == 1)).astype(int)
        
        # Interacciones Bollinger Bands + LIT
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            df['bb_upper_break_lit'] = ((df['close'] > df['bb_upper']) & (df['lit_signal'] == 1)).astype(int)
            df['bb_lower_break_lit'] = ((df['close'] < df['bb_lower']) & (df['lit_signal'] == -1)).astype(int)
        
        # Interacciones de volumen + LIT
        if 'volume_ratio_sma' in df.columns:
            df['high_volume_lit_signal'] = ((df['volume_ratio_sma'] > 1.5) & (df['lit_signal'] != 0)).astype(int)
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame, method: str = "hybrid") -> pd.Series:
        """Crea la variable objetivo."""
        log.info(f"Creando target con método: {method}")
        
        if method == "future_returns":
            # Basado en retornos futuros
            future_returns = df['close'].shift(-5) / df['close'] - 1
            target = np.where(future_returns > 0.002, 1,
                            np.where(future_returns < -0.002, -1, 0))
        
        elif method == "lit_signals":
            # Basado en señales LIT
            target = df['lit_signal'].values
        
        elif method == "hybrid":
            # Combinación de retornos futuros y señales LIT
            future_returns = df['close'].shift(-5) / df['close'] - 1
            returns_signal = np.where(future_returns > 0.002, 1,
                                    np.where(future_returns < -0.002, -1, 0))
            lit_signal = df['lit_signal'].values
            
            # Combinación: acuerdo entre ambos o retornos fuertes con LIT confiable
            target = np.where(
                (returns_signal == lit_signal) & (returns_signal != 0),
                returns_signal,  # Acuerdo
                np.where(
                    (abs(future_returns) > 0.004) & (df['lit_confidence'] > 0.7),
                    returns_signal,  # Retornos fuertes + LIT confiable
                    0  # Hold
                )
            )
        
        else:
            raise ValueError(f"Método no soportado: {method}")
        
        # Convertir a etiquetas categóricas (0=sell, 1=hold, 2=buy)
        target_series = pd.Series(target, index=df.index)
        target_series = target_series.map({-1: 0, 0: 1, 1: 2})
        
        # Log distribución
        distribution = target_series.value_counts().to_dict()
        log.info(f"Distribución del target: {distribution}")
        
        return target_series
    
    def select_features(self, df: pd.DataFrame, target: pd.Series) -> List[str]:
        """Selecciona las mejores características usando XGBoost."""
        log.info("Seleccionando características...")
        
        # Obtener columnas numéricas
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_columns = ['target', 'signal', 'timestamp']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Modelo temporal para importancia
        temp_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            max_depth=3,
            n_estimators=50,
            random_state=42,
            verbosity=0
        )
        
        # Preparar datos
        X_temp = df[feature_columns].fillna(0)
        y_temp = target.fillna(1)  # Hold por defecto
        
        # Asegurar misma longitud
        min_length = min(len(X_temp), len(y_temp))
        X_temp = X_temp.iloc[:min_length]
        y_temp = y_temp.iloc[:min_length]
        
        # Entrenar modelo temporal
        temp_model.fit(X_temp, y_temp)
        
        # Obtener importancias
        importances = temp_model.feature_importances_
        feature_importance = list(zip(feature_columns, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Seleccionar top características
        max_features = self.training_config['max_features']
        selected_features = [feat[0] for feat in feature_importance[:max_features]]
        
        log.info(f"Características seleccionadas: {len(selected_features)}")
        log.info(f"Top 10: {selected_features[:10]}")
        
        return selected_features
    
    def train_model(self, data: pd.DataFrame, target_method: str = "hybrid", 
                   save_path: str = "models/lit_ml_model.pkl") -> Dict[str, Any]:
        """Entrena el modelo completo."""
        log.info("🚀 Iniciando entrenamiento del modelo LIT + ML")
        
        # 1. Crear características mejoradas
        df_features = self.create_enhanced_features(data)
        
        # 2. Crear variable objetivo
        target = self.create_target_variable(df_features, method=target_method)
        
        # 3. Limpiar datos
        df_clean = df_features.dropna()
        target_clean = target.loc[df_clean.index]
        
        if len(df_clean) < self.training_config['min_samples']:
            raise ValueError(f"Datos insuficientes después de limpieza: {len(df_clean)}")
        
        log.info(f"Datos limpios: {len(df_clean)} muestras")
        
        # 4. Seleccionar características
        selected_features = self.select_features(df_clean, target_clean)
        self.feature_names = selected_features
        
        X = df_clean[selected_features]
        y = target_clean
        
        # 5. División temporal de datos
        test_size = self.training_config['test_size']
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        log.info(f"División - Train: {len(X_train)}, Test: {len(X_test)}")
        
        # 6. Normalizar características
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # 7. Entrenar modelo
        log.info("🎓 Entrenando modelo XGBoost...")
        self.model = xgb.XGBClassifier(**self.model_config)
        
        # Entrenar modelo
        self.model.fit(X_train_scaled, y_train)
        
        # 8. Evaluación
        log.info("📊 Evaluando modelo...")
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_metrics = self._evaluate_predictions(y_train, train_pred, "Entrenamiento")
        test_metrics = self._evaluate_predictions(y_test, test_pred, "Prueba")
        
        # 9. Validación cruzada
        cv_scores = self._cross_validate(X_train_scaled, y_train)
        
        # 10. Importancia de características
        feature_importance = self._get_feature_importance()
        
        # 11. Guardar modelo
        log.info(f"💾 Guardando modelo en {save_path}...")
        self._save_model(save_path)
        
        # 12. Compilar resultados
        results = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'feature_names': self.feature_names,
            'data_info': {
                'total_samples': len(df_features),
                'clean_samples': len(df_clean),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features_count': len(selected_features)
            }
        }
        
        self.training_metrics = results
        
        log.info("✅ Entrenamiento completado exitosamente!")
        log.info(f"📊 Test Accuracy: {test_metrics['accuracy']:.4f}")
        log.info(f"📊 Test F1-Score: {test_metrics['f1_score']:.4f}")
        
        return results
    
    def _cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Realiza validación cruzada temporal."""
        tscv = TimeSeriesSplit(n_splits=self.training_config['cv_folds'])
        
        cv_accuracy = cross_val_score(self.model, X, y, cv=tscv, scoring='accuracy')
        cv_precision = cross_val_score(self.model, X, y, cv=tscv, scoring='precision_macro')
        cv_recall = cross_val_score(self.model, X, y, cv=tscv, scoring='recall_macro')
        cv_f1 = cross_val_score(self.model, X, y, cv=tscv, scoring='f1_macro')
        
        return {
            'accuracy_mean': cv_accuracy.mean(),
            'accuracy_std': cv_accuracy.std(),
            'precision_mean': cv_precision.mean(),
            'precision_std': cv_precision.std(),
            'recall_mean': cv_recall.mean(),
            'recall_std': cv_recall.std(),
            'f1_mean': cv_f1.mean(),
            'f1_std': cv_f1.std()
        }
    
    def _evaluate_predictions(self, y_true: pd.Series, y_pred: np.ndarray, dataset_name: str) -> Dict[str, float]:
        """Evalúa predicciones y calcula métricas."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Log métricas
        log.info(f"\n📊 Métricas de {dataset_name}:")
        log.info(f"  Accuracy:  {accuracy:.4f}")
        log.info(f"  Precision: {precision:.4f}")
        log.info(f"  Recall:    {recall:.4f}")
        log.info(f"  F1-Score:  {f1:.4f}")
        
        # Log reporte detallado
        log.info(f"\n{classification_report(y_true, y_pred, target_names=['Sell', 'Hold', 'Buy'], zero_division=0)}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Obtiene importancia de características."""
        importances = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importances))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        log.info("\n🏆 Top 15 características más importantes:")
        for i, (feature, importance) in enumerate(sorted_features[:15]):
            log.info(f"{i+1:2d}. {feature:35s}: {importance:.4f}")
        
        return dict(sorted_features)
    
    def _save_model(self, filepath: str):
        """Guarda el modelo entrenado."""
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'training_metrics': self.training_metrics,
            'feature_engineer': self.feature_engineer,
            'lit_detector': self.lit_detector,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        log.info(f"✅ Modelo guardado exitosamente en: {filepath}")
        
        # Guardar métricas en JSON
        metrics_path = filepath.replace('.pkl', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2, default=str)
        log.info(f"✅ Métricas guardadas en: {metrics_path}")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description="Entrenador del Modelo LIT + ML")
    parser.add_argument('--symbol', default='EURUSD=X', help='Símbolo a entrenar (default: EURUSD=X)')
    parser.add_argument('--timeframe', default='1h', help='Marco temporal (default: 1h)')
    parser.add_argument('--target-method', default='hybrid', 
                       choices=['future_returns', 'lit_signals', 'hybrid'],
                       help='Método para crear target (default: hybrid)')
    parser.add_argument('--config', help='Archivo de configuración personalizada (JSON)')
    parser.add_argument('--output', default='models/lit_ml_model.pkl', 
                       help='Ruta de salida del modelo (default: models/lit_ml_model.pkl)')
    
    args = parser.parse_args()
    
    try:
        log.info("🚀 Iniciando script de entrenamiento LIT + ML")
        log.info(f"Parámetros: symbol={args.symbol}, timeframe={args.timeframe}, target={args.target_method}")
        
        # Inicializar entrenador
        trainer = LITMLTrainer(config_path=args.config)
        
        # Cargar datos
        data = trainer.load_and_prepare_data(
            symbol=args.symbol,
            timeframe=args.timeframe
        )
        
        # Entrenar modelo
        results = trainer.train_model(
            data=data,
            target_method=args.target_method,
            save_path=args.output
        )
        
        # Mostrar resumen final
        print("\n" + "="*60)
        print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*60)
        print(f"📊 Accuracy de prueba:  {results['test_metrics']['accuracy']:.4f}")
        print(f"📊 F1-Score de prueba:  {results['test_metrics']['f1_score']:.4f}")
        print(f"📊 Precisión de prueba: {results['test_metrics']['precision']:.4f}")
        print(f"📊 Recall de prueba:    {results['test_metrics']['recall']:.4f}")
        print(f"📊 CV Accuracy:         {results['cv_scores']['accuracy_mean']:.4f} ± {results['cv_scores']['accuracy_std']:.4f}")
        print(f"📊 CV F1-Score:         {results['cv_scores']['f1_mean']:.4f} ± {results['cv_scores']['f1_std']:.4f}")
        print(f"💾 Modelo guardado en:  {args.output}")
        print(f"📈 Muestras de entrenamiento: {results['data_info']['train_samples']}")
        print(f"📈 Muestras de prueba:        {results['data_info']['test_samples']}")
        print(f"🔧 Características usadas:   {results['data_info']['features_count']}")
        print("="*60)
        
    except Exception as e:
        log.error(f"❌ Error durante el entrenamiento: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main() 