"""
Módulo de entrenamiento del modelo XGBoost.

Este módulo maneja el entrenamiento del modelo de Machine Learning
con validación cruzada, optimización de hiperparámetros y guardado del modelo.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
from typing import Dict, Tuple, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from src.models.feature_engineering import FeatureEngineer
from src.core.config import config
from src.utils.logger import log


class ModelTrainer:
    """
    Entrenador del modelo XGBoost para señales de trading.
    
    Maneja el entrenamiento, validación y optimización del modelo
    de clasificación para generar señales de trading.
    """
    
    def __init__(self):
        """Inicializa el entrenador del modelo."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        
        # Configuración del modelo
        self.model_params = {
            'objective': 'multi:softprob',
            'num_class': 3,  # buy, hold, sell
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
        
        log.info("ModelTrainer inicializado")
    
    def train(self, 
              data: pd.DataFrame, 
              target_column: Optional[str] = None,
              test_size: float = 0.2,
              optimize_hyperparams: bool = True) -> Dict[str, Any]:
        """
        Entrena el modelo XGBoost.
        
        Args:
            data: DataFrame con datos de entrenamiento.
            target_column: Nombre de la columna objetivo.
            test_size: Proporción de datos para prueba.
            optimize_hyperparams: Si optimizar hiperparámetros.
            
        Returns:
            Dict[str, Any]: Métricas de entrenamiento y validación.
        """
        log.info("Iniciando entrenamiento del modelo")
        
        # Preparar características
        log.info("Generando características...")
        features_data = self.feature_engineer.create_features(data)
        
        # Preparar dataset
        X, y = self.feature_engineer.prepare_ml_dataset(features_data, target_column)
        
        if len(X) == 0:
            raise ValueError("No hay datos suficientes para entrenar el modelo")
        
        log.info(f"Dataset preparado: {len(X)} muestras, {len(X.columns)} características")
        
        # División temporal para datos de series temporales
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Normalizar características
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
        
        # Optimizar hiperparámetros si se solicita
        if optimize_hyperparams:
            log.info("Optimizando hiperparámetros...")
            best_params = self._optimize_hyperparameters(X_train_scaled, y_train)
            self.model_params.update(best_params)
        
        # Entrenar modelo
        log.info("Entrenando modelo XGBoost...")
        self.model = xgb.XGBClassifier(**self.model_params)
        
        # Entrenar modelo
        self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        # Validación cruzada
        cv_scores = self._cross_validate(X_train_scaled, y_train)
        
        # Predicciones
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Métricas
        train_metrics = self._calculate_metrics(y_train, y_pred_train, "Entrenamiento")
        test_metrics = self._calculate_metrics(y_test, y_pred_test, "Prueba")
        
        # Importancia de características
        feature_importance = self._get_feature_importance(X_train.columns)
        
        # Resultados
        results = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance,
            'model_params': self.model_params,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        log.info(f"Entrenamiento completado. Accuracy de prueba: {test_metrics['accuracy']:.4f}")
        
        return results
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Optimiza los hiperparámetros del modelo usando GridSearchCV.
        
        Args:
            X: Características de entrenamiento.
            y: Variable objetivo.
            
        Returns:
            Dict[str, Any]: Mejores hiperparámetros encontrados.
        """
        param_grid = {
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [50, 100, 150],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Usar TimeSeriesSplit para validación cruzada temporal
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Modelo base
        base_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # Búsqueda en cuadrícula
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        log.info(f"Mejores parámetros encontrados: {grid_search.best_params_}")
        log.info(f"Mejor score CV: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def _cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Realiza validación cruzada temporal.
        
        Args:
            X: Características.
            y: Variable objetivo.
            
        Returns:
            Dict[str, float]: Métricas de validación cruzada.
        """
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Diferentes métricas
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
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, dataset_name: str) -> Dict[str, float]:
        """
        Calcula métricas de evaluación.
        
        Args:
            y_true: Valores verdaderos.
            y_pred: Predicciones.
            dataset_name: Nombre del dataset.
            
        Returns:
            Dict[str, float]: Métricas calculadas.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        
        # Log del reporte de clasificación
        log.info(f"\nReporte de clasificación - {dataset_name}:")
        log.info(f"\n{classification_report(y_true, y_pred, target_names=['Sell', 'Hold', 'Buy'])}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Obtiene la importancia de las características.
        
        Args:
            feature_names: Nombres de las características.
            
        Returns:
            Dict[str, float]: Importancia de cada característica.
        """
        if not self.is_trained:
            return {}
        
        importances = self.model.feature_importances_
        feature_importance = dict(zip(feature_names, importances))
        
        # Ordenar por importancia
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        log.info("Top 10 características más importantes:")
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            log.info(f"{i+1:2d}. {feature:30s}: {importance:.4f}")
        
        return dict(sorted_features)
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            data: DataFrame con datos para predicción.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (Predicciones, Probabilidades).
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Generar características
        features_data = self.feature_engineer.create_features(data)
        X, _ = self.feature_engineer.prepare_ml_dataset(features_data)
        
        # Normalizar
        X_scaled = pd.DataFrame(
            self.scaler.transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # Predicciones
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def predict_single(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Realiza predicción para una sola muestra.
        
        Args:
            data: DataFrame con datos para predicción.
            
        Returns:
            Dict[str, Any]: Resultado de la predicción.
        """
        predictions, probabilities = self.predict(data)
        
        if len(predictions) == 0:
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'probabilities': {'sell': 0.33, 'hold': 0.34, 'buy': 0.33}
            }
        
        # Última predicción
        last_pred = predictions[-1]
        last_probs = probabilities[-1]
        
        # Mapear predicción a señal
        signal_map = {-1: 'sell', 0: 'hold', 1: 'buy'}
        signal = signal_map.get(last_pred, 'hold')
        
        # Confianza como probabilidad máxima
        confidence = max(last_probs)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'probabilities': {
                'sell': last_probs[0],
                'hold': last_probs[1],
                'buy': last_probs[2]
            }
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Guarda el modelo entrenado.
        
        Args:
            filepath: Ruta donde guardar el modelo.
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_params': self.model_params,
            'feature_engineer': self.feature_engineer
        }
        
        joblib.dump(model_data, filepath)
        log.info(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Carga un modelo previamente entrenado.
        
        Args:
            filepath: Ruta del modelo a cargar.
        """
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_params = model_data['model_params']
            self.feature_engineer = model_data.get('feature_engineer', FeatureEngineer())
            self.is_trained = True
            
            log.info(f"Modelo cargado desde: {filepath}")
            
        except Exception as e:
            log.error(f"Error cargando modelo: {str(e)}")
            raise
    
    def retrain(self, 
                new_data: pd.DataFrame, 
                retrain_ratio: float = 0.3) -> Dict[str, Any]:
        """
        Reentrena el modelo con nuevos datos.
        
        Args:
            new_data: Nuevos datos para reentrenamiento.
            retrain_ratio: Proporción de datos nuevos vs históricos.
            
        Returns:
            Dict[str, Any]: Métricas del reentrenamiento.
        """
        if not self.is_trained:
            raise ValueError("El modelo debe estar entrenado antes de reentrenar")
        
        log.info("Iniciando reentrenamiento del modelo")
        
        # Por simplicidad, reentrenar con todos los datos nuevos
        # En producción, se podría combinar con datos históricos
        return self.train(new_data, optimize_hyperparams=False)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el modelo entrenado.
        
        Returns:
            Dict[str, Any]: Información del modelo.
        """
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        return {
            'status': 'trained',
            'model_type': 'XGBoost',
            'num_features': self.model.n_features_in_,
            'num_classes': len(self.model.classes_),
            'parameters': self.model_params,
            'training_score': getattr(self.model, 'best_score', None)
        }
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evalúa el modelo con datos de prueba.
        
        Args:
            test_data: Datos de prueba.
            
        Returns:
            Dict[str, Any]: Métricas de evaluación.
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Preparar datos
        features_data = self.feature_engineer.create_features(test_data)
        X, y = self.feature_engineer.prepare_ml_dataset(features_data)
        
        # Predicciones
        predictions, probabilities = self.predict(test_data)
        
        # Asegurar que las longitudes coincidan
        min_length = min(len(y), len(predictions))
        y_test = y.iloc[:min_length]
        y_pred = predictions[:min_length]
        
        # Calcular métricas
        metrics = self._calculate_metrics(y_test, y_pred, "Evaluación")
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'probabilities': probabilities,
            'samples_evaluated': len(y_test)
        }