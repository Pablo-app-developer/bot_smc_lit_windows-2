#!/usr/bin/env python3
"""
Pruebas unitarias para el modelo ML.

Valida que el modelo ML genera predicciones válidas y consistentes,
y que el predictor LIT + ML funciona correctamente.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import tempfile
import joblib
from pathlib import Path
from unittest.mock import Mock, patch

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.predictor import LITMLPredictor
from src.models.model_trainer import ModelTrainer
from src.models.feature_engineering import FeatureEngineer


class TestMLModel:
    """Pruebas para el modelo ML y predictor."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture que crea datos de muestra para las pruebas."""
        dates = pd.date_range('2024-01-01', periods=200, freq='1H')
        
        # Crear datos OHLCV realistas
        data = []
        base_price = 1.0800
        
        for i in range(200):
            # Simular movimiento de precios con tendencia y ruido
            trend = 0.0001 * np.sin(i / 20)  # Tendencia sinusoidal
            noise = np.random.normal(0, 0.0002)
            price = base_price + trend + noise
            
            high = price + np.random.uniform(0, 0.0003)
            low = price - np.random.uniform(0, 0.0003)
            open_price = data[i-1]['close'] if i > 0 else price
            volume = np.random.randint(1000, 5000)
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    @pytest.fixture
    def feature_engineer(self):
        """Fixture que crea un ingeniero de características."""
        return FeatureEngineer()
    
    @pytest.fixture
    def mock_model_file(self):
        """Fixture que crea un archivo de modelo mock."""
        # Crear un modelo mock simple
        mock_model_data = {
            'model': Mock(),
            'scaler': Mock(),
            'feature_columns': ['feature_1', 'feature_2', 'feature_3'],
            'model_info': {
                'model_type': 'XGBoost',
                'version': '1.0',
                'features_count': 3,
                'training_date': '2024-01-01'
            }
        }
        
        # Configurar el mock del modelo
        mock_model_data['model'].predict_proba = Mock(return_value=np.array([[0.2, 0.6, 0.2]]))
        mock_model_data['model'].predict = Mock(return_value=np.array([1]))
        
        # Configurar el mock del scaler
        mock_model_data['scaler'].transform = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
        
        # Guardar en archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            joblib.dump(mock_model_data, f.name)
            return f.name
    
    def test_feature_engineer_initialization(self, feature_engineer):
        """Prueba que el ingeniero de características se inicializa correctamente."""
        assert feature_engineer is not None
        assert hasattr(feature_engineer, 'create_features')
        assert hasattr(feature_engineer, 'prepare_ml_dataset')
    
    def test_feature_creation(self, feature_engineer, sample_data):
        """Prueba creación de características."""
        features = feature_engineer.create_features(sample_data)
        
        # Verificar que retorna un DataFrame
        assert isinstance(features, pd.DataFrame)
        
        # Debe tener más columnas que los datos originales
        assert len(features.columns) > len(sample_data.columns)
        
        # Verificar que no hay valores infinitos
        assert not np.isinf(features.select_dtypes(include=[np.number])).any().any()
        
        # Verificar que el índice se mantiene
        assert len(features) <= len(sample_data)
    
    def test_ml_dataset_preparation(self, feature_engineer, sample_data):
        """Prueba preparación de dataset para ML."""
        features_df, target = feature_engineer.prepare_ml_dataset(sample_data)
        
        # Verificar que retorna DataFrames
        assert isinstance(features_df, pd.DataFrame)
        assert isinstance(target, pd.Series)
        
        # Verificar que tienen el mismo índice
        assert len(features_df) == len(target)
        
        # Verificar que hay características
        assert len(features_df.columns) > 0
    
    def test_predictor_initialization(self):
        """Prueba inicialización del predictor."""
        predictor = LITMLPredictor("dummy_path.pkl")
        
        assert predictor is not None
        assert predictor.model_path == "dummy_path.pkl"
        assert predictor.model is None
        assert predictor.scaler is None
        assert predictor.feature_columns is None
    
    def test_predictor_load_model(self, mock_model_file):
        """Prueba carga del modelo."""
        predictor = LITMLPredictor(mock_model_file)
        
        # Cargar modelo
        success = predictor.load_model()
        
        assert success is True
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert predictor.feature_columns is not None
        assert predictor.model_info is not None
    
    def test_predictor_load_nonexistent_model(self):
        """Prueba carga de modelo inexistente."""
        predictor = LITMLPredictor("nonexistent_model.pkl")
        
        success = predictor.load_model()
        
        assert success is False
        assert predictor.model is None
    
    def test_predict_single_with_mock_model(self, mock_model_file, sample_data):
        """Prueba predicción individual con modelo mock."""
        predictor = LITMLPredictor(mock_model_file)
        predictor.load_model()
        
        # Realizar predicción
        prediction = predictor.predict_single(sample_data)
        
        # Verificar estructura de la predicción
        assert isinstance(prediction, dict)
        
        expected_keys = ['signal', 'confidence', 'probabilities', 'prediction_raw']
        for key in expected_keys:
            assert key in prediction
        
        # Verificar tipos y rangos
        assert prediction['signal'] in ['buy', 'sell', 'hold']
        assert 0 <= prediction['confidence'] <= 1
        assert isinstance(prediction['probabilities'], dict)
        assert len(prediction['probabilities']) == 3  # buy, sell, hold
        
        # Verificar que las probabilidades suman 1
        prob_sum = sum(prediction['probabilities'].values())
        assert abs(prob_sum - 1.0) < 1e-6
    
    def test_predict_batch_with_mock_model(self, mock_model_file, sample_data):
        """Prueba predicción por lotes con modelo mock."""
        predictor = LITMLPredictor(mock_model_file)
        predictor.load_model()
        
        # Crear múltiples ventanas de datos
        batch_data = []
        window_size = 50
        
        for i in range(0, len(sample_data) - window_size, 20):
            window = sample_data.iloc[i:i + window_size]
            batch_data.append(window)
        
        # Realizar predicción por lotes
        predictions = predictor.predict_batch(batch_data)
        
        # Verificar que retorna una lista
        assert isinstance(predictions, list)
        assert len(predictions) == len(batch_data)
        
        # Verificar cada predicción
        for prediction in predictions:
            assert isinstance(prediction, dict)
            assert prediction['signal'] in ['buy', 'sell', 'hold']
            assert 0 <= prediction['confidence'] <= 1
    
    def test_predict_without_loaded_model(self, sample_data):
        """Prueba predicción sin modelo cargado."""
        predictor = LITMLPredictor("dummy_path.pkl")
        
        prediction = predictor.predict_single(sample_data)
        
        # Debe retornar predicción por defecto
        assert prediction['signal'] == 'hold'
        assert prediction['confidence'] == 0.0
    
    def test_prediction_consistency(self, mock_model_file, sample_data):
        """Prueba consistencia de predicciones."""
        predictor = LITMLPredictor(mock_model_file)
        predictor.load_model()
        
        # Realizar múltiples predicciones con los mismos datos
        predictions = []
        for _ in range(5):
            prediction = predictor.predict_single(sample_data)
            predictions.append(prediction)
        
        # Todas las predicciones deben ser idénticas
        first_prediction = predictions[0]
        for prediction in predictions[1:]:
            assert prediction['signal'] == first_prediction['signal']
            assert abs(prediction['confidence'] - first_prediction['confidence']) < 1e-10
    
    def test_feature_engineering_consistency(self, feature_engineer, sample_data):
        """Prueba consistencia en ingeniería de características."""
        # Crear características múltiples veces
        features_list = []
        for _ in range(3):
            features = feature_engineer.create_features(sample_data)
            features_list.append(features)
        
        # Verificar que son idénticas
        first_features = features_list[0]
        for features in features_list[1:]:
            pd.testing.assert_frame_equal(first_features, features)
    
    def test_handle_missing_data(self, feature_engineer):
        """Prueba manejo de datos faltantes."""
        # Crear datos con valores faltantes
        dates = pd.date_range('2024-01-01', periods=50, freq='1H')
        data = []
        
        for i in range(50):
            # Introducir algunos NaN
            close = 1.0800 + np.random.normal(0, 0.001) if i % 10 != 0 else np.nan
            high = close + 0.001 if not np.isnan(close) else np.nan
            low = close - 0.001 if not np.isnan(close) else np.nan
            open_price = close if not np.isnan(close) else np.nan
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': 1000
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # No debe lanzar excepciones
        features = feature_engineer.create_features(df)
        
        assert isinstance(features, pd.DataFrame)
        # Debe manejar los NaN de alguna manera
        assert len(features) > 0
    
    def test_edge_cases_small_dataset(self, feature_engineer):
        """Prueba casos límite con dataset pequeño."""
        # Dataset muy pequeño
        dates = pd.date_range('2024-01-01', periods=5, freq='1H')
        data = []
        
        for i in range(5):
            data.append({
                'timestamp': dates[i],
                'open': 1.0800,
                'high': 1.0810,
                'low': 1.0790,
                'close': 1.0805,
                'volume': 1000
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # No debe lanzar excepciones
        features = feature_engineer.create_features(df)
        
        assert isinstance(features, pd.DataFrame)
    
    def test_signal_distribution(self, mock_model_file, sample_data):
        """Prueba distribución de señales."""
        predictor = LITMLPredictor(mock_model_file)
        predictor.load_model()
        
        # Configurar el mock para retornar diferentes señales
        signals_to_test = [
            ([0.7, 0.2, 0.1], 'buy'),    # buy dominante
            ([0.1, 0.8, 0.1], 'sell'),   # sell dominante
            ([0.3, 0.3, 0.4], 'hold')    # hold dominante
        ]
        
        for probs, expected_signal in signals_to_test:
            predictor.model.predict_proba.return_value = np.array([probs])
            predictor.model.predict.return_value = np.array([np.argmax(probs)])
            
            prediction = predictor.predict_single(sample_data)
            
            assert prediction['signal'] == expected_signal
            assert prediction['confidence'] == max(probs)
    
    def test_model_trainer_initialization(self):
        """Prueba inicialización del entrenador de modelos."""
        trainer = ModelTrainer()
        
        assert trainer is not None
        assert hasattr(trainer, 'train_model')
        assert hasattr(trainer, 'evaluate_model')
    
    @patch('src.models.model_trainer.XGBClassifier')
    def test_model_training_mock(self, mock_xgb, sample_data):
        """Prueba entrenamiento de modelo con mock."""
        # Configurar mock
        mock_model = Mock()
        mock_xgb.return_value = mock_model
        
        trainer = ModelTrainer()
        
        # Crear etiquetas sintéticas
        y = np.random.choice([0, 1, 2], size=len(sample_data))
        
        # Entrenar modelo
        result = trainer.train_model(sample_data, y)
        
        # Verificar que se llamó al entrenamiento
        assert mock_model.fit.called
        assert isinstance(result, dict)
    
    def test_feature_importance_calculation(self, mock_model_file, sample_data):
        """Prueba cálculo de importancia de características."""
        predictor = LITMLPredictor(mock_model_file)
        predictor.load_model()
        
        # Configurar mock para feature importance
        predictor.model.feature_importances_ = np.array([0.3, 0.5, 0.2])
        
        # Obtener importancia (si el método existe)
        if hasattr(predictor, 'get_feature_importance'):
            importance = predictor.get_feature_importance()
            assert isinstance(importance, dict)
            assert len(importance) == len(predictor.feature_columns)
    
    def test_prediction_with_insufficient_data(self, mock_model_file):
        """Prueba predicción con datos insuficientes."""
        predictor = LITMLPredictor(mock_model_file)
        predictor.load_model()
        
        # Datos muy limitados
        dates = pd.date_range('2024-01-01', periods=3, freq='1H')
        data = []
        
        for i in range(3):
            data.append({
                'timestamp': dates[i],
                'open': 1.0800,
                'high': 1.0810,
                'low': 1.0790,
                'close': 1.0805,
                'volume': 1000
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Debe manejar graciosamente
        prediction = predictor.predict_single(df)
        
        assert isinstance(prediction, dict)
        assert prediction['signal'] in ['buy', 'sell', 'hold']
    
    def test_realtime_prediction_simulation(self, mock_model_file, sample_data):
        """Prueba simulación de predicción en tiempo real."""
        predictor = LITMLPredictor(mock_model_file)
        predictor.load_model()
        
        # Simular llegada de datos en tiempo real
        window_size = 100
        predictions = []
        
        for i in range(window_size, len(sample_data), 10):
            # Ventana deslizante
            window = sample_data.iloc[i-window_size:i]
            prediction = predictor.predict_single(window)
            predictions.append(prediction)
        
        # Verificar que todas las predicciones son válidas
        assert len(predictions) > 0
        
        for prediction in predictions:
            assert prediction['signal'] in ['buy', 'sell', 'hold']
            assert 0 <= prediction['confidence'] <= 1


if __name__ == "__main__":
    # Ejecutar pruebas si se ejecuta directamente
    pytest.main([__file__, "-v"]) 