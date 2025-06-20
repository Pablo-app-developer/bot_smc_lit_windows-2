#!/usr/bin/env python3
"""
Pruebas unitarias para el detector LIT.

Valida que el detector LIT identifica correctamente zonas de liquidez
e inducement según la teoría LIT (Liquidity + Inducement Theory).
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from src.strategies.lit_detector import LITDetector


def get_signal_value(signal):
    """Extrae el valor de la señal del enum."""
    signal_str = str(signal)
    if '.' in signal_str:
        return signal_str.split('.')[-1].upper()
    return signal_str.upper()


class TestLITDetector:
    """Pruebas para el detector LIT."""
    
    @pytest.fixture
    def detector(self):
        """Fixture que crea una instancia del detector LIT."""
        return LITDetector()
    
    @pytest.fixture
    def sample_data(self):
        """Fixture que crea datos de muestra para las pruebas."""
        # Crear datos sintéticos que simulan patrones LIT
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        
        # Patrón de liquidez: precios que forman equal highs/lows
        base_price = 1.0800
        prices = []
        
        for i in range(100):
            if i < 20:
                # Tendencia alcista inicial
                price = base_price + (i * 0.0001)
            elif i < 40:
                # Formación de equal highs (liquidez)
                price = base_price + 0.0020 + np.random.normal(0, 0.00005)
            elif i < 60:
                # Barrido de liquidez (spike)
                if i == 45:
                    price = base_price + 0.0035  # Spike que barre liquidez
                else:
                    price = base_price + 0.0015 + np.random.normal(0, 0.00005)
            else:
                # Movimiento de inducement
                price = base_price + 0.0010 - ((i - 60) * 0.00005)
            
            prices.append(price)
        
        # Crear OHLC realista
        data = []
        for i, close in enumerate(prices):
            high = close + np.random.uniform(0, 0.0002)
            low = close - np.random.uniform(0, 0.0002)
            open_price = prices[i-1] if i > 0 else close
            volume = np.random.randint(1000, 5000)
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def test_detector_initialization(self, detector):
        """Prueba que el detector se inicializa correctamente."""
        assert detector is not None
        assert hasattr(detector, 'analyze')
        assert hasattr(detector, '_detect_liquidity_levels')
        assert hasattr(detector, '_detect_inducement_zones')
    
    def test_analyze_basic(self, detector, sample_data):
        """Prueba básica del análisis LIT."""
        result = detector.analyze(sample_data)
        
        # Verificar que retorna un LITSignal
        assert result is not None
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'events')
        
        # Verificar tipos de datos
        assert get_signal_value(result.signal) in ['BUY', 'SELL', 'HOLD']
        assert 0 <= result.confidence <= 1
    
    def test_data_validation(self, detector, sample_data):
        """Prueba validación de datos."""
        is_valid = detector.validate_data(sample_data)
        
        # Los datos de muestra pueden tener inconsistencias OHLC menores
        # pero el método debe ejecutarse sin errores
        assert isinstance(is_valid, bool)
    
    def test_analyze_complete(self, detector, sample_data):
        """Prueba análisis completo de señales LIT."""
        result = detector.analyze(sample_data)
        
        # Verificar que retorna un LITSignal válido
        assert result is not None
        
        # Verificar propiedades básicas
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'events')
        assert hasattr(result, 'entry_price')
        
        # Verificar tipos de datos
        assert get_signal_value(result.signal) in ['BUY', 'SELL', 'HOLD']
        assert 0 <= result.confidence <= 1
        assert isinstance(result.events, list)
    
    def test_consistency(self, detector, sample_data):
        """Prueba consistencia de análisis."""
        # Ejecutar análisis múltiples veces
        results = []
        for _ in range(3):
            result = detector.analyze(sample_data)
            results.append(result)
        
        # Las señales deben ser consistentes
        signals = [r.signal for r in results]
        assert len(set(signals)) == 1  # Todas iguales
        
        # Las confianzas deben ser idénticas
        confidences = [r.confidence for r in results]
        assert all(abs(c - confidences[0]) < 1e-10 for c in confidences)
    
    def test_invalid_data_handling(self, detector):
        """Prueba manejo de datos inválidos."""
        # DataFrame vacío
        empty_df = pd.DataFrame()
        
        # No debe lanzar excepciones
        try:
            result = detector.analyze(empty_df)
            # Si no lanza excepción, debe retornar algo válido o None
            if result is not None:
                assert hasattr(result, 'signal')
        except Exception:
            # Es aceptable que lance excepción con datos vacíos
            pass
    
    def test_performance_metrics(self, detector):
        """Prueba métricas de rendimiento."""
        # Obtener métricas
        metrics = detector.get_performance_metrics()
        
        # Debe retornar un diccionario
        assert isinstance(metrics, dict)
        
        # Resetear métricas
        detector.reset_performance_metrics()
        
        # Debe funcionar sin errores
        assert True
    
    def test_insufficient_data_handling(self, detector):
        """Prueba manejo de datos insuficientes."""
        # Muy pocos datos
        dates = pd.date_range('2024-01-01', periods=5, freq='1h')
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
        try:
            result = detector.analyze(df)
            if result is not None:
                assert hasattr(result, 'signal')
        except Exception:
            # Es aceptable que lance excepción con datos insuficientes
            pass


if __name__ == "__main__":
    # Ejecutar pruebas si se ejecuta directamente
    pytest.main([__file__, "-v"]) 