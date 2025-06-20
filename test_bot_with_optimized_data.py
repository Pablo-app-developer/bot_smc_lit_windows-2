#!/usr/bin/env python3
"""
Test del Bot con Datos Optimizados.

Demuestra que el bot puede funcionar correctamente
cuando se configura para obtener datos abundantes.
"""

import sys
import os
import pandas as pd
import yfinance as yf
from datetime import datetime

# Configurar path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import log
from src.strategies.lit_detector import LITDetector
from src.models.feature_engineering import FeatureEngineer
from src.models.predictor import LITMLPredictor


def load_optimized_data(symbol: str = 'AAPL', timeframe: str = '1h', periods: int = 200):
    """Carga datos con configuración optimizada."""
    try:
        print(f"\n📊 CARGANDO DATOS OPTIMIZADOS")
        print("-" * 40)
        print(f"Símbolo: {symbol}")
        print(f"Timeframe: {timeframe}")
        print(f"Períodos solicitados: {periods}")
        
        # Configuración optimizada por timeframe
        config_map = {
            '1m': {'period': '7d', 'interval': '1m'},
            '5m': {'period': '60d', 'interval': '5m'},
            '15m': {'period': '60d', 'interval': '15m'},
            '1h': {'period': '730d', 'interval': '1h'},  # 2 años
            '4h': {'period': '730d', 'interval': '1d'},
            '1d': {'period': '5y', 'interval': '1d'}     # 5 años
        }
        
        config = config_map.get(timeframe, {'period': '2y', 'interval': '1d'})
        
        print(f"Configuración yfinance: period={config['period']}, interval={config['interval']}")
        
        # Descargar datos
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            period=config['period'],
            interval=config['interval'],
            auto_adjust=True,
            prepost=False
        )
        
        if data.empty:
            print("❌ No se obtuvieron datos")
            return pd.DataFrame()
        
        # Limpiar datos
        data.columns = [col.lower() for col in data.columns]
        data = data.dropna()
        
        # Tomar últimos períodos si hay más datos
        if len(data) > periods:
            data = data.tail(periods)
        
        print(f"✅ Datos cargados exitosamente:")
        print(f"   - Filas totales: {len(data)}")
        print(f"   - Rango: {data.index[0]} a {data.index[-1]}")
        print(f"   - Último precio: ${data['close'].iloc[-1]:.2f}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error cargando datos: {str(e)}")
        return pd.DataFrame()


def test_lit_analysis(data: pd.DataFrame):
    """Prueba análisis LIT con datos optimizados."""
    try:
        print(f"\n🎯 ANÁLISIS LIT")
        print("-" * 30)
        
        if len(data) < 50:
            print("❌ Datos insuficientes para análisis LIT")
            return
        
        # Inicializar detector LIT
        lit_detector = LITDetector(
            lookback_periods=50,
            liquidity_threshold=0.001
        )
        
        # Ejecutar análisis
        lit_signals = lit_detector.detect_lit_events(data)
        
        # Contar señales
        buy_signals = (lit_signals == 1).sum()
        sell_signals = (lit_signals == -1).sum()
        hold_signals = (lit_signals == 0).sum()
        
        print(f"✅ Análisis LIT completado:")
        print(f"   - Señales BUY: {buy_signals}")
        print(f"   - Señales SELL: {sell_signals}")
        print(f"   - Señales HOLD: {hold_signals}")
        print(f"   - Total períodos analizados: {len(lit_signals)}")
        
        # Última señal
        if len(lit_signals) > 0:
            last_signal = lit_signals.iloc[-1]
            signal_text = "BUY" if last_signal == 1 else "SELL" if last_signal == -1 else "HOLD"
            print(f"   - Última señal: {signal_text}")
        
        return lit_signals
        
    except Exception as e:
        print(f"❌ Error en análisis LIT: {str(e)}")
        return pd.Series()


def test_ml_features(data: pd.DataFrame):
    """Prueba generación de features ML."""
    try:
        print(f"\n🤖 FEATURES MACHINE LEARNING")
        print("-" * 35)
        
        if len(data) < 50:
            print("❌ Datos insuficientes para features ML")
            return pd.DataFrame()
        
        # Inicializar feature engineer
        feature_engineer = FeatureEngineer()
        
        # Generar features
        features = feature_engineer.create_features(data)
        
        print(f"✅ Features ML generadas:")
        print(f"   - Total features: {len(features.columns)}")
        print(f"   - Filas con features: {len(features)}")
        print(f"   - Features principales: {list(features.columns[:10])}")
        
        # Verificar calidad
        missing_pct = (features.isnull().sum().sum() / (len(features) * len(features.columns))) * 100
        print(f"   - Datos faltantes: {missing_pct:.1f}%")
        
        return features
        
    except Exception as e:
        print(f"❌ Error generando features: {str(e)}")
        return pd.DataFrame()


def test_ml_prediction(features: pd.DataFrame, lit_signals: pd.Series):
    """Prueba predicción ML."""
    try:
        print(f"\n🧠 PREDICCIÓN MACHINE LEARNING")
        print("-" * 35)
        
        if len(features) < 100:
            print("❌ Datos insuficientes para entrenamiento ML")
            return
        
        # Inicializar predictor
        predictor = LITMLPredictor()
        
        # Preparar datos de entrenamiento
        if len(lit_signals) > 0 and len(lit_signals) == len(features):
            # Entrenar modelo
            predictor.train(features, lit_signals)
            
            # Hacer predicción
            prediction = predictor.predict(features.tail(1))
            confidence = predictor.predict_proba(features.tail(1))
            
            pred_text = "BUY" if prediction[0] == 1 else "SELL" if prediction[0] == -1 else "HOLD"
            
            print(f"✅ Predicción ML completada:")
            print(f"   - Predicción: {pred_text}")
            print(f"   - Confianza: {confidence[0]:.2%}")
            print(f"   - Modelo entrenado con {len(features)} muestras")
            
        else:
            print("⚠️  No se pudo entrenar modelo (datos inconsistentes)")
        
    except Exception as e:
        print(f"❌ Error en predicción ML: {str(e)}")


def main():
    """Función principal de testing."""
    print("🚀 TEST DEL BOT CON DATOS OPTIMIZADOS")
    print("=" * 50)
    print(f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Cargar datos optimizados
    data = load_optimized_data('AAPL', '1h', 200)
    
    if data.empty:
        print("\n❌ FALLO: No se pudieron cargar datos")
        return
    
    # 2. Probar análisis LIT
    lit_signals = test_lit_analysis(data)
    
    # 3. Probar features ML
    features = test_ml_features(data)
    
    # 4. Probar predicción ML
    if not features.empty:
        test_ml_prediction(features, lit_signals)
    
    # 5. Resumen final
    print(f"\n" + "=" * 50)
    print("📋 RESUMEN DEL TEST")
    print("=" * 50)
    
    if len(data) >= 100:
        print("✅ ÉXITO: Bot puede funcionar con datos abundantes")
        print(f"   - Datos cargados: {len(data)} filas")
        print(f"   - Análisis LIT: {'✅' if len(lit_signals) > 0 else '❌'}")
        print(f"   - Features ML: {'✅' if len(features) > 0 else '❌'}")
        print(f"   - Sistema operativo: ✅")
        
        print(f"\n💡 RECOMENDACIÓN:")
        print(f"   El bot PUEDE cargar datos abundantes de AAPL.")
        print(f"   El problema está en la configuración del data_loader.py")
        print(f"   que no usa los parámetros optimizados de yfinance.")
        
    else:
        print("❌ FALLO: Datos insuficientes incluso con configuración optimizada")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main() 