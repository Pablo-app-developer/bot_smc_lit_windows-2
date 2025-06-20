#!/usr/bin/env python3
"""
Ejemplos de Uso del Predictor LIT + ML.

Este archivo contiene ejemplos prácticos de cómo usar el sistema
de predicciones LIT + ML en diferentes escenarios.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.predictor import LITMLPredictor, load_and_predict, batch_predict_for_backtesting
from src.integrations.mt5_predictor import create_mt5_predictor
from src.data.data_loader import DataLoader
from src.utils.logger import log


def ejemplo_prediccion_simple():
    """
    Ejemplo 1: Predicción simple para un símbolo.
    
    Muestra cómo realizar una predicción básica usando datos históricos.
    """
    print("="*60)
    print("📊 EJEMPLO 1: PREDICCIÓN SIMPLE")
    print("="*60)
    
    try:
        # Configuración
        model_path = "models/test_model.pkl"  # Usar el modelo de prueba
        symbol = "AAPL"
        
        # Verificar que el modelo existe
        if not os.path.exists(model_path):
            print(f"❌ Modelo no encontrado: {model_path}")
            print("Ejecuta primero: python scripts/train_model.py")
            return
        
        # Crear predictor
        predictor = LITMLPredictor(model_path)
        
        # Cargar modelo
        print("🔄 Cargando modelo...")
        if not predictor.load_model():
            print("❌ Error cargando modelo")
            return
        
        # Obtener datos
        print(f"📈 Obteniendo datos para {symbol}...")
        data_loader = DataLoader()
        data = data_loader.load_data(symbol=symbol, timeframe="1d", periods=100)
        
        if data.empty:
            print(f"❌ No se pudieron obtener datos para {symbol}")
            return
        
        print(f"✅ Datos obtenidos: {len(data)} velas")
        
        # Realizar predicción
        print("🎯 Realizando predicción...")
        prediction = predictor.predict_single(data)
        
        # Mostrar resultado
        print("\n📊 RESULTADO:")
        print(f"  Símbolo: {symbol}")
        print(f"  Señal: {prediction['signal'].upper()}")
        print(f"  Confianza: {prediction.get('confidence', 0):.3f}")
        print(f"  Precio actual: {data['close'].iloc[-1]:.2f}")
        
        if 'probabilities' in prediction:
            probs = prediction['probabilities']
            print(f"  Probabilidades:")
            print(f"    Compra: {probs.get('buy', 0):.3f}")
            print(f"    Venta: {probs.get('sell', 0):.3f}")
            print(f"    Mantener: {probs.get('hold', 0):.3f}")
        
        print("✅ Predicción completada")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def ejemplo_backtesting_basico():
    """
    Ejemplo 2: Backtesting básico.
    
    Muestra cómo realizar backtesting con datos históricos.
    """
    print("\n" + "="*60)
    print("📈 EJEMPLO 2: BACKTESTING BÁSICO")
    print("="*60)
    
    try:
        # Configuración
        model_path = "models/test_model.pkl"
        symbol = "AAPL"
        days = 30
        
        # Verificar modelo
        if not os.path.exists(model_path):
            print(f"❌ Modelo no encontrado: {model_path}")
            return
        
        # Obtener datos históricos
        print(f"📊 Obteniendo {days} días de datos para {symbol}...")
        data_loader = DataLoader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = data_loader.load_data(
            symbol=symbol,
            timeframe="1d",
            start_date=start_date,
            end_date=end_date
        )
        
        if data.empty:
            print(f"❌ No se pudieron obtener datos para {symbol}")
            return
        
        print(f"✅ Datos obtenidos: {len(data)} velas")
        
        # Ejecutar backtesting
        print("🔄 Ejecutando backtesting...")
        predictions_df = batch_predict_for_backtesting(model_path, data, window_size=20)
        
        if predictions_df.empty:
            print("❌ No se pudieron generar predicciones")
            return
        
        print(f"✅ Predicciones generadas: {len(predictions_df)}")
        
        # Análisis básico
        signal_counts = predictions_df['signal'].value_counts()
        avg_confidence = predictions_df['confidence'].mean()
        
        print("\n📊 RESULTADOS DEL BACKTESTING:")
        print(f"  Total predicciones: {len(predictions_df)}")
        print(f"  Confianza promedio: {avg_confidence:.3f}")
        print(f"  Distribución de señales:")
        for signal, count in signal_counts.items():
            percentage = (count / len(predictions_df)) * 100
            print(f"    {signal.upper()}: {count} ({percentage:.1f}%)")
        
        # Predicciones de alta confianza
        high_conf = predictions_df[predictions_df['confidence'] > 0.7]
        print(f"  Alta confianza (>0.7): {len(high_conf)} ({len(high_conf)/len(predictions_df)*100:.1f}%)")
        
        print("✅ Backtesting completado")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def ejemplo_prediccion_multiple():
    """
    Ejemplo 3: Predicciones múltiples para varios símbolos.
    
    Muestra cómo realizar predicciones para múltiples símbolos.
    """
    print("\n" + "="*60)
    print("🎯 EJEMPLO 3: PREDICCIONES MÚLTIPLES")
    print("="*60)
    
    try:
        # Configuración
        model_path = "models/test_model.pkl"
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        
        # Verificar modelo
        if not os.path.exists(model_path):
            print(f"❌ Modelo no encontrado: {model_path}")
            return
        
        # Crear predictor
        predictor = LITMLPredictor(model_path)
        if not predictor.load_model():
            print("❌ Error cargando modelo")
            return
        
        print("✅ Modelo cargado")
        
        # Realizar predicciones para cada símbolo
        results = {}
        data_loader = DataLoader()
        
        for symbol in symbols:
            try:
                print(f"🔄 Procesando {symbol}...")
                
                # Obtener datos
                data = data_loader.load_data(symbol=symbol, timeframe="1d", periods=100)
                
                if data.empty:
                    print(f"  ⚠️ Sin datos para {symbol}")
                    continue
                
                # Realizar predicción
                prediction = predictor.predict_single(data)
                
                # Guardar resultado
                results[symbol] = {
                    'signal': prediction['signal'],
                    'confidence': prediction.get('confidence', 0),
                    'price': data['close'].iloc[-1],
                    'probabilities': prediction.get('probabilities', {})
                }
                
                print(f"  ✅ {symbol}: {prediction['signal']} (conf: {prediction.get('confidence', 0):.3f})")
                
            except Exception as e:
                print(f"  ❌ Error con {symbol}: {str(e)}")
                continue
        
        # Mostrar resumen
        if results:
            print("\n📊 RESUMEN DE PREDICCIONES:")
            print("-" * 60)
            print(f"{'Símbolo':<8} {'Señal':<8} {'Confianza':<10} {'Precio':<10}")
            print("-" * 60)
            
            for symbol, result in results.items():
                print(f"{symbol:<8} {result['signal'].upper():<8} {result['confidence']:<10.3f} {result['price']:<10.2f}")
            
            # Estadísticas
            signals = [r['signal'] for r in results.values()]
            confidences = [r['confidence'] for r in results.values()]
            
            print(f"\n📈 ESTADÍSTICAS:")
            print(f"  Símbolos procesados: {len(results)}")
            print(f"  Confianza promedio: {sum(confidences)/len(confidences):.3f}")
            print(f"  Señales BUY: {signals.count('buy')}")
            print(f"  Señales SELL: {signals.count('sell')}")
            print(f"  Señales HOLD: {signals.count('hold')}")
        
        print("✅ Predicciones múltiples completadas")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def ejemplo_uso_funciones_utilidad():
    """
    Ejemplo 4: Uso de funciones de utilidad.
    
    Muestra cómo usar las funciones de utilidad para casos simples.
    """
    print("\n" + "="*60)
    print("🛠️ EJEMPLO 4: FUNCIONES DE UTILIDAD")
    print("="*60)
    
    try:
        # Configuración
        model_path = "models/test_model.pkl"
        symbol = "AAPL"
        
        # Verificar modelo
        if not os.path.exists(model_path):
            print(f"❌ Modelo no encontrado: {model_path}")
            return
        
        # Obtener datos
        print(f"📊 Obteniendo datos para {symbol}...")
        data_loader = DataLoader()
        data = data_loader.load_data(symbol=symbol, timeframe="1d", periods=100)
        
        if data.empty:
            print(f"❌ No se pudieron obtener datos para {symbol}")
            return
        
        # Usar función de utilidad para predicción rápida
        print("🎯 Usando función de utilidad load_and_predict()...")
        prediction = load_and_predict(model_path, data)
        
        print(f"✅ Predicción rápida:")
        print(f"  Señal: {prediction['signal']}")
        print(f"  Confianza: {prediction.get('confidence', 0):.3f}")
        
        # Ejemplo de información del modelo
        predictor = LITMLPredictor(model_path)
        if predictor.load_model():
            model_info = predictor.get_model_info()
            
            print(f"\n📋 INFORMACIÓN DEL MODELO:")
            print(f"  Estado: {model_info['status']}")
            print(f"  Tipo: {model_info['model_type']}")
            print(f"  Características: {model_info['features_count']}")
            print(f"  Predicciones realizadas: {model_info['predictions_made']}")
        
        print("✅ Funciones de utilidad demostradas")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def ejemplo_mt5_demo():
    """
    Ejemplo 5: Demo de integración MT5 (sin trading real).
    
    Muestra cómo configurar la integración MT5 sin ejecutar operaciones.
    """
    print("\n" + "="*60)
    print("🔌 EJEMPLO 5: DEMO INTEGRACIÓN MT5")
    print("="*60)
    
    try:
        # Configuración
        model_path = "models/test_model.pkl"
        
        # Verificar modelo
        if not os.path.exists(model_path):
            print(f"❌ Modelo no encontrado: {model_path}")
            return
        
        print("🔄 Creando integrador MT5...")
        
        # Crear integrador (sin inicializar MT5 real)
        integrator = create_mt5_predictor(model_path)
        
        print("✅ Integrador MT5 creado")
        print(f"  Login configurado: {integrator.login}")
        print(f"  Servidor: {integrator.server}")
        print(f"  Símbolos: {integrator.symbols}")
        print(f"  Intervalo predicción: {integrator.prediction_interval}s")
        
        # Mostrar configuración
        print(f"\n⚙️ CONFIGURACIÓN:")
        print(f"  Modelo: {integrator.model_path}")
        print(f"  Trading habilitado: {integrator.trading_enabled}")
        print(f"  Riesgo por operación: {integrator.risk_per_trade * 100}%")
        
        # Nota sobre uso real
        print(f"\n📝 NOTA:")
        print(f"  Para usar MT5 real, ejecuta:")
        print(f"  python scripts/run_predictions.py realtime --hours 1")
        print(f"  (Requiere MetaTrader 5 instalado y configurado)")
        
        print("✅ Demo MT5 completada")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def main():
    """Ejecuta todos los ejemplos."""
    print("🚀 EJEMPLOS DE USO DEL PREDICTOR LIT + ML")
    print("=" * 80)
    
    # Ejecutar ejemplos
    ejemplo_prediccion_simple()
    ejemplo_backtesting_basico()
    ejemplo_prediccion_multiple()
    ejemplo_uso_funciones_utilidad()
    ejemplo_mt5_demo()
    
    print("\n" + "="*80)
    print("✅ TODOS LOS EJEMPLOS COMPLETADOS")
    print("="*80)
    
    print("\n📚 PRÓXIMOS PASOS:")
    print("1. Entrena tu propio modelo: python scripts/train_model.py")
    print("2. Ejecuta predicciones: python scripts/run_predictions.py single --symbol AAPL")
    print("3. Prueba backtesting: python scripts/run_predictions.py backtest --symbol AAPL --days 30")
    print("4. Configura MT5 para trading real")


if __name__ == "__main__":
    main() 