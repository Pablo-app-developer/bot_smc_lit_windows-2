"""
Ejemplo de uso del LITDetector mejorado.

Este script demuestra cómo usar el detector LIT profesional para:
- Detectar eventos de liquidez, inducement e inefficiencies
- Configurar parámetros personalizados
- Analizar señales generadas
- Monitorear métricas de performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

from src.strategies.lit_detector import LITDetector, EventType, Direction, SignalType
from src.utils.logger import log


def create_sample_data():
    """Crea datos de muestra para demostración."""
    print("📊 Creando datos de muestra...")
    
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
    np.random.seed(42)
    
    # Simular datos realistas con patrones LIT
    base_price = 100.0
    data = []
    
    for i in range(200):
        # Movimiento base
        change = np.random.normal(0, 0.015)
        base_price *= (1 + change)
        
        # Crear algunos patrones específicos
        if i == 50:  # Liquidity sweep
            high = base_price * 1.03  # Spike
            low = base_price * 0.99
            close = base_price * 0.995  # Retroceso
            volume = 15000  # Alto volumen
        elif i == 100:  # Inducement zone
            high = base_price * 1.02
            low = base_price * 0.985
            close = base_price * 1.015
            volume = 12000
        elif i == 150:  # Gap (inefficiency)
            high = base_price * 1.025
            low = base_price * 1.01  # Gap up
            close = base_price * 1.02
            volume = 8000
        else:  # Datos normales
            high = base_price * (1 + abs(np.random.normal(0, 0.008)))
            low = base_price * (1 - abs(np.random.normal(0, 0.008)))
            close = base_price * (1 + np.random.normal(0, 0.005))
            volume = np.random.randint(3000, 8000)
        
        open_price = base_price + np.random.normal(0, 0.003)
        
        data.append([open_price, high, low, close, volume])
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'], index=dates)
    print(f"✅ Datos creados: {len(df)} velas desde {df.index[0]} hasta {df.index[-1]}")
    
    return df


def demonstrate_basic_usage():
    """Demuestra el uso básico del detector."""
    print("\n🔍 DEMOSTRACIÓN BÁSICA DEL LIT DETECTOR")
    print("=" * 50)
    
    # Crear detector con configuración por defecto
    detector = LITDetector()
    
    # Crear datos de muestra
    data = create_sample_data()
    
    # Analizar datos
    print("\n📈 Analizando datos con LIT Detector...")
    signal = detector.analyze(data)
    
    # Mostrar resultados
    print(f"\n📊 RESULTADOS DEL ANÁLISIS:")
    print(f"   Señal: {signal.signal.value.upper()}")
    print(f"   Confianza: {signal.confidence:.2%}")
    print(f"   Precio de entrada: ${signal.entry_price:.2f}")
    
    if signal.stop_loss:
        print(f"   Stop Loss: ${signal.stop_loss:.2f}")
    if signal.take_profit:
        print(f"   Take Profit: ${signal.take_profit:.2f}")
    if signal.risk_reward_ratio:
        print(f"   Risk/Reward: 1:{signal.risk_reward_ratio:.2f}")
    
    print(f"\n🎯 EVENTOS DETECTADOS ({len(signal.events)}):")
    for i, event in enumerate(signal.events, 1):
        print(f"   {i}. {event.event_type.value.title()}")
        print(f"      Dirección: {event.direction.value}")
        print(f"      Confianza: {event.confidence:.2%}")
        print(f"      Precio: ${event.price:.2f}")
        print(f"      Calidad: {event.pattern_quality:.2%}")
        print(f"      Volumen confirmado: {'✅' if event.volume_confirmation else '❌'}")
        print()
    
    print(f"📋 CONTEXTO ADICIONAL:")
    for key, value in signal.context.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"     - {k}: {v}")
        else:
            print(f"   {key}: {value}")


def demonstrate_custom_configuration():
    """Demuestra configuración personalizada."""
    print("\n⚙️ CONFIGURACIÓN PERSONALIZADA")
    print("=" * 50)
    
    # Configuración personalizada para trading más agresivo
    custom_config = {
        'lookback_candles': 30,  # Menos historial
        'volume_confirmation_threshold': 1.2,  # Menos exigente con volumen
        'pattern_quality_threshold': 0.5,  # Menor calidad mínima
        'fake_breakout_retracement': 0.5  # Menos retroceso para fake breakout
    }
    
    detector = LITDetector(custom_config)
    data = create_sample_data()
    
    print(f"🔧 Configuración aplicada:")
    for key, value in custom_config.items():
        print(f"   {key}: {value}")
    
    signal = detector.analyze(data)
    
    print(f"\n📊 Resultados con configuración personalizada:")
    print(f"   Señal: {signal.signal.value.upper()}")
    print(f"   Eventos detectados: {len(signal.events)}")
    print(f"   Confianza promedio: {np.mean([e.confidence for e in signal.events]):.2%}")


def demonstrate_performance_monitoring():
    """Demuestra monitoreo de performance."""
    print("\n📈 MONITOREO DE PERFORMANCE")
    print("=" * 50)
    
    detector = LITDetector()
    data = create_sample_data()
    
    # Ejecutar múltiples análisis
    print("🔄 Ejecutando múltiples análisis...")
    for i in range(10):
        # Usar diferentes ventanas de datos
        window_data = data.iloc[i*10:(i*10)+100]
        if len(window_data) >= detector.lookback_candles:
            signal = detector.analyze(window_data)
            print(f"   Análisis {i+1}: {signal.signal.value} (confianza: {signal.confidence:.2%})")
    
    # Obtener métricas
    metrics = detector.get_performance_metrics()
    
    print(f"\n📊 MÉTRICAS DE PERFORMANCE:")
    print(f"   Total de señales: {metrics['total_signals']}")
    print(f"   Tiempo promedio: {metrics['avg_processing_time_ms']:.2f}ms")
    print(f"   Tiempo máximo: {metrics['max_processing_time_ms']:.2f}ms")
    print(f"   Tasa de éxito: {metrics['success_rate']:.1f}%")


def demonstrate_event_filtering():
    """Demuestra filtrado de eventos por tipo."""
    print("\n🔍 FILTRADO DE EVENTOS POR TIPO")
    print("=" * 50)
    
    detector = LITDetector()
    data = create_sample_data()
    signal = detector.analyze(data)
    
    # Filtrar por tipo de evento
    event_types = {}
    for event in signal.events:
        event_type = event.event_type.value
        if event_type not in event_types:
            event_types[event_type] = []
        event_types[event_type].append(event)
    
    print(f"📋 Eventos por tipo:")
    for event_type, events in event_types.items():
        print(f"\n   {event_type.upper()} ({len(events)} eventos):")
        for event in events:
            direction_icon = "🟢" if event.direction == Direction.BULLISH else "🔴"
            print(f"     {direction_icon} Confianza: {event.confidence:.2%}, "
                  f"Precio: ${event.price:.2f}")


def demonstrate_real_data():
    """Demuestra uso con datos reales (requiere conexión a internet)."""
    print("\n🌐 ANÁLISIS CON DATOS REALES")
    print("=" * 50)
    
    try:
        # Descargar datos reales de Yahoo Finance
        print("📥 Descargando datos de EURUSD...")
        ticker = yf.Ticker("EURUSD=X")
        data = ticker.history(period="5d", interval="1h")
        
        if data.empty:
            print("❌ No se pudieron obtener datos reales")
            return
        
        # Renombrar columnas para compatibilidad
        data.columns = [col.lower() for col in data.columns]
        
        detector = LITDetector()
        signal = detector.analyze(data)
        
        print(f"✅ Datos obtenidos: {len(data)} velas")
        print(f"📊 Análisis completado:")
        print(f"   Par: EUR/USD")
        print(f"   Señal: {signal.signal.value.upper()}")
        print(f"   Confianza: {signal.confidence:.2%}")
        print(f"   Eventos detectados: {len(signal.events)}")
        print(f"   Estructura de mercado: {signal.context.get('market_structure', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error al obtener datos reales: {str(e)}")
        print("💡 Asegúrate de tener conexión a internet y yfinance instalado")


def main():
    """Función principal que ejecuta todas las demostraciones."""
    print("🚀 DEMOSTRACIÓN DEL LIT DETECTOR PROFESIONAL")
    print("=" * 60)
    print("Este ejemplo muestra las capacidades avanzadas del detector LIT")
    print("para identificar patrones de liquidez, inducement e inefficiencies.")
    
    try:
        # Ejecutar demostraciones
        demonstrate_basic_usage()
        demonstrate_custom_configuration()
        demonstrate_performance_monitoring()
        demonstrate_event_filtering()
        demonstrate_real_data()
        
        print("\n✅ DEMOSTRACIÓN COMPLETADA")
        print("=" * 60)
        print("El LIT Detector está listo para usar en tu estrategia de trading!")
        
    except Exception as e:
        print(f"\n❌ Error durante la demostración: {str(e)}")
        log.error(f"Error en demostración LIT Detector: {str(e)}")


if __name__ == "__main__":
    main() 