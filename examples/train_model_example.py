#!/usr/bin/env python3
"""
Ejemplo de uso del script de entrenamiento del modelo LIT + ML.

Este ejemplo demuestra cómo usar el entrenador para crear un modelo
que combina indicadores técnicos con señales LIT.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_model import LITMLTrainer
from src.utils.logger import log


def ejemplo_entrenamiento_basico():
    """Ejemplo básico de entrenamiento."""
    print("🚀 Ejemplo 1: Entrenamiento Básico")
    print("="*50)
    
    # Inicializar entrenador
    trainer = LITMLTrainer()
    
    # Cargar datos
    data = trainer.load_and_prepare_data(
        symbol="EURUSD=X",
        timeframe="1h"
    )
    
    # Entrenar modelo
    results = trainer.train_model(
        data=data,
        target_method="hybrid",
        save_path="models/ejemplo_basico.pkl"
    )
    
    print(f"✅ Modelo entrenado con accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"✅ F1-Score: {results['test_metrics']['f1_score']:.4f}")
    print()


def ejemplo_entrenamiento_personalizado():
    """Ejemplo con configuración personalizada."""
    print("🚀 Ejemplo 2: Entrenamiento Personalizado")
    print("="*50)
    
    # Inicializar con configuración personalizada
    trainer = LITMLTrainer(config_path="config/training_config.json")
    
    # Cargar datos
    data = trainer.load_and_prepare_data(
        symbol="GBPUSD=X",
        timeframe="4h"
    )
    
    # Entrenar modelo con método de target diferente
    results = trainer.train_model(
        data=data,
        target_method="future_returns",
        save_path="models/ejemplo_personalizado.pkl"
    )
    
    print(f"✅ Modelo personalizado entrenado")
    print(f"✅ Características usadas: {results['data_info']['features_count']}")
    print(f"✅ Muestras de entrenamiento: {results['data_info']['train_samples']}")
    print()


def ejemplo_comparacion_metodos():
    """Ejemplo comparando diferentes métodos de target."""
    print("🚀 Ejemplo 3: Comparación de Métodos")
    print("="*50)
    
    trainer = LITMLTrainer()
    
    # Cargar datos una vez
    data = trainer.load_and_prepare_data(
        symbol="EURUSD=X",
        timeframe="1h"
    )
    
    metodos = ["future_returns", "lit_signals", "hybrid"]
    resultados = {}
    
    for metodo in metodos:
        print(f"Entrenando con método: {metodo}")
        
        # Crear nuevo entrenador para cada método
        trainer_metodo = LITMLTrainer()
        
        results = trainer_metodo.train_model(
            data=data,
            target_method=metodo,
            save_path=f"models/comparacion_{metodo}.pkl"
        )
        
        resultados[metodo] = {
            'accuracy': results['test_metrics']['accuracy'],
            'f1_score': results['test_metrics']['f1_score'],
            'precision': results['test_metrics']['precision'],
            'recall': results['test_metrics']['recall']
        }
    
    # Mostrar comparación
    print("\n📊 Comparación de Resultados:")
    print("-" * 60)
    print(f"{'Método':<15} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 60)
    
    for metodo, metricas in resultados.items():
        print(f"{metodo:<15} {metricas['accuracy']:<10.4f} {metricas['f1_score']:<10.4f} "
              f"{metricas['precision']:<10.4f} {metricas['recall']:<10.4f}")
    
    print("-" * 60)
    print()


def ejemplo_analisis_caracteristicas():
    """Ejemplo de análisis de importancia de características."""
    print("🚀 Ejemplo 4: Análisis de Características")
    print("="*50)
    
    trainer = LITMLTrainer()
    
    # Cargar datos
    data = trainer.load_and_prepare_data(
        symbol="EURUSD=X",
        timeframe="1h"
    )
    
    # Entrenar modelo
    results = trainer.train_model(
        data=data,
        target_method="hybrid",
        save_path="models/analisis_features.pkl"
    )
    
    # Mostrar top características
    print("\n🏆 Top 20 Características Más Importantes:")
    print("-" * 50)
    
    feature_importance = results['feature_importance']
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(sorted_features[:20]):
        print(f"{i+1:2d}. {feature:<35}: {importance:.4f}")
    
    # Análisis por categorías
    categorias = {
        'LIT': [f for f in feature_importance.keys() if 'lit_' in f.lower()],
        'RSI': [f for f in feature_importance.keys() if 'rsi' in f.lower()],
        'MACD': [f for f in feature_importance.keys() if 'macd' in f.lower()],
        'Bollinger': [f for f in feature_importance.keys() if 'bb_' in f.lower()],
        'Volume': [f for f in feature_importance.keys() if 'volume' in f.lower()],
        'Candle': [f for f in feature_importance.keys() if any(x in f.lower() for x in ['body', 'wick', 'candle'])],
    }
    
    print("\n📊 Importancia por Categorías:")
    print("-" * 40)
    
    for categoria, features in categorias.items():
        if features:
            importancia_total = sum(feature_importance.get(f, 0) for f in features)
            print(f"{categoria:<12}: {importancia_total:.4f} ({len(features)} features)")
    
    print()


def main():
    """Función principal con todos los ejemplos."""
    print("🎯 Ejemplos de Entrenamiento del Modelo LIT + ML")
    print("=" * 60)
    print()
    
    try:
        # Ejemplo 1: Entrenamiento básico
        ejemplo_entrenamiento_basico()
        
        # Ejemplo 2: Configuración personalizada
        ejemplo_entrenamiento_personalizado()
        
        # Ejemplo 3: Comparación de métodos
        ejemplo_comparacion_metodos()
        
        # Ejemplo 4: Análisis de características
        ejemplo_analisis_caracteristicas()
        
        print("🎉 Todos los ejemplos completados exitosamente!")
        print("💾 Modelos guardados en el directorio 'models/'")
        
    except Exception as e:
        log.error(f"❌ Error en los ejemplos: {str(e)}")
        print(f"❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main() 