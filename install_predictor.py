#!/usr/bin/env python3
"""
Script de Instalación - Sistema Predictor LIT + ML.

Este script instala y configura automáticamente el sistema completo
de predicciones LIT + ML con integración MetaTrader 5.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_header():
    """Imprime el header del instalador."""
    print("=" * 80)
    print("🚀 INSTALADOR SISTEMA PREDICTOR LIT + ML")
    print("=" * 80)
    print("Instalando dependencias y configurando el sistema...")
    print()


def check_python_version():
    """Verifica la versión de Python."""
    print("🔍 Verificando versión de Python...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True


def install_dependencies():
    """Instala las dependencias requeridas."""
    print("\n📦 Instalando dependencias...")
    
    # Dependencias principales
    main_deps = [
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "xgboost>=1.6.0",
        "yfinance>=0.2.0",
        "ccxt>=3.0.0",
        "ta>=0.10.0",
        "joblib>=1.2.0",
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0"
    ]
    
    # Dependencias opcionales
    optional_deps = [
        "MetaTrader5",  # Para integración MT5
        "TA-Lib"        # Para indicadores técnicos avanzados
    ]
    
    try:
        # Instalar dependencias principales
        print("🔄 Instalando dependencias principales...")
        for dep in main_deps:
            print(f"   Instalando {dep}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep, "--quiet"
            ])
        
        print("✅ Dependencias principales instaladas")
        
        # Instalar dependencias opcionales
        print("\n🔄 Instalando dependencias opcionales...")
        for dep in optional_deps:
            try:
                print(f"   Instalando {dep}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", dep, "--quiet"
                ])
                print(f"   ✅ {dep} instalado")
            except subprocess.CalledProcessError:
                print(f"   ⚠️ {dep} no se pudo instalar (opcional)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False


def create_directories():
    """Crea los directorios necesarios."""
    print("\n📁 Creando estructura de directorios...")
    
    directories = [
        "models",
        "logs",
        "data",
        "config",
        "results"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Creado: {directory}/")
        else:
            print(f"   ✓ Existe: {directory}/")
    
    return True


def create_env_file():
    """Crea archivo de configuración .env."""
    print("\n⚙️ Creando archivo de configuración...")
    
    env_content = """# Configuración Sistema Predictor LIT + ML
# Generado automáticamente

# Configuración General
DEBUG=False
LOG_LEVEL=INFO

# Configuración de Datos
DEFAULT_SYMBOL=AAPL
DEFAULT_TIMEFRAME=1h
DEFAULT_PERIODS=100

# Configuración MT5
MT5_LOGIN=5036791117
MT5_PASSWORD=BtUvF-X8
MT5_SERVER=MetaQuotes-Demo

# Configuración de Trading
RISK_PER_TRADE=0.02
MAX_DAILY_TRADES=10
MIN_CONFIDENCE=0.6
MAX_SPREAD=3

# Configuración del Modelo
MODEL_PATH=models/lit_ml_model.pkl
RETRAIN_INTERVAL_DAYS=30
FEATURE_SELECTION_THRESHOLD=0.01

# Configuración de Backtesting
BACKTEST_WINDOW_SIZE=50
BACKTEST_MIN_SAMPLES=100
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(env_content)
        print("   ✅ Archivo .env creado")
    else:
        print("   ✓ Archivo .env ya existe")
    
    return True


def test_installation():
    """Prueba la instalación."""
    print("\n🧪 Probando instalación...")
    
    try:
        # Probar imports principales
        print("   Probando imports...")
        import pandas as pd
        import numpy as np
        import sklearn
        import xgboost as xgb
        import yfinance as yf
        print("   ✅ Imports principales - OK")
        
        # Probar estructura del proyecto
        print("   Verificando estructura del proyecto...")
        required_files = [
            "src/models/predictor.py",
            "src/integrations/mt5_predictor.py",
            "scripts/run_predictions.py",
            "scripts/train_model.py"
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                print(f"   ❌ Archivo faltante: {file_path}")
                return False
            
        print("   ✅ Estructura del proyecto - OK")
        
        # Probar carga de módulos
        print("   Probando carga de módulos...")
        sys.path.append(str(Path.cwd()))
        
        try:
            from src.models.predictor import LITMLPredictor
            from src.integrations.mt5_predictor import create_mt5_predictor
            print("   ✅ Módulos del sistema - OK")
        except ImportError as e:
            print(f"   ⚠️ Advertencia en módulos: {e}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Error en imports: {e}")
        return False


def show_next_steps():
    """Muestra los próximos pasos."""
    print("\n" + "=" * 80)
    print("✅ INSTALACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 80)
    
    print("\n📚 PRÓXIMOS PASOS:")
    print()
    print("1. 🎯 Entrenar tu primer modelo:")
    print("   python scripts/train_model.py")
    print()
    print("2. 🔍 Realizar una predicción de prueba:")
    print("   python scripts/run_predictions.py single --symbol AAPL")
    print()
    print("3. 📈 Ejecutar backtesting:")
    print("   python scripts/run_predictions.py backtest --symbol AAPL --days 30")
    print()
    print("4. 🚀 Predicciones en tiempo real (requiere MT5):")
    print("   python scripts/run_predictions.py realtime --hours 1")
    print()
    print("5. 📖 Ver ejemplos de uso:")
    print("   python examples/predictor_examples.py")
    print()
    
    print("📋 ARCHIVOS DE CONFIGURACIÓN:")
    print("   .env                    - Configuración general")
    print("   config/training_config.json - Configuración de entrenamiento")
    print()
    
    print("📁 DIRECTORIOS IMPORTANTES:")
    print("   models/                 - Modelos entrenados")
    print("   logs/                   - Archivos de log")
    print("   results/                - Resultados de backtesting")
    print()
    
    print("📚 DOCUMENTACIÓN:")
    print("   PREDICTOR_LIT_ML.md     - Documentación completa")
    print("   ENTRENAMIENTO_MODELO_LIT_ML.md - Guía de entrenamiento")
    print()
    
    print("⚠️ IMPORTANTE:")
    print("   - Siempre prueba en cuenta demo antes de usar capital real")
    print("   - Configura adecuadamente los parámetros de riesgo")
    print("   - Monitorea constantemente el sistema en producción")
    print()


def main():
    """Función principal del instalador."""
    print_header()
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Instalar dependencias
    if not install_dependencies():
        print("\n❌ Error en la instalación de dependencias")
        sys.exit(1)
    
    # Crear directorios
    if not create_directories():
        print("\n❌ Error creando directorios")
        sys.exit(1)
    
    # Crear archivo .env
    if not create_env_file():
        print("\n❌ Error creando configuración")
        sys.exit(1)
    
    # Probar instalación
    if not test_installation():
        print("\n⚠️ Instalación completada con advertencias")
    
    # Mostrar próximos pasos
    show_next_steps()


if __name__ == "__main__":
    main() 