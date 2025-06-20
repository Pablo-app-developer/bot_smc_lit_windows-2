#!/usr/bin/env python3
"""
Script de Instalación del Trade Executor.

Este script instala y configura automáticamente el sistema
de trading automático LIT + ML.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_header():
    """Imprime el header del instalador."""
    print("=" * 60)
    print("🤖 INSTALADOR DEL TRADE EXECUTOR LIT + ML")
    print("=" * 60)
    print("Este script instalará y configurará el sistema de trading automático.")
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


def install_requirements():
    """Instala los paquetes requeridos."""
    print("\n📦 Instalando dependencias...")
    
    requirements = [
        "MetaTrader5",
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "joblib",
        "ta",
        "yfinance"
    ]
    
    for package in requirements:
        try:
            print(f"   Instalando {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ])
            print(f"   ✅ {package} instalado")
        except subprocess.CalledProcessError:
            print(f"   ❌ Error instalando {package}")
            return False
    
    print("✅ Todas las dependencias instaladas correctamente")
    return True


def verify_mt5_installation():
    """Verifica la instalación de MetaTrader5."""
    print("\n🔌 Verificando MetaTrader5...")
    
    try:
        import MetaTrader5 as mt5
        
        # Intentar inicializar
        if mt5.initialize():
            print("✅ MetaTrader5 disponible y funcional")
            mt5.shutdown()
            return True
        else:
            print("⚠️ MetaTrader5 instalado pero no se puede inicializar")
            print("   Asegúrate de tener MetaTrader 5 instalado en el sistema")
            return False
            
    except ImportError:
        print("❌ MetaTrader5 no está disponible")
        print("   Instala MetaTrader 5 desde: https://www.metatrader5.com/")
        return False


def create_directories():
    """Crea los directorios necesarios."""
    print("\n📁 Creando estructura de directorios...")
    
    directories = [
        "logs",
        "models",
        "data/raw",
        "data/processed",
        "results",
        "config"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    print("✅ Estructura de directorios creada")
    return True


def create_config_files():
    """Crea archivos de configuración."""
    print("\n⚙️ Creando archivos de configuración...")
    
    # Archivo .env.example
    env_content = """# Configuración del Trade Executor
MT5_LOGIN=5036791117
MT5_PASSWORD=BtUvF-X8
MT5_SERVER=MetaQuotes-Demo

# Configuración de riesgo
RISK_LEVEL=moderate
MIN_CONFIDENCE=0.65
MAX_SPREAD=3.0

# Configuración del bot
PREDICTION_INTERVAL=300
SYMBOLS=EURUSD,GBPUSD,USDJPY,AUDUSD
TIMEFRAME=1h

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading.log
"""
    
    with open(".env.example", "w") as f:
        f.write(env_content)
    print("   ✅ .env.example creado")
    
    # Archivo de configuración YAML
    config_content = """# Configuración del Sistema de Trading LIT + ML

# MetaTrader 5
mt5:
  login: 5036791117
  password: "BtUvF-X8"
  server: "MetaQuotes-Demo"

# Gestión de Riesgos
risk:
  level: "moderate"  # conservative, moderate, aggressive
  max_risk_per_trade: 0.02
  max_daily_risk: 0.10
  max_open_positions: 5
  min_confidence: 0.65
  max_spread: 3.0

# Trading Bot
bot:
  symbols: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
  timeframe: "1h"
  prediction_interval: 300  # segundos
  trading_enabled: false  # cambiar a true para trading real

# Modelo
model:
  path: "models/lit_ml_model.pkl"
  retrain_interval: 24  # horas

# Logging
logging:
  level: "INFO"
  file: "logs/trading.log"
  max_size: "10MB"
  backup_count: 5
"""
    
    with open("config/trading_config.yaml", "w") as f:
        f.write(config_content)
    print("   ✅ config/trading_config.yaml creado")
    
    print("✅ Archivos de configuración creados")
    return True


def test_system():
    """Prueba el sistema básico."""
    print("\n🧪 Probando el sistema...")
    
    try:
        # Importar módulos principales
        sys.path.append(str(Path.cwd()))
        
        from src.trading.trade_executor import create_trade_executor
        print("   ✅ Trade Executor importado correctamente")
        
        from src.utils.logger import log
        print("   ✅ Sistema de logging funcional")
        
        # Probar conexión MT5 (sin ejecutar operaciones)
        executor = create_trade_executor("moderate")
        print("   ✅ Ejecutor creado correctamente")
        
        print("✅ Sistema básico funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba del sistema: {str(e)}")
        return False


def show_usage_examples():
    """Muestra ejemplos de uso."""
    print("\n📚 EJEMPLOS DE USO:")
    print("-" * 40)
    
    print("\n1. Probar ejemplos:")
    print("   python examples/trade_executor_examples.py")
    
    print("\n2. Ejecutar bot en modo análisis:")
    print("   python scripts/run_trading_bot.py --mode analysis --duration 1")
    
    print("\n3. Ejecutar bot en modo demo:")
    print("   python scripts/run_trading_bot.py --mode demo --risk conservative")
    
    print("\n4. Trading real (¡CUIDADO!):")
    print("   python scripts/run_trading_bot.py --mode trading --risk moderate")
    
    print("\n5. Uso programático:")
    print("""   from src.trading.trade_executor import execute_signal_simple
   
   success = execute_signal_simple({
       'symbol': 'EURUSD',
       'signal': 'buy',
       'confidence': 0.75,
       'price': 1.0850
   })""")


def show_next_steps():
    """Muestra los próximos pasos."""
    print("\n🚀 PRÓXIMOS PASOS:")
    print("-" * 40)
    
    print("\n1. ✅ Revisar configuración:")
    print("   - Editar .env.example y renombrar a .env")
    print("   - Modificar config/trading_config.yaml según necesidades")
    
    print("\n2. ✅ Probar en modo demo:")
    print("   - Ejecutar ejemplos para familiarizarse")
    print("   - Usar modo 'analysis' para ver predicciones")
    print("   - Probar modo 'demo' con cuenta demo")
    
    print("\n3. ✅ Configurar cuenta real (opcional):")
    print("   - Obtener credenciales de broker real")
    print("   - Actualizar configuración MT5")
    print("   - Empezar con configuración conservadora")
    
    print("\n4. ✅ Monitoreo y mantenimiento:")
    print("   - Revisar logs regularmente")
    print("   - Monitorear rendimiento")
    print("   - Ajustar parámetros según resultados")


def main():
    """Función principal del instalador."""
    print_header()
    
    # Verificaciones previas
    if not check_python_version():
        return False
    
    # Instalación
    steps = [
        ("Instalando dependencias", install_requirements),
        ("Verificando MetaTrader5", verify_mt5_installation),
        ("Creando directorios", create_directories),
        ("Creando configuración", create_config_files),
        ("Probando sistema", test_system)
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Error en: {step_name}")
            print("   La instalación no se completó correctamente.")
            return False
    
    # Instalación exitosa
    print("\n" + "=" * 60)
    print("🎉 ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!")
    print("=" * 60)
    
    show_usage_examples()
    show_next_steps()
    
    print("\n" + "=" * 60)
    print("📞 SOPORTE:")
    print("   - Documentación: TRADE_EXECUTOR_GUIDE.md")
    print("   - Ejemplos: examples/trade_executor_examples.py")
    print("   - Logs: logs/trading.log")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Instalación cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
        sys.exit(1) 