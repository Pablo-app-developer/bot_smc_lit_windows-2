#!/usr/bin/env python3
"""
Ejemplo de uso del sistema de configuración con python-dotenv.

Este script demuestra cómo cargar y usar la configuración del bot
desde variables de entorno usando el módulo config.py.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio src al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import config, ConfigurationError


def main():
    """Función principal del ejemplo."""
    print("=" * 60)
    print("EJEMPLO DE CONFIGURACIÓN CON PYTHON-DOTENV")
    print("=" * 60)
    
    try:
        # 1. Mostrar configuración general
        print("\n🔧 CONFIGURACIÓN GENERAL:")
        print(f"  Bot: {config.general.bot_name}")
        print(f"  Versión: {config.general.version}")
        print(f"  Entorno: {config.general.environment}")
        print(f"  Debug: {config.general.debug}")
        print(f"  Modo desarrollo: {config.general.development_mode}")
        
        # 2. Mostrar configuración de trading
        print("\n💰 CONFIGURACIÓN DE TRADING:")
        print(f"  Símbolo: {config.trading.symbol}")
        print(f"  Timeframe: {config.trading.timeframe}")
        print(f"  Balance inicial: ${config.trading.balance_inicial:,.2f}")
        print(f"  Riesgo por trade: {config.trading.risk_per_trade * 100:.1f}%")
        print(f"  Max posiciones: {config.trading.max_positions}")
        print(f"  Max drawdown: {config.trading.max_drawdown * 100:.1f}%")
        print(f"  Intervalo de chequeo: {config.trading.check_interval}s")
        
        # 3. Mostrar configuración de ML
        print("\n🤖 CONFIGURACIÓN DE MACHINE LEARNING:")
        print(f"  Tipo de modelo: {config.ml.model_type}")
        print(f"  Confianza mínima: {config.ml.min_confidence * 100:.1f}%")
        print(f"  Feature lookback: {config.ml.feature_lookback}")
        print(f"  Frecuencia re-entrenamiento: {config.ml.retrain_frequency}")
        print(f"  Optimizar hiperparámetros: {config.ml.optimize_hyperparams}")
        print(f"  Trials Optuna: {config.ml.optuna_trials}")
        
        # 4. Mostrar configuración LIT
        print("\n🎯 CONFIGURACIÓN DE ESTRATEGIA LIT:")
        print(f"  Lookback periods: {config.lit.lookback_periods}")
        print(f"  Confianza mínima: {config.lit.min_confidence * 100:.1f}%")
        print(f"  Umbral liquidez: {config.lit.liquidity_threshold}")
        print(f"  Min toques inducement: {config.lit.inducement_min_touches}")
        print(f"  Tamaño min ineficiencia: {config.lit.inefficiency_min_size}")
        print(f"  Multiplicador ATR: {config.lit.atr_multiplier}")
        
        # 5. Mostrar configuración de riesgo
        print("\n🛡️ CONFIGURACIÓN DE GESTIÓN DE RIESGO:")
        print(f"  Ratio TP/SL: {config.risk.tp_sl_ratio}:1")
        print(f"  Trailing stop: {config.risk.use_trailing_stop}")
        print(f"  Trailing stop ATR: {config.risk.trailing_stop_atr}")
        print(f"  Scale out en profit: {config.risk.scale_out_profit}")
        print(f"  Max riesgo portafolio: {config.risk.max_portfolio_risk * 100:.1f}%")
        
        # 6. Mostrar configuración de datos
        print("\n📊 CONFIGURACIÓN DE FUENTES DE DATOS:")
        print(f"  Fuente principal: {config.data.source}")
        print(f"  Ruta CSV: {config.data.csv_path}")
        if config.data.source == 'ccxt':
            print(f"  Exchange CCXT: {config.data.ccxt_exchange}")
            print(f"  Sandbox: {config.data.ccxt_sandbox}")
        
        # 7. Mostrar configuración de broker
        print("\n🏦 CONFIGURACIÓN DE BROKER:")
        print(f"  Tipo: {config.broker.type}")
        if config.broker.type == 'alpaca':
            print(f"  Base URL: {config.broker.alpaca_base_url}")
        elif config.broker.type == 'interactive_brokers':
            print(f"  Host: {config.broker.ib_host}:{config.broker.ib_port}")
        
        # 8. Mostrar configuración de logging
        print("\n📝 CONFIGURACIÓN DE LOGGING:")
        print(f"  Nivel: {config.logging.level}")
        print(f"  Archivo: {config.logging.file}")
        print(f"  Rotación: {config.logging.rotation}")
        print(f"  Retención: {config.logging.retention}")
        print(f"  Formato: {config.logging.format}")
        print(f"  A consola: {config.logging.to_console}")
        
        # 9. Mostrar configuración de notificaciones
        print("\n🔔 CONFIGURACIÓN DE NOTIFICACIONES:")
        print(f"  Telegram habilitado: {config.notifications.telegram_enabled}")
        print(f"  Email habilitado: {config.notifications.email_enabled}")
        print(f"  Discord habilitado: {config.notifications.discord_enabled}")
        
        # 10. Mostrar rutas del sistema
        print("\n📁 RUTAS DEL SISTEMA:")
        paths = config.get_paths()
        for name, path in paths.items():
            print(f"  {name.capitalize()}: {path}")
        
        # 11. Mostrar configuración de backtesting
        print("\n📈 CONFIGURACIÓN DE BACKTESTING:")
        print(f"  Comisión: {config.backtest.commission}")
        print(f"  Incluir slippage: {config.backtest.include_slippage}")
        print(f"  Fecha inicio: {config.backtest.start_date}")
        print(f"  Fecha fin: {config.backtest.end_date}")
        
        # 12. Mostrar configuración de monitoreo
        print("\n📊 CONFIGURACIÓN DE MONITOREO:")
        print(f"  Métricas de sistema: {config.monitoring.enable_system_metrics}")
        print(f"  Intervalo reporte: {config.monitoring.metrics_report_interval}s")
        print(f"  Alertas performance: {config.monitoring.enable_performance_alerts}")
        print(f"  Umbral CPU: {config.monitoring.cpu_alert_threshold}%")
        print(f"  Umbral memoria: {config.monitoring.memory_alert_threshold}%")
        
        # 13. Verificar estado del entorno
        print("\n🔍 ESTADO DEL ENTORNO:")
        print(f"  Es producción: {config.is_production()}")
        print(f"  Es desarrollo: {config.is_development()}")
        
        # 14. Mostrar credenciales (sin valores sensibles)
        print("\n🔐 CREDENCIALES CONFIGURADAS:")
        broker_creds = config.get_broker_credentials()
        data_creds = config.get_data_credentials()
        
        if broker_creds:
            print(f"  Broker ({config.broker.type}): {len(broker_creds)} credenciales")
        else:
            print(f"  Broker ({config.broker.type}): Sin credenciales")
            
        if data_creds:
            print(f"  Datos ({config.data.source}): {len(data_creds)} credenciales")
        else:
            print(f"  Datos ({config.data.source}): Sin credenciales")
        
        # 15. Validar configuración
        print("\n✅ VALIDACIÓN DE CONFIGURACIÓN:")
        is_valid = config.validate()
        print(f"  Estado: {'✅ VÁLIDA' if is_valid else '❌ INVÁLIDA'}")
        
        # 16. Crear directorios necesarios
        print("\n📂 CREANDO DIRECTORIOS:")
        config.create_directories()
        paths = config.get_paths()
        for name in ['data', 'models', 'logs', 'results']:
            if name in paths and paths[name].exists():
                print(f"  ✅ {name}: {paths[name]}")
            else:
                print(f"  ❌ {name}: No se pudo crear")
        
        # 17. Mostrar rutas específicas
        print("\n🎯 RUTAS ESPECÍFICAS:")
        print(f"  Modelo principal: {config.get_model_path()}")
        print(f"  Scaler: {config.get_scaler_path()}")
        print(f"  Log principal: {config.get_log_path()}")
        
        # 18. Convertir a diccionario
        print("\n📋 CONFIGURACIÓN COMO DICCIONARIO:")
        config_dict = config.to_dict()
        print(f"  Secciones: {list(config_dict.keys())}")
        print(f"  Total parámetros: {sum(len(section) for section in config_dict.values())}")
        
        print("\n" + "=" * 60)
        print("✅ EJEMPLO COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        
    except ConfigurationError as e:
        print(f"\n❌ ERROR DE CONFIGURACIÓN: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        return 1
    
    return 0


def demo_environment_variables():
    """Demuestra cómo las variables de entorno afectan la configuración."""
    print("\n" + "=" * 60)
    print("DEMO: VARIABLES DE ENTORNO")
    print("=" * 60)
    
    # Mostrar algunas variables de entorno importantes
    important_vars = [
        'TRADING_SYMBOL',
        'TRADING_BALANCE_INICIAL',
        'TRADING_RISK_PER_TRADE',
        'ML_MIN_CONFIDENCE',
        'LIT_MIN_CONFIDENCE',
        'BROKER_TYPE',
        'DATA_SOURCE',
        'LOG_LEVEL'
    ]
    
    print("\n📋 VARIABLES DE ENTORNO IMPORTANTES:")
    for var in important_vars:
        value = os.getenv(var, 'NO DEFINIDA')
        print(f"  {var}: {value}")
    
    print("\n💡 PARA CAMBIAR LA CONFIGURACIÓN:")
    print("  1. Edita el archivo .env")
    print("  2. O define variables de entorno:")
    print("     export TRADING_SYMBOL=GBPUSD")
    print("     export TRADING_RISK_PER_TRADE=0.01")
    print("  3. Reinicia el bot para aplicar cambios")


def demo_config_validation():
    """Demuestra la validación de configuración."""
    print("\n" + "=" * 60)
    print("DEMO: VALIDACIÓN DE CONFIGURACIÓN")
    print("=" * 60)
    
    # Simular configuración inválida
    print("\n🧪 SIMULANDO CONFIGURACIÓN INVÁLIDA:")
    
    # Guardar valores originales
    original_risk = config.trading.risk_per_trade
    original_balance = config.trading.balance_inicial
    
    try:
        # Configurar valores inválidos
        config.trading.risk_per_trade = 1.5  # > 100%
        config.trading.balance_inicial = -1000  # Negativo
        
        print(f"  Riesgo por trade: {config.trading.risk_per_trade * 100}% (INVÁLIDO)")
        print(f"  Balance inicial: ${config.trading.balance_inicial} (INVÁLIDO)")
        
        # Intentar validar
        is_valid = config.validate()
        print(f"  Resultado validación: {'✅ VÁLIDA' if is_valid else '❌ INVÁLIDA'}")
        
    finally:
        # Restaurar valores originales
        config.trading.risk_per_trade = original_risk
        config.trading.balance_inicial = original_balance
        
        print(f"\n🔄 VALORES RESTAURADOS:")
        print(f"  Riesgo por trade: {config.trading.risk_per_trade * 100:.1f}% (VÁLIDO)")
        print(f"  Balance inicial: ${config.trading.balance_inicial:,.2f} (VÁLIDO)")


if __name__ == "__main__":
    # Ejecutar ejemplo principal
    exit_code = main()
    
    # Ejecutar demos adicionales
    demo_environment_variables()
    demo_config_validation()
    
    sys.exit(exit_code) 