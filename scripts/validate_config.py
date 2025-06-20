#!/usr/bin/env python3
"""
Script de validación de configuración con python-dotenv.

Este script verifica que el sistema de configuración esté funcionando
correctamente y que todas las variables de entorno estén bien configuradas.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.core.config import config, ConfigurationError
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    print("💡 Asegúrate de instalar las dependencias: pip install -r requirements.txt")
    sys.exit(1)


class ConfigValidator:
    """Validador de configuración del sistema."""
    
    def __init__(self):
        """Inicializa el validador."""
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
    
    def validate_all(self) -> bool:
        """
        Ejecuta todas las validaciones.
        
        Returns:
            bool: True si todas las validaciones pasan.
        """
        print("🔍 INICIANDO VALIDACIÓN COMPLETA DE CONFIGURACIÓN")
        print("=" * 60)
        
        # Ejecutar todas las validaciones
        validations = [
            self._validate_dotenv_file,
            self._validate_environment_variables,
            self._validate_config_loading,
            self._validate_config_values,
            self._validate_paths,
            self._validate_credentials,
            self._validate_dependencies,
            self._validate_file_permissions
        ]
        
        for validation in validations:
            try:
                validation()
            except Exception as e:
                self.errors.append(f"Error en {validation.__name__}: {str(e)}")
        
        # Mostrar resultados
        self._show_results()
        
        return len(self.errors) == 0
    
    def _validate_dotenv_file(self) -> None:
        """Valida la existencia y formato del archivo .env."""
        print("\n📄 Validando archivo .env...")
        
        env_file = Path('.env')
        env_example = Path('.env.example')
        
        if not env_example.exists():
            self.errors.append("Archivo .env.example no encontrado")
            return
        else:
            self.info.append("✅ Archivo .env.example encontrado")
        
        if not env_file.exists():
            self.warnings.append("Archivo .env no encontrado, usando valores por defecto")
            self.info.append("💡 Copia .env.example a .env y configura tus valores")
        else:
            self.info.append("✅ Archivo .env encontrado")
            
            # Verificar que .env no esté vacío
            if env_file.stat().st_size == 0:
                self.warnings.append("Archivo .env está vacío")
            
            # Verificar formato básico
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                valid_lines = 0
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' not in line:
                            self.warnings.append(f"Línea {line_num} en .env no tiene formato válido: {line}")
                        else:
                            valid_lines += 1
                
                self.info.append(f"✅ {valid_lines} variables válidas en .env")
                
            except Exception as e:
                self.errors.append(f"Error leyendo .env: {str(e)}")
    
    def _validate_environment_variables(self) -> None:
        """Valida las variables de entorno críticas."""
        print("\n🔧 Validando variables de entorno...")
        
        # Variables críticas que deben estar definidas
        critical_vars = [
            'TRADING_SYMBOL',
            'TRADING_BALANCE_INICIAL',
            'TRADING_RISK_PER_TRADE',
            'ML_MIN_CONFIDENCE',
            'LIT_MIN_CONFIDENCE'
        ]
        
        # Variables importantes pero opcionales
        important_vars = [
            'BROKER_TYPE',
            'DATA_SOURCE',
            'LOG_LEVEL',
            'TELEGRAM_BOT_TOKEN',
            'ALPACA_API_KEY'
        ]
        
        # Verificar variables críticas
        for var in critical_vars:
            value = os.getenv(var)
            if value is None:
                self.warnings.append(f"Variable crítica {var} no definida, usando valor por defecto")
            else:
                self.info.append(f"✅ {var} = {value}")
        
        # Verificar variables importantes
        defined_important = 0
        for var in important_vars:
            value = os.getenv(var)
            if value is not None:
                defined_important += 1
                # No mostrar valores sensibles
                if 'TOKEN' in var or 'KEY' in var or 'PASSWORD' in var:
                    self.info.append(f"✅ {var} = ***DEFINIDA***")
                else:
                    self.info.append(f"✅ {var} = {value}")
        
        self.info.append(f"📊 Variables importantes definidas: {defined_important}/{len(important_vars)}")
    
    def _validate_config_loading(self) -> None:
        """Valida que la configuración se cargue correctamente."""
        print("\n⚙️ Validando carga de configuración...")
        
        try:
            # Verificar que config se haya inicializado
            if config is None:
                self.errors.append("Objeto config no inicializado")
                return
            
            # Verificar secciones principales
            sections = [
                'general', 'trading', 'ml', 'lit', 'risk',
                'data', 'broker', 'logging', 'notifications',
                'paths', 'database', 'backtest', 'security', 'monitoring'
            ]
            
            for section in sections:
                if hasattr(config, section):
                    self.info.append(f"✅ Sección {section} cargada")
                else:
                    self.errors.append(f"Sección {section} no encontrada")
            
            # Verificar propiedades de compatibilidad
            compat_props = ['bot_name', 'version', 'data_source']
            for prop in compat_props:
                if hasattr(config, prop):
                    value = getattr(config, prop)
                    self.info.append(f"✅ Propiedad {prop} = {value}")
                else:
                    self.warnings.append(f"Propiedad de compatibilidad {prop} no encontrada")
            
        except Exception as e:
            self.errors.append(f"Error cargando configuración: {str(e)}")
    
    def _validate_config_values(self) -> None:
        """Valida los valores de configuración."""
        print("\n🎯 Validando valores de configuración...")
        
        try:
            # Validar trading
            if config.trading.balance_inicial <= 0:
                self.errors.append("Balance inicial debe ser positivo")
            else:
                self.info.append(f"✅ Balance inicial: ${config.trading.balance_inicial:,.2f}")
            
            if not (0 < config.trading.risk_per_trade <= 0.1):
                self.errors.append("Riesgo por trade debe estar entre 0 y 10%")
            else:
                self.info.append(f"✅ Riesgo por trade: {config.trading.risk_per_trade * 100:.1f}%")
            
            if config.trading.max_positions <= 0:
                self.errors.append("Máximo de posiciones debe ser positivo")
            else:
                self.info.append(f"✅ Max posiciones: {config.trading.max_positions}")
            
            # Validar ML
            if not (0 < config.ml.min_confidence <= 1):
                self.errors.append("Confianza mínima ML debe estar entre 0 y 100%")
            else:
                self.info.append(f"✅ Confianza ML: {config.ml.min_confidence * 100:.1f}%")
            
            # Validar LIT
            if not (0 < config.lit.min_confidence <= 1):
                self.errors.append("Confianza mínima LIT debe estar entre 0 y 100%")
            else:
                self.info.append(f"✅ Confianza LIT: {config.lit.min_confidence * 100:.1f}%")
            
            # Validar timeframe
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
            if config.trading.timeframe not in valid_timeframes:
                self.warnings.append(f"Timeframe {config.trading.timeframe} puede no ser válido")
            else:
                self.info.append(f"✅ Timeframe: {config.trading.timeframe}")
            
            # Validar símbolo
            if len(config.trading.symbol) < 3:
                self.warnings.append("Símbolo de trading parece muy corto")
            else:
                self.info.append(f"✅ Símbolo: {config.trading.symbol}")
            
        except Exception as e:
            self.errors.append(f"Error validando valores: {str(e)}")
    
    def _validate_paths(self) -> None:
        """Valida las rutas del sistema."""
        print("\n📁 Validando rutas del sistema...")
        
        try:
            paths = config.get_paths()
            
            # Verificar que el directorio base existe
            if not paths['base'].exists():
                self.errors.append(f"Directorio base no existe: {paths['base']}")
            else:
                self.info.append(f"✅ Directorio base: {paths['base']}")
            
            # Verificar directorios críticos
            critical_dirs = ['data', 'models', 'logs']
            for dir_name in critical_dirs:
                if dir_name in paths:
                    path = paths[dir_name]
                    if path.exists():
                        self.info.append(f"✅ Directorio {dir_name}: {path}")
                    else:
                        self.warnings.append(f"Directorio {dir_name} no existe: {path}")
                        # Intentar crear
                        try:
                            path.mkdir(parents=True, exist_ok=True)
                            self.info.append(f"✅ Directorio {dir_name} creado: {path}")
                        except Exception as e:
                            self.errors.append(f"No se pudo crear {dir_name}: {str(e)}")
            
            # Verificar rutas específicas
            model_path = config.get_model_path()
            scaler_path = config.get_scaler_path()
            log_path = config.get_log_path()
            
            self.info.append(f"📄 Ruta modelo: {model_path}")
            self.info.append(f"📄 Ruta scaler: {scaler_path}")
            self.info.append(f"📄 Ruta log: {log_path}")
            
        except Exception as e:
            self.errors.append(f"Error validando rutas: {str(e)}")
    
    def _validate_credentials(self) -> None:
        """Valida las credenciales configuradas."""
        print("\n🔐 Validando credenciales...")
        
        try:
            # Validar credenciales de broker
            broker_creds = config.get_broker_credentials()
            if broker_creds:
                self.info.append(f"✅ Credenciales de broker ({config.broker.type}): {len(broker_creds)} campos")
                
                # Verificar campos específicos por tipo de broker
                if config.broker.type == 'mt5':
                    required = ['login', 'password', 'server']
                    missing = [field for field in required if not broker_creds.get(field)]
                    if missing:
                        self.warnings.append(f"Credenciales MT5 incompletas: {missing}")
                
                elif config.broker.type == 'alpaca':
                    required = ['api_key', 'secret_key']
                    missing = [field for field in required if not broker_creds.get(field)]
                    if missing:
                        self.warnings.append(f"Credenciales Alpaca incompletas: {missing}")
            else:
                if config.broker.type != 'demo':
                    self.warnings.append(f"Sin credenciales para broker {config.broker.type}")
            
            # Validar credenciales de datos
            data_creds = config.get_data_credentials()
            if data_creds:
                self.info.append(f"✅ Credenciales de datos ({config.data.source}): {len(data_creds)} campos")
            else:
                if config.data.source in ['ccxt', 'mt5']:
                    self.warnings.append(f"Sin credenciales para fuente de datos {config.data.source}")
            
            # Validar notificaciones
            if config.notifications.telegram_enabled:
                if config.notifications.telegram_bot_token and config.notifications.telegram_chat_id:
                    self.info.append("✅ Credenciales Telegram configuradas")
                else:
                    self.errors.append("Telegram habilitado pero faltan credenciales")
            
            if config.notifications.email_enabled:
                if config.notifications.email_username and config.notifications.email_password:
                    self.info.append("✅ Credenciales Email configuradas")
                else:
                    self.errors.append("Email habilitado pero faltan credenciales")
            
        except Exception as e:
            self.errors.append(f"Error validando credenciales: {str(e)}")
    
    def _validate_dependencies(self) -> None:
        """Valida las dependencias del sistema."""
        print("\n📦 Validando dependencias...")
        
        # Dependencias críticas
        critical_deps = [
            'pandas', 'numpy', 'scikit-learn', 'xgboost',
            'joblib', 'python-dotenv', 'loguru'
        ]
        
        # Dependencias opcionales
        optional_deps = [
            'yfinance', 'ccxt', 'matplotlib', 'plotly',
            'pytest', 'ta-lib'
        ]
        
        missing_critical = []
        missing_optional = []
        
        for dep in critical_deps:
            try:
                __import__(dep.replace('-', '_'))
                self.info.append(f"✅ {dep}")
            except ImportError:
                missing_critical.append(dep)
        
        for dep in optional_deps:
            try:
                __import__(dep.replace('-', '_'))
                self.info.append(f"✅ {dep}")
            except ImportError:
                missing_optional.append(dep)
        
        if missing_critical:
            self.errors.append(f"Dependencias críticas faltantes: {missing_critical}")
        
        if missing_optional:
            self.warnings.append(f"Dependencias opcionales faltantes: {missing_optional}")
        
        # Verificar versión de Python
        python_version = sys.version_info
        if python_version < (3, 8):
            self.errors.append(f"Python {python_version.major}.{python_version.minor} no soportado, requiere >= 3.8")
        else:
            self.info.append(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    def _validate_file_permissions(self) -> None:
        """Valida los permisos de archivos."""
        print("\n🔒 Validando permisos de archivos...")
        
        try:
            paths = config.get_paths()
            
            # Verificar permisos de escritura en directorios críticos
            write_dirs = ['data', 'models', 'logs', 'results']
            for dir_name in write_dirs:
                if dir_name in paths:
                    path = paths[dir_name]
                    if path.exists():
                        if os.access(path, os.W_OK):
                            self.info.append(f"✅ Escritura en {dir_name}: OK")
                        else:
                            self.errors.append(f"Sin permisos de escritura en {dir_name}: {path}")
            
            # Verificar archivo .env
            env_file = Path('.env')
            if env_file.exists():
                if os.access(env_file, os.R_OK):
                    self.info.append("✅ Lectura de .env: OK")
                else:
                    self.errors.append("Sin permisos de lectura en .env")
                
                # Verificar que .env no sea ejecutable (seguridad)
                if os.access(env_file, os.X_OK):
                    self.warnings.append("Archivo .env es ejecutable (riesgo de seguridad)")
            
        except Exception as e:
            self.errors.append(f"Error validando permisos: {str(e)}")
    
    def _show_results(self) -> None:
        """Muestra los resultados de la validación."""
        print("\n" + "=" * 60)
        print("📊 RESULTADOS DE LA VALIDACIÓN")
        print("=" * 60)
        
        # Mostrar información
        if self.info:
            print(f"\n✅ INFORMACIÓN ({len(self.info)}):")
            for info in self.info:
                print(f"  {info}")
        
        # Mostrar advertencias
        if self.warnings:
            print(f"\n⚠️ ADVERTENCIAS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  ⚠️ {warning}")
        
        # Mostrar errores
        if self.errors:
            print(f"\n❌ ERRORES ({len(self.errors)}):")
            for error in self.errors:
                print(f"  ❌ {error}")
        
        # Resumen final
        print(f"\n📈 RESUMEN:")
        print(f"  ✅ Información: {len(self.info)}")
        print(f"  ⚠️ Advertencias: {len(self.warnings)}")
        print(f"  ❌ Errores: {len(self.errors)}")
        
        if len(self.errors) == 0:
            print(f"\n🎉 VALIDACIÓN EXITOSA - El sistema está listo para usar")
        else:
            print(f"\n🚨 VALIDACIÓN FALLIDA - Corrige los errores antes de continuar")


def main():
    """Función principal."""
    print("🔍 VALIDADOR DE CONFIGURACIÓN - BOT TRADING LIT + ML")
    print("Usando python-dotenv para gestión de variables de entorno")
    
    validator = ConfigValidator()
    success = validator.validate_all()
    
    if success:
        print("\n💡 PRÓXIMOS PASOS:")
        print("  1. Ejecuta: python examples/config_example.py")
        print("  2. Ejecuta: python main.py validate")
        print("  3. Ejecuta: python main.py status")
        print("  4. Configura tus credenciales en .env")
        print("  5. Ejecuta: python main.py train EURUSD 1000")
        return 0
    else:
        print("\n🔧 PARA CORREGIR ERRORES:")
        print("  1. Revisa el archivo .env.example")
        print("  2. Copia .env.example a .env")
        print("  3. Configura las variables necesarias")
        print("  4. Instala dependencias: pip install -r requirements.txt")
        print("  5. Ejecuta este script nuevamente")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 