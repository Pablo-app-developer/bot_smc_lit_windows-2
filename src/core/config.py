"""
Módulo de configuración del bot de trading LIT + ML.

Este módulo maneja la carga, validación y gestión de toda la configuración del bot
desde variables de entorno usando python-dotenv. Proporciona una interfaz robusta
y type-safe para acceder a la configuración en todo el sistema.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from dotenv import load_dotenv
import logging

# Cargar variables de entorno desde .env
load_dotenv()


def str_to_bool(value: str) -> bool:
    """Convierte string a boolean de forma segura."""
    return value.lower() in ('true', '1', 'yes', 'on', 'enabled')


def get_env_float(key: str, default: float) -> float:
    """Obtiene variable de entorno como float con valor por defecto."""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_env_int(key: str, default: int) -> int:
    """Obtiene variable de entorno como int con valor por defecto."""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def get_env_bool(key: str, default: bool) -> bool:
    """Obtiene variable de entorno como bool con valor por defecto."""
    value = os.getenv(key, str(default))
    return str_to_bool(value)


@dataclass
class GeneralConfig:
    """Configuración general del bot."""
    
    bot_name: str = field(default_factory=lambda: os.getenv("BOT_NAME", "LIT_ML_Trading_Bot"))
    version: str = field(default_factory=lambda: os.getenv("VERSION", "1.0.0"))
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: get_env_bool("DEBUG", True))
    development_mode: bool = field(default_factory=lambda: get_env_bool("DEVELOPMENT_MODE", True))


@dataclass
class TradingConfig:
    """Configuración de trading del bot."""
    
    symbol: str = field(default_factory=lambda: os.getenv("TRADING_SYMBOL", "EURUSD"))
    timeframe: str = field(default_factory=lambda: os.getenv("TRADING_TIMEFRAME", "15m"))
    balance_inicial: float = field(default_factory=lambda: get_env_float("TRADING_BALANCE_INICIAL", 10000.0))
    risk_per_trade: float = field(default_factory=lambda: get_env_float("TRADING_RISK_PER_TRADE", 0.02))
    max_positions: int = field(default_factory=lambda: get_env_int("TRADING_MAX_POSITIONS", 3))
    leverage: float = field(default_factory=lambda: get_env_float("TRADING_LEVERAGE", 1.0))
    max_drawdown: float = field(default_factory=lambda: get_env_float("TRADING_MAX_DRAWDOWN", 0.10))
    check_interval: int = field(default_factory=lambda: get_env_int("TRADING_CHECK_INTERVAL", 300))
    lookback_periods: int = field(default_factory=lambda: get_env_int("TRADING_LOOKBACK_PERIODS", 1000))
    slippage_pips: float = field(default_factory=lambda: get_env_float("TRADING_SLIPPAGE_PIPS", 1.0))
    spread_pips: float = field(default_factory=lambda: get_env_float("TRADING_SPREAD_PIPS", 1.5))


@dataclass
class MLConfig:
    """Configuración del modelo de Machine Learning."""
    
    model_type: str = field(default_factory=lambda: os.getenv("ML_MODEL_TYPE", "xgboost"))
    min_confidence: float = field(default_factory=lambda: get_env_float("ML_MIN_CONFIDENCE", 0.70))
    feature_lookback: int = field(default_factory=lambda: get_env_int("ML_FEATURE_LOOKBACK", 100))
    retrain_frequency: str = field(default_factory=lambda: os.getenv("ML_RETRAIN_FREQUENCY", "weekly"))
    min_training_samples: int = field(default_factory=lambda: get_env_int("ML_MIN_TRAINING_SAMPLES", 1000))
    cv_folds: int = field(default_factory=lambda: get_env_int("ML_CV_FOLDS", 5))
    optimize_hyperparams: bool = field(default_factory=lambda: get_env_bool("ML_OPTIMIZE_HYPERPARAMS", True))
    optuna_trials: int = field(default_factory=lambda: get_env_int("ML_OPTUNA_TRIALS", 100))


@dataclass
class LITConfig:
    """Configuración de la estrategia LIT."""
    
    lookback_periods: int = field(default_factory=lambda: get_env_int("LIT_LOOKBACK_PERIODS", 50))
    min_confidence: float = field(default_factory=lambda: get_env_float("LIT_MIN_CONFIDENCE", 0.60))
    liquidity_threshold: float = field(default_factory=lambda: get_env_float("LIT_LIQUIDITY_THRESHOLD", 0.001))
    inducement_min_touches: int = field(default_factory=lambda: get_env_int("LIT_INDUCEMENT_MIN_TOUCHES", 2))
    inefficiency_min_size: float = field(default_factory=lambda: get_env_float("LIT_INEFFICIENCY_MIN_SIZE", 0.0005))
    equal_level_window: int = field(default_factory=lambda: get_env_int("LIT_EQUAL_LEVEL_WINDOW", 10))
    equal_level_tolerance: float = field(default_factory=lambda: get_env_float("LIT_EQUAL_LEVEL_TOLERANCE", 2.0))
    atr_multiplier: float = field(default_factory=lambda: get_env_float("LIT_ATR_MULTIPLIER", 2.0))


@dataclass
class RiskConfig:
    """Configuración de gestión de riesgo."""
    
    tp_sl_ratio: float = field(default_factory=lambda: get_env_float("RISK_TP_SL_RATIO", 2.0))
    use_trailing_stop: bool = field(default_factory=lambda: get_env_bool("RISK_USE_TRAILING_STOP", True))
    trailing_stop_atr: float = field(default_factory=lambda: get_env_float("RISK_TRAILING_STOP_ATR", 1.5))
    scale_out_profit: bool = field(default_factory=lambda: get_env_bool("RISK_SCALE_OUT_PROFIT", False))
    scale_out_percentage: float = field(default_factory=lambda: get_env_float("RISK_SCALE_OUT_PERCENTAGE", 0.5))
    max_portfolio_risk: float = field(default_factory=lambda: get_env_float("RISK_MAX_PORTFOLIO_RISK", 0.06))


@dataclass
class DataConfig:
    """Configuración de fuentes de datos."""
    
    source: str = field(default_factory=lambda: os.getenv("DATA_SOURCE", "yfinance"))
    csv_path: str = field(default_factory=lambda: os.getenv("DATA_CSV_PATH", "data/historical_data.csv"))
    
    # CCXT Configuration
    ccxt_exchange: str = field(default_factory=lambda: os.getenv("CCXT_EXCHANGE", "binance"))
    ccxt_api_key: Optional[str] = field(default_factory=lambda: os.getenv("CCXT_API_KEY"))
    ccxt_secret_key: Optional[str] = field(default_factory=lambda: os.getenv("CCXT_SECRET_KEY"))
    ccxt_sandbox: bool = field(default_factory=lambda: get_env_bool("CCXT_SANDBOX", True))


@dataclass
class BrokerConfig:
    """Configuración del broker."""
    
    type: str = field(default_factory=lambda: os.getenv("BROKER_TYPE", "demo"))
    
    # MetaTrader 5
    mt5_login: Optional[str] = field(default_factory=lambda: os.getenv("MT5_LOGIN"))
    mt5_password: Optional[str] = field(default_factory=lambda: os.getenv("MT5_PASSWORD"))
    mt5_server: Optional[str] = field(default_factory=lambda: os.getenv("MT5_SERVER"))
    
    # Alpaca
    alpaca_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ALPACA_API_KEY"))
    alpaca_secret_key: Optional[str] = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY"))
    alpaca_base_url: str = field(default_factory=lambda: os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))
    
    # Interactive Brokers
    ib_host: str = field(default_factory=lambda: os.getenv("IB_HOST", "127.0.0.1"))
    ib_port: int = field(default_factory=lambda: get_env_int("IB_PORT", 7497))
    ib_client_id: int = field(default_factory=lambda: get_env_int("IB_CLIENT_ID", 1))


@dataclass
class LoggingConfig:
    """Configuración del sistema de logging."""
    
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "logs/trading_bot.log"))
    rotation: str = field(default_factory=lambda: os.getenv("LOG_ROTATION", "1 day"))
    retention: str = field(default_factory=lambda: os.getenv("LOG_RETENTION", "30 days"))
    format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "detailed"))
    to_console: bool = field(default_factory=lambda: get_env_bool("LOG_TO_CONSOLE", True))


@dataclass
class NotificationConfig:
    """Configuración de notificaciones."""
    
    # Telegram
    telegram_enabled: bool = field(default_factory=lambda: get_env_bool("TELEGRAM_ENABLED", False))
    telegram_bot_token: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN"))
    telegram_chat_id: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID"))
    
    # Email
    email_enabled: bool = field(default_factory=lambda: get_env_bool("EMAIL_ENABLED", False))
    email_smtp_server: str = field(default_factory=lambda: os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com"))
    email_smtp_port: int = field(default_factory=lambda: get_env_int("EMAIL_SMTP_PORT", 587))
    email_username: Optional[str] = field(default_factory=lambda: os.getenv("EMAIL_USERNAME"))
    email_password: Optional[str] = field(default_factory=lambda: os.getenv("EMAIL_PASSWORD"))
    email_to: Optional[str] = field(default_factory=lambda: os.getenv("EMAIL_TO"))
    
    # Discord
    discord_enabled: bool = field(default_factory=lambda: get_env_bool("DISCORD_ENABLED", False))
    discord_webhook_url: Optional[str] = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL"))


@dataclass
class PathsConfig:
    """Configuración de rutas y archivos."""
    
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "data"))
    models_dir: str = field(default_factory=lambda: os.getenv("MODELS_DIR", "models"))
    logs_dir: str = field(default_factory=lambda: os.getenv("LOGS_DIR", "logs"))
    results_dir: str = field(default_factory=lambda: os.getenv("RESULTS_DIR", "results"))
    model_file: str = field(default_factory=lambda: os.getenv("MODEL_FILE", "trained_model.joblib"))
    scaler_file: str = field(default_factory=lambda: os.getenv("SCALER_FILE", "feature_scaler.joblib"))
    features_config_file: str = field(default_factory=lambda: os.getenv("FEATURES_CONFIG_FILE", "features_config.json"))


@dataclass
class DatabaseConfig:
    """Configuración de base de datos."""
    
    type: str = field(default_factory=lambda: os.getenv("DATABASE_TYPE", "sqlite"))
    url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///data/trading_bot.db"))


@dataclass
class BacktestConfig:
    """Configuración de backtesting."""
    
    commission: float = field(default_factory=lambda: get_env_float("BACKTEST_COMMISSION", 0.0001))
    include_slippage: bool = field(default_factory=lambda: get_env_bool("BACKTEST_INCLUDE_SLIPPAGE", True))
    start_date: str = field(default_factory=lambda: os.getenv("BACKTEST_START_DATE", "2024-01-01"))
    end_date: str = field(default_factory=lambda: os.getenv("BACKTEST_END_DATE", "2024-12-31"))


@dataclass
class SecurityConfig:
    """Configuración de seguridad."""
    
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "change_this_in_production"))
    security_salt: str = field(default_factory=lambda: os.getenv("SECURITY_SALT", "default_salt"))
    operation_timeout: int = field(default_factory=lambda: get_env_int("OPERATION_TIMEOUT", 30))


@dataclass
class MonitoringConfig:
    """Configuración de monitoreo."""
    
    enable_system_metrics: bool = field(default_factory=lambda: get_env_bool("ENABLE_SYSTEM_METRICS", True))
    metrics_report_interval: int = field(default_factory=lambda: get_env_int("METRICS_REPORT_INTERVAL", 300))
    enable_performance_alerts: bool = field(default_factory=lambda: get_env_bool("ENABLE_PERFORMANCE_ALERTS", True))
    cpu_alert_threshold: int = field(default_factory=lambda: get_env_int("CPU_ALERT_THRESHOLD", 80))
    memory_alert_threshold: int = field(default_factory=lambda: get_env_int("MEMORY_ALERT_THRESHOLD", 85))


class ConfigurationError(Exception):
    """Excepción personalizada para errores de configuración."""
    pass


class Config:
    """
    Clase principal de configuración del bot.
    
    Centraliza toda la configuración del sistema y proporciona métodos
    para validación, acceso a paths y gestión de configuración.
    """
    
    def __init__(self):
        """Inicializa la configuración del bot."""
        self._load_config()
        self._validate_config()
    
    def _load_config(self) -> None:
        """Carga toda la configuración desde variables de entorno."""
        try:
            # Cargar todas las secciones de configuración
            self.general = GeneralConfig()
            self.trading = TradingConfig()
            self.ml = MLConfig()
            self.lit = LITConfig()
            self.risk = RiskConfig()
            self.data = DataConfig()
            self.broker = BrokerConfig()
            self.logging = LoggingConfig()
            self.notifications = NotificationConfig()
            self.paths = PathsConfig()
            self.database = DatabaseConfig()
            self.backtest = BacktestConfig()
            self.security = SecurityConfig()
            self.monitoring = MonitoringConfig()
            
            # Propiedades de compatibilidad con código existente
            self.bot_name = self.general.bot_name
            self.version = self.general.version
            self.data_source = self.data.source
            
        except Exception as e:
            raise ConfigurationError(f"Error cargando configuración: {str(e)}")
    
    def _validate_config(self) -> None:
        """Valida la configuración cargada."""
        errors = []
        
        try:
            # Validaciones de trading
            if self.trading.balance_inicial <= 0:
                errors.append("Balance inicial debe ser positivo")
            
            if not (0 < self.trading.risk_per_trade <= 1):
                errors.append("Riesgo por trade debe estar entre 0 y 1")
            
            if self.trading.max_positions <= 0:
                errors.append("Máximo de posiciones debe ser positivo")
            
            if not (0 < self.trading.max_drawdown <= 1):
                errors.append("Max drawdown debe estar entre 0 y 1")
            
            # Validaciones de ML
            if not (0 < self.ml.min_confidence <= 1):
                errors.append("Confianza mínima ML debe estar entre 0 y 1")
            
            if self.ml.feature_lookback <= 0:
                errors.append("Feature lookback debe ser positivo")
            
            # Validaciones de LIT
            if not (0 < self.lit.min_confidence <= 1):
                errors.append("Confianza mínima LIT debe estar entre 0 y 1")
            
            if self.lit.lookback_periods <= 0:
                errors.append("Lookback periods LIT debe ser positivo")
            
            # Validaciones de riesgo
            if self.risk.tp_sl_ratio <= 0:
                errors.append("Ratio TP/SL debe ser positivo")
            
            # Validaciones de broker
            if self.broker.type not in ['demo', 'mt5', 'alpaca', 'interactive_brokers']:
                errors.append("Tipo de broker no válido")
            
            # Validaciones de notificaciones
            if self.notifications.telegram_enabled:
                if not self.notifications.telegram_bot_token:
                    errors.append("Token de Telegram requerido si está habilitado")
                if not self.notifications.telegram_chat_id:
                    errors.append("Chat ID de Telegram requerido si está habilitado")
            
            if self.notifications.email_enabled:
                if not self.notifications.email_username:
                    errors.append("Username de email requerido si está habilitado")
                if not self.notifications.email_password:
                    errors.append("Password de email requerido si está habilitado")
            
            # Validaciones de seguridad
            if self.security.secret_key == "change_this_in_production" and not self.general.development_mode:
                errors.append("Debe cambiar la clave secreta en producción")
            
            if errors:
                raise ConfigurationError(f"Errores de validación: {'; '.join(errors)}")
                
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Error durante validación: {str(e)}")
    
    def get_paths(self) -> Dict[str, Path]:
        """
        Retorna los paths principales del proyecto.
        
        Returns:
            Dict[str, Path]: Diccionario con los paths principales.
        """
        base_path = Path(__file__).parent.parent.parent
        
        return {
            'base': base_path,
            'data': base_path / self.paths.data_dir,
            'models': base_path / self.paths.models_dir,
            'logs': base_path / self.paths.logs_dir,
            'results': base_path / self.paths.results_dir,
            'src': base_path / 'src',
            'tests': base_path / 'tests'
        }
    
    def create_directories(self) -> None:
        """Crea los directorios necesarios si no existen."""
        paths = self.get_paths()
        
        directories_to_create = ['data', 'models', 'logs', 'results']
        
        for dir_name in directories_to_create:
            if dir_name in paths:
                paths[dir_name].mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: Optional[str] = None) -> Path:
        """
        Retorna la ruta completa del modelo.
        
        Args:
            model_name: Nombre específico del modelo (opcional).
            
        Returns:
            Path: Ruta completa del modelo.
        """
        models_dir = self.get_paths()['models']
        filename = model_name or self.paths.model_file
        return models_dir / filename
    
    def get_scaler_path(self, scaler_name: Optional[str] = None) -> Path:
        """
        Retorna la ruta completa del scaler.
        
        Args:
            scaler_name: Nombre específico del scaler (opcional).
            
        Returns:
            Path: Ruta completa del scaler.
        """
        models_dir = self.get_paths()['models']
        filename = scaler_name or self.paths.scaler_file
        return models_dir / filename
    
    def get_log_path(self, log_name: Optional[str] = None) -> Path:
        """
        Retorna la ruta completa del archivo de log.
        
        Args:
            log_name: Nombre específico del log (opcional).
            
        Returns:
            Path: Ruta completa del log.
        """
        logs_dir = self.get_paths()['logs']
        filename = log_name or Path(self.logging.file).name
        return logs_dir / filename
    
    def is_production(self) -> bool:
        """Verifica si el bot está en modo producción."""
        return self.general.environment.lower() == 'production'
    
    def is_development(self) -> bool:
        """Verifica si el bot está en modo desarrollo."""
        return self.general.development_mode or self.general.environment.lower() == 'development'
    
    def get_broker_credentials(self) -> Dict[str, Any]:
        """
        Retorna las credenciales del broker configurado.
        
        Returns:
            Dict[str, Any]: Credenciales del broker.
        """
        if self.broker.type == 'mt5':
            return {
                'login': self.broker.mt5_login,
                'password': self.broker.mt5_password,
                'server': self.broker.mt5_server
            }
        elif self.broker.type == 'alpaca':
            return {
                'api_key': self.broker.alpaca_api_key,
                'secret_key': self.broker.alpaca_secret_key,
                'base_url': self.broker.alpaca_base_url
            }
        elif self.broker.type == 'interactive_brokers':
            return {
                'host': self.broker.ib_host,
                'port': self.broker.ib_port,
                'client_id': self.broker.ib_client_id
            }
        else:
            return {}
    
    def get_data_credentials(self) -> Dict[str, Any]:
        """
        Retorna las credenciales de la fuente de datos.
        
        Returns:
            Dict[str, Any]: Credenciales de la fuente de datos.
        """
        if self.data.source == 'ccxt':
            return {
                'exchange': self.data.ccxt_exchange,
                'api_key': self.data.ccxt_api_key,
                'secret_key': self.data.ccxt_secret_key,
                'sandbox': self.data.ccxt_sandbox
            }
        else:
            return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la configuración a diccionario.
        
        Returns:
            Dict[str, Any]: Configuración como diccionario.
        """
        return {
            'general': self.general.__dict__,
            'trading': self.trading.__dict__,
            'ml': self.ml.__dict__,
            'lit': self.lit.__dict__,
            'risk': self.risk.__dict__,
            'data': self.data.__dict__,
            'broker': self.broker.__dict__,
            'logging': self.logging.__dict__,
            'notifications': self.notifications.__dict__,
            'paths': self.paths.__dict__,
            'database': self.database.__dict__,
            'backtest': self.backtest.__dict__,
            'security': self.security.__dict__,
            'monitoring': self.monitoring.__dict__
        }
    
    def validate(self) -> bool:
        """
        Valida la configuración (método de compatibilidad).
        
        Returns:
            bool: True si la configuración es válida.
        """
        try:
            self._validate_config()
            return True
        except ConfigurationError:
            return False
    
    def __str__(self) -> str:
        """Representación string de la configuración."""
        return f"Config(bot={self.bot_name}, version={self.version}, env={self.general.environment})"
    
    def __repr__(self) -> str:
        """Representación detallada de la configuración."""
        return self.__str__()


# Instancia global de configuración
try:
    config = Config()
except ConfigurationError as e:
    print(f"Error crítico en configuración: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error inesperado cargando configuración: {e}")
    sys.exit(1) 