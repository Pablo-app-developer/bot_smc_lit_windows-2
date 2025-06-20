# Configuración con Python-Dotenv

## Descripción

El bot de trading LIT + ML utiliza **python-dotenv** para gestionar la configuración a través de variables de entorno. Este enfoque proporciona:

- ✅ **Seguridad**: Credenciales separadas del código
- ✅ **Flexibilidad**: Configuración por entorno (dev/prod)
- ✅ **Simplicidad**: Un solo archivo `.env` para toda la configuración
- ✅ **Escalabilidad**: Fácil gestión de múltiples configuraciones

## Instalación

Python-dotenv ya está incluido en `requirements.txt`:

```bash
pip install python-dotenv==1.0.0
```

## Estructura de Archivos

```
Bot_Trading_LIT_ML/
├── .env.example          # Plantilla de configuración
├── .env                  # Tu configuración (NO subir a Git)
├── src/core/config.py    # Módulo de configuración
├── examples/config_example.py    # Ejemplo de uso
└── scripts/validate_config.py   # Validador de configuración
```

## Configuración Inicial

### 1. Crear archivo .env

```bash
# Copiar plantilla
cp .env.example .env

# Editar con tus valores
nano .env  # o tu editor preferido
```

### 2. Configurar variables básicas

```bash
# .env
BOT_NAME=Mi_Bot_Trading
TRADING_SYMBOL=EURUSD
TRADING_BALANCE_INICIAL=10000.00
TRADING_RISK_PER_TRADE=0.02
ML_MIN_CONFIDENCE=0.70
LIT_MIN_CONFIDENCE=0.60
```

### 3. Validar configuración

```bash
python scripts/validate_config.py
```

## Secciones de Configuración

### 🔧 General
```bash
BOT_NAME=LIT_ML_Trading_Bot
VERSION=1.0.0
ENVIRONMENT=development
DEBUG=true
DEVELOPMENT_MODE=true
```

### 💰 Trading
```bash
TRADING_SYMBOL=EURUSD
TRADING_TIMEFRAME=15m
TRADING_BALANCE_INICIAL=10000.00
TRADING_RISK_PER_TRADE=0.02
TRADING_MAX_POSITIONS=3
TRADING_LEVERAGE=1.0
TRADING_MAX_DRAWDOWN=0.10
TRADING_CHECK_INTERVAL=300
```

### 🤖 Machine Learning
```bash
ML_MODEL_TYPE=xgboost
ML_MIN_CONFIDENCE=0.70
ML_FEATURE_LOOKBACK=100
ML_RETRAIN_FREQUENCY=weekly
ML_OPTIMIZE_HYPERPARAMS=true
ML_OPTUNA_TRIALS=100
```

### 🎯 Estrategia LIT
```bash
LIT_LOOKBACK_PERIODS=50
LIT_MIN_CONFIDENCE=0.60
LIT_LIQUIDITY_THRESHOLD=0.001
LIT_INDUCEMENT_MIN_TOUCHES=2
LIT_INEFFICIENCY_MIN_SIZE=0.0005
LIT_ATR_MULTIPLIER=2.0
```

### 🛡️ Gestión de Riesgo
```bash
RISK_TP_SL_RATIO=2.0
RISK_USE_TRAILING_STOP=true
RISK_TRAILING_STOP_ATR=1.5
RISK_MAX_PORTFOLIO_RISK=0.06
```

### 📊 Fuentes de Datos
```bash
DATA_SOURCE=yfinance
# Para CCXT
CCXT_EXCHANGE=binance
CCXT_API_KEY=your_api_key
CCXT_SECRET_KEY=your_secret_key
CCXT_SANDBOX=true
```

### 🏦 Broker
```bash
BROKER_TYPE=demo
# Para Alpaca
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 🔔 Notificaciones
```bash
TELEGRAM_ENABLED=false
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

EMAIL_ENABLED=false
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

## Uso en Código

### Importar configuración

```python
from src.core.config import config

# Acceder a configuración
print(f"Bot: {config.general.bot_name}")
print(f"Símbolo: {config.trading.symbol}")
print(f"Confianza ML: {config.ml.min_confidence}")
```

### Ejemplo completo

```python
from src.core.config import config

def main():
    # Configuración de trading
    symbol = config.trading.symbol
    balance = config.trading.balance_inicial
    risk = config.trading.risk_per_trade
    
    print(f"Iniciando bot para {symbol}")
    print(f"Balance: ${balance:,.2f}")
    print(f"Riesgo por trade: {risk * 100:.1f}%")
    
    # Configuración de ML
    model_type = config.ml.model_type
    confidence = config.ml.min_confidence
    
    print(f"Modelo: {model_type}")
    print(f"Confianza mínima: {confidence * 100:.1f}%")
    
    # Rutas del sistema
    paths = config.get_paths()
    print(f"Directorio de modelos: {paths['models']}")
    
    # Validar configuración
    if config.validate():
        print("✅ Configuración válida")
    else:
        print("❌ Configuración inválida")

if __name__ == "__main__":
    main()
```

## Configuraciones por Entorno

### Desarrollo (.env.dev)
```bash
ENVIRONMENT=development
DEBUG=true
BROKER_TYPE=demo
TRADING_BALANCE_INICIAL=10000.00
LOG_LEVEL=DEBUG
```

### Producción (.env.prod)
```bash
ENVIRONMENT=production
DEBUG=false
BROKER_TYPE=alpaca
TRADING_BALANCE_INICIAL=50000.00
LOG_LEVEL=INFO
SECRET_KEY=your_production_secret_key
```

### Cargar configuración específica

```python
from dotenv import load_dotenv

# Cargar configuración específica
load_dotenv('.env.prod')  # Para producción
load_dotenv('.env.dev')   # Para desarrollo
```

## Validación de Configuración

### Validación automática

```python
from src.core.config import config, ConfigurationError

try:
    # La configuración se valida automáticamente al importar
    print("Configuración cargada correctamente")
except ConfigurationError as e:
    print(f"Error de configuración: {e}")
```

### Validación manual

```python
# Validar configuración actual
is_valid = config.validate()
if not is_valid:
    print("Configuración inválida")

# Obtener errores específicos
try:
    config._validate_config()
except ConfigurationError as e:
    print(f"Errores: {e}")
```

### Script de validación

```bash
# Validación completa del sistema
python scripts/validate_config.py

# Ejemplo de configuración
python examples/config_example.py
```

## Gestión de Credenciales

### Variables sensibles

```bash
# Nunca hardcodear en el código
ALPACA_API_KEY=PKTEST_abc123...
ALPACA_SECRET_KEY=xyz789...
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
EMAIL_PASSWORD=your_app_password
SECRET_KEY=your_secret_key_here
```

### Acceso seguro

```python
# Obtener credenciales del broker
broker_creds = config.get_broker_credentials()
if broker_creds:
    api_key = broker_creds.get('api_key')
    secret_key = broker_creds.get('secret_key')

# Obtener credenciales de datos
data_creds = config.get_data_credentials()
if data_creds:
    exchange = data_creds.get('exchange')
    api_key = data_creds.get('api_key')
```

## Mejores Prácticas

### 1. Seguridad

```bash
# ✅ Hacer
- Usar .env para credenciales
- Agregar .env al .gitignore
- Usar valores por defecto seguros
- Validar configuración al inicio

# ❌ Evitar
- Hardcodear credenciales en código
- Subir .env a control de versiones
- Usar credenciales de producción en desarrollo
```

### 2. Organización

```bash
# ✅ Estructura clara
[SECCION]_PARAMETRO=valor

# Ejemplos
TRADING_SYMBOL=EURUSD
ML_MIN_CONFIDENCE=0.70
LIT_LOOKBACK_PERIODS=50
```

### 3. Documentación

```bash
# ✅ Comentarios descriptivos
# Símbolo principal a tradear (ej: EURUSD, GBPUSD)
TRADING_SYMBOL=EURUSD

# Confianza mínima para ejecutar señal (0.7 = 70%)
ML_MIN_CONFIDENCE=0.70
```

### 4. Valores por defecto

```python
# ✅ Siempre proporcionar defaults
symbol = os.getenv("TRADING_SYMBOL", "EURUSD")
confidence = float(os.getenv("ML_MIN_CONFIDENCE", "0.70"))
```

## Troubleshooting

### Error: "No module named 'dotenv'"

```bash
# Instalar python-dotenv
pip install python-dotenv

# O desde requirements.txt
pip install -r requirements.txt
```

### Error: "ConfigurationError"

```bash
# Validar configuración
python scripts/validate_config.py

# Verificar archivo .env
ls -la .env

# Verificar formato
cat .env | grep -v "^#" | grep "="
```

### Variables no se cargan

```python
# Verificar carga manual
from dotenv import load_dotenv
import os

load_dotenv()
print(os.getenv("TRADING_SYMBOL"))  # Debe mostrar el valor
```

### Configuración no válida

```bash
# Ver errores específicos
python -c "from src.core.config import config; config.validate()"

# Revisar valores
python examples/config_example.py
```

## Ejemplos Avanzados

### Configuración dinámica

```python
import os
from src.core.config import config

# Cambiar configuración en runtime (solo para testing)
original_symbol = config.trading.symbol
config.trading.symbol = "GBPUSD"

# Restaurar
config.trading.symbol = original_symbol
```

### Múltiples archivos .env

```python
from dotenv import load_dotenv

# Cargar configuración base
load_dotenv('.env')

# Sobrescribir con configuración específica
load_dotenv('.env.local', override=True)
```

### Configuración condicional

```python
import os
from src.core.config import config

if config.is_production():
    # Configuración de producción
    log_level = "INFO"
    broker_type = "alpaca"
else:
    # Configuración de desarrollo
    log_level = "DEBUG"
    broker_type = "demo"
```

## Referencias

- [Python-dotenv Documentation](https://pypi.org/project/python-dotenv/) según los resultados de búsqueda
- [GeeksforGeeks: Python Environment Variables](https://www.geeksforgeeks.org/using-python-environment-variables-with-python-dotenv/) según los resultados de búsqueda
- [12-Factor App Methodology](https://12factor.net/config)

## Scripts Útiles

```bash
# Validar configuración completa
python scripts/validate_config.py

# Ejemplo de uso
python examples/config_example.py

# Ver configuración actual
python main.py status

# Validar sistema completo
python main.py validate
```

---

**⚠️ IMPORTANTE**: Nunca subas el archivo `.env` a control de versiones. Siempre usa `.env.example` como plantilla y configura tus valores específicos en `.env`. 