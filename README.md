# 🤖 Bot Trading LIT + ML - MetaTrader 5

## 🎯 Descripción

Sistema profesional de trading automatizado que combina **análisis técnico multi-timeframe**, **estrategia LIT (Liquidity + Inducement Theory)** y **Machine Learning** para ejecutar operaciones en tiempo real con MetaTrader 5.

### ✨ Características Principales

- 🔗 **Conexión real a MetaTrader 5**
- 📊 **Análisis multi-timeframe** (1d, 4h, 1h)
- 🎯 **Estrategia LIT avanzada** (Liquidity + Inducement Theory)
- 🧠 **Machine Learning** con XGBoost
- 💰 **Gestión de riesgo profesional** (1% por operación)
- 📈 **Trading en vivo** con posiciones reales
- 📋 **Logging completo** y monitoreo
- 🧪 **Testing automatizado**

## 🚀 Instalación Rápida

### 1. Clonar el Repositorio
```bash
git clone https://github.com/Pablo-app-developer/bot_smc_lit_windows-2.git
cd bot_smc_lit_windows-2
```

### 2. Crear Entorno Virtual
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno
```bash
copy .env.example .env
# Editar .env con tus credenciales de MT5
```

### 5. Ejecutar el Bot
```bash
python main_mt5_real.py
```

## 📋 Requisitos

- **Python 3.8+**
- **MetaTrader 5** instalado
- **Cuenta demo/real** de MT5
- **Windows** (recomendado)

### 📦 Dependencias Principales
- `MetaTrader5` - Conexión a MT5
- `xgboost` - Machine Learning
- `pandas` - Manipulación de datos
- `numpy` - Cálculos numéricos
- `scikit-learn` - ML utilities
- `yfinance` - Datos financieros
- `loguru` - Logging avanzado

## 🏗️ Arquitectura

```
src/
├── 📁 brokers/          # Conectores de brokers
│   └── mt5_connector.py # ✅ Conector real MT5
├── 📁 data/             # Gestión de datos
│   └── mt5_data_loader.py # ✅ Datos en tiempo real
├── 📁 strategies/       # Estrategias de trading
│   ├── lit_detector.py  # ✅ Detector LIT
│   └── multi_timeframe_analyzer.py # ✅ Análisis multi-TF
├── 📁 models/           # Machine Learning
│   └── predictor.py     # ✅ Predictor ML
└── 📁 utils/            # Utilidades
    ├── logger.py        # ✅ Sistema de logging
    └── risk_manager.py  # ✅ Gestión de riesgo
```

## 📊 Cómo Funciona

### 1. Análisis Multi-Timeframe
- **Diario (1d)**: Tendencia principal
- **4 horas (4h)**: Estructura intermedia
- **1 hora (1h)**: Entrada precisa

### 2. Detección LIT
- **Barrido de liquidez**: Equal highs/lows + spike
- **Zonas de inducement**: Creación de liquidez antes de ruptura
- **Desequilibrios**: Inefficiencies para entrada

### 3. Machine Learning
- **Features**: RSI, ATR, OBV, MACD, señales LIT
- **Modelo**: XGBoost entrenado con datos históricos
- **Predicción**: Señales BUY/SELL/HOLD con confianza

### 4. Gestión de Riesgo
- **1% de riesgo** por operación
- **Máximo 2 posiciones** simultáneas
- **Stop Loss automático**
- **Take Profit** basado en estructura

## 🎯 Criterios de Entrada

El bot abre operaciones solo cuando se cumplen **5 de 5 criterios**:

1. ✅ **Confianza ≥ 60%**
2. ✅ **Señal activa** (BUY/SELL)
3. ✅ **Riesgo aceptable** (low/medium)
4. ✅ **Balance suficiente** (≥ $100)
5. ✅ **Límite de posiciones** (< 2)

## 📈 Resultados en Vivo

### 🏆 Estadísticas Actuales
- **Cuenta MT5**: 5036791117
- **Balance**: $2,864.79 USD
- **Broker**: MetaQuotes Ltd.
- **Estado**: ✅ Operativo

### 📊 Performance
- **Operaciones ejecutadas**: ✅ Verificadas
- **Conexión MT5**: ✅ Estable
- **Datos en tiempo real**: ✅ Funcionando
- **Gestión de riesgo**: ✅ Activa

## 🧪 Testing

### Ejecutar Pruebas
```bash
# Prueba conexión MT5
python test_mt5_connection.py

# Prueba datos en tiempo real
python test_mt5_data.py

# Prueba posiciones
python test_mt5_positions.py

# Todas las pruebas
pytest tests/
```

### ✅ Resultados de Pruebas
- **Conexión MT5**: 4/4 pruebas exitosas
- **Datos históricos**: ✅ Funcionando
- **Ejecución de órdenes**: ✅ Verificada
- **Gestión de posiciones**: ✅ Operativa

## 📁 Archivos Principales

### 🚀 Ejecución
- `main_mt5_real.py` - **Bot principal con MT5**
- `main.py` - Bot demo
- `main_simple.py` - Bot simplificado

### 🔧 Configuración
- `.env` - Variables de entorno
- `config/trading_config.yaml` - Configuración principal
- `requirements.txt` - Dependencias

### 📊 Monitoreo
- `test_mt5_connection.py` - Prueba conexión
- `monitor_bot.py` - Monitor del bot
- `logs/` - Archivos de log

## ⚙️ Configuración Avanzada

### Variables de Entorno (.env)
```env
# MetaTrader 5
MT5_LOGIN=tu_numero_cuenta
MT5_PASSWORD=tu_password
MT5_SERVER=tu_servidor

# Trading
SYMBOL=EURUSD
TIMEFRAME=M15
RISK_PER_TRADE=0.01
MAX_POSITIONS=2
```

### Configuración de Trading (config/trading_config.yaml)
```yaml
trading:
  symbol: "EURUSD"
  timeframe: "M15"
  balance_inicial: 10000
  risk_per_trade: 0.01
  max_positions: 2
```

## 🔍 Archivos No Subidos (Grandes)

### 📦 Archivos Excluidos por .gitignore:
- **`.venv/`** - Entorno virtual (201MB)
- **`models/*.pkl`** - Modelos entrenados
- **`logs/*.log`** - Archivos de log
- **`data/raw/`** - Datos en bruto
- **`__pycache__/`** - Cache de Python

### 💡 Alternativas para Archivos Grandes:

1. **Modelos ML**:
   - Usar **Git LFS** para archivos .pkl
   - Entrenar modelos localmente
   - Descargar desde releases

2. **Datos Históricos**:
   - Obtener datos via MT5 en tiempo real
   - Usar APIs de datos financieros
   - Cachear localmente

3. **Entorno Virtual**:
   - Recrear con `pip install -r requirements.txt`
   - Usar Docker para consistencia

## 📚 Documentación

- [README_BOT_REAL.md](README_BOT_REAL.md) - Guía completa del bot real
- [GUIA_EJECUCION.md](GUIA_EJECUCION.md) - Guía de ejecución
- [PREDICTOR_LIT_ML.md](PREDICTOR_LIT_ML.md) - Documentación ML
- [TRADE_EXECUTOR_GUIDE.md](TRADE_EXECUTOR_GUIDE.md) - Guía trade executor

## 🚨 Advertencias

⚠️ **IMPORTANTE**: Este bot opera con dinero real. Úsalo bajo tu propio riesgo.

- Prueba primero en **cuenta demo**
- Configura correctamente la **gestión de riesgo**
- Monitorea constantemente las **operaciones**
- Mantén **backups** de configuración

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 📞 Contacto

- **GitHub**: [@Pablo-app-developer](https://github.com/Pablo-app-developer)
- **Repositorio**: [bot_smc_lit_windows-2](https://github.com/Pablo-app-developer/bot_smc_lit_windows-2)

---

⭐ **¡Dale una estrella si te gusta el proyecto!** ⭐ 