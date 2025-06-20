# Sistema Predictor LIT + ML - Documentación Completa

## 📋 Descripción General

El **Sistema Predictor LIT + ML** es un módulo profesional que integra el modelo entrenado LIT (Liquidity + Inducement Theory) con Machine Learning para realizar predicciones de trading en tiempo real y backtesting. Incluye integración completa con MetaTrader 5 para trading automático.

## 🏗️ Arquitectura del Sistema

### Componentes Principales

1. **`LITMLPredictor`** - Predictor principal que carga y ejecuta el modelo
2. **`MT5PredictorIntegration`** - Integración con MetaTrader 5 para trading real
3. **Scripts de Ejecución** - Interfaces de línea de comandos para diferentes usos
4. **Funciones de Utilidad** - Helpers para casos de uso específicos

### Estructura de Archivos

```
src/models/
├── predictor.py              # Predictor principal LIT + ML
└── feature_engineering.py    # Ingeniería de características

src/integrations/
└── mt5_predictor.py          # Integración MetaTrader 5

scripts/
├── train_model.py            # Entrenamiento del modelo
└── run_predictions.py        # Ejecución de predicciones

examples/
├── predictor_examples.py     # Ejemplos de uso
└── train_model_example.py    # Ejemplos de entrenamiento

config/
└── training_config.json     # Configuración de entrenamiento

models/
└── *.pkl                    # Modelos entrenados
```

## 🚀 Instalación y Configuración

### Dependencias Requeridas

```bash
# Dependencias principales
pip install xgboost scikit-learn pandas numpy
pip install yfinance ccxt ta-lib
pip install MetaTrader5  # Para integración MT5
```

### Configuración MT5

Las credenciales de MetaTrader 5 están configuradas por defecto:

```python
# Credenciales MT5
MT5_LOGIN = 5036791117
MT5_PASSWORD = "BtUvF-X8"
MT5_SERVER = "MetaQuotes-Demo"
```

## 📊 Uso del Sistema

### 1. Predicción Única

Realiza una predicción para un símbolo específico:

```bash
# Predicción básica
python scripts/run_predictions.py single --symbol AAPL --model models/test_model.pkl

# Con marco temporal específico
python scripts/run_predictions.py single --symbol EURUSD --timeframe 1h
```

### 2. Backtesting

Ejecuta backtesting con datos históricos:

```bash
# Backtesting básico (30 días)
python scripts/run_predictions.py backtest --symbol AAPL --days 30

# Backtesting extendido
python scripts/run_predictions.py backtest --symbol EURUSD --days 90 --timeframe 1h
```

### 3. Predicciones en Tiempo Real con MT5

#### Solo Predicciones (Sin Trading)

```bash
# Ejecutar por 2 horas
python scripts/run_predictions.py realtime --hours 2

# Con modelo específico
python scripts/run_predictions.py realtime --hours 1 --model models/my_model.pkl
```

#### Con Trading Automático

```bash
# ⚠️ CUIDADO: Esto ejecuta operaciones reales
python scripts/run_predictions.py realtime --hours 1 --trading
```

## 🔧 API del Predictor

### Clase LITMLPredictor

#### Predicción Única

```python
from src.models.predictor import LITMLPredictor

# Crear y cargar predictor
predictor = LITMLPredictor("models/lit_ml_model.pkl")
predictor.load_model()

# Realizar predicción
prediction = predictor.predict_single(data)

# Resultado:
{
    'signal': 'buy',           # 'buy', 'sell', 'hold'
    'confidence': 0.742,       # Confianza [0-1]
    'probabilities': {
        'buy': 0.742,
        'sell': 0.158,
        'hold': 0.100
    }
}
```

## 📈 Ejemplos Prácticos

### Análisis de Múltiples Símbolos

```python
symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
predictor = LITMLPredictor("models/test_model.pkl")
predictor.load_model()

for symbol in symbols:
    data = data_loader.load_data(symbol=symbol, periods=100)
    prediction = predictor.predict_single(data)
    print(f"{symbol}: {prediction['signal']} (conf: {prediction['confidence']:.3f})")
```

## ⚠️ Consideraciones Importantes

- **SIEMPRE** prueba primero en cuenta demo
- Configura límites de riesgo apropiados
- Monitorea constantemente el sistema
- Reentrena el modelo periódicamente

---

**Nota**: Este sistema está diseñado para uso profesional. Siempre prueba en entorno demo antes de usar capital real. 