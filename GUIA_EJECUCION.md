# Guía de Ejecución - Bot de Trading LIT + ML

## Descripción del Pipeline

El bot integra todos los módulos en un pipeline claro y eficiente:

```
DATOS → SEÑALES LIT → FEATURES ML → PREDICCIÓN → EJECUCIÓN
```

### Componentes Integrados

1. **DataLoader**: Carga datos de mercado desde múltiples fuentes
2. **LITDetector**: Detecta patrones de liquidez e inducement
3. **FeatureEngineer**: Genera features técnicos y de LIT para ML
4. **TradingPredictor**: Combina señales LIT + ML para predicciones
5. **TradeExecutor**: Ejecuta órdenes con gestión de riesgo

## Instalación y Configuración

### 1. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno
```bash
cp .env.example .env
# Editar .env con tus configuraciones
```

### 3. Configurar Parámetros
Editar `src/core/config.py` o usar variables de entorno:

- `TRADING_SYMBOL`: Símbolo a tradear (ej: EURUSD)
- `TRADING_TIMEFRAME`: Timeframe (M5, M15, H1, etc.)
- `TRADING_BALANCE_INICIAL`: Balance inicial
- `TRADING_RISK_PER_TRADE`: Riesgo por trade (0.01 = 1%)
- `ML_MIN_CONFIDENCE`: Confianza mínima para ejecutar (0.7)
- `LIT_MIN_CONFIDENCE`: Confianza mínima LIT (0.6)

## Comandos de Ejecución

### Trading en Vivo
```bash
# Ejecutar bot en modo live trading
python main.py

# Con logging detallado
python main.py --log-level DEBUG
```

### Backtesting
```bash
# Backtest básico
python main.py backtest EURUSD 2024-01-01 2024-06-01

# Backtest con múltiples símbolos
python main.py backtest GBPUSD 2024-01-01 2024-03-01
python main.py backtest USDJPY 2024-01-01 2024-03-01
```

### Entrenamiento de Modelos
```bash
# Entrenar modelo básico
python main.py train EURUSD 2000

# Entrenar con optimización de hiperparámetros
python main.py optimize EURUSD 5000
```

### Validación y Estado
```bash
# Validar configuración del sistema
python main.py validate

# Ver configuración actual
python main.py status

# Ver ayuda
python main.py --help
```

## Flujo de Ejecución Detallado

### 1. Modo Trading en Vivo

```python
# 1. Inicialización
bot = LITMLTradingBot()
├── DataLoader()          # Conexión a fuentes de datos  
├── LITDetector()         # Configurar parámetros LIT
├── FeatureEngineer()     # Preparar generación de features
├── TradingPredictor()    # Cargar modelo ML entrenado
└── TradeExecutor()       # Configurar gestión de riesgo

# 2. Loop Principal (cada 5 minutos por defecto)
while running:
    ├── Obtener datos de mercado actualizados
    ├── Detectar señales LIT en los datos
    ├── Generar features técnicos + LIT
    ├── Generar predicción ML combinada
    ├── Validar señal y condiciones de ejecución
    ├── Ejecutar orden si cumple criterios
    ├── Actualizar posiciones existentes
    └── Log de métricas y estado
```

### 2. Validaciones de Seguridad

Antes de ejecutar cada señal:
- ✅ Confianza ML >= umbral mínimo
- ✅ Confianza LIT >= umbral mínimo  
- ✅ Número de posiciones < máximo permitido
- ✅ Suficiente balance disponible
- ✅ Validación de riesgo por trade
- ✅ Verificación de drawdown máximo

### 3. Gestión de Riesgo Automática

- **Stop Loss**: Calculado automáticamente basado en ATR
- **Take Profit**: 2:1 ratio riesgo/beneficio por defecto
- **Position Sizing**: Basado en % de riesgo del balance
- **Max Drawdown**: Cierre automático si se excede límite
- **Trailing Stop**: Activado en trades ganadores

## Archivos de Salida

### Resultados de Trading en Vivo
- `live_trading_results.csv`: Historial completo de trades
- `performance_history.csv`: Métricas por ciclo
- `logs/trading_YYYYMMDD.log`: Logs detallados

### Resultados de Backtest
- `backtest_[SYMBOL]_[START]_[END]_[TIMESTAMP].csv`: Trades del backtest
- `metrics_backtest_[SYMBOL]_[START]_[END]_[TIMESTAMP].json`: Métricas de rendimiento

### Modelos Entrenados
- `models/model_[SYMBOL]_[TIMESTAMP].joblib`: Modelo ML entrenado
- `models/feature_scaler_[TIMESTAMP].joblib`: Scaler de features

## Monitoreo en Tiempo Real

### Métricas Clave
- **Balance Actual**: Balance en tiempo real
- **Posiciones Abiertas**: Número de trades activos
- **Win Rate**: % de trades ganadores
- **Return**: Retorno acumulado
- **Max Drawdown**: Máxima pérdida desde el pico
- **Sharpe Ratio**: Ratio rendimiento/riesgo

### Logs de Estado
```
=== ESTADO DEL BOT ===
Ciclo: #142
Precio: 1.08457
Balance: $10,234.50
Posiciones abiertas: 2
Total trades: 28
Win rate: 64.3%
Retorno: 2.35%
Max drawdown: -1.2%
Última señal: buy (0.823)
```

## Configuración Avanzada

### Parámetros LIT
```python
LIT_LOOKBACK_PERIODS = 50        # Períodos para análisis LIT
LIT_MIN_CONFIDENCE = 0.6         # Confianza mínima señales LIT
LIT_LIQUIDITY_THRESHOLD = 0.001  # Umbral detección liquidez
```

### Parámetros ML
```python
ML_FEATURE_LOOKBACK = 100        # Períodos para features
ML_MIN_CONFIDENCE = 0.7          # Confianza mínima ML
ML_MODEL_UPDATE_FREQ = 1000      # Frecuencia re-entrenamiento
```

### Parámetros Trading
```python
TRADING_RISK_PER_TRADE = 0.02    # 2% riesgo por trade
TRADING_MAX_POSITIONS = 3        # Máx 3 posiciones simultáneas
TRADING_MAX_DRAWDOWN = 0.1       # 10% drawdown máximo
TRADING_CHECK_INTERVAL = 300     # 5 minutos entre checks
```

## Solución de Problemas

### Error: "No se pueden cargar datos de mercado"
```bash
# Verificar configuración de datos
python main.py validate

# Verificar conexión a internet y APIs
# Revisar logs en logs/trading_YYYYMMDD.log
```

### Error: "Modelo ML no encontrado"  
```bash
# Entrenar nuevo modelo
python main.py train EURUSD 2000

# Verificar archivos en models/
ls models/*.joblib
```

### Performance Insuficiente
```bash
# Optimizar hiperparámetros
python main.py optimize EURUSD 5000

# Ajustar parámetros de confianza
# Revisar configuración de riesgo
```

## Buenas Prácticas

### 1. Antes de Ejecutar
- ✅ Validar sistema completo: `python main.py validate`
- ✅ Revisar configuración: `python main.py status`  
- ✅ Hacer backtest histórico primero
- ✅ Comenzar con balance pequeño de prueba

### 2. Durante la Ejecución
- 📊 Monitorear logs regularmente
- 📈 Revisar métricas cada hora
- 🛑 Detener si drawdown > límite
- 💾 Backup de datos importantes

### 3. Optimización Continua
- 🔄 Re-entrenar modelo semanalmente
- 📊 Analizar performance mensualmente
- ⚙️ Ajustar parámetros según mercado
- 🧪 Probar en demo antes de live

## Ejemplos de Uso Completos

### Ejemplo 1: Setup Inicial Completo
```bash
# 1. Validar sistema
python main.py validate

# 2. Entrenar modelo inicial
python main.py train EURUSD 2000

# 3. Hacer backtest de validación
python main.py backtest EURUSD 2024-01-01 2024-03-01

# 4. Si resultados OK, iniciar live trading
python main.py
```

### Ejemplo 2: Optimización de Performance
```bash
# 1. Optimizar hiperparámetros con más datos
python main.py optimize EURUSD 5000

# 2. Probar en diferentes períodos
python main.py backtest EURUSD 2024-01-01 2024-02-01
python main.py backtest EURUSD 2024-02-01 2024-03-01
python main.py backtest EURUSD 2024-03-01 2024-04-01

# 3. Ajustar configuración basado en resultados
# 4. Re-iniciar trading con nuevo modelo
```

### Ejemplo 3: Multi-Symbol Trading
```bash
# Entrenar modelos para múltiples pares
python main.py train EURUSD 3000
python main.py train GBPUSD 3000
python main.py train USDJPY 3000

# Configurar main.py para multi-symbol
# Ejecutar instancias separadas por símbolo
```

## Contacto y Soporte

Para dudas técnicas o mejoras:
- Revisar logs detallados en `logs/`
- Verificar configuración con `python main.py validate`
- Consultar documentación de módulos en `src/`

---

**⚠️ IMPORTANTE**: Este bot opera con dinero real. Siempre probar exhaustivamente en demo antes de usar fondos reales. Nunca arriesgar más de lo que puedes permitirte perder. 