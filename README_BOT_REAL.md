# 🤖 Bot de Trading REAL con MetaTrader 5

## 🎯 ADVERTENCIA IMPORTANTE

**⚠️ ESTE BOT EJECUTA OPERACIONES REALES EN TU CUENTA DE TRADING**

- ✅ **USAR SOLO EN CUENTA DEMO** para pruebas iniciales
- ❌ **NO usar en cuenta real** hasta estar completamente seguro
- 📊 Siempre verifica todas las operaciones manualmente
- 💰 Configura límites de riesgo apropiados

## 🚀 Características del Bot Real

### ✅ Conexión Real a MetaTrader 5
- Conecta directamente a tu terminal MT5
- Ejecuta órdenes reales en tiempo real
- Monitorea posiciones activas continuamente
- Gestión automática de stop loss y take profit

### 📊 Estrategia LIT (Liquidity + Inducement Theory)
- Detecta zonas de liquidez en tiempo real
- Identifica patrones de inducement
- Confirma señales con múltiples timeframes
- Análisis de desequilibrios de mercado

### 🧠 Machine Learning Integrado
- Predicciones basadas en XGBoost
- Features técnicos avanzados
- Aprendizaje continuo con nuevos datos
- Validación de señales con IA

### 🛡️ Gestión Profesional de Riesgo
- Máximo 1% de riesgo por operación
- Stop loss automático en todas las operaciones
- Límite de posiciones simultáneas
- Monitoreo continuo de drawdown

## 📋 Requisitos Previos

### 1. MetaTrader 5 Instalado
```bash
# Descargar desde: https://www.metatrader5.com/
# O desde tu broker preferido
```

### 2. Cuenta Demo/Real Configurada
- Cuenta demo para pruebas (RECOMENDADO)
- Cuenta real solo después de validación completa
- Trading debe estar habilitado en la cuenta

### 3. Dependencias de Python
```bash
pip install -r requirements.txt
```

## 🛠️ Instalación y Configuración

### Paso 1: Clonar el Repositorio
```bash
git clone <tu-repositorio>
cd Bot_Trading_LIT_ML
```

### Paso 2: Instalar Dependencias
```bash
pip install -r requirements.txt
```

### Paso 3: Configurar MetaTrader 5
1. Abrir MetaTrader 5
2. Hacer login con tu cuenta demo
3. Verificar que el trading esté habilitado
4. Asegurarse de que los símbolos EURUSD, GBPUSD, USDJPY estén disponibles

### Paso 4: Probar la Conexión
```bash
python test_mt5_connection.py
```

## 🚀 Ejecución del Bot

### Modo de Prueba (RECOMENDADO)
```bash
python main_mt5_real.py
```

### Configuración Personalizada
```python
# En main_mt5_real.py, puedes modificar:
self.symbols = ["EURUSD", "GBPUSD", "USDJPY"]  # Pares a tradear
self.min_confidence = 0.65                      # Confianza mínima
self.max_positions = 2                          # Posiciones máximas
self.risk_per_trade = 0.01                      # 1% riesgo por trade
```

## 📊 Monitoreo en Tiempo Real

El bot muestra información detallada:

```
📊 CICLO 15 - 14:30:45
🔍 Analizando mercado para oportunidades REALES...
🎯 OPORTUNIDAD: EURUSD - 🟢 COMPRAR
   💰 Precio: $1.08750
   📊 Confianza: 72.5%

💰 OPERACIÓN REAL: COMPRAR EURUSD
   📊 Volumen: 0.01 lotes
   💰 Precio: 1.08750
   🛡️  SL: 1.08580
   🎯 TP: 1.09020

✅ ¡OPERACIÓN EJECUTADA EN VIVO!
   🎫 Ticket: 123456789
   💰 Precio ejecución: 1.08752

👁️  1 posiciones activas:
   📊 EURUSD #123456789: BUY | P&L: +$2.50
```

## 🛡️ Características de Seguridad

### Límites de Riesgo
- **Riesgo por operación**: Máximo 1% del balance
- **Posiciones simultáneas**: Máximo 2 posiciones
- **Stop Loss obligatorio**: En todas las operaciones
- **Take Profit automático**: Ratio 2:1 mínimo

### Validaciones de Seguridad
- Verificación de balance antes de cada operación
- Validación de símbolos disponibles
- Confirmación de permisos de trading
- Reconexión automática en caso de desconexión

### Controles de Emergencia
- Detención inmediata con `Ctrl+C`
- Cierre automático de posiciones al detener el bot
- Logs detallados de todas las operaciones
- Resumen financiero al finalizar

## 📈 Métricas y Estadísticas

El bot proporciona métricas en tiempo real:

- **Balance inicial vs actual**
- **P&L total y porcentual**
- **Número de operaciones ejecutadas**
- **Tasa de éxito de las operaciones**
- **Tiempo de ejecución**
- **Posiciones activas**

## 🔧 Personalización Avanzada

### Agregar Nuevos Símbolos
```python
# En RealTradingBot.__init__():
self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF"]
```

### Modificar Parámetros de Riesgo
```python
# Cambiar riesgo por operación (1% = 0.01)
self.risk_per_trade = 0.005  # 0.5% más conservador

# Cambiar número máximo de posiciones
self.max_positions = 1  # Solo 1 posición a la vez
```

### Ajustar Confianza Mínima
```python
# Requerir mayor confianza para las señales
self.min_confidence = 0.70  # 70% en lugar de 65%
```

## 🧪 Testing y Validación

### Script de Pruebas Automatizadas
```bash
python test_mt5_connection.py
```

### Pruebas Manuales
1. **Verificar conexión**: ¿MT5 se conecta correctamente?
2. **Probar símbolos**: ¿Están disponibles todos los pares?
3. **Validar órdenes**: ¿Se ejecutan las órdenes correctamente?
4. **Confirmar cierre**: ¿Se cierran las posiciones automáticamente?

### Backtesting
```bash
# Para validar la estrategia antes del trading real
python scripts/backtest_strategy.py
```

## 📞 Soporte y Resolución de Problemas

### Problemas Comunes

**❌ "MetaTrader5 no está instalado"**
```bash
pip install MetaTrader5
```

**❌ "No se pudo conectar a MT5"**
- Verificar que MT5 esté abierto
- Confirmar que tienes una cuenta configurada
- Revisar que el trading esté habilitado

**❌ "Trading no permitido"**
- Usar cuenta demo en lugar de real
- Verificar configuración de la cuenta
- Contactar al broker si es necesario

**❌ "Símbolo no disponible"**
- Verificar que el símbolo esté en Market Watch
- Agregar manualmente desde Symbols
- Verificar disponibilidad con tu broker

### Logs y Depuración
```bash
# Los logs se guardan automáticamente en:
tail -f logs/trading_bot.log

# Para más detalles:
export LOG_LEVEL=DEBUG
python main_mt5_real.py
```

## 📚 Referencias y Recursos

### Documentación Técnica
- [MetaTrader 5 Python Integration](https://www.mql5.com/en/docs/python_metatrader5)
- [Libro Trading Python ES](https://github.com/Pablo-app-developer/libro-trading-python-es)
- [TopForex Trading Guide](https://topforex.trade/academy/build-forex-trading-bot-python-guide)

### Estrategias de Trading
- [LIT Strategy Documentation](docs/LIT_DETECTOR_README.md)
- [Risk Management Guide](docs/RISK_MANAGEMENT.md)
- [Machine Learning Integration](docs/ML_INTEGRATION.md)

## ⚖️ Responsabilidad Legal

**DISCLAIMER**: Este software es solo para fines educativos y de investigación. El trading en Forex conlleva riesgos significativos y puede resultar en pérdidas sustanciales. Los desarrolladores no se hacen responsables de ninguna pérdida financiera resultante del uso de este software.

**RECOMENDACIONES**:
- Usar solo en cuentas demo para aprendizaje
- Consultar con un asesor financiero antes de trading real
- Entender completamente los riesgos del trading automatizado
- Nunca invertir más de lo que puedes permitirte perder

## 🤝 Contribuciones

Las contribuciones son bienvenidas:

1. Fork el repositorio
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus changes (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

---

**¡RECUERDA: SIEMPRE USA CUENTA DEMO PRIMERO!** 🛡️ 