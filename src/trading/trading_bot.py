#!/usr/bin/env python3
"""
Trading Bot - Bot de Trading Automático LIT + ML.

Este módulo integra el predictor LIT + ML con el ejecutor de trading
para crear un bot de trading completamente automático.
"""

import os
import sys
import time
import threading
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.predictor import LITMLPredictor
from src.trading.trade_executor import TradeExecutor, TradeSignal, RiskLevel, create_trade_executor
from src.data.data_loader import DataLoader
from src.utils.logger import log


class TradingBot:
    """
    Bot de trading automático que integra predicciones LIT + ML con ejecución real.
    """
    
    def __init__(self,
                 model_path: str = "models/lit_ml_model.pkl",
                 symbols: List[str] = None,
                 timeframe: str = "1h",
                 prediction_interval: int = 300,  # 5 minutos
                 risk_level: str = "moderate",
                 min_confidence: float = 0.65,
                 max_spread: float = 3.0,
                 trading_enabled: bool = False):
        """
        Inicializa el bot de trading.
        
        Args:
            model_path: Ruta al modelo entrenado
            symbols: Lista de símbolos a operar
            timeframe: Marco temporal para análisis
            prediction_interval: Intervalo entre predicciones (segundos)
            risk_level: Nivel de riesgo ('conservative', 'moderate', 'aggressive')
            min_confidence: Confianza mínima para ejecutar operaciones
            max_spread: Spread máximo permitido
            trading_enabled: Si habilitar trading real
        """
        self.model_path = model_path
        self.symbols = symbols or ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        self.timeframe = timeframe
        self.prediction_interval = prediction_interval
        self.min_confidence = min_confidence
        self.max_spread = max_spread
        self.trading_enabled = trading_enabled
        
        # Componentes principales
        self.predictor = LITMLPredictor(model_path)
        self.executor = create_trade_executor(risk_level)
        self.data_loader = DataLoader()
        
        # Estado del bot
        self.running = False
        self.paused = False
        self.emergency_stop_triggered = False
        
        # Hilos de ejecución
        self.prediction_thread = None
        self.monitoring_thread = None
        
        # Datos y predicciones
        self.current_predictions: Dict[str, Dict[str, Any]] = {}
        self.prediction_history: List[Dict[str, Any]] = []
        self.last_signals: Dict[str, TradeSignal] = {}
        
        # Callbacks y eventos
        self.signal_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # Estadísticas
        self.stats = {
            'start_time': None,
            'predictions_made': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'successful_trades': 0,
            'total_profit': 0.0,
            'uptime_hours': 0.0
        }
        
        # Configuración de filtros
        self.filters = {
            'min_confidence': min_confidence,
            'max_spread': max_spread,
            'min_volume': 0.01,
            'max_correlation': 0.8,
            'trading_hours': None,  # None = 24/7
            'news_filter': False
        }
        
        log.info(f"TradingBot inicializado")
        log.info(f"  Modelo: {model_path}")
        log.info(f"  Símbolos: {self.symbols}")
        log.info(f"  Timeframe: {timeframe}")
        log.info(f"  Intervalo: {prediction_interval}s")
        log.info(f"  Trading: {'✅ HABILITADO' if trading_enabled else '❌ DESHABILITADO'}")
        log.info(f"  Confianza mínima: {min_confidence}")
    
    def initialize(self) -> bool:
        """
        Inicializa el bot de trading.
        
        Returns:
            bool: True si se inicializó correctamente
        """
        try:
            log.info("🚀 Inicializando TradingBot...")
            
            # 1. Cargar modelo predictor
            log.info("📊 Cargando modelo predictor...")
            if not self.predictor.load_model():
                log.error("❌ No se pudo cargar el modelo")
                return False
            
            log.info("✅ Modelo cargado exitosamente")
            
            # 2. Conectar executor (solo si trading está habilitado)
            if self.trading_enabled:
                log.info("🔌 Conectando executor de trading...")
                if not self.executor.connect():
                    log.error("❌ No se pudo conectar el executor")
                    return False
                
                log.info("✅ Executor conectado exitosamente")
                
                # Mostrar información de la cuenta
                account_info = self.executor.get_account_summary()
                if account_info:
                    log.info(f"💰 Cuenta: {account_info['login']}")
                    log.info(f"💰 Balance: {account_info['balance']:.2f} {account_info['currency']}")
                    log.info(f"💰 Equity: {account_info['equity']:.2f} {account_info['currency']}")
            else:
                log.info("📊 Modo análisis - Trading deshabilitado")
            
            # 3. Verificar símbolos
            log.info("🔍 Verificando símbolos...")
            valid_symbols = self._verify_symbols()
            if not valid_symbols:
                log.error("❌ No hay símbolos válidos")
                return False
            
            self.symbols = valid_symbols
            log.info(f"✅ Símbolos válidos: {self.symbols}")
            
            # 4. Configurar manejadores de señales
            self._setup_signal_handlers()
            
            log.info("✅ TradingBot inicializado correctamente")
            return True
            
        except Exception as e:
            log.error(f"❌ Error inicializando bot: {str(e)}")
            return False
    
    def start(self) -> bool:
        """
        Inicia el bot de trading.
        
        Returns:
            bool: True si se inició correctamente
        """
        if self.running:
            log.warning("⚠️ El bot ya está ejecutándose")
            return True
        
        if not self.initialize():
            return False
        
        try:
            self.running = True
            self.stats['start_time'] = datetime.now()
            
            # Iniciar hilo de predicciones
            self.prediction_thread = threading.Thread(
                target=self._prediction_loop,
                name="PredictionThread",
                daemon=True
            )
            self.prediction_thread.start()
            
            # Iniciar hilo de monitoreo
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="MonitoringThread",
                daemon=True
            )
            self.monitoring_thread.start()
            
            log.info("🎯 TradingBot iniciado exitosamente")
            log.info(f"  Modo: {'TRADING REAL' if self.trading_enabled else 'ANÁLISIS'}")
            log.info(f"  Símbolos: {len(self.symbols)}")
            log.info(f"  Intervalo: {self.prediction_interval}s")
            
            return True
            
        except Exception as e:
            log.error(f"❌ Error iniciando bot: {str(e)}")
            self.running = False
            return False
    
    def stop(self, emergency: bool = False):
        """
        Detiene el bot de trading.
        
        Args:
            emergency: Si es una parada de emergencia
        """
        if emergency:
            log.warning("🚨 PARADA DE EMERGENCIA ACTIVADA")
            self.emergency_stop_triggered = True
            
            # Cerrar todas las posiciones si trading está habilitado
            if self.trading_enabled and self.executor.connected:
                self.executor.emergency_stop()
        
        self.running = False
        self.paused = False
        
        # Esperar a que terminen los hilos
        if self.prediction_thread and self.prediction_thread.is_alive():
            self.prediction_thread.join(timeout=10)
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)
        
        # Desconectar executor
        if self.trading_enabled:
            self.executor.disconnect()
        
        # Mostrar estadísticas finales
        self._print_final_stats()
        
        log.info("🛑 TradingBot detenido")
    
    def pause(self):
        """Pausa el bot (detiene nuevas operaciones pero mantiene monitoreo)."""
        self.paused = True
        log.info("⏸️ TradingBot pausado")
    
    def resume(self):
        """Reanuda el bot."""
        self.paused = False
        log.info("▶️ TradingBot reanudado")
    
    def _prediction_loop(self):
        """Loop principal de predicciones."""
        log.info("🔄 Iniciando loop de predicciones...")
        
        while self.running:
            try:
                if self.paused:
                    time.sleep(10)
                    continue
                
                start_time = time.time()
                
                # Realizar predicciones para todos los símbolos
                predictions = self._make_predictions_all_symbols()
                
                # Procesar predicciones
                if predictions:
                    self._process_predictions(predictions)
                
                # Calcular tiempo de ejecución
                execution_time = time.time() - start_time
                
                # Esperar hasta el próximo ciclo
                sleep_time = max(0, self.prediction_interval - execution_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                log.error(f"❌ Error en loop de predicciones: {str(e)}")
                self._execute_error_callbacks(e)
                time.sleep(30)  # Esperar antes de reintentar
    
    def _monitoring_loop(self):
        """Loop de monitoreo del sistema."""
        log.info("👁️ Iniciando loop de monitoreo...")
        
        while self.running:
            try:
                # Actualizar estadísticas
                self._update_stats()
                
                # Verificar conexiones
                if self.trading_enabled:
                    if not self._check_connections():
                        log.warning("⚠️ Problemas de conexión detectados")
                
                # Imprimir estadísticas cada 10 minutos
                if self.stats['predictions_made'] % 20 == 0 and self.stats['predictions_made'] > 0:
                    self._print_current_stats()
                
                time.sleep(60)  # Monitoreo cada minuto
                
            except Exception as e:
                log.error(f"❌ Error en loop de monitoreo: {str(e)}")
                time.sleep(60)
    
    def _make_predictions_all_symbols(self) -> Dict[str, Dict[str, Any]]:
        """
        Realiza predicciones para todos los símbolos.
        
        Returns:
            Dict[str, Dict[str, Any]]: Predicciones por símbolo
        """
        predictions = {}
        
        for symbol in self.symbols:
            try:
                # Obtener datos
                data = self.data_loader.load_data(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    periods=100
                )
                
                if data.empty or len(data) < 50:
                    log.warning(f"⚠️ Datos insuficientes para {symbol}")
                    continue
                
                # Realizar predicción
                prediction = self.predictor.predict_single(data)
                
                # Agregar información adicional
                prediction['symbol'] = symbol
                prediction['last_price'] = float(data['close'].iloc[-1])
                prediction['data_timestamp'] = data.index[-1].isoformat() if hasattr(data.index[-1], 'isoformat') else str(data.index[-1])
                
                # Obtener spread si trading está habilitado
                if self.trading_enabled and self.executor.connected:
                    spread = self._get_spread(symbol)
                    prediction['spread'] = spread
                
                predictions[symbol] = prediction
                self.stats['predictions_made'] += 1
                
                log.debug(f"📊 {symbol}: {prediction['signal']} (conf: {prediction.get('confidence', 0):.3f})")
                
            except Exception as e:
                log.error(f"❌ Error predicción {symbol}: {str(e)}")
                continue
        
        return predictions
    
    def _process_predictions(self, predictions: Dict[str, Dict[str, Any]]):
        """
        Procesa las predicciones y genera señales de trading.
        
        Args:
            predictions: Diccionario de predicciones por símbolo
        """
        for symbol, prediction in predictions.items():
            try:
                # Guardar predicción
                self.current_predictions[symbol] = prediction
                self.prediction_history.append(prediction)
                
                # Mantener solo las últimas 1000 predicciones
                if len(self.prediction_history) > 1000:
                    self.prediction_history = self.prediction_history[-1000:]
                
                # Verificar si generar señal de trading
                if self._should_generate_signal(prediction):
                    signal = self._create_trading_signal(prediction)
                    
                    if signal:
                        self._handle_trading_signal(signal)
                
                # Ejecutar callbacks de señal
                self._execute_signal_callbacks(symbol, prediction)
                
            except Exception as e:
                log.error(f"❌ Error procesando predicción {symbol}: {str(e)}")
    
    def _should_generate_signal(self, prediction: Dict[str, Any]) -> bool:
        """
        Determina si se debe generar una señal de trading.
        
        Args:
            prediction: Predicción del modelo
            
        Returns:
            bool: True si se debe generar señal
        """
        # Filtro básico: no generar señales HOLD
        if prediction['signal'] == 'hold':
            return False
        
        # Filtro de confianza
        if prediction.get('confidence', 0) < self.filters['min_confidence']:
            return False
        
        # Filtro de spread (si está disponible)
        if 'spread' in prediction and prediction['spread'] > self.filters['max_spread']:
            return False
        
        # Filtro de horario de trading (si está configurado)
        if self.filters['trading_hours']:
            current_hour = datetime.now().hour
            if current_hour not in self.filters['trading_hours']:
                return False
        
        return True
    
    def _create_trading_signal(self, prediction: Dict[str, Any]) -> Optional[TradeSignal]:
        """
        Crea una señal de trading a partir de una predicción.
        
        Args:
            prediction: Predicción del modelo
            
        Returns:
            Optional[TradeSignal]: Señal de trading o None
        """
        try:
            signal = TradeSignal(
                symbol=prediction['symbol'],
                signal=prediction['signal'],
                confidence=prediction.get('confidence', 0),
                price=prediction.get('last_price', 0),
                timestamp=datetime.now(),
                probabilities=prediction.get('probabilities', {}),
                metadata={
                    'model_prediction': prediction.get('prediction_raw'),
                    'data_timestamp': prediction.get('data_timestamp'),
                    'spread': prediction.get('spread', 0)
                }
            )
            
            return signal
            
        except Exception as e:
            log.error(f"❌ Error creando señal: {str(e)}")
            return None
    
    def _handle_trading_signal(self, signal: TradeSignal):
        """
        Maneja una señal de trading.
        
        Args:
            signal: Señal de trading
        """
        try:
            log.info(f"🎯 Nueva señal: {signal.symbol} {signal.signal.upper()}")
            log.info(f"  Confianza: {signal.confidence:.3f}")
            log.info(f"  Precio: {signal.price:.5f}")
            
            # Guardar señal
            self.last_signals[signal.symbol] = signal
            self.stats['signals_generated'] += 1
            
            # Ejecutar trading si está habilitado
            if self.trading_enabled and not self.paused:
                order = self.executor.execute_signal(signal)
                
                if order:
                    self.stats['trades_executed'] += 1
                    log.info(f"✅ Operación ejecutada: {order.ticket}")
                    
                    # Ejecutar callbacks de trading
                    self._execute_trade_callbacks(signal, order)
                else:
                    log.warning(f"⚠️ No se pudo ejecutar operación para {signal.symbol}")
            else:
                log.info("📊 Modo análisis - Señal registrada pero no ejecutada")
            
        except Exception as e:
            log.error(f"❌ Error manejando señal: {str(e)}")
            self._execute_error_callbacks(e)
    
    def _verify_symbols(self) -> List[str]:
        """
        Verifica que los símbolos sean válidos.
        
        Returns:
            List[str]: Lista de símbolos válidos
        """
        valid_symbols = []
        
        for symbol in self.symbols:
            try:
                # Intentar obtener datos
                data = self.data_loader.load_data(symbol=symbol, timeframe=self.timeframe, periods=10)
                
                if not data.empty:
                    valid_symbols.append(symbol)
                    log.debug(f"✅ Símbolo válido: {symbol}")
                else:
                    log.warning(f"⚠️ Símbolo sin datos: {symbol}")
                    
            except Exception as e:
                log.warning(f"⚠️ Error verificando {symbol}: {str(e)}")
        
        return valid_symbols
    
    def _get_spread(self, symbol: str) -> float:
        """
        Obtiene el spread actual del símbolo.
        
        Args:
            symbol: Símbolo a consultar
            
        Returns:
            float: Spread en puntos
        """
        try:
            if self.trading_enabled and self.executor.connected:
                import MetaTrader5 as mt5
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    return symbol_info.spread
            return 0.0
        except:
            return 0.0
    
    def _check_connections(self) -> bool:
        """
        Verifica el estado de las conexiones.
        
        Returns:
            bool: True si todas las conexiones están OK
        """
        try:
            if self.trading_enabled:
                # Verificar conexión MT5
                if not self.executor.connected:
                    log.warning("⚠️ Conexión MT5 perdida, reintentando...")
                    return self.executor.connect()
            
            return True
            
        except Exception as e:
            log.error(f"❌ Error verificando conexiones: {str(e)}")
            return False
    
    def _update_stats(self):
        """Actualiza las estadísticas del bot."""
        if self.stats['start_time']:
            self.stats['uptime_hours'] = (datetime.now() - self.stats['start_time']).total_seconds() / 3600
    
    def _print_current_stats(self):
        """Imprime estadísticas actuales."""
        log.info("📈 ESTADÍSTICAS ACTUALES:")
        log.info(f"  Tiempo activo: {self.stats['uptime_hours']:.1f} horas")
        log.info(f"  Predicciones: {self.stats['predictions_made']}")
        log.info(f"  Señales generadas: {self.stats['signals_generated']}")
        log.info(f"  Operaciones ejecutadas: {self.stats['trades_executed']}")
        
        if self.trading_enabled:
            account_summary = self.executor.get_account_summary()
            if account_summary:
                log.info(f"  Balance: {account_summary.get('balance', 0):.2f}")
                log.info(f"  Posiciones abiertas: {account_summary.get('open_positions', 0)}")
    
    def _print_final_stats(self):
        """Imprime estadísticas finales."""
        log.info("📊 ESTADÍSTICAS FINALES:")
        log.info(f"  Duración total: {self.stats['uptime_hours']:.1f} horas")
        log.info(f"  Predicciones totales: {self.stats['predictions_made']}")
        log.info(f"  Señales generadas: {self.stats['signals_generated']}")
        log.info(f"  Operaciones ejecutadas: {self.stats['trades_executed']}")
        
        if self.stats['signals_generated'] > 0:
            execution_rate = (self.stats['trades_executed'] / self.stats['signals_generated']) * 100
            log.info(f"  Tasa de ejecución: {execution_rate:.1f}%")
    
    def _setup_signal_handlers(self):
        """Configura manejadores de señales del sistema."""
        def signal_handler(signum, frame):
            log.info(f"🛑 Señal recibida: {signum}")
            self.stop(emergency=True)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    # Métodos para callbacks
    
    def add_signal_callback(self, callback: Callable):
        """Agrega callback para señales generadas."""
        self.signal_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable):
        """Agrega callback para operaciones ejecutadas."""
        self.trade_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Agrega callback para errores."""
        self.error_callbacks.append(callback)
    
    def _execute_signal_callbacks(self, symbol: str, prediction: Dict[str, Any]):
        """Ejecuta callbacks de señales."""
        for callback in self.signal_callbacks:
            try:
                callback(symbol, prediction)
            except Exception as e:
                log.error(f"❌ Error en callback de señal: {str(e)}")
    
    def _execute_trade_callbacks(self, signal: TradeSignal, order):
        """Ejecuta callbacks de trading."""
        for callback in self.trade_callbacks:
            try:
                callback(signal, order)
            except Exception as e:
                log.error(f"❌ Error en callback de trading: {str(e)}")
    
    def _execute_error_callbacks(self, error: Exception):
        """Ejecuta callbacks de errores."""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                log.error(f"❌ Error en callback de error: {str(e)}")
    
    # Métodos de consulta
    
    def get_current_predictions(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene las predicciones actuales."""
        return self.current_predictions.copy()
    
    def get_last_signals(self) -> Dict[str, TradeSignal]:
        """Obtiene las últimas señales generadas."""
        return self.last_signals.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del bot."""
        return self.stats.copy()
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de la cuenta."""
        if self.trading_enabled:
            return self.executor.get_account_summary()
        return {}
    
    def is_running(self) -> bool:
        """Verifica si el bot está ejecutándose."""
        return self.running
    
    def is_paused(self) -> bool:
        """Verifica si el bot está pausado."""
        return self.paused


# Funciones de utilidad

def create_trading_bot(model_path: str = "models/lit_ml_model.pkl",
                      symbols: List[str] = None,
                      risk_level: str = "moderate",
                      trading_enabled: bool = False) -> TradingBot:
    """
    Crea un bot de trading con configuración predeterminada.
    
    Args:
        model_path: Ruta al modelo entrenado
        symbols: Lista de símbolos a operar
        risk_level: Nivel de riesgo
        trading_enabled: Si habilitar trading real
        
    Returns:
        TradingBot: Instancia configurada
    """
    return TradingBot(
        model_path=model_path,
        symbols=symbols or ['EURUSD', 'GBPUSD', 'USDJPY'],
        risk_level=risk_level,
        trading_enabled=trading_enabled
    )


def run_trading_bot(duration_hours: int = 24,
                   model_path: str = "models/lit_ml_model.pkl",
                   trading_enabled: bool = False) -> bool:
    """
    Ejecuta el bot de trading por un período determinado.
    
    Args:
        duration_hours: Duración en horas
        model_path: Ruta al modelo
        trading_enabled: Si habilitar trading real
        
    Returns:
        bool: True si se ejecutó correctamente
    """
    bot = create_trading_bot(model_path, trading_enabled=trading_enabled)
    
    try:
        if not bot.start():
            return False
        
        log.info(f"🕐 Ejecutando por {duration_hours} horas...")
        time.sleep(duration_hours * 3600)
        
        return True
        
    except KeyboardInterrupt:
        log.info("⏹️ Detenido por el usuario")
        return True
        
    except Exception as e:
        log.error(f"❌ Error en ejecución: {str(e)}")
        return False
        
    finally:
        bot.stop() 