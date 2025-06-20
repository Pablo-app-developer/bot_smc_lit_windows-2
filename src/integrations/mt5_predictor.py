#!/usr/bin/env python3
"""
Integrador MT5 + Predictor LIT ML.

Este módulo integra el predictor LIT + ML con MetaTrader 5 para
realizar predicciones en tiempo real y ejecutar operaciones automáticas.
"""

import os
import sys
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 no disponible. Instalando...")

from src.models.predictor import LITMLPredictor
from src.utils.logger import log


class MT5PredictorIntegration:
    """
    Integración profesional entre el predictor LIT + ML y MetaTrader 5.
    
    Maneja conexión MT5, obtención de datos en tiempo real,
    predicciones automáticas y ejecución de operaciones.
    """
    
    def __init__(self, 
                 model_path: str = "models/lit_ml_model.pkl",
                 login: int = 5036791117,
                 password: str = "BtUvF-X8",
                 server: str = "MetaQuotes-Demo"):
        """
        Inicializa la integración MT5 + Predictor.
        
        Args:
            model_path: Ruta al modelo entrenado.
            login: Login de MT5.
            password: Password de MT5.
            server: Servidor de MT5.
        """
        self.model_path = model_path
        self.login = login
        self.password = password
        self.server = server
        
        # Componentes principales
        self.predictor = LITMLPredictor(model_path)
        self.mt5_connected = False
        
        # Configuración de trading
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF']
        self.timeframe = mt5.TIMEFRAME_H1 if MT5_AVAILABLE else None
        self.prediction_interval = 300  # 5 minutos
        
        # Estado del sistema
        self.is_running = False
        self.last_predictions = {}
        self.trading_enabled = False
        self.risk_per_trade = 0.02  # 2% por operación
        
        # Hilos de ejecución
        self.prediction_thread = None
        self.monitoring_thread = None
        
        # Estadísticas
        self.stats = {
            'predictions_made': 0,
            'trades_executed': 0,
            'successful_predictions': 0,
            'start_time': None
        }
        
        log.info(f"MT5PredictorIntegration inicializado")
        log.info(f"  Modelo: {model_path}")
        log.info(f"  Login: {login}")
        log.info(f"  Servidor: {server}")
    
    def initialize(self) -> bool:
        """
        Inicializa la conexión MT5 y carga el modelo.
        
        Returns:
            bool: True si se inicializó correctamente.
        """
        log.info("🚀 Inicializando MT5PredictorIntegration...")
        
        # 1. Verificar disponibilidad de MT5
        if not MT5_AVAILABLE:
            log.error("MetaTrader5 no está disponible")
            return False
        
        # 2. Conectar a MT5
        if not self._connect_mt5():
            log.error("No se pudo conectar a MT5")
            return False
        
        # 3. Cargar modelo
        if not self.predictor.load_model():
            log.error("No se pudo cargar el modelo")
            return False
        
        # 4. Verificar símbolos
        if not self._verify_symbols():
            log.warning("Algunos símbolos no están disponibles")
        
        log.info("✅ MT5PredictorIntegration inicializado correctamente")
        return True
    
    def _connect_mt5(self) -> bool:
        """
        Establece conexión con MetaTrader 5.
        
        Returns:
            bool: True si se conectó exitosamente.
        """
        try:
            log.info("Conectando a MetaTrader 5...")
            
            # Inicializar MT5
            if not mt5.initialize():
                log.error(f"Error inicializando MT5: {mt5.last_error()}")
                return False
            
            # Conectar con credenciales
            if not mt5.login(self.login, password=self.password, server=self.server):
                log.error(f"Error login MT5: {mt5.last_error()}")
                mt5.shutdown()
                return False
            
            # Verificar conexión
            account_info = mt5.account_info()
            if account_info is None:
                log.error("No se pudo obtener información de la cuenta")
                return False
            
            self.mt5_connected = True
            
            log.info("✅ Conectado a MT5 exitosamente")
            log.info(f"  Cuenta: {account_info.login}")
            log.info(f"  Servidor: {account_info.server}")
            log.info(f"  Balance: {account_info.balance}")
            log.info(f"  Equity: {account_info.equity}")
            
            return True
            
        except Exception as e:
            log.error(f"Error conectando a MT5: {str(e)}")
            return False
    
    def _verify_symbols(self) -> bool:
        """
        Verifica que los símbolos estén disponibles en MT5.
        
        Returns:
            bool: True si todos los símbolos están disponibles.
        """
        available_symbols = []
        
        for symbol in self.symbols:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is not None:
                available_symbols.append(symbol)
                log.info(f"✅ Símbolo disponible: {symbol}")
            else:
                log.warning(f"⚠️ Símbolo no disponible: {symbol}")
        
        self.symbols = available_symbols
        
        if len(self.symbols) == 0:
            log.error("No hay símbolos disponibles para trading")
            return False
        
        log.info(f"Símbolos configurados: {self.symbols}")
        return True
    
    def start_realtime_predictions(self, trading_enabled: bool = False) -> bool:
        """
        Inicia las predicciones en tiempo real.
        
        Args:
            trading_enabled: Si habilitar trading automático.
            
        Returns:
            bool: True si se inició correctamente.
        """
        if not self.mt5_connected:
            log.error("MT5 no está conectado")
            return False
        
        if self.is_running:
            log.warning("Las predicciones ya están ejecutándose")
            return True
        
        self.trading_enabled = trading_enabled
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        # Iniciar hilo de predicciones
        self.prediction_thread = threading.Thread(
            target=self._prediction_loop,
            daemon=True
        )
        self.prediction_thread.start()
        
        # Iniciar hilo de monitoreo
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        log.info("🎯 Predicciones en tiempo real iniciadas")
        log.info(f"  Trading automático: {'✅ HABILITADO' if trading_enabled else '❌ DESHABILITADO'}")
        log.info(f"  Intervalo: {self.prediction_interval}s")
        log.info(f"  Símbolos: {len(self.symbols)}")
        
        return True
    
    def stop_realtime_predictions(self):
        """Detiene las predicciones en tiempo real."""
        self.is_running = False
        
        # Esperar a que terminen los hilos
        if self.prediction_thread and self.prediction_thread.is_alive():
            self.prediction_thread.join(timeout=10)
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)
        
        log.info("🛑 Predicciones en tiempo real detenidas")
        self._print_session_stats()
    
    def _prediction_loop(self):
        """Loop principal de predicciones."""
        log.info("Iniciando loop de predicciones...")
        
        while self.is_running:
            try:
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
                log.error(f"Error en loop de predicciones: {str(e)}")
                time.sleep(30)  # Esperar 30s antes de reintentar
    
    def _monitoring_loop(self):
        """Loop de monitoreo del sistema."""
        log.info("Iniciando loop de monitoreo...")
        
        while self.is_running:
            try:
                # Verificar conexión MT5
                if not self._check_mt5_connection():
                    log.warning("Conexión MT5 perdida, reintentando...")
                    self._connect_mt5()
                
                # Imprimir estadísticas cada 10 minutos
                if self.stats['predictions_made'] % 10 == 0 and self.stats['predictions_made'] > 0:
                    self._print_current_stats()
                
                time.sleep(60)  # Monitoreo cada minuto
                
            except Exception as e:
                log.error(f"Error en loop de monitoreo: {str(e)}")
                time.sleep(60)
    
    def _make_predictions_all_symbols(self) -> Dict[str, Dict[str, Any]]:
        """
        Realiza predicciones para todos los símbolos.
        
        Returns:
            Dict[str, Dict[str, Any]]: Predicciones por símbolo.
        """
        predictions = {}
        
        for symbol in self.symbols:
            try:
                # Obtener datos de MT5
                data = self._get_mt5_data(symbol, self.timeframe, 200)
                
                if data is None or len(data) < 50:
                    log.warning(f"Datos insuficientes para {symbol}")
                    continue
                
                # Realizar predicción
                prediction = self.predictor.predict_single(data)
                
                # Agregar información del símbolo
                prediction['symbol'] = symbol
                prediction['last_price'] = float(data['close'].iloc[-1])
                prediction['spread'] = self._get_spread(symbol)
                
                predictions[symbol] = prediction
                
                self.stats['predictions_made'] += 1
                
                log.info(f"📊 {symbol}: {prediction['signal']} (conf: {prediction.get('confidence', 0):.3f})")
                
            except Exception as e:
                log.error(f"Error predicción {symbol}: {str(e)}")
                continue
        
        return predictions
    
    def _get_mt5_data(self, symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos de MT5.
        
        Args:
            symbol: Símbolo a consultar.
            timeframe: Marco temporal.
            count: Número de velas.
            
        Returns:
            Optional[pd.DataFrame]: DataFrame con datos OHLCV.
        """
        try:
            # Obtener datos de MT5
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                log.warning(f"No se pudieron obtener datos para {symbol}")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            
            # Convertir timestamp a datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Renombrar columnas para compatibilidad
            df.rename(columns={
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            }, inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            log.error(f"Error obteniendo datos MT5 para {symbol}: {str(e)}")
            return None
    
    def _get_spread(self, symbol: str) -> float:
        """
        Obtiene el spread actual del símbolo.
        
        Args:
            symbol: Símbolo a consultar.
            
        Returns:
            float: Spread en puntos.
        """
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is not None:
                return symbol_info.spread
            return 0.0
        except:
            return 0.0
    
    def _process_predictions(self, predictions: Dict[str, Dict[str, Any]]):
        """
        Procesa las predicciones y ejecuta acciones.
        
        Args:
            predictions: Diccionario de predicciones por símbolo.
        """
        for symbol, prediction in predictions.items():
            try:
                # Guardar predicción
                self.last_predictions[symbol] = prediction
                
                # Ejecutar trading si está habilitado
                if self.trading_enabled:
                    self._execute_trading_decision(symbol, prediction)
                
            except Exception as e:
                log.error(f"Error procesando predicción {symbol}: {str(e)}")
    
    def _execute_trading_decision(self, symbol: str, prediction: Dict[str, Any]):
        """
        Ejecuta decisión de trading basada en la predicción.
        
        Args:
            symbol: Símbolo a operar.
            prediction: Predicción del modelo.
        """
        signal = prediction['signal']
        confidence = prediction.get('confidence', 0)
        
        # Filtros de calidad
        if confidence < 0.6:  # Confianza mínima 60%
            return
        
        if prediction.get('spread', 0) > 3:  # Spread máximo 3 puntos
            log.warning(f"Spread muy alto para {symbol}: {prediction['spread']}")
            return
        
        try:
            if signal == 'buy':
                self._place_buy_order(symbol, prediction)
            elif signal == 'sell':
                self._place_sell_order(symbol, prediction)
            # 'hold' no ejecuta operación
            
        except Exception as e:
            log.error(f"Error ejecutando trading {symbol}: {str(e)}")
    
    def _place_buy_order(self, symbol: str, prediction: Dict[str, Any]):
        """
        Coloca orden de compra.
        
        Args:
            symbol: Símbolo a comprar.
            prediction: Predicción del modelo.
        """
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return
            
            # Calcular volumen basado en riesgo
            lot_size = self._calculate_lot_size(symbol, 'buy')
            
            # Precio actual
            price = mt5.symbol_info_tick(symbol).ask
            
            # Stop Loss y Take Profit
            sl = price - (50 * symbol_info.point)  # 50 puntos SL
            tp = price + (100 * symbol_info.point)  # 100 puntos TP
            
            # Crear orden
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 12345,
                "comment": f"LIT_ML_BUY_{prediction.get('confidence', 0):.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Enviar orden
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                log.error(f"Error orden BUY {symbol}: {result.retcode}")
            else:
                log.info(f"✅ Orden BUY ejecutada: {symbol} @ {price} (Vol: {lot_size})")
                self.stats['trades_executed'] += 1
            
        except Exception as e:
            log.error(f"Error colocando orden BUY {symbol}: {str(e)}")
    
    def _place_sell_order(self, symbol: str, prediction: Dict[str, Any]):
        """
        Coloca orden de venta.
        
        Args:
            symbol: Símbolo a vender.
            prediction: Predicción del modelo.
        """
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return
            
            # Calcular volumen basado en riesgo
            lot_size = self._calculate_lot_size(symbol, 'sell')
            
            # Precio actual
            price = mt5.symbol_info_tick(symbol).bid
            
            # Stop Loss y Take Profit
            sl = price + (50 * symbol_info.point)  # 50 puntos SL
            tp = price - (100 * symbol_info.point)  # 100 puntos TP
            
            # Crear orden
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 12345,
                "comment": f"LIT_ML_SELL_{prediction.get('confidence', 0):.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Enviar orden
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                log.error(f"Error orden SELL {symbol}: {result.retcode}")
            else:
                log.info(f"✅ Orden SELL ejecutada: {symbol} @ {price} (Vol: {lot_size})")
                self.stats['trades_executed'] += 1
            
        except Exception as e:
            log.error(f"Error colocando orden SELL {symbol}: {str(e)}")
    
    def _calculate_lot_size(self, symbol: str, direction: str) -> float:
        """
        Calcula el tamaño del lote basado en gestión de riesgo.
        
        Args:
            symbol: Símbolo a operar.
            direction: Dirección ('buy' o 'sell').
            
        Returns:
            float: Tamaño del lote.
        """
        try:
            account_info = mt5.account_info()
            symbol_info = mt5.symbol_info(symbol)
            
            if account_info is None or symbol_info is None:
                return 0.01  # Lote mínimo por defecto
            
            # Capital disponible
            balance = account_info.balance
            
            # Riesgo por operación
            risk_amount = balance * self.risk_per_trade
            
            # Stop Loss en puntos (50 puntos)
            sl_points = 50
            
            # Valor por punto
            point_value = symbol_info.trade_tick_value
            
            # Calcular lote
            lot_size = risk_amount / (sl_points * point_value)
            
            # Ajustar a límites del símbolo
            lot_size = max(symbol_info.volume_min, lot_size)
            lot_size = min(symbol_info.volume_max, lot_size)
            
            # Redondear al step del símbolo
            lot_step = symbol_info.volume_step
            lot_size = round(lot_size / lot_step) * lot_step
            
            return lot_size
            
        except Exception as e:
            log.error(f"Error calculando lote para {symbol}: {str(e)}")
            return 0.01
    
    def _check_mt5_connection(self) -> bool:
        """
        Verifica si la conexión MT5 está activa.
        
        Returns:
            bool: True si está conectado.
        """
        try:
            account_info = mt5.account_info()
            return account_info is not None
        except:
            return False
    
    def _print_current_stats(self):
        """Imprime estadísticas actuales."""
        if self.stats['start_time']:
            runtime = datetime.now() - self.stats['start_time']
            
            log.info("📈 ESTADÍSTICAS ACTUALES:")
            log.info(f"  Tiempo ejecutándose: {runtime}")
            log.info(f"  Predicciones realizadas: {self.stats['predictions_made']}")
            log.info(f"  Operaciones ejecutadas: {self.stats['trades_executed']}")
            log.info(f"  Símbolos activos: {len(self.symbols)}")
    
    def _print_session_stats(self):
        """Imprime estadísticas de la sesión."""
        if self.stats['start_time']:
            runtime = datetime.now() - self.stats['start_time']
            
            log.info("📊 RESUMEN DE SESIÓN:")
            log.info(f"  Duración total: {runtime}")
            log.info(f"  Predicciones totales: {self.stats['predictions_made']}")
            log.info(f"  Operaciones ejecutadas: {self.stats['trades_executed']}")
            
            if self.stats['predictions_made'] > 0:
                trade_rate = (self.stats['trades_executed'] / self.stats['predictions_made']) * 100
                log.info(f"  Tasa de ejecución: {trade_rate:.1f}%")
    
    def get_current_predictions(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene las predicciones actuales.
        
        Returns:
            Dict[str, Dict[str, Any]]: Predicciones por símbolo.
        """
        return self.last_predictions.copy()
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de la cuenta MT5.
        
        Returns:
            Optional[Dict[str, Any]]: Información de la cuenta.
        """
        if not self.mt5_connected:
            return None
        
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return None
            
            return {
                'login': account_info.login,
                'server': account_info.server,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'currency': account_info.currency
            }
            
        except Exception as e:
            log.error(f"Error obteniendo info cuenta: {str(e)}")
            return None
    
    def shutdown(self):
        """Cierra la conexión y limpia recursos."""
        log.info("🔄 Cerrando MT5PredictorIntegration...")
        
        # Detener predicciones
        if self.is_running:
            self.stop_realtime_predictions()
        
        # Cerrar conexión MT5
        if self.mt5_connected:
            mt5.shutdown()
            self.mt5_connected = False
            log.info("✅ Conexión MT5 cerrada")
        
        log.info("✅ MT5PredictorIntegration cerrado correctamente")


# Funciones de utilidad

def install_mt5():
    """Instala MetaTrader5 si no está disponible."""
    try:
        import subprocess
        import sys
        
        log.info("Instalando MetaTrader5...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "MetaTrader5"])
        log.info("✅ MetaTrader5 instalado correctamente")
        return True
    except Exception as e:
        log.error(f"Error instalando MetaTrader5: {str(e)}")
        return False


def create_mt5_predictor(model_path: str = "models/lit_ml_model.pkl") -> MT5PredictorIntegration:
    """
    Crea una instancia de MT5PredictorIntegration con configuración por defecto.
    
    Args:
        model_path: Ruta al modelo entrenado.
        
    Returns:
        MT5PredictorIntegration: Instancia configurada.
    """
    return MT5PredictorIntegration(
        model_path=model_path,
        login=5036791117,
        password="BtUvF-X8",
        server="MetaQuotes-Demo"
    )


def run_realtime_predictions(model_path: str = "models/lit_ml_model.pkl",
                           trading_enabled: bool = False,
                           duration_hours: int = 24) -> bool:
    """
    Ejecuta predicciones en tiempo real por un período determinado.
    
    Args:
        model_path: Ruta al modelo.
        trading_enabled: Si habilitar trading automático.
        duration_hours: Duración en horas.
        
    Returns:
        bool: True si se ejecutó correctamente.
    """
    # Verificar/instalar MT5
    if not MT5_AVAILABLE:
        if not install_mt5():
            return False
    
    # Crear integrador
    integrator = create_mt5_predictor(model_path)
    
    try:
        # Inicializar
        if not integrator.initialize():
            return False
        
        # Iniciar predicciones
        if not integrator.start_realtime_predictions(trading_enabled):
            return False
        
        # Ejecutar por el tiempo especificado
        log.info(f"🕐 Ejecutando por {duration_hours} horas...")
        time.sleep(duration_hours * 3600)
        
        return True
        
    except KeyboardInterrupt:
        log.info("⏹️ Detenido por el usuario")
        return True
        
    except Exception as e:
        log.error(f"Error en ejecución: {str(e)}")
        return False
        
    finally:
        integrator.shutdown() 