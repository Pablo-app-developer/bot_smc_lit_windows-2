"""
DataLoader Real para MetaTrader 5.

Obtiene datos históricos directamente desde MetaTrader 5
en lugar de Yahoo Finance para mayor precisión y disponibilidad.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

from src.utils.logger import log


class MT5DataLoader:
    """
    DataLoader que obtiene datos reales desde MetaTrader 5.
    
    Ventajas sobre Yahoo Finance:
    - Datos en tiempo real
    - Spreads reales
    - Símbolos Forex correctos
    - Mayor precisión
    """
    
    def __init__(self):
        """Inicializa el DataLoader de MT5."""
        self.is_connected = False
        self.timeframe_map = {
            '1m': mt5.TIMEFRAME_M1,
            '5m': mt5.TIMEFRAME_M5,
            '15m': mt5.TIMEFRAME_M15,
            '30m': mt5.TIMEFRAME_M30,
            '1h': mt5.TIMEFRAME_H1,
            '4h': mt5.TIMEFRAME_H4,
            '1d': mt5.TIMEFRAME_D1,
            '1w': mt5.TIMEFRAME_W1
        }
        
        # Cache para mejorar rendimiento
        self._cache = {}
        self._cache_ttl = 60  # 1 minuto
        
        log.info("MT5DataLoader inicializado")
    
    def connect(self) -> bool:
        """Establece conexión con MT5."""
        try:
            if not mt5.initialize():
                log.error("Error inicializando MT5 para DataLoader")
                return False
            
            self.is_connected = True
            log.info("✅ MT5DataLoader conectado")
            return True
            
        except Exception as e:
            log.error(f"Error conectando MT5DataLoader: {str(e)}")
            return False
    
    def disconnect(self):
        """Desconecta de MT5."""
        if self.is_connected:
            mt5.shutdown()
            self.is_connected = False
            log.info("MT5DataLoader desconectado")
    
    def load_data(self, symbol: str, timeframe: str = "1h", periods: int = 100) -> pd.DataFrame:
        """
        Carga datos históricos reales desde MT5.
        
        Args:
            symbol: Símbolo (EURUSD, GBPUSD, etc.)
            timeframe: Marco temporal (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
            periods: Número de períodos
            
        Returns:
            DataFrame con datos OHLCV
        """
        try:
            log.info(f"Cargando datos REALES MT5: {symbol} - {timeframe} - {periods} períodos")
            
            # Verificar conexión
            if not self.is_connected:
                if not self.connect():
                    log.error("No se pudo conectar a MT5")
                    return pd.DataFrame()
            
            # Verificar cache
            cache_key = f"{symbol}_{timeframe}_{periods}"
            now = time.time()
            
            if cache_key in self._cache:
                cached_data, cache_time = self._cache[cache_key]
                if (now - cache_time) < self._cache_ttl:
                    log.debug(f"Usando datos en cache para {symbol}")
                    return cached_data.copy()
            
            # Obtener timeframe de MT5
            mt5_timeframe = self.timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Verificar símbolo
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                log.error(f"Símbolo {symbol} no disponible en MT5")
                return pd.DataFrame()
            
            # Seleccionar símbolo si no está seleccionado
            if not symbol_info.select:
                if not mt5.symbol_select(symbol, True):
                    log.error(f"No se pudo seleccionar símbolo {symbol}")
                    return pd.DataFrame()
            
            # Obtener datos históricos
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, periods)
            
            if rates is None or len(rates) == 0:
                log.warning(f"No se obtuvieron datos para {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            
            # Convertir tiempo
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Renombrar columnas para compatibilidad
            df = df.rename(columns={
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            })
            
            # Seleccionar columnas principales
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Verificar calidad de datos
            if df.isnull().any().any():
                log.warning(f"Datos con valores nulos para {symbol}")
                df = df.dropna()
            
            # Guardar en cache
            self._cache[cache_key] = (df.copy(), now)
            
            log.info(f"✅ Datos MT5 obtenidos: {len(df)} velas de {symbol} {timeframe}")
            log.info(f"   Rango: {df.index[0]} a {df.index[-1]}")
            log.info(f"   Último precio: {df['close'].iloc[-1]:.5f}")
            
            return df
            
        except Exception as e:
            log.error(f"Error cargando datos MT5 para {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Obtiene precio actual del símbolo.
        
        Args:
            symbol: Símbolo
            
        Returns:
            Precio actual o None
        """
        try:
            if not self.is_connected:
                if not self.connect():
                    return None
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            # Retornar precio medio (bid + ask) / 2
            return (tick.bid + tick.ask) / 2
            
        except Exception as e:
            log.error(f"Error obteniendo precio actual de {symbol}: {str(e)}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Obtiene información del símbolo.
        
        Args:
            symbol: Símbolo
            
        Returns:
            Diccionario con información del símbolo
        """
        try:
            if not self.is_connected:
                if not self.connect():
                    return None
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
            
            tick = mt5.symbol_info_tick(symbol)
            
            return {
                'symbol': symbol,
                'description': symbol_info.description,
                'currency_base': symbol_info.currency_base,
                'currency_profit': symbol_info.currency_profit,
                'point': symbol_info.point,
                'digits': symbol_info.digits,
                'spread': symbol_info.spread,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step,
                'current_bid': tick.bid if tick else None,
                'current_ask': tick.ask if tick else None,
                'current_spread': tick.ask - tick.bid if tick else None
            }
            
        except Exception as e:
            log.error(f"Error obteniendo info de {symbol}: {str(e)}")
            return None
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Valida si un símbolo está disponible.
        
        Args:
            symbol: Símbolo a validar
            
        Returns:
            True si está disponible
        """
        try:
            if not self.is_connected:
                if not self.connect():
                    return False
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
            
            # Intentar seleccionar
            if not symbol_info.select:
                return mt5.symbol_select(symbol, True)
            
            return True
            
        except Exception as e:
            log.error(f"Error validando símbolo {symbol}: {str(e)}")
            return False
    
    def get_available_symbols(self) -> List[str]:
        """
        Obtiene lista de símbolos disponibles.
        
        Returns:
            Lista de símbolos disponibles
        """
        try:
            if not self.is_connected:
                if not self.connect():
                    return []
            
            symbols = mt5.symbols_get()
            if symbols is None:
                return []
            
            # Filtrar solo símbolos Forex principales
            forex_symbols = []
            forex_pairs = [
                'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 
                'USDCAD', 'NZDUSD', 'EURJPY', 'GBPJPY', 'EURGBP',
                'AUDCAD', 'AUDCHF', 'AUDJPY', 'CADCHF', 'CADJPY',
                'CHFJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD',
                'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD', 'NZDCAD',
                'NZDCHF', 'NZDJPY'
            ]
            
            for symbol in symbols:
                if symbol.name in forex_pairs:
                    forex_symbols.append(symbol.name)
            
            return sorted(forex_symbols)
            
        except Exception as e:
            log.error(f"Error obteniendo símbolos disponibles: {str(e)}")
            return []
    
    def clear_cache(self):
        """Limpia el cache de datos."""
        self._cache.clear()
        log.info("Cache de datos limpiado")


# Función de conveniencia
def create_mt5_data_loader() -> MT5DataLoader:
    """Crea una instancia del MT5DataLoader."""
    return MT5DataLoader() 