"""
Proveedor de Datos Forex Profesional.

Sistema completo para obtener datos de los principales pares de divisas
con múltiples fuentes y configuración optimizada para trading 24/7.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import log


class ForexDataProvider:
    """
    Proveedor de datos Forex profesional.
    
    Características:
    - Soporte para pares principales, menores y exóticos
    - Datos 24/7 con múltiples fuentes
    - Spreads y comisiones realistas
    - Sesiones de mercado (Londres, Nueva York, Tokio, Sydney)
    """
    
    def __init__(self):
        """Inicializa el proveedor de datos Forex."""
        
        # PARES FOREX PRINCIPALES (MAJORS)
        self.major_pairs = {
            'EURUSD': {'symbol': 'EURUSD=X', 'pip_value': 0.0001, 'spread': 1.2},
            'GBPUSD': {'symbol': 'GBPUSD=X', 'pip_value': 0.0001, 'spread': 1.5},
            'USDJPY': {'symbol': 'USDJPY=X', 'pip_value': 0.01, 'spread': 1.0},
            'USDCHF': {'symbol': 'USDCHF=X', 'pip_value': 0.0001, 'spread': 1.8},
            'AUDUSD': {'symbol': 'AUDUSD=X', 'pip_value': 0.0001, 'spread': 1.4},
            'USDCAD': {'symbol': 'USDCAD=X', 'pip_value': 0.0001, 'spread': 1.6},
            'NZDUSD': {'symbol': 'NZDUSD=X', 'pip_value': 0.0001, 'spread': 2.0}
        }
        
        # PARES MENORES (MINORS/CROSSES)
        self.minor_pairs = {
            'EURGBP': {'symbol': 'EURGBP=X', 'pip_value': 0.0001, 'spread': 1.8},
            'EURJPY': {'symbol': 'EURJPY=X', 'pip_value': 0.01, 'spread': 1.6},
            'GBPJPY': {'symbol': 'GBPJPY=X', 'pip_value': 0.01, 'spread': 2.2},
            'EURCHF': {'symbol': 'EURCHF=X', 'pip_value': 0.0001, 'spread': 2.0},
            'GBPCHF': {'symbol': 'GBPCHF=X', 'pip_value': 0.0001, 'spread': 2.5},
            'AUDCAD': {'symbol': 'AUDCAD=X', 'pip_value': 0.0001, 'spread': 2.8},
            'AUDCHF': {'symbol': 'AUDCHF=X', 'pip_value': 0.0001, 'spread': 2.4},
            'AUDJPY': {'symbol': 'AUDJPY=X', 'pip_value': 0.01, 'spread': 2.0},
            'CADJPY': {'symbol': 'CADJPY=X', 'pip_value': 0.01, 'spread': 2.6},
            'CHFJPY': {'symbol': 'CHFJPY=X', 'pip_value': 0.01, 'spread': 2.8}
        }
        
        # PARES EXÓTICOS (EXOTICS)
        self.exotic_pairs = {
            'USDMXN': {'symbol': 'USDMXN=X', 'pip_value': 0.0001, 'spread': 8.0},
            'USDZAR': {'symbol': 'USDZAR=X', 'pip_value': 0.0001, 'spread': 12.0},
            'USDTRY': {'symbol': 'USDTRY=X', 'pip_value': 0.0001, 'spread': 15.0},
            'USDBRL': {'symbol': 'USDBRL=X', 'pip_value': 0.0001, 'spread': 20.0}
        }
        
        # Combinar todos los pares
        self.all_pairs = {**self.major_pairs, **self.minor_pairs, **self.exotic_pairs}
        
        # SESIONES DE MERCADO FOREX (UTC)
        self.market_sessions = {
            'Sydney': {'start': 21, 'end': 6},      # 21:00-06:00 UTC
            'Tokyo': {'start': 23, 'end': 8},       # 23:00-08:00 UTC
            'London': {'start': 7, 'end': 16},      # 07:00-16:00 UTC
            'New_York': {'start': 12, 'end': 21}    # 12:00-21:00 UTC
        }
        
        # Configuración optimizada por timeframe
        self.timeframe_config = {
            '1m': {'period': '7d', 'interval': '1m'},
            '5m': {'period': '60d', 'interval': '5m'},
            '15m': {'period': '60d', 'interval': '15m'},
            '1h': {'period': '730d', 'interval': '1h'},    # 2 años
            '4h': {'period': '730d', 'interval': '1d'},    # Fallback
            '1d': {'period': '5y', 'interval': '1d'},      # 5 años
            '1w': {'period': '10y', 'interval': '1wk'}     # 10 años
        }
        
        log.info("ForexDataProvider inicializado con 21 pares de divisas")
    
    def get_forex_data(self, pair: str, timeframe: str = '1h', 
                      periods: int = 200) -> pd.DataFrame:
        """
        Obtiene datos de un par de divisas.
        
        Args:
            pair: Par de divisas (ej: 'EURUSD', 'GBPUSD')
            timeframe: Marco temporal
            periods: Número de períodos
            
        Returns:
            DataFrame con datos OHLCV + información Forex
        """
        try:
            if pair not in self.all_pairs:
                log.error(f"Par {pair} no soportado")
                return pd.DataFrame()
            
            pair_info = self.all_pairs[pair]
            symbol = pair_info['symbol']
            
            log.info(f"Cargando datos Forex: {pair} ({symbol}) - {timeframe} - {periods} períodos")
            
            # Obtener configuración
            config = self.timeframe_config.get(timeframe, {
                'period': '2y',
                'interval': '1d'
            })
            
            # Descargar datos
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                period=config['period'],
                interval=config['interval'],
                auto_adjust=True,
                prepost=False
            )
            
            if data.empty:
                log.warning(f"No se encontraron datos para {pair}")
                return pd.DataFrame()
            
            # Limpiar datos
            data.columns = [col.lower() for col in data.columns]
            data = data.dropna()
            
            # Agregar información Forex específica
            data = self._add_forex_features(data, pair_info)
            
            # Tomar últimos períodos
            if len(data) > periods:
                data = data.tail(periods)
            
            log.info(f"✅ Datos Forex cargados: {len(data)} filas para {pair}")
            return data
            
        except Exception as e:
            log.error(f"Error cargando datos Forex {pair}: {str(e)}")
            return pd.DataFrame()
    
    def _add_forex_features(self, data: pd.DataFrame, pair_info: Dict) -> pd.DataFrame:
        """Agrega características específicas de Forex."""
        try:
            # Información del par
            data['pip_value'] = pair_info['pip_value']
            data['spread_pips'] = pair_info['spread']
            data['spread_price'] = pair_info['spread'] * pair_info['pip_value']
            
            # Precios bid/ask simulados
            data['bid'] = data['close'] - (data['spread_price'] / 2)
            data['ask'] = data['close'] + (data['spread_price'] / 2)
            
            # Rangos en pips
            data['range_pips'] = (data['high'] - data['low']) / pair_info['pip_value']
            data['body_pips'] = abs(data['close'] - data['open']) / pair_info['pip_value']
            
            # Sesión de mercado activa
            data['market_session'] = data.index.to_series().apply(self._get_market_session)
            
            # Volatilidad horaria
            data['hourly_volatility'] = data['close'].rolling(24).std() if len(data) > 24 else 0
            
            return data
            
        except Exception as e:
            log.error(f"Error agregando features Forex: {str(e)}")
            return data
    
    def _get_market_session(self, timestamp) -> str:
        """Determina la sesión de mercado activa."""
        try:
            hour = timestamp.hour
            
            # Verificar cada sesión
            for session, times in self.market_sessions.items():
                start, end = times['start'], times['end']
                
                if start <= end:  # Sesión normal
                    if start <= hour < end:
                        return session
                else:  # Sesión que cruza medianoche
                    if hour >= start or hour < end:
                        return session
            
            return 'Closed'
            
        except:
            return 'Unknown'
    
    def get_multiple_pairs(self, pairs: List[str], timeframe: str = '1h', 
                          periods: int = 200) -> Dict[str, pd.DataFrame]:
        """
        Obtiene datos para múltiples pares de divisas.
        
        Args:
            pairs: Lista de pares de divisas
            timeframe: Marco temporal
            periods: Períodos por par
            
        Returns:
            Diccionario con datos por par
        """
        results = {}
        
        log.info(f"Cargando {len(pairs)} pares de divisas...")
        
        for pair in pairs:
            try:
                data = self.get_forex_data(pair, timeframe, periods)
                if len(data) >= 50:  # Mínimo aceptable
                    results[pair] = data
                    log.info(f"✅ {pair}: {len(data)} filas cargadas")
                else:
                    log.warning(f"⚠️  {pair}: datos insuficientes ({len(data)} filas)")
            except Exception as e:
                log.error(f"❌ Error cargando {pair}: {str(e)}")
        
        return results
    
    def get_current_forex_price(self, pair: str) -> Dict[str, float]:
        """
        Obtiene precio actual de un par con bid/ask.
        
        Args:
            pair: Par de divisas
            
        Returns:
            Diccionario con precios bid, ask, mid y spread
        """
        try:
            if pair not in self.all_pairs:
                return {}
            
            pair_info = self.all_pairs[pair]
            symbol = pair_info['symbol']
            
            # Obtener precio actual
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            
            if data.empty:
                return {}
            
            current_price = float(data['Close'].iloc[-1])
            spread_price = pair_info['spread'] * pair_info['pip_value']
            
            return {
                'pair': pair,
                'mid': current_price,
                'bid': current_price - (spread_price / 2),
                'ask': current_price + (spread_price / 2),
                'spread_pips': pair_info['spread'],
                'spread_price': spread_price,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            log.error(f"Error obteniendo precio actual {pair}: {str(e)}")
            return {}
    
    def get_market_status(self) -> Dict[str, any]:
        """
        Obtiene el estado actual del mercado Forex.
        
        Returns:
            Información sobre sesiones activas y volatilidad
        """
        try:
            now = datetime.utcnow()
            hour = now.hour
            
            # Sesiones activas
            active_sessions = []
            for session, times in self.market_sessions.items():
                start, end = times['start'], times['end']
                
                if start <= end:
                    if start <= hour < end:
                        active_sessions.append(session)
                else:
                    if hour >= start or hour < end:
                        active_sessions.append(session)
            
            # Determinar volatilidad esperada
            volatility_level = 'Low'
            if len(active_sessions) >= 2:
                volatility_level = 'High'  # Solapamiento de sesiones
            elif len(active_sessions) == 1:
                volatility_level = 'Medium'
            
            return {
                'timestamp': now,
                'utc_hour': hour,
                'active_sessions': active_sessions,
                'expected_volatility': volatility_level,
                'market_open': len(active_sessions) > 0,
                'session_overlap': len(active_sessions) >= 2
            }
            
        except Exception as e:
            log.error(f"Error obteniendo estado del mercado: {str(e)}")
            return {}
    
    def get_correlation_matrix(self, pairs: List[str], timeframe: str = '1d', 
                             periods: int = 100) -> pd.DataFrame:
        """
        Calcula matriz de correlación entre pares de divisas.
        
        Args:
            pairs: Lista de pares
            timeframe: Marco temporal
            periods: Períodos para cálculo
            
        Returns:
            Matriz de correlación
        """
        try:
            log.info(f"Calculando correlaciones para {len(pairs)} pares...")
            
            # Obtener datos de todos los pares
            all_data = self.get_multiple_pairs(pairs, timeframe, periods)
            
            if len(all_data) < 2:
                log.warning("Datos insuficientes para correlación")
                return pd.DataFrame()
            
            # Crear DataFrame con precios de cierre
            price_data = pd.DataFrame()
            for pair, data in all_data.items():
                if len(data) > 0:
                    price_data[pair] = data['close']
            
            # Calcular correlación
            correlation_matrix = price_data.corr()
            
            log.info(f"✅ Matriz de correlación calculada: {correlation_matrix.shape}")
            return correlation_matrix
            
        except Exception as e:
            log.error(f"Error calculando correlaciones: {str(e)}")
            return pd.DataFrame()
    
    def get_recommended_pairs(self, risk_level: str = 'medium') -> List[str]:
        """
        Recomienda pares según nivel de riesgo.
        
        Args:
            risk_level: 'low', 'medium', 'high'
            
        Returns:
            Lista de pares recomendados
        """
        recommendations = {
            'low': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF'],  # Majors con menor spread
            'medium': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP', 'EURJPY'],
            'high': list(self.major_pairs.keys()) + list(self.minor_pairs.keys())
        }
        
        return recommendations.get(risk_level, recommendations['medium'])


# Función de conveniencia
def create_forex_provider() -> ForexDataProvider:
    """Crea una instancia del proveedor Forex."""
    return ForexDataProvider() 