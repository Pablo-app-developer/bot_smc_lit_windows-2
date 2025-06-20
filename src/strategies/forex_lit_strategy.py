"""
Estrategia LIT Adaptada para Forex.

Implementa la estrategia Liquidity + Inducement Theory específicamente
optimizada para el mercado Forex con sus características únicas:
- Mercado 24/7
- Alta liquidez
- Spreads variables
- Sesiones de mercado
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from src.strategies.lit_detector import LITDetector
from src.brokers.forex_data_provider import ForexDataProvider
from src.utils.logger import log


class ForexLITStrategy:
    """
    Estrategia LIT optimizada para Forex.
    
    Características específicas para divisas:
    - Detección de liquidez en niveles psicológicos (00, 50)
    - Análisis de sesiones de mercado
    - Gestión de spreads variables
    - Correlaciones entre pares
    """
    
    def __init__(self, lookback_periods: int = 50, 
                 liquidity_threshold: float = 0.0001):
        """
        Inicializa la estrategia Forex LIT.
        
        Args:
            lookback_periods: Períodos para análisis histórico
            liquidity_threshold: Umbral de liquidez en pips
        """
        self.lookback_periods = lookback_periods
        self.liquidity_threshold = liquidity_threshold
        
        # Inicializar componentes
        self.lit_detector = LITDetector()
        self.forex_provider = ForexDataProvider()
        
        # NIVELES PSICOLÓGICOS FOREX (en pips)
        self.psychological_levels = {
            'major': [0, 50],  # .0000, .0050
            'minor': [20, 30, 70, 80],  # .0020, .0030, etc.
            'round_numbers': [0, 25, 50, 75]  # Números redondos
        }
        
        # CONFIGURACIÓN POR SESIÓN
        self.session_config = {
            'London': {
                'volatility_multiplier': 1.2,
                'preferred_pairs': ['EURUSD', 'GBPUSD', 'EURGBP'],
                'liquidity_threshold': 0.0001
            },
            'New_York': {
                'volatility_multiplier': 1.3,
                'preferred_pairs': ['EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY'],
                'liquidity_threshold': 0.0001
            },
            'Tokyo': {
                'volatility_multiplier': 0.8,
                'preferred_pairs': ['USDJPY', 'AUDJPY', 'EURJPY'],
                'liquidity_threshold': 0.00015
            },
            'Sydney': {
                'volatility_multiplier': 0.6,
                'preferred_pairs': ['AUDUSD', 'NZDUSD', 'AUDCAD'],
                'liquidity_threshold': 0.00015
            }
        }
        
        log.info("ForexLITStrategy inicializada para trading de divisas")
    
    def analyze_forex_pair(self, pair: str, timeframe: str = '1h', 
                          periods: int = 200) -> Dict[str, any]:
        """
        Analiza un par de divisas con estrategia LIT.
        
        Args:
            pair: Par de divisas (ej: 'EURUSD')
            timeframe: Marco temporal
            periods: Períodos de análisis
            
        Returns:
            Análisis completo con señales LIT
        """
        try:
            log.info(f"Analizando par Forex: {pair} - {timeframe}")
            
            # 1. Obtener datos del par
            data = self.forex_provider.get_forex_data(pair, timeframe, periods)
            
            if data.empty or len(data) < self.lookback_periods:
                log.warning(f"Datos insuficientes para {pair}")
                return {'signal': 0, 'confidence': 0.0, 'reason': 'Datos insuficientes'}
            
            # 2. Obtener estado del mercado
            market_status = self.forex_provider.get_market_status()
            
            # 3. Detectar eventos LIT básicos
            lit_signals = self.lit_detector.detect_lit_events(data)
            
            # 4. Análisis específico Forex
            forex_analysis = self._analyze_forex_specific(data, pair, market_status)
            
            # 5. Detectar niveles psicológicos
            psychological_analysis = self._analyze_psychological_levels(data, pair)
            
            # 6. Análisis de sesión de mercado
            session_analysis = self._analyze_market_session(data, market_status)
            
            # 7. Combinar análisis
            final_signal = self._combine_forex_signals(
                lit_signals, forex_analysis, psychological_analysis, session_analysis
            )
            
            # 8. Calcular niveles de entrada y salida
            entry_levels = self._calculate_forex_levels(data, final_signal, pair)
            
            result = {
                'pair': pair,
                'timeframe': timeframe,
                'signal': final_signal['signal'],
                'confidence': final_signal['confidence'],
                'reason': final_signal['reason'],
                'entry_levels': entry_levels,
                'market_session': market_status.get('active_sessions', []),
                'spread_impact': forex_analysis.get('spread_impact', 0),
                'psychological_level': psychological_analysis.get('nearest_level', None),
                'session_strength': session_analysis.get('strength', 0),
                'timestamp': datetime.now()
            }
            
            log.info(f"✅ Análisis {pair} completado - Señal: {final_signal['signal']}")
            return result
            
        except Exception as e:
            log.error(f"Error analizando {pair}: {str(e)}")
            return {'signal': 0, 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    def _analyze_forex_specific(self, data: pd.DataFrame, pair: str, 
                               market_status: Dict) -> Dict[str, any]:
        """Análisis específico para características Forex."""
        try:
            analysis = {}
            
            # 1. Análisis de spread
            if 'spread_pips' in data.columns:
                avg_spread = data['spread_pips'].mean()
                current_spread = data['spread_pips'].iloc[-1]
                
                analysis['spread_impact'] = current_spread / avg_spread
                analysis['spread_favorable'] = current_spread <= avg_spread * 1.1
            
            # 2. Análisis de volatilidad por sesión
            if 'market_session' in data.columns:
                session_volatility = {}
                for session in ['London', 'New_York', 'Tokyo', 'Sydney']:
                    session_data = data[data['market_session'] == session]
                    if len(session_data) > 0:
                        session_volatility[session] = session_data['range_pips'].mean()
                
                analysis['session_volatility'] = session_volatility
            
            # 3. Análisis de momentum intraday
            if len(data) >= 24:  # Al menos 24 períodos
                recent_data = data.tail(24)
                momentum = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
                analysis['intraday_momentum'] = momentum
            
            # 4. Análisis de rango de trading
            if 'range_pips' in data.columns:
                avg_range = data['range_pips'].tail(20).mean()
                current_range = data['range_pips'].iloc[-1]
                analysis['range_expansion'] = current_range > avg_range * 1.2
            
            return analysis
            
        except Exception as e:
            log.error(f"Error en análisis Forex específico: {str(e)}")
            return {}
    
    def _analyze_psychological_levels(self, data: pd.DataFrame, pair: str) -> Dict[str, any]:
        """Analiza proximidad a niveles psicológicos."""
        try:
            current_price = data['close'].iloc[-1]
            
            # Obtener información del par para calcular pips
            pair_info = self.forex_provider.all_pairs.get(pair, {})
            pip_value = pair_info.get('pip_value', 0.0001)
            
            # Convertir precio a pips (últimos 2-3 dígitos)
            if pip_value == 0.01:  # Pares JPY
                price_pips = int((current_price % 1) * 100)
            else:  # Otros pares
                price_pips = int((current_price % 0.01) * 10000)
            
            # Encontrar nivel psicológico más cercano
            all_levels = (self.psychological_levels['major'] + 
                         self.psychological_levels['minor'])
            
            nearest_level = min(all_levels, key=lambda x: abs(x - price_pips))
            distance_to_level = abs(price_pips - nearest_level)
            
            # Determinar si está cerca de un nivel importante
            is_near_major = distance_to_level <= 5 and nearest_level in self.psychological_levels['major']
            is_near_minor = distance_to_level <= 3 and nearest_level in self.psychological_levels['minor']
            
            return {
                'current_price_pips': price_pips,
                'nearest_level': nearest_level,
                'distance_to_level': distance_to_level,
                'is_near_major_level': is_near_major,
                'is_near_minor_level': is_near_minor,
                'psychological_strength': 1.0 if is_near_major else 0.5 if is_near_minor else 0.0
            }
            
        except Exception as e:
            log.error(f"Error analizando niveles psicológicos: {str(e)}")
            return {}
    
    def _analyze_market_session(self, data: pd.DataFrame, 
                               market_status: Dict) -> Dict[str, any]:
        """Analiza la fuerza de la sesión de mercado actual."""
        try:
            active_sessions = market_status.get('active_sessions', [])
            
            if not active_sessions:
                return {'strength': 0.0, 'preferred': False}
            
            # Calcular fuerza basada en sesiones activas
            total_strength = 0.0
            session_multipliers = []
            
            for session in active_sessions:
                if session in self.session_config:
                    multiplier = self.session_config[session]['volatility_multiplier']
                    total_strength += multiplier
                    session_multipliers.append(multiplier)
            
            # Normalizar fuerza
            avg_strength = total_strength / len(active_sessions) if active_sessions else 0.0
            
            # Determinar si hay solapamiento de sesiones (mayor volatilidad)
            session_overlap = len(active_sessions) >= 2
            
            return {
                'strength': avg_strength,
                'session_overlap': session_overlap,
                'active_sessions': active_sessions,
                'volatility_expected': 'High' if session_overlap else 'Medium' if avg_strength > 1.0 else 'Low'
            }
            
        except Exception as e:
            log.error(f"Error analizando sesión de mercado: {str(e)}")
            return {'strength': 0.0}
    
    def _combine_forex_signals(self, lit_signals: pd.Series, forex_analysis: Dict,
                              psychological_analysis: Dict, session_analysis: Dict) -> Dict[str, any]:
        """Combina todas las señales para generar señal final."""
        try:
            if lit_signals.empty:
                return {'signal': 0, 'confidence': 0.0, 'reason': 'Sin señales LIT'}
            
            # Señal LIT base
            base_signal = lit_signals.iloc[-1]
            base_confidence = 0.4  # Confianza base
            
            # Factores de ajuste
            confidence_adjustments = []
            reasons = []
            
            # 1. Ajuste por spread
            spread_impact = forex_analysis.get('spread_impact', 1.0)
            if spread_impact <= 1.1:  # Spread favorable
                confidence_adjustments.append(0.1)
                reasons.append('Spread favorable')
            elif spread_impact >= 1.5:  # Spread desfavorable
                confidence_adjustments.append(-0.15)
                reasons.append('Spread elevado')
            
            # 2. Ajuste por niveles psicológicos
            psych_strength = psychological_analysis.get('psychological_strength', 0.0)
            if psych_strength >= 1.0:  # Cerca de nivel mayor
                confidence_adjustments.append(0.15)
                reasons.append('Cerca de nivel psicológico mayor')
            elif psych_strength >= 0.5:  # Cerca de nivel menor
                confidence_adjustments.append(0.08)
                reasons.append('Cerca de nivel psicológico menor')
            
            # 3. Ajuste por sesión de mercado
            session_strength = session_analysis.get('strength', 0.0)
            if session_strength >= 1.2:  # Sesión fuerte
                confidence_adjustments.append(0.12)
                reasons.append('Sesión de alta volatilidad')
            elif session_strength <= 0.8:  # Sesión débil
                confidence_adjustments.append(-0.08)
                reasons.append('Sesión de baja volatilidad')
            
            # 4. Ajuste por solapamiento de sesiones
            if session_analysis.get('session_overlap', False):
                confidence_adjustments.append(0.1)
                reasons.append('Solapamiento de sesiones')
            
            # 5. Ajuste por momentum
            momentum = forex_analysis.get('intraday_momentum', 0.0)
            if abs(momentum) > 0.005:  # Momentum fuerte (>0.5%)
                if (base_signal > 0 and momentum > 0) or (base_signal < 0 and momentum < 0):
                    confidence_adjustments.append(0.1)
                    reasons.append('Momentum alineado')
                else:
                    confidence_adjustments.append(-0.1)
                    reasons.append('Momentum contrario')
            
            # Calcular confianza final
            final_confidence = base_confidence + sum(confidence_adjustments)
            final_confidence = max(0.0, min(1.0, final_confidence))  # Limitar entre 0 y 1
            
            # Determinar señal final
            if final_confidence < 0.5:
                final_signal = 0  # Hold si confianza baja
                reason = 'Confianza insuficiente: ' + ', '.join(reasons)
            else:
                final_signal = base_signal
                reason = f'Señal LIT confirmada: {", ".join(reasons)}'
            
            return {
                'signal': final_signal,
                'confidence': final_confidence,
                'reason': reason,
                'base_signal': base_signal,
                'adjustments': confidence_adjustments
            }
            
        except Exception as e:
            log.error(f"Error combinando señales: {str(e)}")
            return {'signal': 0, 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    def _calculate_forex_levels(self, data: pd.DataFrame, signal: Dict, 
                               pair: str) -> Dict[str, float]:
        """Calcula niveles de entrada, stop loss y take profit para Forex."""
        try:
            if signal['signal'] == 0:
                return {}
            
            current_price = data['close'].iloc[-1]
            
            # Obtener información del par
            pair_info = self.forex_provider.all_pairs.get(pair, {})
            pip_value = pair_info.get('pip_value', 0.0001)
            spread = pair_info.get('spread', 2.0)
            
            # Calcular ATR para niveles dinámicos
            if len(data) >= 14:
                atr = data['range_pips'].tail(14).mean()
            else:
                atr = data['range_pips'].mean() if 'range_pips' in data.columns else 20
            
            # Niveles base en pips
            if signal['signal'] == 1:  # BUY
                entry_price = current_price + (spread * pip_value / 2)  # Ask price
                stop_loss_pips = min(max(atr * 0.8, 15), 50)  # Entre 15-50 pips
                take_profit_pips = min(max(atr * 1.5, 25), 100)  # Entre 25-100 pips
                
                stop_loss = entry_price - (stop_loss_pips * pip_value)
                take_profit = entry_price + (take_profit_pips * pip_value)
                
            else:  # SELL
                entry_price = current_price - (spread * pip_value / 2)  # Bid price
                stop_loss_pips = min(max(atr * 0.8, 15), 50)
                take_profit_pips = min(max(atr * 1.5, 25), 100)
                
                stop_loss = entry_price + (stop_loss_pips * pip_value)
                take_profit = entry_price - (take_profit_pips * pip_value)
            
            return {
                'entry_price': round(entry_price, 5),
                'stop_loss': round(stop_loss, 5),
                'take_profit': round(take_profit, 5),
                'stop_loss_pips': round(stop_loss_pips, 1),
                'take_profit_pips': round(take_profit_pips, 1),
                'risk_reward_ratio': round(take_profit_pips / stop_loss_pips, 2),
                'atr_pips': round(atr, 1)
            }
            
        except Exception as e:
            log.error(f"Error calculando niveles Forex: {str(e)}")
            return {}
    
    def analyze_multiple_pairs(self, pairs: List[str], timeframe: str = '1h') -> Dict[str, Dict]:
        """
        Analiza múltiples pares de divisas simultáneamente.
        
        Args:
            pairs: Lista de pares a analizar
            timeframe: Marco temporal
            
        Returns:
            Diccionario con análisis por par
        """
        results = {}
        
        log.info(f"Analizando {len(pairs)} pares de divisas...")
        
        for pair in pairs:
            try:
                analysis = self.analyze_forex_pair(pair, timeframe)
                results[pair] = analysis
                
                signal_text = "BUY" if analysis['signal'] == 1 else "SELL" if analysis['signal'] == -1 else "HOLD"
                log.info(f"📊 {pair}: {signal_text} (Confianza: {analysis['confidence']:.1%})")
                
            except Exception as e:
                log.error(f"Error analizando {pair}: {str(e)}")
                results[pair] = {'signal': 0, 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
        
        return results
    
    def get_best_opportunities(self, pairs: List[str], min_confidence: float = 0.6) -> List[Dict]:
        """
        Encuentra las mejores oportunidades de trading.
        
        Args:
            pairs: Pares a analizar
            min_confidence: Confianza mínima requerida
            
        Returns:
            Lista de oportunidades ordenadas por confianza
        """
        try:
            # Analizar todos los pares
            all_analysis = self.analyze_multiple_pairs(pairs)
            
            # Filtrar oportunidades válidas
            opportunities = []
            for pair, analysis in all_analysis.items():
                if (analysis['signal'] != 0 and 
                    analysis['confidence'] >= min_confidence):
                    opportunities.append({
                        'pair': pair,
                        'signal': analysis['signal'],
                        'confidence': analysis['confidence'],
                        'reason': analysis['reason'],
                        'entry_levels': analysis.get('entry_levels', {}),
                        'market_session': analysis.get('market_session', [])
                    })
            
            # Ordenar por confianza
            opportunities.sort(key=lambda x: x['confidence'], reverse=True)
            
            log.info(f"✅ Encontradas {len(opportunities)} oportunidades de {len(pairs)} pares analizados")
            
            return opportunities
            
        except Exception as e:
            log.error(f"Error buscando oportunidades: {str(e)}")
            return []


# Función de conveniencia
def create_forex_lit_strategy(lookback: int = 50, threshold: float = 0.0001) -> ForexLITStrategy:
    """Crea una instancia de la estrategia Forex LIT."""
    return ForexLITStrategy(lookback, threshold) 