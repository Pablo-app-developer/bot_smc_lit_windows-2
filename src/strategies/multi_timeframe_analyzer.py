"""
Analizador Multi-Timeframe para Trading Profesional.

Implementa análisis top-down combinando múltiples marcos temporales
para mejorar la precisión de las señales de trading, siguiendo las
mejores prácticas de traders institucionales.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from src.data.data_loader import DataLoader
from src.strategies.lit_detector import LITDetector, LITSignal, SignalType
from src.data.indicators import calculate_all_indicators
from src.utils.logger import log
from src.core.config import config


class MarketRegime(Enum):
    """Regímenes de mercado identificados."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"


class TimeframeBias(Enum):
    """Sesgo direccional por timeframe."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class TimeframeAnalysis:
    """Análisis de un timeframe específico."""
    timeframe: str
    regime: MarketRegime
    bias: TimeframeBias
    strength: float  # 0.0 - 1.0
    key_levels: Dict[str, float]
    lit_signals: List[LITSignal]
    trend_direction: str
    volatility_level: str
    volume_profile: str
    confidence: float


@dataclass
class MultiTimeframeSignal:
    """Señal combinada de múltiples timeframes."""
    primary_signal: SignalType
    confidence: float
    timeframe_alignment: float  # Porcentaje de timeframes alineados
    regime_consistency: float
    risk_level: str
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe_analysis: Dict[str, TimeframeAnalysis]
    timestamp: datetime


class MultiTimeframeAnalyzer:
    """
    Analizador Multi-Timeframe Profesional.
    
    Implementa análisis top-down siguiendo las mejores prácticas:
    1. Timeframe superior para contexto y sesgo
    2. Timeframe medio para confirmación
    3. Timeframe inferior para entrada precisa
    """
    
    def __init__(self):
        """Inicializa el analizador multi-timeframe."""
        self.data_loader = DataLoader()
        
        # Configuración de timeframes (top-down approach)
        self.timeframes = {
            'higher': '1d',      # Contexto y sesgo principal
            'medium': '4h',      # Confirmación de tendencia
            'lower': '1h'        # Entrada y timing preciso
        }
        
        # Detectores LIT por timeframe
        self.lit_detectors = {
            tf: LITDetector() for tf in self.timeframes.values()
        }
        
        # Cache para optimización
        self._analysis_cache = {}
        self._last_update = {}
        
        log.info("MultiTimeframeAnalyzer inicializado")
        log.info(f"Timeframes configurados: {self.timeframes}")
    
    def analyze_market(self, symbol: str) -> MultiTimeframeSignal:
        """
        Realiza análisis multi-timeframe completo.
        
        Args:
            symbol: Símbolo a analizar.
            
        Returns:
            MultiTimeframeSignal: Señal combinada de todos los timeframes.
        """
        try:
            log.info(f"Iniciando análisis multi-timeframe para {symbol}")
            
            # 1. Análisis individual por timeframe (top-down)
            timeframe_analyses = {}
            
            for role, timeframe in self.timeframes.items():
                analysis = self._analyze_single_timeframe(symbol, timeframe, role)
                timeframe_analyses[timeframe] = analysis
                
                log.info(f"Timeframe {timeframe} ({role}): "
                        f"Régimen={analysis.regime.value}, "
                        f"Sesgo={analysis.bias.value}, "
                        f"Fuerza={analysis.strength:.2f}")
            
            # 2. Combinar análisis de timeframes
            combined_signal = self._combine_timeframe_signals(timeframe_analyses)
            
            # 3. Calcular niveles de riesgo
            combined_signal = self._calculate_risk_levels(combined_signal, timeframe_analyses)
            
            log.info(f"Señal multi-timeframe generada: "
                    f"Señal={combined_signal.primary_signal.value}, "
                    f"Confianza={combined_signal.confidence:.2f}, "
                    f"Alineación={combined_signal.timeframe_alignment:.2f}")
            
            return combined_signal
            
        except Exception as e:
            log.error(f"Error en análisis multi-timeframe: {str(e)}")
            return self._get_default_signal(symbol)
    
    def _analyze_single_timeframe(self, symbol: str, timeframe: str, role: str) -> TimeframeAnalysis:
        """Analiza un timeframe específico."""
        try:
            # Obtener datos del timeframe
            periods = self._get_periods_for_timeframe(timeframe)
            data = self.data_loader.load_data(
                symbol=symbol,
                timeframe=timeframe,
                periods=periods
            )
            
            if len(data) < 50:
                log.warning(f"Datos insuficientes para {timeframe}: {len(data)} velas")
                return self._get_default_timeframe_analysis(timeframe)
            
            # Calcular indicadores
            data_with_indicators = calculate_all_indicators(data)
            
            # Análisis LIT
            lit_detector = self.lit_detectors[timeframe]
            lit_signals = lit_detector.detect_signals(data_with_indicators)
            
            # Determinar régimen de mercado
            regime = self._identify_market_regime(data_with_indicators)
            
            # Determinar sesgo direccional
            bias = self._determine_directional_bias(data_with_indicators, role)
            
            # Calcular fuerza de la señal
            strength = self._calculate_signal_strength(data_with_indicators, lit_signals)
            
            # Identificar niveles clave
            key_levels = self._identify_key_levels(data_with_indicators)
            
            # Análisis adicionales
            trend_direction = self._analyze_trend_direction(data_with_indicators)
            volatility_level = self._analyze_volatility_level(data_with_indicators)
            volume_profile = self._analyze_volume_profile(data_with_indicators)
            
            # Calcular confianza
            confidence = self._calculate_timeframe_confidence(
                regime, bias, strength, lit_signals
            )
            
            return TimeframeAnalysis(
                timeframe=timeframe,
                regime=regime,
                bias=bias,
                strength=strength,
                key_levels=key_levels,
                lit_signals=lit_signals,
                trend_direction=trend_direction,
                volatility_level=volatility_level,
                volume_profile=volume_profile,
                confidence=confidence
            )
            
        except Exception as e:
            log.error(f"Error analizando timeframe {timeframe}: {str(e)}")
            return self._get_default_timeframe_analysis(timeframe)
    
    def _identify_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Identifica el régimen de mercado actual."""
        try:
            # Calcular métricas para identificar régimen
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std()
            
            # Tendencia (ADX si está disponible)
            trend_strength = 0.5
            if 'adx' in data.columns:
                trend_strength = data['adx'].iloc[-1] / 100.0
            
            # Volatilidad actual vs histórica
            current_vol = volatility.iloc[-1]
            avg_vol = volatility.mean()
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            # Rango vs tendencia
            high_20 = data['high'].rolling(20).max()
            low_20 = data['low'].rolling(20).min()
            range_size = (high_20.iloc[-1] - low_20.iloc[-1]) / data['close'].iloc[-1]
            
            # Determinar régimen
            if trend_strength > 0.7 and vol_ratio < 1.5:
                # Tendencia fuerte
                if returns.iloc[-5:].mean() > 0:
                    return MarketRegime.TRENDING_UP
                else:
                    return MarketRegime.TRENDING_DOWN
            elif vol_ratio > 2.0:
                return MarketRegime.VOLATILE
            elif vol_ratio < 0.5:
                return MarketRegime.QUIET
            else:
                return MarketRegime.RANGING
                
        except Exception as e:
            log.error(f"Error identificando régimen: {str(e)}")
            return MarketRegime.RANGING
    
    def _determine_directional_bias(self, data: pd.DataFrame, role: str) -> TimeframeBias:
        """Determina el sesgo direccional del timeframe."""
        try:
            # Múltiples indicadores para sesgo
            signals = []
            
            # 1. Medias móviles
            if 'sma_20' in data.columns and 'sma_50' in data.columns:
                if data['sma_20'].iloc[-1] > data['sma_50'].iloc[-1]:
                    signals.append(1)  # Bullish
                else:
                    signals.append(-1)  # Bearish
            
            # 2. Precio vs medias
            if 'sma_20' in data.columns:
                if data['close'].iloc[-1] > data['sma_20'].iloc[-1]:
                    signals.append(1)
                else:
                    signals.append(-1)
            
            # 3. MACD
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                if data['macd'].iloc[-1] > data['macd_signal'].iloc[-1]:
                    signals.append(1)
                else:
                    signals.append(-1)
            
            # 4. RSI (para timeframes superiores)
            if role == 'higher' and 'rsi' in data.columns:
                rsi = data['rsi'].iloc[-1]
                if rsi > 55:
                    signals.append(1)
                elif rsi < 45:
                    signals.append(-1)
                else:
                    signals.append(0)
            
            # Combinar señales
            if not signals:
                return TimeframeBias.NEUTRAL
            
            avg_signal = np.mean(signals)
            
            if avg_signal > 0.3:
                return TimeframeBias.BULLISH
            elif avg_signal < -0.3:
                return TimeframeBias.BEARISH
            else:
                return TimeframeBias.NEUTRAL
                
        except Exception as e:
            log.error(f"Error determinando sesgo: {str(e)}")
            return TimeframeBias.NEUTRAL
    
    def _calculate_signal_strength(self, data: pd.DataFrame, lit_signals: List[LITSignal]) -> float:
        """Calcula la fuerza de la señal."""
        try:
            strength_factors = []
            
            # 1. Fuerza de señales LIT
            if lit_signals:
                lit_strength = np.mean([signal.confidence for signal in lit_signals])
                strength_factors.append(lit_strength)
            
            # 2. Momentum
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[-1]
                momentum_strength = abs(rsi - 50) / 50  # 0-1
                strength_factors.append(momentum_strength)
            
            # 3. Volumen
            if 'volume_ratio' in data.columns:
                vol_strength = min(data['volume_ratio'].iloc[-1], 2.0) / 2.0
                strength_factors.append(vol_strength)
            
            # 4. Volatilidad
            if 'atr' in data.columns:
                atr_norm = data['atr'].iloc[-1] / data['close'].iloc[-1]
                vol_strength = min(atr_norm * 100, 1.0)  # Normalizar
                strength_factors.append(vol_strength)
            
            return np.mean(strength_factors) if strength_factors else 0.5
            
        except Exception as e:
            log.error(f"Error calculando fuerza: {str(e)}")
            return 0.5
    
    def _identify_key_levels(self, data: pd.DataFrame) -> Dict[str, float]:
        """Identifica niveles clave de soporte y resistencia."""
        try:
            levels = {}
            
            # Niveles de precio
            levels['current'] = data['close'].iloc[-1]
            levels['high_20'] = data['high'].rolling(20).max().iloc[-1]
            levels['low_20'] = data['low'].rolling(20).min().iloc[-1]
            
            # Medias móviles como niveles
            if 'sma_20' in data.columns:
                levels['sma_20'] = data['sma_20'].iloc[-1]
            if 'sma_50' in data.columns:
                levels['sma_50'] = data['sma_50'].iloc[-1]
            
            # Bollinger Bands
            if 'bb_upper' in data.columns:
                levels['bb_upper'] = data['bb_upper'].iloc[-1]
                levels['bb_lower'] = data['bb_lower'].iloc[-1]
            
            return levels
            
        except Exception as e:
            log.error(f"Error identificando niveles: {str(e)}")
            return {'current': data['close'].iloc[-1] if len(data) > 0 else 0}
    
    def _analyze_trend_direction(self, data: pd.DataFrame) -> str:
        """Analiza la dirección de la tendencia."""
        try:
            if 'sma_20' in data.columns and 'sma_50' in data.columns:
                sma_20 = data['sma_20'].iloc[-1]
                sma_50 = data['sma_50'].iloc[-1]
                
                if sma_20 > sma_50:
                    return "uptrend"
                elif sma_20 < sma_50:
                    return "downtrend"
                else:
                    return "sideways"
            
            # Fallback: comparar precios
            if len(data) >= 10:
                recent_avg = data['close'].iloc[-5:].mean()
                older_avg = data['close'].iloc[-10:-5].mean()
                
                if recent_avg > older_avg * 1.01:
                    return "uptrend"
                elif recent_avg < older_avg * 0.99:
                    return "downtrend"
                else:
                    return "sideways"
            
            return "sideways"
            
        except Exception as e:
            log.error(f"Error analizando tendencia: {str(e)}")
            return "sideways"
    
    def _analyze_volatility_level(self, data: pd.DataFrame) -> str:
        """Analiza el nivel de volatilidad."""
        try:
            if 'atr' in data.columns:
                atr_current = data['atr'].iloc[-1]
                atr_avg = data['atr'].rolling(20).mean().iloc[-1]
                
                ratio = atr_current / atr_avg if atr_avg > 0 else 1.0
                
                if ratio > 1.5:
                    return "high"
                elif ratio < 0.7:
                    return "low"
                else:
                    return "normal"
            
            return "normal"
            
        except Exception as e:
            log.error(f"Error analizando volatilidad: {str(e)}")
            return "normal"
    
    def _analyze_volume_profile(self, data: pd.DataFrame) -> str:
        """Analiza el perfil de volumen."""
        try:
            if 'volume_ratio' in data.columns:
                vol_ratio = data['volume_ratio'].iloc[-1]
                
                if vol_ratio > 1.5:
                    return "high"
                elif vol_ratio < 0.7:
                    return "low"
                else:
                    return "normal"
            
            return "normal"
            
        except Exception as e:
            log.error(f"Error analizando volumen: {str(e)}")
            return "normal"
    
    def _calculate_timeframe_confidence(self, regime: MarketRegime, bias: TimeframeBias, 
                                      strength: float, lit_signals: List[LITSignal]) -> float:
        """Calcula la confianza del análisis del timeframe."""
        try:
            confidence_factors = []
            
            # Factor de régimen
            regime_confidence = {
                MarketRegime.TRENDING_UP: 0.8,
                MarketRegime.TRENDING_DOWN: 0.8,
                MarketRegime.RANGING: 0.6,
                MarketRegime.VOLATILE: 0.4,
                MarketRegime.QUIET: 0.5
            }
            confidence_factors.append(regime_confidence.get(regime, 0.5))
            
            # Factor de sesgo
            bias_confidence = {
                TimeframeBias.BULLISH: 0.7,
                TimeframeBias.BEARISH: 0.7,
                TimeframeBias.NEUTRAL: 0.4
            }
            confidence_factors.append(bias_confidence.get(bias, 0.4))
            
            # Factor de fuerza
            confidence_factors.append(strength)
            
            # Factor de señales LIT
            if lit_signals:
                lit_confidence = np.mean([signal.confidence for signal in lit_signals])
                confidence_factors.append(lit_confidence)
            else:
                confidence_factors.append(0.3)
            
            return np.mean(confidence_factors)
            
        except Exception as e:
            log.error(f"Error calculando confianza: {str(e)}")
            return 0.5
    
    def _combine_timeframe_signals(self, analyses: Dict[str, TimeframeAnalysis]) -> MultiTimeframeSignal:
        """Combina las señales de múltiples timeframes."""
        try:
            # Pesos por timeframe (higher tiene más peso para sesgo)
            weights = {
                self.timeframes['higher']: 0.5,  # Mayor peso para contexto
                self.timeframes['medium']: 0.3,   # Peso medio para confirmación
                self.timeframes['lower']: 0.2     # Menor peso para timing
            }
            
            # Calcular señal combinada
            bullish_score = 0.0
            bearish_score = 0.0
            total_weight = 0.0
            
            for timeframe, analysis in analyses.items():
                weight = weights.get(timeframe, 0.2)
                confidence_weight = weight * analysis.confidence
                
                if analysis.bias == TimeframeBias.BULLISH:
                    bullish_score += confidence_weight * analysis.strength
                elif analysis.bias == TimeframeBias.BEARISH:
                    bearish_score += confidence_weight * analysis.strength
                
                total_weight += confidence_weight
            
            # Determinar señal principal
            if total_weight == 0:
                primary_signal = SignalType.HOLD
                confidence = 0.0
            elif bullish_score > bearish_score * 1.2:  # Requiere ventaja clara
                primary_signal = SignalType.BUY
                confidence = bullish_score / total_weight
            elif bearish_score > bullish_score * 1.2:
                primary_signal = SignalType.SELL
                confidence = bearish_score / total_weight
            else:
                primary_signal = SignalType.HOLD
                confidence = max(bullish_score, bearish_score) / total_weight if total_weight > 0 else 0.0
            
            # Calcular alineación de timeframes
            aligned_count = 0
            total_count = len(analyses)
            
            for analysis in analyses.values():
                if primary_signal == SignalType.BUY and analysis.bias == TimeframeBias.BULLISH:
                    aligned_count += 1
                elif primary_signal == SignalType.SELL and analysis.bias == TimeframeBias.BEARISH:
                    aligned_count += 1
                elif primary_signal == SignalType.HOLD and analysis.bias == TimeframeBias.NEUTRAL:
                    aligned_count += 1
            
            timeframe_alignment = aligned_count / total_count if total_count > 0 else 0.0
            
            # Calcular consistencia de régimen
            regimes = [analysis.regime for analysis in analyses.values()]
            regime_consistency = len(set(regimes)) / len(regimes) if regimes else 0.0
            regime_consistency = 1.0 - regime_consistency  # Invertir para que más consistencia = mayor valor
            
            return MultiTimeframeSignal(
                primary_signal=primary_signal,
                confidence=confidence,
                timeframe_alignment=timeframe_alignment,
                regime_consistency=regime_consistency,
                risk_level="medium",  # Se calculará después
                entry_price=0.0,      # Se calculará después
                stop_loss=0.0,        # Se calculará después
                take_profit=0.0,      # Se calculará después
                timeframe_analysis=analyses,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            log.error(f"Error combinando señales: {str(e)}")
            return self._get_default_signal("UNKNOWN")
    
    def _calculate_risk_levels(self, signal: MultiTimeframeSignal, 
                             analyses: Dict[str, TimeframeAnalysis]) -> MultiTimeframeSignal:
        """Calcula niveles de riesgo y precios objetivo."""
        try:
            # Obtener precio actual del timeframe inferior
            lower_analysis = analyses.get(self.timeframes['lower'])
            if not lower_analysis:
                return signal
            
            current_price = lower_analysis.key_levels.get('current', 0)
            if current_price == 0:
                return signal
            
            signal.entry_price = current_price
            
            # Calcular ATR para stop loss dinámico
            atr_multiplier = 2.0  # Conservador para cuenta pequeña
            
            # Usar niveles clave para stop loss y take profit
            if signal.primary_signal == SignalType.BUY:
                # Stop loss debajo del soporte
                support_level = lower_analysis.key_levels.get('low_20', current_price * 0.98)
                signal.stop_loss = min(support_level, current_price * 0.98)  # Máximo 2% de pérdida
                
                # Take profit en resistencia
                resistance_level = lower_analysis.key_levels.get('high_20', current_price * 1.03)
                signal.take_profit = max(resistance_level, current_price * 1.03)  # Mínimo 3% de ganancia
                
            elif signal.primary_signal == SignalType.SELL:
                # Stop loss arriba de la resistencia
                resistance_level = lower_analysis.key_levels.get('high_20', current_price * 1.02)
                signal.stop_loss = max(resistance_level, current_price * 1.02)  # Máximo 2% de pérdida
                
                # Take profit en soporte
                support_level = lower_analysis.key_levels.get('low_20', current_price * 0.97)
                signal.take_profit = min(support_level, current_price * 0.97)  # Mínimo 3% de ganancia
            
            # Determinar nivel de riesgo
            risk_factors = []
            
            # Factor de volatilidad
            vol_count = sum(1 for analysis in analyses.values() 
                          if analysis.volatility_level == "high")
            risk_factors.append(vol_count / len(analyses))
            
            # Factor de alineación
            risk_factors.append(1.0 - signal.timeframe_alignment)
            
            # Factor de confianza
            risk_factors.append(1.0 - signal.confidence)
            
            avg_risk = np.mean(risk_factors)
            
            if avg_risk < 0.3:
                signal.risk_level = "low"
            elif avg_risk < 0.6:
                signal.risk_level = "medium"
            else:
                signal.risk_level = "high"
            
            return signal
            
        except Exception as e:
            log.error(f"Error calculando niveles de riesgo: {str(e)}")
            return signal
    
    def _get_lookback_for_timeframe(self, timeframe: str) -> int:
        """Obtiene el lookback apropiado para cada timeframe."""
        lookback_map = {
            '1m': 100, '5m': 100, '15m': 100, '30m': 100,
            '1h': 80, '2h': 80, '4h': 60,
            '1d': 50, '1w': 30, '1M': 20
        }
        return lookback_map.get(timeframe, 50)
    
    def _get_periods_for_timeframe(self, timeframe: str) -> int:
        """Obtiene el número de períodos para cada timeframe."""
        periods_map = {
            '1m': 200, '5m': 200, '15m': 200, '30m': 200,
            '1h': 150, '2h': 150, '4h': 100,
            '1d': 100, '1w': 50, '1M': 30
        }
        return periods_map.get(timeframe, 100)
    
    def _get_default_timeframe_analysis(self, timeframe: str) -> TimeframeAnalysis:
        """Retorna análisis por defecto para un timeframe."""
        return TimeframeAnalysis(
            timeframe=timeframe,
            regime=MarketRegime.RANGING,
            bias=TimeframeBias.NEUTRAL,
            strength=0.0,
            key_levels={'current': 0.0},
            lit_signals=[],
            trend_direction="sideways",
            volatility_level="normal",
            volume_profile="normal",
            confidence=0.0
        )
    
    def _get_default_signal(self, symbol: str) -> MultiTimeframeSignal:
        """Retorna señal por defecto."""
        return MultiTimeframeSignal(
            primary_signal=SignalType.HOLD,
            confidence=0.0,
            timeframe_alignment=0.0,
            regime_consistency=0.0,
            risk_level="high",
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            timeframe_analysis={},
            timestamp=datetime.now()
        )
    
    def get_timeframe_summary(self, signal: MultiTimeframeSignal) -> Dict[str, Any]:
        """Genera resumen del análisis multi-timeframe."""
        try:
            summary = {
                'signal': signal.primary_signal.value,
                'confidence': signal.confidence,
                'timeframe_alignment': signal.timeframe_alignment,
                'regime_consistency': signal.regime_consistency,
                'risk_level': signal.risk_level,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'timestamp': signal.timestamp.isoformat(),
                'timeframes': {}
            }
            
            for timeframe, analysis in signal.timeframe_analysis.items():
                summary['timeframes'][timeframe] = {
                    'regime': analysis.regime.value,
                    'bias': analysis.bias.value,
                    'strength': analysis.strength,
                    'confidence': analysis.confidence,
                    'trend': analysis.trend_direction,
                    'volatility': analysis.volatility_level,
                    'volume': analysis.volume_profile,
                    'lit_signals_count': len(analysis.lit_signals)
                }
            
            return summary
            
        except Exception as e:
            log.error(f"Error generando resumen: {str(e)}")
            return {'error': str(e)} 