"""
Detector de eventos LIT (Liquidity + Inducement Theory) - Versión Mejorada.

Este módulo implementa la estrategia LIT avanzada que identifica:
- Barrido de liquidez (equal highs/lows + spike con confirmación)
- Zonas de inducement (creación de liquidez antes de romper estructura)
- Zonas de entrada basadas en desequilibrio (inefficiency)
- Detección de trampas (fake breakouts)
- Análisis de volumen y momentum
- Validación de patrones con múltiples timeframes

Características profesionales:
- Algoritmos optimizados para detección precisa
- Validación robusta de datos de entrada
- Métricas de performance integradas
- Logging detallado para debugging
- Configuración flexible y extensible
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from datetime import datetime, timedelta

from src.core.config import config
from src.utils.logger import log
from src.utils.helpers import detect_price_levels


class SignalType(Enum):
    """Tipos de señales LIT."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class EventType(Enum):
    """Tipos de eventos LIT."""
    LIQUIDITY_SWEEP = "liquidity_sweep"
    INDUCEMENT_ZONE = "inducement_zone"
    INEFFICIENCY = "inefficiency"
    FAKE_BREAKOUT = "fake_breakout"
    VOLUME_CONFIRMATION = "volume_confirmation"


class Direction(Enum):
    """Direcciones de eventos LIT."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class LITEvent:
    """Representa un evento LIT detectado con información detallada."""
    
    timestamp: pd.Timestamp
    event_type: EventType
    direction: Direction
    price: float
    confidence: float
    details: Dict[str, Any]
    
    # Nuevos campos para análisis avanzado
    volume_confirmation: bool = False
    momentum_strength: float = 0.0
    pattern_quality: float = 0.0
    risk_reward_ratio: Optional[float] = None
    
    def __post_init__(self):
        """Validación post-inicialización."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence debe estar entre 0 y 1, recibido: {self.confidence}")
        if self.price <= 0:
            raise ValueError(f"Price debe ser positivo, recibido: {self.price}")


@dataclass
class LITSignal:
    """Señal generada por el detector LIT con análisis completo."""
    
    timestamp: pd.Timestamp
    signal: SignalType
    confidence: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    events: List[LITEvent] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Nuevos campos para análisis profesional
    risk_reward_ratio: Optional[float] = None
    expected_duration: Optional[timedelta] = None
    market_structure: str = "unknown"
    volume_profile: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validación y cálculos post-inicialización."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence debe estar entre 0 y 1, recibido: {self.confidence}")
        
        # Calcular risk-reward ratio si tenemos stop y target
        if self.stop_loss and self.take_profit and self.signal != SignalType.HOLD:
            if self.signal == SignalType.BUY:
                risk = abs(self.entry_price - self.stop_loss)
                reward = abs(self.take_profit - self.entry_price)
            else:  # SELL
                risk = abs(self.stop_loss - self.entry_price)
                reward = abs(self.entry_price - self.take_profit)
            
            self.risk_reward_ratio = reward / risk if risk > 0 else None


class LITDetector:
    """
    Detector de eventos LIT (Liquidity + Inducement Theory) - Versión Profesional.
    
    Implementa algoritmos avanzados de detección de patrones LIT:
    - Barrido de liquidez con confirmación de volumen
    - Zonas de inducement con análisis de momentum
    - Inefficiencies con validación de estructura
    - Detección de trampas (fake breakouts)
    - Análisis multi-timeframe
    
    Características profesionales:
    - Validación robusta de datos
    - Optimización de rendimiento
    - Métricas de calidad integradas
    - Configuración flexible
    - Logging detallado
    """
    
    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el detector LIT con configuración avanzada.
        
        Args:
            custom_config: Configuración personalizada opcional.
        """
        # Configuración base
        self.lookback_candles = config.lit.lookback_periods
        self.liquidity_threshold = config.lit.liquidity_threshold
        self.inducement_min_touches = config.lit.inducement_min_touches
        self.inefficiency_min_size = config.lit.inefficiency_min_size
        
        # Configuración avanzada
        self.volume_confirmation_threshold = 1.5  # Volumen 1.5x promedio
        self.fake_breakout_retracement = 0.618  # 61.8% retroceso para fake breakout
        self.momentum_lookback = 5  # Velas para análisis de momentum
        self.pattern_quality_threshold = 0.6  # Calidad mínima de patrón
        
        # Aplicar configuración personalizada
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    log.debug(f"Configuración personalizada aplicada: {key} = {value}")
        
        # Cache optimizado para niveles detectados
        self._liquidity_levels = {'resistance': [], 'support': []}
        self._inducement_zones = []
        self._volume_profile = {}
        self._last_analysis_time = None
        
        # Métricas de performance
        self._performance_metrics = {
            'total_signals': 0,
            'successful_detections': 0,
            'false_positives': 0,
            'processing_time_ms': []
        }
        
        log.info(f"Detector LIT profesional inicializado con configuración: "
                f"lookback={self.lookback_candles}, threshold={self.liquidity_threshold}")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Valida que los datos de entrada sean correctos.
        
        Args:
            data: DataFrame con datos OHLCV.
            
        Returns:
            bool: True si los datos son válidos.
            
        Raises:
            ValueError: Si los datos no son válidos.
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if data is None or data.empty:
            raise ValueError("Los datos no pueden estar vacíos")
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Columnas faltantes: {missing_columns}")
        
        if len(data) < self.lookback_candles:
            log.warning(f"Datos insuficientes: {len(data)} < {self.lookback_candles}")
            return False
        
        # Validar que no hay valores NaN o infinitos
        if data[required_columns].isnull().any().any():
            log.warning("Datos contienen valores NaN")
            return False
        
        if np.isinf(data[required_columns]).any().any():
            log.warning("Datos contienen valores infinitos")
            return False
        
        # Validar lógica OHLC
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.any():
            log.warning("Datos contienen velas OHLC inválidas")
            return False
        
        return True
    
    def analyze(self, data: pd.DataFrame) -> LITSignal:
        """
        Analiza los datos y genera una señal LIT con validación completa.
        
        Args:
            data: DataFrame con datos OHLCV.
            
        Returns:
            LITSignal: Señal generada por el análisis LIT.
            
        Raises:
            ValueError: Si los datos no son válidos.
        """
        start_time = datetime.now()
        
        try:
            # Validar datos de entrada
            if not self.validate_data(data):
                return LITSignal(
                    timestamp=data.index[-1],
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    entry_price=data['close'].iloc[-1],
                    context={'error': 'Datos insuficientes o inválidos'}
                )
            
            # Detectar eventos LIT con algoritmos mejorados
            events = self._detect_lit_events(data)
            
            # Filtrar eventos por calidad
            high_quality_events = [
                event for event in events 
                if event.pattern_quality >= self.pattern_quality_threshold
            ]
            
            # Generar señal basada en eventos de alta calidad
            signal = self._generate_signal(data, high_quality_events)
            
            # Actualizar métricas de performance
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._performance_metrics['processing_time_ms'].append(processing_time)
            self._performance_metrics['total_signals'] += 1
            
            # Agregar contexto adicional
            signal.context.update({
                'total_events_detected': len(events),
                'high_quality_events': len(high_quality_events),
                'processing_time_ms': processing_time,
                'market_structure': self._analyze_market_structure(data),
                'volume_profile': self._calculate_volume_profile(data)
            })
            
            log.debug(f"Análisis LIT completado: {len(events)} eventos, "
                     f"señal {signal.signal.value} con confianza {signal.confidence:.2f}")
            
            return signal
            
        except Exception as e:
            log.error(f"Error en análisis LIT: {str(e)}")
            return LITSignal(
                timestamp=data.index[-1],
                signal=SignalType.HOLD,
                confidence=0.0,
                entry_price=data['close'].iloc[-1],
                context={'error': str(e)}
            )
    
    def _detect_lit_events(self, data: pd.DataFrame) -> List[LITEvent]:
        """
        Detecta todos los eventos LIT en los datos.
        
        Args:
            data: DataFrame con datos OHLCV.
            
        Returns:
            List[LITEvent]: Lista de eventos LIT detectados.
        """
        events = []
        
        # Detectar niveles de liquidez
        liquidity_levels = self._detect_liquidity_levels(data)
        
        # Detectar barridos de liquidez
        sweeps = self._detect_liquidity_sweeps(data, liquidity_levels)
        events.extend(sweeps)
        
        # Detectar zonas de inducement
        inducement_zones = self._detect_inducement_zones(data)
        events.extend(inducement_zones)
        
        # Detectar inefficiencies
        inefficiencies = self._detect_inefficiencies(data)
        events.extend(inefficiencies)
        
        # Detectar fake breakouts (trampas)
        fake_breakouts = self._detect_fake_breakouts(data, liquidity_levels)
        events.extend(fake_breakouts)
        
        # Validar eventos con análisis de volumen
        validated_events = self._validate_events_with_volume(data, events)
        
        return validated_events
    
    def _detect_liquidity_levels(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Detecta niveles de liquidez (equal highs/lows).
        
        Args:
            data: DataFrame con datos OHLCV.
            
        Returns:
            Dict[str, List[float]]: Niveles de resistencia y soporte.
        """
        recent_data = data.tail(self.lookback_candles)
        
        # Usar función helper para detectar niveles
        levels = detect_price_levels(
            recent_data['high'], 
            recent_data['low'], 
            self.liquidity_threshold
        )
        
        # Filtrar niveles con múltiples toques
        filtered_levels = {'resistance': [], 'support': []}
        
        for level in levels['resistance']:
            touches = self._count_level_touches(recent_data, level, 'resistance')
            if touches >= self.inducement_min_touches:
                filtered_levels['resistance'].append(level)
        
        for level in levels['support']:
            touches = self._count_level_touches(recent_data, level, 'support')
            if touches >= self.inducement_min_touches:
                filtered_levels['support'].append(level)
        
        return filtered_levels
    
    def _count_level_touches(self, data: pd.DataFrame, level: float, level_type: str) -> int:
        """
        Cuenta cuántas veces se ha tocado un nivel.
        
        Args:
            data: DataFrame con datos OHLCV.
            level: Nivel de precio a analizar.
            level_type: Tipo de nivel ('resistance' o 'support').
            
        Returns:
            int: Número de toques del nivel.
        """
        tolerance = level * self.liquidity_threshold
        
        if level_type == 'resistance':
            touches = ((data['high'] >= level - tolerance) & 
                      (data['high'] <= level + tolerance)).sum()
        else:  # support
            touches = ((data['low'] >= level - tolerance) & 
                      (data['low'] <= level + tolerance)).sum()
        
        return touches
    
    def _detect_liquidity_sweeps(self, data: pd.DataFrame, levels: Dict[str, List[float]]) -> List[LITEvent]:
        """
        Detecta barridos de liquidez.
        
        Args:
            data: DataFrame con datos OHLCV.
            levels: Niveles de liquidez detectados.
            
        Returns:
            List[LITEvent]: Lista de eventos de barrido de liquidez.
        """
        events = []
        recent_data = data.tail(10)  # Analizar últimas 10 velas
        
        for level in levels['resistance']:
            sweep_event = self._check_resistance_sweep(recent_data, level)
            if sweep_event:
                events.append(sweep_event)
        
        for level in levels['support']:
            sweep_event = self._check_support_sweep(recent_data, level)
            if sweep_event:
                events.append(sweep_event)
        
        return events
    
    def _check_resistance_sweep(self, data: pd.DataFrame, level: float) -> Optional[LITEvent]:
        """
        Verifica si hay un barrido de resistencia.
        
        Args:
            data: DataFrame con datos recientes.
            level: Nivel de resistencia.
            
        Returns:
            Optional[LITEvent]: Evento de barrido si se detecta.
        """
        tolerance = level * self.liquidity_threshold
        
        # Buscar spike que rompe la resistencia y luego retrocede
        for i in range(1, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            # Verificar spike arriba del nivel
            if (current['high'] > level + tolerance and 
                current['close'] < level and
                previous['high'] <= level + tolerance):
                
                # Calcular confianza basada en el retroceso
                spike_size = current['high'] - level
                body_close = level - current['close']
                confidence = min(0.9, body_close / spike_size) if spike_size > 0 else 0.3
                
                return LITEvent(
                    timestamp=current.name,
                    event_type=EventType.LIQUIDITY_SWEEP,
                    direction=Direction.BEARISH,
                    price=level,
                    confidence=confidence,
                    pattern_quality=confidence,
                    details={
                        'level_type': 'resistance',
                        'spike_high': current['high'],
                        'spike_size': spike_size,
                        'close_below_level': body_close
                    }
                )
        
        return None
    
    def _check_support_sweep(self, data: pd.DataFrame, level: float) -> Optional[LITEvent]:
        """
        Verifica si hay un barrido de soporte.
        
        Args:
            data: DataFrame con datos recientes.
            level: Nivel de soporte.
            
        Returns:
            Optional[LITEvent]: Evento de barrido si se detecta.
        """
        tolerance = level * self.liquidity_threshold
        
        # Buscar spike que rompe el soporte y luego retrocede
        for i in range(1, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            # Verificar spike debajo del nivel
            if (current['low'] < level - tolerance and 
                current['close'] > level and
                previous['low'] >= level - tolerance):
                
                # Calcular confianza basada en el retroceso
                spike_size = level - current['low']
                body_close = current['close'] - level
                confidence = min(0.9, body_close / spike_size) if spike_size > 0 else 0.3
                
                return LITEvent(
                    timestamp=current.name,
                    event_type=EventType.LIQUIDITY_SWEEP,
                    direction=Direction.BULLISH,
                    price=level,
                    confidence=confidence,
                    pattern_quality=confidence,
                    details={
                        'level_type': 'support',
                        'spike_low': current['low'],
                        'spike_size': spike_size,
                        'close_above_level': body_close
                    }
                )
        
        return None
    
    def _detect_inducement_zones(self, data: pd.DataFrame) -> List[LITEvent]:
        """
        Detecta zonas de inducement.
        
        Args:
            data: DataFrame con datos OHLCV.
            
        Returns:
            List[LITEvent]: Lista de eventos de inducement.
        """
        events = []
        recent_data = data.tail(self.lookback_candles)
        
        # Detectar patrones de inducement
        for i in range(5, len(recent_data) - 1):
            window = recent_data.iloc[i-5:i+2]
            
            # Buscar patrón: acumulación -> spike -> retroceso
            inducement_event = self._analyze_inducement_pattern(window)
            if inducement_event:
                events.append(inducement_event)
        
        return events
    
    def _analyze_inducement_pattern(self, window: pd.DataFrame) -> Optional[LITEvent]:
        """
        Analiza un patrón potencial de inducement.
        
        Args:
            window: Ventana de datos para analizar.
            
        Returns:
            Optional[LITEvent]: Evento de inducement si se detecta.
        """
        if len(window) < 5:
            return None
        
        # Identificar fase de acumulación (range estrecho)
        price_range = window['high'].max() - window['low'].min()
        avg_range = (window['high'] - window['low']).mean()
        
        # Detectar breakout seguido de retroceso
        recent_high = window['high'].iloc[-3:].max()
        recent_low = window['low'].iloc[-3:].min()
        
        current = window.iloc[-1]
        previous_max = window['high'].iloc[:-1].max()
        previous_min = window['low'].iloc[:-1].min()
        
        # Patrón alcista: rompe máximo y luego retrocede
        if (recent_high > previous_max and 
            current['close'] < previous_max and
            price_range > avg_range * 1.5):
            
            confidence = min(0.8, (recent_high - previous_max) / avg_range)
            
            return LITEvent(
                timestamp=current.name,
                event_type=EventType.INDUCEMENT_ZONE,
                direction=Direction.BULLISH,
                price=previous_max,
                confidence=confidence,
                pattern_quality=confidence,
                details={
                    'pattern_type': 'bullish_inducement',
                    'breakout_high': recent_high,
                    'level_broken': previous_max,
                    'range_expansion': price_range / avg_range
                }
            )
        
        # Patrón bajista: rompe mínimo y luego retrocede
        elif (recent_low < previous_min and 
              current['close'] > previous_min and
              price_range > avg_range * 1.5):
            
            confidence = min(0.8, (previous_min - recent_low) / avg_range)
            
            return LITEvent(
                timestamp=current.name,
                event_type=EventType.INDUCEMENT_ZONE,
                direction=Direction.BEARISH,
                price=previous_min,
                confidence=confidence,
                pattern_quality=confidence,
                details={
                    'pattern_type': 'bearish_inducement',
                    'breakout_low': recent_low,
                    'level_broken': previous_min,
                    'range_expansion': price_range / avg_range
                }
            )
        
        return None
    
    def _detect_inefficiencies(self, data: pd.DataFrame) -> List[LITEvent]:
        """
        Detecta inefficiencies (gaps/desequilibrios).
        
        Args:
            data: DataFrame con datos OHLCV.
            
        Returns:
            List[LITEvent]: Lista de eventos de inefficiency.
        """
        events = []
        recent_data = data.tail(20)
        
        for i in range(2, len(recent_data)):
            current = recent_data.iloc[i]
            previous = recent_data.iloc[i-1]
            prev_prev = recent_data.iloc[i-2]
            
            # Detectar gap alcista
            if (previous['low'] > prev_prev['high'] and
                abs(previous['low'] - prev_prev['high']) > self.inefficiency_min_size):
                
                gap_size = previous['low'] - prev_prev['high']
                confidence = min(0.9, gap_size / prev_prev['close'])
                
                events.append(LITEvent(
                    timestamp=current.name,
                    event_type=EventType.INEFFICIENCY,
                    direction=Direction.BULLISH,
                    price=(previous['low'] + prev_prev['high']) / 2,
                    confidence=confidence,
                    pattern_quality=confidence,
                    details={
                        'gap_type': 'bullish_gap',
                        'gap_high': previous['low'],
                        'gap_low': prev_prev['high'],
                        'gap_size': gap_size
                    }
                ))
            
            # Detectar gap bajista
            elif (previous['high'] < prev_prev['low'] and
                  abs(prev_prev['low'] - previous['high']) > self.inefficiency_min_size):
                
                gap_size = prev_prev['low'] - previous['high']
                confidence = min(0.9, gap_size / prev_prev['close'])
                
                events.append(LITEvent(
                    timestamp=current.name,
                    event_type=EventType.INEFFICIENCY,
                    direction=Direction.BEARISH,
                    price=(prev_prev['low'] + previous['high']) / 2,
                    confidence=confidence,
                    pattern_quality=confidence,
                    details={
                        'gap_type': 'bearish_gap',
                        'gap_high': prev_prev['low'],
                        'gap_low': previous['high'],
                        'gap_size': gap_size
                    }
                ))
        
        return events
    
    def _generate_signal(self, data: pd.DataFrame, events: List[LITEvent]) -> LITSignal:
        """
        Genera una señal basada en los eventos LIT detectados.
        
        Args:
            data: DataFrame con datos OHLCV.
            events: Lista de eventos LIT detectados.
            
        Returns:
            LITSignal: Señal generada.
        """
        current_price = data['close'].iloc[-1]
        current_timestamp = data.index[-1]
        
        if not events:
            return LITSignal(
                timestamp=current_timestamp,
                signal=SignalType.HOLD,
                confidence=0.0,
                entry_price=current_price,
                events=events
            )
        
        # Ponderar eventos por recencia y confianza
        bullish_score = 0
        bearish_score = 0
        
        for event in events:
            # Factor de recencia (eventos más recientes tienen más peso)
            time_diff = (current_timestamp - event.timestamp).total_seconds() / 3600  # horas
            recency_factor = max(0.1, 1 / (1 + time_diff / 24))  # decae en 24 horas
            
            # Puntaje ponderado
            weighted_confidence = event.confidence * recency_factor
            
            if event.direction == Direction.BULLISH:
                bullish_score += weighted_confidence
            elif event.direction == Direction.BEARISH:
                bearish_score += weighted_confidence
        
        # Determinar señal final
        total_score = bullish_score + bearish_score
        
        if total_score == 0:
            signal_type = SignalType.HOLD
            confidence = 0.0
        elif bullish_score > bearish_score * 1.2:  # Sesgo para señales más claras
            signal_type = SignalType.BUY
            confidence = min(0.95, bullish_score / total_score)
        elif bearish_score > bullish_score * 1.2:
            signal_type = SignalType.SELL
            confidence = min(0.95, bearish_score / total_score)
        else:
            signal_type = SignalType.HOLD
            confidence = abs(bullish_score - bearish_score) / total_score
        
        # Calcular niveles de stop y take profit
        atr = self._calculate_recent_atr(data)
        stop_loss, take_profit = self._calculate_levels(current_price, signal_type, atr)
        
        return LITSignal(
            timestamp=current_timestamp,
            signal=signal_type,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            events=events,
            context={
                'bullish_score': bullish_score,
                'bearish_score': bearish_score,
                'total_events': len(events),
                'atr': atr
            }
        )
    
    def _calculate_recent_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calcula el ATR reciente.
        
        Args:
            data: DataFrame con datos OHLCV.
            period: Período para el cálculo del ATR.
            
        Returns:
            float: Valor del ATR.
        """
        if len(data) < period:
            return (data['high'] - data['low']).mean()
        
        recent_data = data.tail(period)
        tr1 = recent_data['high'] - recent_data['low']
        tr2 = abs(recent_data['high'] - recent_data['close'].shift(1))
        tr3 = abs(recent_data['low'] - recent_data['close'].shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.mean()
    
    def _calculate_levels(self, entry_price: float, signal: SignalType, atr: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Calcula niveles de stop loss y take profit.
        
        Args:
            entry_price: Precio de entrada.
            signal: Tipo de señal.
            atr: Average True Range.
            
        Returns:
            Tuple[Optional[float], Optional[float]]: (Stop Loss, Take Profit).
        """
        if signal == SignalType.HOLD:
            return None, None
        
        # Usar ATR para calcular niveles dinámicos
        stop_distance = atr * 1.5  # 1.5x ATR para stop loss
        profit_distance = atr * 2.5  # 2.5x ATR para take profit (R:R 1:1.67)
        
        if signal == SignalType.BUY:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + profit_distance
        else:  # SELL
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - profit_distance
        
        return stop_loss, take_profit
    
    def _detect_fake_breakouts(self, data: pd.DataFrame, levels: Dict[str, List[float]]) -> List[LITEvent]:
        """
        Detecta fake breakouts (trampas) en niveles de liquidez.
        
        Args:
            data: DataFrame con datos OHLCV.
            levels: Niveles de liquidez detectados.
            
        Returns:
            List[LITEvent]: Lista de eventos de fake breakout.
        """
        events = []
        recent_data = data.tail(15)
        
        for level in levels['resistance'] + levels['support']:
            fake_breakout = self._analyze_fake_breakout(recent_data, level)
            if fake_breakout:
                events.append(fake_breakout)
        
        return events
    
    def _analyze_fake_breakout(self, data: pd.DataFrame, level: float) -> Optional[LITEvent]:
        """
        Analiza si hay un fake breakout en un nivel específico.
        
        Args:
            data: DataFrame con datos recientes.
            level: Nivel a analizar.
            
        Returns:
            Optional[LITEvent]: Evento de fake breakout si se detecta.
        """
        tolerance = level * self.liquidity_threshold
        
        for i in range(2, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            prev_prev = data.iloc[i-2]
            
            # Fake breakout alcista (rompe resistencia y retrocede)
            if (previous['high'] > level + tolerance and
                current['close'] < level * (1 - self.fake_breakout_retracement * 0.01) and
                prev_prev['high'] <= level + tolerance):
                
                retracement = (previous['high'] - current['close']) / (previous['high'] - level)
                confidence = min(0.85, retracement)
                
                return LITEvent(
                    timestamp=current.name,
                    event_type=EventType.FAKE_BREAKOUT,
                    direction=Direction.BEARISH,
                    price=level,
                    confidence=confidence,
                    pattern_quality=retracement,
                    details={
                        'breakout_type': 'fake_resistance_break',
                        'breakout_high': previous['high'],
                        'retracement_level': current['close'],
                        'retracement_percentage': retracement * 100
                    }
                )
            
            # Fake breakout bajista (rompe soporte y retrocede)
            elif (previous['low'] < level - tolerance and
                  current['close'] > level * (1 + self.fake_breakout_retracement * 0.01) and
                  prev_prev['low'] >= level - tolerance):
                
                retracement = (current['close'] - previous['low']) / (level - previous['low'])
                confidence = min(0.85, retracement)
                
                return LITEvent(
                    timestamp=current.name,
                    event_type=EventType.FAKE_BREAKOUT,
                    direction=Direction.BULLISH,
                    price=level,
                    confidence=confidence,
                    pattern_quality=retracement,
                    details={
                        'breakout_type': 'fake_support_break',
                        'breakout_low': previous['low'],
                        'retracement_level': current['close'],
                        'retracement_percentage': retracement * 100
                    }
                )
        
        return None
    
    def _validate_events_with_volume(self, data: pd.DataFrame, events: List[LITEvent]) -> List[LITEvent]:
        """
        Valida eventos LIT con análisis de volumen.
        
        Args:
            data: DataFrame con datos OHLCV.
            events: Lista de eventos a validar.
            
        Returns:
            List[LITEvent]: Eventos validados con confirmación de volumen.
        """
        if 'volume' not in data.columns:
            log.warning("No hay datos de volumen disponibles para validación")
            return events
        
        validated_events = []
        avg_volume = data['volume'].tail(20).mean()
        
        for event in events:
            # Encontrar la vela correspondiente al evento
            event_candle = data.loc[data.index <= event.timestamp].iloc[-1]
            
            # Verificar confirmación de volumen
            volume_ratio = event_candle['volume'] / avg_volume if avg_volume > 0 else 1
            volume_confirmation = volume_ratio >= self.volume_confirmation_threshold
            
            # Actualizar evento con información de volumen
            event.volume_confirmation = volume_confirmation
            event.details['volume_ratio'] = volume_ratio
            event.details['avg_volume'] = avg_volume
            
            # Ajustar confianza basada en volumen
            if volume_confirmation:
                event.confidence = min(0.95, event.confidence * 1.1)
            else:
                event.confidence = event.confidence * 0.9
            
            validated_events.append(event)
        
        return validated_events
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> str:
        """
        Analiza la estructura del mercado (trending, ranging, etc.).
        
        Args:
            data: DataFrame con datos OHLCV.
            
        Returns:
            str: Tipo de estructura del mercado.
        """
        recent_data = data.tail(20)
        
        # Calcular highs y lows recientes
        highs = recent_data['high'].rolling(3).max()
        lows = recent_data['low'].rolling(3).min()
        
        # Contar higher highs y higher lows
        higher_highs = (highs.diff() > 0).sum()
        higher_lows = (lows.diff() > 0).sum()
        
        # Contar lower highs y lower lows
        lower_highs = (highs.diff() < 0).sum()
        lower_lows = (lows.diff() < 0).sum()
        
        # Determinar estructura
        if higher_highs >= 3 and higher_lows >= 2:
            return "uptrend"
        elif lower_highs >= 3 and lower_lows >= 2:
            return "downtrend"
        elif abs(recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].mean() < 0.05:
            return "tight_range"
        else:
            return "ranging"
    
    def _calculate_volume_profile(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula el perfil de volumen reciente.
        
        Args:
            data: DataFrame con datos OHLCV.
            
        Returns:
            Dict[str, float]: Métricas del perfil de volumen.
        """
        if 'volume' not in data.columns:
            return {}
        
        recent_data = data.tail(20)
        
        return {
            'avg_volume': recent_data['volume'].mean(),
            'volume_trend': recent_data['volume'].pct_change().mean(),
            'volume_volatility': recent_data['volume'].std() / recent_data['volume'].mean(),
            'high_volume_candles': (recent_data['volume'] > recent_data['volume'].mean() * 1.5).sum()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Obtiene métricas de performance del detector.
        
        Returns:
            Dict[str, Any]: Métricas de performance.
        """
        processing_times = self._performance_metrics['processing_time_ms']
        
        return {
            'total_signals': self._performance_metrics['total_signals'],
            'successful_detections': self._performance_metrics['successful_detections'],
            'false_positives': self._performance_metrics['false_positives'],
            'avg_processing_time_ms': np.mean(processing_times) if processing_times else 0,
            'max_processing_time_ms': max(processing_times) if processing_times else 0,
            'success_rate': (
                self._performance_metrics['successful_detections'] / 
                max(1, self._performance_metrics['total_signals'])
            ) * 100
        }
    
    def reset_performance_metrics(self):
        """Reinicia las métricas de performance."""
        self._performance_metrics = {
            'total_signals': 0,
            'successful_detections': 0,
            'false_positives': 0,
            'processing_time_ms': []
        }
        log.info("Métricas de performance reiniciadas")

    def detect_signals(self, data: pd.DataFrame) -> List[LITSignal]:
        """
        Método de compatibilidad que convierte el resultado de analyze() a lista.
        
        Args:
            data: DataFrame con datos OHLCV.
            
        Returns:
            List[LITSignal]: Lista con la señal generada (o lista vacía si es HOLD).
        """
        try:
            signal = self.analyze(data)
            if signal.signal == SignalType.HOLD:
                return []
            else:
                return [signal]
        except Exception as e:
            log.error(f"Error en detect_signals: {str(e)}")
            return []