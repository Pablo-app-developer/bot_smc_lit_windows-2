"""
Sistema de logging profesional para el bot de trading LIT.

Este módulo configura y maneja el sistema de logging centralizado
del bot, incluyendo rotación de archivos y formateo estructurado.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger

from src.core.config import config


class TradingLogger:
    """
    Sistema de logging profesional para el bot de trading.
    
    Maneja logging estructurado con rotación automática de archivos
    y diferentes niveles de logging según el entorno.
    """
    
    def __init__(self):
        """Inicializa el sistema de logging."""
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Configura el logger con rotación y formateo."""
        # Crear directorio de logs si no existe
        log_dir = Path(config.logging.file).parent
        log_dir.mkdir(exist_ok=True, parents=True)
        
        # Remover configuración por defecto
        logger.remove()
        
        # Configurar formato para consola
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        # Configurar formato para archivo
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        )
        
        # Logger para consola
        logger.add(
            sys.stdout,
            format=console_format,
            level=config.logging.level,
            colorize=True
        )
        
        # Logger para archivo con rotación
        logger.add(
            config.logging.file,
            format=file_format,
            level=config.logging.level,
            rotation=config.logging.rotation,
            retention="1 month",
            compression="zip",
            encoding="utf-8"
        )
        
        # Configurar contexto del bot
        logger.configure(extra={
            "bot_name": config.bot_name,
            "version": config.version
        })
    
    def get_logger(self) -> "logger":
        """
        Retorna el logger configurado.
        
        Returns:
            logger: Instancia del logger de loguru.
        """
        return logger
    
    def log_trade(self, 
                  action: str, 
                  symbol: str, 
                  price: float, 
                  quantity: float, 
                  signal_type: str) -> None:
        """
        Registra información específica de trades.
        
        Args:
            action: Acción realizada (BUY/SELL/CLOSE).
            symbol: Símbolo del instrumento.
            price: Precio de ejecución.
            quantity: Cantidad operada.
            signal_type: Tipo de señal (LIT/ML/MANUAL).
        """
        logger.info(
            f"TRADE | {action} | {symbol} | "
            f"Price: {price:.5f} | Qty: {quantity:.4f} | "
            f"Signal: {signal_type}"
        )
    
    def log_signal(self, 
                   signal: str, 
                   symbol: str, 
                   confidence: float, 
                   strategy: str) -> None:
        """
        Registra señales de trading generadas.
        
        Args:
            signal: Señal generada (BUY/SELL/HOLD).
            symbol: Símbolo del instrumento.
            confidence: Nivel de confianza de la señal.
            strategy: Estrategia que generó la señal.
        """
        logger.info(
            f"SIGNAL | {signal} | {symbol} | "
            f"Confidence: {confidence:.3f} | Strategy: {strategy}"
        )
    
    def log_error(self, 
                  error: Exception, 
                  context: Optional[str] = None) -> None:
        """
        Registra errores con contexto adicional.
        
        Args:
            error: Excepción capturada.
            context: Contexto adicional del error.
        """
        error_msg = f"ERROR | {type(error).__name__}: {str(error)}"
        if context:
            error_msg += f" | Context: {context}"
        
        logger.error(error_msg)
    
    def log_performance(self, 
                       symbol: str, 
                       pnl: float, 
                       return_pct: float, 
                       trades_count: int) -> None:
        """
        Registra métricas de rendimiento.
        
        Args:
            symbol: Símbolo del instrumento.
            pnl: PnL acumulado.
            return_pct: Retorno porcentual.
            trades_count: Número de trades ejecutados.
        """
        logger.info(
            f"PERFORMANCE | {symbol} | "
            f"PnL: {pnl:.2f} | Return: {return_pct:.2f}% | "
            f"Trades: {trades_count}"
        )


# Instancia global del logger
trading_logger = TradingLogger()
log = trading_logger.get_logger() 