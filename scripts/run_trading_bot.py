#!/usr/bin/env python3
"""
Script para ejecutar el bot de trading LIT + ML.

Este script permite ejecutar el bot de trading con diferentes configuraciones
y modos de operación.
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from src.trading.trading_bot import TradingBot, create_trading_bot
from src.utils.logger import log


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Bot de Trading LIT + ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Modo análisis (sin trading real)
  python scripts/run_trading_bot.py --mode analysis --duration 2

  # Trading real con riesgo moderado
  python scripts/run_trading_bot.py --mode trading --risk moderate --duration 24

  # Trading con símbolos específicos
  python scripts/run_trading_bot.py --mode trading --symbols EURUSD GBPUSD --duration 8

  # Modo demo con configuración conservadora
  python scripts/run_trading_bot.py --mode demo --risk conservative --confidence 0.75
        """
    )
    
    # Argumentos principales
    parser.add_argument(
        '--mode',
        choices=['analysis', 'demo', 'trading'],
        default='analysis',
        help='Modo de operación (default: analysis)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/lit_ml_model.pkl',
        help='Ruta al modelo entrenado (default: models/lit_ml_model.pkl)'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
        help='Símbolos a operar (default: EURUSD GBPUSD USDJPY AUDUSD)'
    )
    
    parser.add_argument(
        '--timeframe',
        choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
        default='1h',
        help='Marco temporal (default: 1h)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=24.0,
        help='Duración en horas (default: 24.0)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Intervalo entre predicciones en segundos (default: 300)'
    )
    
    # Configuración de riesgo
    parser.add_argument(
        '--risk',
        choices=['conservative', 'moderate', 'aggressive'],
        default='moderate',
        help='Nivel de riesgo (default: moderate)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.65,
        help='Confianza mínima para ejecutar operaciones (default: 0.65)'
    )
    
    parser.add_argument(
        '--max-spread',
        type=float,
        default=3.0,
        help='Spread máximo permitido (default: 3.0)'
    )
    
    # Opciones adicionales
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Mostrar información detallada'
    )
    
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='No mostrar estadísticas periódicas'
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Mostrar configuración
    print("🤖 BOT DE TRADING LIT + ML")
    print("=" * 50)
    print(f"Modo: {args.mode.upper()}")
    print(f"Modelo: {args.model}")
    print(f"Símbolos: {', '.join(args.symbols)}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Duración: {args.duration} horas")
    print(f"Intervalo: {args.interval} segundos")
    print(f"Riesgo: {args.risk}")
    print(f"Confianza mínima: {args.confidence}")
    print(f"Spread máximo: {args.max_spread}")
    print("=" * 50)
    
    # Determinar si habilitar trading
    trading_enabled = args.mode in ['demo', 'trading']
    
    if trading_enabled:
        if args.mode == 'trading':
            print("⚠️  ADVERTENCIA: TRADING REAL HABILITADO")
            print("   Las operaciones se ejecutarán con dinero real.")
        else:
            print("📊 MODO DEMO: Trading en cuenta demo")
        
        response = input("\n¿Continuar? (s/N): ").lower().strip()
        if response not in ['s', 'si', 'sí', 'y', 'yes']:
            print("❌ Operación cancelada")
            return
    else:
        print("📈 MODO ANÁLISIS: Solo predicciones, sin trading")
    
    print("\n🚀 Iniciando bot...")
    
    try:
        # Crear bot
        bot = TradingBot(
            model_path=args.model,
            symbols=args.symbols,
            timeframe=args.timeframe,
            prediction_interval=args.interval,
            risk_level=args.risk,
            min_confidence=args.confidence,
            max_spread=args.max_spread,
            trading_enabled=trading_enabled
        )
        
        # Agregar callbacks si no se deshabilitaron las estadísticas
        if not args.no_stats:
            def stats_callback(symbol, prediction):
                if prediction['signal'] != 'hold':
                    print(f"📊 {symbol}: {prediction['signal'].upper()} "
                          f"(conf: {prediction.get('confidence', 0):.3f})")
            
            def trade_callback(signal, order):
                print(f"💰 OPERACIÓN: {signal.symbol} {signal.signal.upper()} "
                      f"- Ticket: {order.ticket}")
            
            bot.add_signal_callback(stats_callback)
            bot.add_trade_callback(trade_callback)
        
        # Iniciar bot
        if not bot.start():
            print("❌ No se pudo iniciar el bot")
            return
        
        print(f"✅ Bot iniciado exitosamente")
        print(f"⏰ Ejecutando por {args.duration} horas...")
        print("   Presiona Ctrl+C para detener")
        
        # Ejecutar por el tiempo especificado
        start_time = time.time()
        end_time = start_time + (args.duration * 3600)
        
        try:
            while time.time() < end_time and bot.is_running():
                time.sleep(60)  # Verificar cada minuto
                
                # Mostrar progreso cada 30 minutos
                if not args.no_stats:
                    elapsed = (time.time() - start_time) / 3600
                    if int(elapsed * 2) % 1 == 0:  # Cada 30 minutos
                        remaining = args.duration - elapsed
                        print(f"⏰ Tiempo restante: {remaining:.1f} horas")
        
        except KeyboardInterrupt:
            print("\n⏹️ Deteniendo bot...")
        
        # Detener bot
        bot.stop()
        
        print("✅ Bot detenido exitosamente")
        
    except Exception as e:
        print(f"❌ Error ejecutando bot: {str(e)}")
        log.error(f"Error en run_trading_bot: {str(e)}")


if __name__ == "__main__":
    main() 