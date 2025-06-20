#!/usr/bin/env python3
"""
Script Principal - Predicciones LIT + ML en Tiempo Real.

Este script ejecuta el sistema completo de predicciones LIT + ML,
incluyendo integración con MetaTrader 5 y backtesting.
"""

import os
import sys
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.predictor import LITMLPredictor, batch_predict_for_backtesting
from src.integrations.mt5_predictor import MT5PredictorIntegration, create_mt5_predictor
from src.data.data_loader import DataLoader
from src.utils.logger import log


def run_single_prediction(model_path: str, symbol: str, timeframe: str = "1h"):
    """
    Ejecuta una predicción única para un símbolo.
    
    Args:
        model_path: Ruta al modelo entrenado.
        symbol: Símbolo a analizar.
        timeframe: Marco temporal.
    """
    log.info(f"🎯 Predicción única: {symbol} {timeframe}")
    
    try:
        # Crear predictor
        predictor = LITMLPredictor(model_path)
        
        # Cargar modelo
        if not predictor.load_model():
            log.error("No se pudo cargar el modelo")
            return
        
        # Realizar predicción en tiempo real
        prediction = predictor.predict_realtime(symbol, timeframe)
        
        # Mostrar resultado
        print("\n" + "="*60)
        print(f"📊 PREDICCIÓN PARA {symbol}")
        print("="*60)
        print(f"Señal: {prediction['signal'].upper()}")
        print(f"Confianza: {prediction.get('confidence', 0):.3f}")
        print(f"Precio actual: {prediction.get('last_price', 0):.5f}")
        print(f"Timestamp: {prediction.get('timestamp', 'N/A')}")
        
        if 'probabilities' in prediction:
            probs = prediction['probabilities']
            print(f"\nProbabilidades:")
            print(f"  Compra: {probs.get('buy', 0):.3f}")
            print(f"  Venta: {probs.get('sell', 0):.3f}")
            print(f"  Mantener: {probs.get('hold', 0):.3f}")
        
        print("="*60)
        
    except Exception as e:
        log.error(f"Error en predicción única: {str(e)}")


def run_backtesting(model_path: str, symbol: str, days: int = 30, timeframe: str = "1h"):
    """
    Ejecuta backtesting del modelo.
    
    Args:
        model_path: Ruta al modelo entrenado.
        symbol: Símbolo a analizar.
        days: Días de datos históricos.
        timeframe: Marco temporal.
    """
    log.info(f"📈 Backtesting: {symbol} - {days} días")
    
    try:
        # Cargar datos históricos
        data_loader = DataLoader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Obtener datos
        data = data_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if data.empty:
            log.error(f"No se pudieron cargar datos para {symbol}")
            return
        
        log.info(f"Datos cargados: {len(data)} velas")
        
        # Ejecutar backtesting
        predictions_df = batch_predict_for_backtesting(model_path, data)
        
        if predictions_df.empty:
            log.error("No se pudieron generar predicciones")
            return
        
        # Análisis de resultados
        analyze_backtesting_results(predictions_df, data)
        
    except Exception as e:
        log.error(f"Error en backtesting: {str(e)}")


def analyze_backtesting_results(predictions_df, data):
    """
    Analiza los resultados del backtesting.
    
    Args:
        predictions_df: DataFrame con predicciones.
        data: DataFrame con datos históricos.
    """
    print("\n" + "="*80)
    print("📊 RESULTADOS DEL BACKTESTING")
    print("="*80)
    
    # Estadísticas básicas
    total_predictions = len(predictions_df)
    signal_counts = predictions_df['signal'].value_counts()
    
    print(f"Total de predicciones: {total_predictions}")
    print(f"\nDistribución de señales:")
    for signal, count in signal_counts.items():
        percentage = (count / total_predictions) * 100
        print(f"  {signal.upper()}: {count} ({percentage:.1f}%)")
    
    # Estadísticas de confianza
    confidence_stats = predictions_df['confidence'].describe()
    print(f"\nEstadísticas de confianza:")
    print(f"  Media: {confidence_stats['mean']:.3f}")
    print(f"  Mediana: {confidence_stats['50%']:.3f}")
    print(f"  Desv. Estándar: {confidence_stats['std']:.3f}")
    print(f"  Mínimo: {confidence_stats['min']:.3f}")
    print(f"  Máximo: {confidence_stats['max']:.3f}")
    
    # Predicciones de alta confianza
    high_confidence = predictions_df[predictions_df['confidence'] > 0.7]
    print(f"\nPredicciones alta confianza (>0.7): {len(high_confidence)} ({len(high_confidence)/total_predictions*100:.1f}%)")
    
    if len(high_confidence) > 0:
        hc_signals = high_confidence['signal'].value_counts()
        print("Distribución alta confianza:")
        for signal, count in hc_signals.items():
            print(f"  {signal.upper()}: {count}")
    
    # Análisis de rendimiento simple
    if 'price' in predictions_df.columns:
        analyze_simple_performance(predictions_df)
    
    print("="*80)


def analyze_simple_performance(predictions_df):
    """
    Análisis simple de rendimiento de las predicciones.
    
    Args:
        predictions_df: DataFrame con predicciones y precios.
    """
    print(f"\n📈 ANÁLISIS DE RENDIMIENTO SIMPLE:")
    
    # Filtrar predicciones de compra y venta con alta confianza
    buy_signals = predictions_df[
        (predictions_df['signal'] == 'buy') & 
        (predictions_df['confidence'] > 0.6)
    ].copy()
    
    sell_signals = predictions_df[
        (predictions_df['signal'] == 'sell') & 
        (predictions_df['confidence'] > 0.6)
    ].copy()
    
    print(f"Señales BUY alta confianza: {len(buy_signals)}")
    print(f"Señales SELL alta confianza: {len(sell_signals)}")
    
    # Calcular rendimientos simples (próximas 5 velas)
    total_return = 0
    successful_trades = 0
    total_trades = 0
    
    for idx, row in buy_signals.iterrows():
        if idx + 5 < len(predictions_df):
            entry_price = row['price']
            exit_price = predictions_df.iloc[idx + 5]['price']
            return_pct = ((exit_price - entry_price) / entry_price) * 100
            total_return += return_pct
            total_trades += 1
            if return_pct > 0:
                successful_trades += 1
    
    for idx, row in sell_signals.iterrows():
        if idx + 5 < len(predictions_df):
            entry_price = row['price']
            exit_price = predictions_df.iloc[idx + 5]['price']
            return_pct = ((entry_price - exit_price) / entry_price) * 100
            total_return += return_pct
            total_trades += 1
            if return_pct > 0:
                successful_trades += 1
    
    if total_trades > 0:
        avg_return = total_return / total_trades
        success_rate = (successful_trades / total_trades) * 100
        
        print(f"Operaciones simuladas: {total_trades}")
        print(f"Operaciones exitosas: {successful_trades}")
        print(f"Tasa de éxito: {success_rate:.1f}%")
        print(f"Rendimiento promedio: {avg_return:.3f}%")
        print(f"Rendimiento total: {total_return:.3f}%")


def run_realtime_mt5(model_path: str, duration_hours: int = 1, trading_enabled: bool = False):
    """
    Ejecuta predicciones en tiempo real con MT5.
    
    Args:
        model_path: Ruta al modelo entrenado.
        duration_hours: Duración en horas.
        trading_enabled: Si habilitar trading automático.
    """
    log.info(f"🚀 Predicciones MT5 en tiempo real - {duration_hours}h")
    
    try:
        # Crear integrador MT5
        integrator = create_mt5_predictor(model_path)
        
        # Inicializar
        if not integrator.initialize():
            log.error("No se pudo inicializar MT5")
            return
        
        # Mostrar información de la cuenta
        account_info = integrator.get_account_info()
        if account_info:
            print("\n" + "="*60)
            print("💰 INFORMACIÓN DE LA CUENTA MT5")
            print("="*60)
            print(f"Login: {account_info['login']}")
            print(f"Servidor: {account_info['server']}")
            print(f"Balance: {account_info['balance']:.2f} {account_info['currency']}")
            print(f"Equity: {account_info['equity']:.2f} {account_info['currency']}")
            print(f"Margen libre: {account_info['free_margin']:.2f} {account_info['currency']}")
            print("="*60)
        
        # Iniciar predicciones
        if not integrator.start_realtime_predictions(trading_enabled):
            log.error("No se pudieron iniciar las predicciones")
            return
        
        # Ejecutar por el tiempo especificado
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        print(f"\n🕐 Ejecutando hasta: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Presiona Ctrl+C para detener...")
        
        try:
            while datetime.now() < end_time:
                time.sleep(60)  # Verificar cada minuto
                
                # Mostrar predicciones actuales cada 10 minutos
                if datetime.now().minute % 10 == 0:
                    current_predictions = integrator.get_current_predictions()
                    if current_predictions:
                        print(f"\n📊 Predicciones actuales ({datetime.now().strftime('%H:%M:%S')}):")
                        for symbol, pred in current_predictions.items():
                            print(f"  {symbol}: {pred['signal']} (conf: {pred.get('confidence', 0):.3f})")
        
        except KeyboardInterrupt:
            log.info("⏹️ Detenido por el usuario")
        
        # Detener predicciones
        integrator.stop_realtime_predictions()
        
        # Mostrar estadísticas finales
        final_predictions = integrator.get_current_predictions()
        if final_predictions:
            print(f"\n📈 PREDICCIONES FINALES:")
            for symbol, pred in final_predictions.items():
                print(f"  {symbol}: {pred['signal']} @ {pred.get('last_price', 0):.5f}")
        
    except Exception as e:
        log.error(f"Error en predicciones MT5: {str(e)}")
    
    finally:
        try:
            integrator.shutdown()
        except:
            pass


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Sistema de Predicciones LIT + ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Predicción única
  python run_predictions.py single --symbol EURUSD --model models/lit_ml_model.pkl

  # Backtesting
  python run_predictions.py backtest --symbol AAPL --days 30 --model models/lit_ml_model.pkl

  # Tiempo real con MT5 (solo predicciones)
  python run_predictions.py realtime --hours 2 --model models/lit_ml_model.pkl

  # Tiempo real con MT5 (con trading automático)
  python run_predictions.py realtime --hours 1 --model models/lit_ml_model.pkl --trading
        """
    )
    
    # Subcomandos
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Comando: single
    single_parser = subparsers.add_parser('single', help='Predicción única')
    single_parser.add_argument('--symbol', required=True, help='Símbolo a analizar')
    single_parser.add_argument('--timeframe', default='1h', help='Marco temporal')
    single_parser.add_argument('--model', default='models/lit_ml_model.pkl', help='Ruta al modelo')
    
    # Comando: backtest
    backtest_parser = subparsers.add_parser('backtest', help='Backtesting')
    backtest_parser.add_argument('--symbol', required=True, help='Símbolo a analizar')
    backtest_parser.add_argument('--days', type=int, default=30, help='Días de datos históricos')
    backtest_parser.add_argument('--timeframe', default='1h', help='Marco temporal')
    backtest_parser.add_argument('--model', default='models/lit_ml_model.pkl', help='Ruta al modelo')
    
    # Comando: realtime
    realtime_parser = subparsers.add_parser('realtime', help='Predicciones en tiempo real con MT5')
    realtime_parser.add_argument('--hours', type=int, default=1, help='Duración en horas')
    realtime_parser.add_argument('--model', default='models/lit_ml_model.pkl', help='Ruta al modelo')
    realtime_parser.add_argument('--trading', action='store_true', help='Habilitar trading automático')
    
    # Parsear argumentos
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Verificar que el modelo existe
    if not os.path.exists(args.model):
        print(f"❌ Error: Modelo no encontrado: {args.model}")
        print("Ejecuta primero el entrenamiento del modelo:")
        print("python scripts/train_model.py")
        return
    
    # Ejecutar comando
    try:
        if args.command == 'single':
            run_single_prediction(args.model, args.symbol, args.timeframe)
        
        elif args.command == 'backtest':
            run_backtesting(args.model, args.symbol, args.days, args.timeframe)
        
        elif args.command == 'realtime':
            run_realtime_mt5(args.model, args.hours, args.trading)
        
    except KeyboardInterrupt:
        print("\n⏹️ Operación cancelada por el usuario")
    
    except Exception as e:
        log.error(f"Error ejecutando comando: {str(e)}")


if __name__ == "__main__":
    main() 