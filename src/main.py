"""
Bot de Trading LIT + ML - Archivo Principal.

Este es el punto de entrada principal del bot de trading automatizado
que combina la estrategia LIT con Machine Learning.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional

from src.core.config import config
from src.data.data_loader import DataLoader
from src.models.predictor import TradingPredictor
from src.core.trade_executor import TradeExecutor
from src.utils.logger import log


class LITMLBot:
    """
    Bot de Trading LIT + ML.
    
    Orquesta todos los componentes del bot para trading automatizado.
    """
    
    def __init__(self):
        """Inicializa el bot de trading."""
        log.info("=== Iniciando Bot de Trading LIT + ML ===")
        
        # Crear directorios necesarios
        config.create_directories()
        
        # Inicializar componentes
        self.data_loader = DataLoader()
        self.predictor = TradingPredictor()
        self.trade_executor = TradeExecutor()
        
        # Variables de estado
        self.running = False
        self.last_prediction = None
        
        log.info("Bot inicializado correctamente")
    
    async def start(self) -> None:
        """Inicia el bot de trading."""
        log.info("Iniciando bot de trading...")
        self.running = True
        
        try:
            await self._main_loop()
        except KeyboardInterrupt:
            log.info("Deteniendo bot por interrupción del usuario...")
        except Exception as e:
            log.error(f"Error en el bot: {str(e)}")
        finally:
            await self.stop()
    
    async def _main_loop(self) -> None:
        """Bucle principal del bot."""
        while self.running:
            try:
                # Obtener datos de mercado
                market_data = self.data_loader.get_latest_data()
                
                if len(market_data) == 0:
                    log.warning("No se pudieron obtener datos de mercado")
                    await asyncio.sleep(60)  # Esperar 1 minuto
                    continue
                
                current_price = market_data['close'].iloc[-1]
                
                # Generar predicción
                prediction = self.predictor.predict(market_data)
                
                # Validar predicción
                validation = self.predictor.validate_prediction(prediction)
                
                if validation['is_valid']:
                    # Ejecutar señal
                    order_id = self.trade_executor.execute_signal(
                        prediction, current_price
                    )
                    
                    if order_id:
                        log.info(f"Señal ejecutada: {prediction['signal']} | "
                               f"Confianza: {prediction['confidence']:.3f} | "
                               f"Orden: {order_id}")
                    
                    self.last_prediction = prediction
                else:
                    log.info(f"Señal rechazada: {validation['warnings']}")
                
                # Log de estado
                await self._log_status(current_price)
                
                # Esperar antes de la siguiente iteración
                await asyncio.sleep(300)  # 5 minutos
                
            except Exception as e:
                log.error(f"Error en bucle principal: {str(e)}")
                await asyncio.sleep(60)
    
    async def _log_status(self, current_price: float) -> None:
        """
        Log del estado actual del bot.
        
        Args:
            current_price: Precio actual del mercado.
        """
        # Métricas de rendimiento
        metrics = self.trade_executor.get_performance_metrics()
        open_positions = self.trade_executor.get_open_positions()
        
        log.info(f"Estado del Bot:")
        log.info(f"  Precio actual: {current_price:.5f}")
        log.info(f"  Balance: {metrics['balance']:.2f}")
        log.info(f"  Posiciones abiertas: {len(open_positions)}")
        log.info(f"  Total trades: {metrics['total_trades']}")
        log.info(f"  Win rate: {metrics['win_rate']:.1f}%")
        log.info(f"  Return: {metrics['return_percentage']:.2f}%")
        
        if self.last_prediction:
            log.info(f"  Última señal: {self.last_prediction['signal']} "
                   f"({self.last_prediction['confidence']:.3f})")
    
    async def stop(self) -> None:
        """Detiene el bot de trading."""
        log.info("Deteniendo bot de trading...")
        self.running = False
        
        # Cerrar todas las posiciones abiertas
        if len(self.trade_executor.get_open_positions()) > 0:
            current_data = self.data_loader.get_latest_data()
            if len(current_data) > 0:
                current_price = current_data['close'].iloc[-1]
                closed_count = self.trade_executor.close_all_positions(
                    current_price, "bot_shutdown"
                )
                log.info(f"Cerradas {closed_count} posiciones por cierre del bot")
        
        # Log final de métricas
        await self._log_final_metrics()
        
        log.info("Bot detenido")
    
    async def _log_final_metrics(self) -> None:
        """Log de métricas finales."""
        metrics = self.trade_executor.get_performance_metrics()
        
        log.info("=== MÉTRICAS FINALES ===")
        log.info(f"Balance inicial: {metrics['initial_balance']:.2f}")
        log.info(f"Balance final: {metrics['balance']:.2f}")
        log.info(f"Total trades: {metrics['total_trades']}")
        log.info(f"Trades ganadores: {metrics['winning_trades']}")
        log.info(f"Trades perdedores: {metrics['losing_trades']}")
        log.info(f"Win rate: {metrics['win_rate']:.2f}%")
        log.info(f"PnL total: {metrics['total_pnl']:.2f}")
        log.info(f"Retorno: {metrics['return_percentage']:.2f}%")
        log.info(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del bot.
        
        Returns:
            Dict[str, Any]: Estado del bot.
        """
        return {
            'running': self.running,
            'metrics': self.trade_executor.get_performance_metrics(),
            'open_positions': len(self.trade_executor.get_open_positions()),
            'model_status': self.predictor.get_model_status(),
            'last_prediction': self.last_prediction
        }


async def run_backtest(symbol: str = None, 
                      start_date: str = None, 
                      end_date: str = None) -> Dict[str, Any]:
    """
    Ejecuta un backtest del bot.
    
    Args:
        symbol: Símbolo a testear.
        start_date: Fecha de inicio (YYYY-MM-DD).
        end_date: Fecha de fin (YYYY-MM-DD).
        
    Returns:
        Dict[str, Any]: Resultados del backtest.
    """
    log.info("=== Iniciando Backtest ===")
    
    # Inicializar componentes
    data_loader = DataLoader()
    predictor = TradingPredictor()
    trade_executor = TradeExecutor()
    
    # Cargar datos históricos
    symbol = symbol or config.trading.symbol
    
    if start_date and end_date:
        from datetime import datetime
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        historical_data = data_loader.load_data(
            symbol=symbol, start_date=start, end_date=end
        )
    else:
        historical_data = data_loader.load_data(symbol=symbol)
    
    log.info(f"Datos cargados: {len(historical_data)} períodos")
    
    # Ejecutar backtest
    window_size = config.ml.feature_lookback
    
    for i in range(window_size, len(historical_data)):
        # Datos hasta el punto actual
        current_data = historical_data.iloc[:i+1]
        current_price = current_data['close'].iloc[-1]
        
        # Generar predicción
        prediction = predictor.predict(current_data)
        
        # Validar y ejecutar
        validation = predictor.validate_prediction(prediction)
        
        if validation['is_valid']:
            trade_executor.execute_signal(prediction, current_price, symbol)
        
        # Actualizar posiciones existentes
        trade_executor._check_existing_positions(current_price)
    
    # Cerrar posiciones restantes
    final_price = historical_data['close'].iloc[-1]
    trade_executor.close_all_positions(final_price, "backtest_end")
    
    # Resultados
    results = {
        'metrics': trade_executor.get_performance_metrics(),
        'trade_history': trade_executor.get_trade_history_df(),
        'data_periods': len(historical_data),
        'symbol': symbol
    }
    
    log.info("=== Backtest Completado ===")
    log.info(f"Return: {results['metrics']['return_percentage']:.2f}%")
    log.info(f"Total trades: {results['metrics']['total_trades']}")
    log.info(f"Win rate: {results['metrics']['win_rate']:.2f}%")
    
    return results


async def train_model() -> None:
    """Entrena un nuevo modelo ML."""
    log.info("=== Entrenando Modelo ML ===")
    
    from src.models.model_trainer import ModelTrainer
    
    # Cargar datos para entrenamiento
    data_loader = DataLoader()
    training_data = data_loader.load_data(periods=2000)  # Más datos para entrenamiento
    
    # Entrenar modelo
    trainer = ModelTrainer()
    results = trainer.train(training_data, optimize_hyperparams=True)
    
    # Guardar modelo
    model_path = config.get_paths()['models'] / 'trained_model.joblib'
    trainer.save_model(str(model_path))
    
    log.info(f"Modelo entrenado y guardado en: {model_path}")
    log.info(f"Accuracy de prueba: {results['test_metrics']['accuracy']:.4f}")


def main():
    """Función principal."""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "backtest":
            # Ejecutar backtest
            symbol = sys.argv[2] if len(sys.argv) > 2 else None
            start_date = sys.argv[3] if len(sys.argv) > 3 else None
            end_date = sys.argv[4] if len(sys.argv) > 4 else None
            
            results = asyncio.run(run_backtest(symbol, start_date, end_date))
            
            # Guardar resultados
            import pandas as pd
            results_path = config.get_paths()['base'] / 'backtest_results.csv'
            if not results['trade_history'].empty:
                results['trade_history'].to_csv(results_path, index=False)
                log.info(f"Resultados guardados en: {results_path}")
        
        elif command == "train":
            # Entrenar modelo
            asyncio.run(train_model())
        
        elif command == "status":
            # Mostrar configuración actual
            log.info("=== Configuración Actual ===")
            log.info(f"Bot: {config.bot_name} v{config.version}")
            log.info(f"Símbolo: {config.trading.symbol}")
            log.info(f"Timeframe: {config.trading.timeframe}")
            log.info(f"Balance inicial: {config.trading.balance_inicial}")
            log.info(f"Riesgo por trade: {config.trading.risk_per_trade * 100}%")
            log.info(f"Max posiciones: {config.trading.max_positions}")
        
        else:
            print("Comandos disponibles:")
            print("  python -m src.main backtest [symbol] [start_date] [end_date]")
            print("  python -m src.main train")
            print("  python -m src.main status")
            print("  python -m src.main  (ejecutar bot en vivo)")
    
    else:
        # Ejecutar bot en vivo
        try:
            bot = LITMLBot()
            asyncio.run(bot.start())
        except KeyboardInterrupt:
            log.info("Bot detenido por el usuario")
        except Exception as e:
            log.error(f"Error ejecutando bot: {str(e)}")


if __name__ == "__main__":
    main() 