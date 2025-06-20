#!/usr/bin/env python3
"""
Monitor del Bot de Trading LIT + ML

Script para monitorear el estado del bot en trading en vivo,
mostrando métricas en tiempo real y estado del aprendizaje automático.
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent))

from src.core.config import config
from src.utils.logger import log
from src.data.data_loader import DataLoader
from src.models.predictor import LITMLPredictor


class TradingBotMonitor:
    """Monitor del bot de trading en tiempo real."""
    
    def __init__(self):
        """Inicializa el monitor."""
        self.data_loader = DataLoader()
        self.predictor = LITMLPredictor()
        self.start_time = datetime.now()
        
        # Cargar modelo si existe
        try:
            self.predictor.load_model()
            log.info("Modelo cargado para monitoreo")
        except:
            log.warning("No se pudo cargar modelo para monitoreo")
    
    def show_live_status(self):
        """Muestra el estado actual del bot."""
        print("\n" + "="*80)
        print(f"🤖 MONITOR BOT TRADING LIT + ML - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Configuración básica
        print(f"📊 CONFIGURACIÓN:")
        print(f"   Símbolo: {config.trading.symbol}")
        print(f"   Timeframe: {config.trading.timeframe}")
        print(f"   Balance inicial: ${config.trading.balance_inicial:,.2f}")
        print(f"   Riesgo por operación: {config.trading.risk_per_trade:.1%}")
        print(f"   Máximo posiciones: {config.trading.max_positions}")
        
        # Estado del mercado
        try:
            market_data = self.data_loader.get_latest_data(periods=5)
            if len(market_data) > 0:
                current_price = market_data['close'].iloc[-1]
                prev_price = market_data['close'].iloc[-2] if len(market_data) > 1 else current_price
                change = ((current_price - prev_price) / prev_price) * 100
                
                print(f"\n📈 MERCADO:")
                print(f"   Precio actual: {current_price:.5f}")
                print(f"   Cambio: {change:+.2f}%")
                print(f"   Última actualización: {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"\n❌ Error obteniendo datos de mercado: {str(e)}")
        
        # Estado del modelo ML
        if hasattr(self.predictor, 'get_learning_status'):
            try:
                learning_status = self.predictor.get_learning_status()
                print(f"\n🧠 APRENDIZAJE AUTOMÁTICO:")
                print(f"   Estado: {'🟢 ACTIVO' if learning_status.get('learning_thread_active', False) else '🔴 INACTIVO'}")
                print(f"   Accuracy actual: {learning_status.get('current_accuracy', 0):.1%}")
                print(f"   Predicciones realizadas: {learning_status.get('predictions_made', 0):,}")
                print(f"   Predicciones correctas: {learning_status.get('correct_predictions', 0):,}")
                print(f"   Buffer de datos: {learning_status.get('data_buffer_size', 0)} puntos")
                print(f"   Último re-entrenamiento: {learning_status.get('last_retrain_time', 'N/A')}")
            except Exception as e:
                print(f"\n❌ Error obteniendo estado de aprendizaje: {str(e)}")
        
        # Archivos de log
        print(f"\n📝 LOGS:")
        log_file = Path(config.logging.file)
        if log_file.exists():
            size_mb = log_file.stat().st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(log_file.stat().st_mtime)
            print(f"   Archivo: {log_file}")
            print(f"   Tamaño: {size_mb:.2f} MB")
            print(f"   Última modificación: {modified.strftime('%H:%M:%S')}")
        else:
            print(f"   ❌ Archivo de log no encontrado: {log_file}")
        
        # Modelos disponibles
        models_dir = Path("models")
        if models_dir.exists():
            models = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
            print(f"\n🎯 MODELOS DISPONIBLES ({len(models)}):")
            for model in sorted(models, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                size_kb = model.stat().st_size / 1024
                modified = datetime.fromtimestamp(model.stat().st_mtime)
                print(f"   {model.name} ({size_kb:.1f} KB) - {modified.strftime('%d/%m %H:%M')}")
        
        # Tiempo de ejecución
        uptime = datetime.now() - self.start_time
        print(f"\n⏱️  TIEMPO DE MONITOREO: {str(uptime).split('.')[0]}")
        
        print("="*80)
    
    def show_recent_logs(self, lines: int = 10):
        """Muestra las últimas líneas del log."""
        log_file = Path(config.logging.file)
        
        if not log_file.exists():
            print("❌ Archivo de log no encontrado")
            return
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:]
            
            print(f"\n📋 ÚLTIMAS {len(recent_lines)} LÍNEAS DEL LOG:")
            print("-" * 80)
            for line in recent_lines:
                print(line.rstrip())
            print("-" * 80)
            
        except Exception as e:
            print(f"❌ Error leyendo log: {str(e)}")
    
    def run_continuous_monitor(self, interval: int = 30):
        """Ejecuta monitoreo continuo."""
        print("🚀 INICIANDO MONITOR CONTINUO DEL BOT")
        print(f"Actualizando cada {interval} segundos...")
        print("Presiona Ctrl+C para detener")
        
        try:
            while True:
                # Limpiar pantalla (Windows/Linux compatible)
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Mostrar estado
                self.show_live_status()
                
                # Mostrar logs recientes
                self.show_recent_logs(5)
                
                # Esperar
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitor detenido por el usuario")
        except Exception as e:
            print(f"\n❌ Error en monitor: {str(e)}")


def main():
    """Función principal del monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor del Bot de Trading LIT + ML")
    parser.add_argument('--continuous', '-c', action='store_true', 
                       help='Ejecutar monitoreo continuo')
    parser.add_argument('--interval', '-i', type=int, default=30,
                       help='Intervalo de actualización en segundos (default: 30)')
    parser.add_argument('--logs', '-l', type=int, default=10,
                       help='Número de líneas de log a mostrar (default: 10)')
    
    args = parser.parse_args()
    
    # Crear monitor
    monitor = TradingBotMonitor()
    
    if args.continuous:
        # Monitoreo continuo
        monitor.run_continuous_monitor(args.interval)
    else:
        # Mostrar estado una vez
        monitor.show_live_status()
        monitor.show_recent_logs(args.logs)


if __name__ == "__main__":
    main() 