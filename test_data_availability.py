#!/usr/bin/env python3
"""
Test de Disponibilidad de Datos.

Verifica qué datos están disponibles para diferentes
símbolos y timeframes, identificando las mejores opciones.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Configurar path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import log


def test_symbol_data(symbol: str, timeframes: list) -> dict:
    """Prueba disponibilidad de datos para un símbolo."""
    results = {}
    
    print(f"\n📊 PROBANDO SÍMBOLO: {symbol}")
    print("-" * 50)
    
    for timeframe in timeframes:
        try:
            # Configuración por timeframe
            config = {
                '1m': {'period': '7d', 'interval': '1m'},
                '5m': {'period': '60d', 'interval': '5m'},
                '15m': {'period': '60d', 'interval': '15m'},
                '1h': {'period': '730d', 'interval': '1h'},
                '4h': {'period': '730d', 'interval': '1d'},  # Fallback
                '1d': {'period': '5y', 'interval': '1d'},
                '1w': {'period': '10y', 'interval': '1wk'}
            }
            
            tf_config = config.get(timeframe, {'period': '2y', 'interval': '1d'})
            
            # Descargar datos
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                period=tf_config['period'],
                interval=tf_config['interval'],
                auto_adjust=True
            )
            
            if not data.empty:
                # Información de los datos
                start_date = data.index[0].strftime('%Y-%m-%d')
                end_date = data.index[-1].strftime('%Y-%m-%d')
                rows = len(data)
                
                # Calcular calidad
                missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
                
                status = "✅ EXCELENTE" if rows >= 200 else "⚠️  LIMITADO" if rows >= 50 else "❌ INSUFICIENTE"
                
                print(f"{timeframe:>4}: {status} - {rows:>4} filas ({start_date} a {end_date}) - Missing: {missing_pct:.1f}%")
                
                results[timeframe] = {
                    'status': 'success',
                    'rows': rows,
                    'start_date': start_date,
                    'end_date': end_date,
                    'missing_pct': missing_pct,
                    'quality': 'excellent' if rows >= 200 else 'limited' if rows >= 50 else 'insufficient'
                }
            else:
                print(f"{timeframe:>4}: ❌ SIN DATOS")
                results[timeframe] = {'status': 'no_data', 'rows': 0}
                
        except Exception as e:
            print(f"{timeframe:>4}: ❌ ERROR - {str(e)}")
            results[timeframe] = {'status': 'error', 'error': str(e)}
    
    return results


def test_current_market_status():
    """Verifica el estado actual del mercado."""
    print("\n🕐 ESTADO DEL MERCADO")
    print("-" * 30)
    
    now = datetime.now()
    print(f"Hora actual: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Horario de mercado US (aproximado)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    is_weekday = now.weekday() < 5  # 0-4 = Lunes-Viernes
    is_market_hours = market_open <= now <= market_close
    
    if is_weekday and is_market_hours:
        print("🟢 Mercado: ABIERTO")
    elif is_weekday:
        print("🟡 Mercado: CERRADO (fuera de horario)")
    else:
        print("🔴 Mercado: CERRADO (fin de semana)")
    
    return is_weekday and is_market_hours


def recommend_optimal_config(results: dict) -> dict:
    """Recomienda la configuración óptima basada en resultados."""
    recommendations = {}
    
    print("\n🎯 RECOMENDACIONES ÓPTIMAS")
    print("-" * 40)
    
    for symbol, symbol_results in results.items():
        best_timeframes = []
        
        for tf, data in symbol_results.items():
            if data.get('status') == 'success' and data.get('rows', 0) >= 100:
                best_timeframes.append((tf, data['rows']))
        
        if best_timeframes:
            # Ordenar por cantidad de datos
            best_timeframes.sort(key=lambda x: x[1], reverse=True)
            recommended_tf = best_timeframes[0][0]
            
            print(f"📈 {symbol}: Usar {recommended_tf} ({best_timeframes[0][1]} filas)")
            
            recommendations[symbol] = {
                'recommended_timeframe': recommended_tf,
                'data_rows': best_timeframes[0][1],
                'alternatives': [tf for tf, _ in best_timeframes[1:3]]  # Top 3
            }
        else:
            print(f"⚠️  {symbol}: Datos insuficientes en todos los timeframes")
            recommendations[symbol] = {
                'recommended_timeframe': '1d',  # Fallback
                'data_rows': 0,
                'alternatives': []
            }
    
    return recommendations


def main():
    """Función principal de testing."""
    print("🔍 TEST DE DISPONIBILIDAD DE DATOS - BOT TRADING LIT + ML")
    print("=" * 70)
    
    # Símbolos a probar
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d', '1w']
    
    # Verificar estado del mercado
    market_open = test_current_market_status()
    
    # Probar cada símbolo
    all_results = {}
    
    for symbol in symbols:
        try:
            results = test_symbol_data(symbol, timeframes)
            all_results[symbol] = results
        except Exception as e:
            print(f"❌ Error probando {symbol}: {str(e)}")
    
    # Generar recomendaciones
    recommendations = recommend_optimal_config(all_results)
    
    # Resumen final
    print("\n" + "=" * 70)
    print("📋 RESUMEN Y CONFIGURACIÓN RECOMENDADA")
    print("=" * 70)
    
    # Encontrar el mejor símbolo y timeframe
    best_configs = []
    for symbol, rec in recommendations.items():
        if rec['data_rows'] > 0:
            best_configs.append((symbol, rec['recommended_timeframe'], rec['data_rows']))
    
    if best_configs:
        best_configs.sort(key=lambda x: x[2], reverse=True)
        best_symbol, best_tf, best_rows = best_configs[0]
        
        print(f"🏆 CONFIGURACIÓN ÓPTIMA:")
        print(f"   Símbolo: {best_symbol}")
        print(f"   Timeframe: {best_tf}")
        print(f"   Datos disponibles: {best_rows} filas")
        print(f"   Estado del mercado: {'ABIERTO' if market_open else 'CERRADO'}")
        
        # Generar configuración .env
        print(f"\n📝 CONFIGURACIÓN RECOMENDADA PARA .env:")
        print(f"TRADING_SYMBOL={best_symbol}")
        print(f"TRADING_TIMEFRAME={best_tf}")
        print(f"TRADING_LOOKBACK_PERIODS={min(best_rows, 200)}")
        
        if not market_open:
            print(f"\n⚠️  NOTA: El mercado está cerrado.")
            print(f"   Para trading 24/7, considera:")
            print(f"   - Timeframe diario (1d) para datos históricos")
            print(f"   - Criptomonedas (BTC, ETH) para mercados 24/7")
    else:
        print("❌ No se encontraron configuraciones viables")
        print("💡 Recomendaciones:")
        print("   - Verificar conexión a internet")
        print("   - Intentar más tarde")
        print("   - Considerar datos sintéticos para desarrollo")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main() 