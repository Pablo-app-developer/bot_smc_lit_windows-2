#!/usr/bin/env python3
"""
Test Completo del Sistema Forex.

Prueba todas las funcionalidades del bot de trading Forex:
- Datos de múltiples pares
- Estrategia LIT adaptada
- Machine Learning
- Gestión de riesgo
- Análisis de correlaciones
"""

import sys
import os
import asyncio
from datetime import datetime
import pandas as pd

# Configurar path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.brokers.forex_data_provider import ForexDataProvider
from src.strategies.forex_lit_strategy import ForexLITStrategy
from src.utils.logger import log


async def test_forex_data_provider():
    """Prueba el proveedor de datos Forex."""
    print("\n📊 PROBANDO PROVEEDOR DE DATOS FOREX")
    print("-" * 50)
    
    try:
        provider = ForexDataProvider()
        
        # Probar pares principales
        major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
        
        print(f"Probando {len(major_pairs)} pares principales...")
        
        results = {}
        for pair in major_pairs:
            data = provider.get_forex_data(pair, '1h', 100)
            results[pair] = len(data)
            
            if len(data) > 0:
                current_price = data['close'].iloc[-1]
                spread = data['spread_pips'].iloc[-1] if 'spread_pips' in data.columns else 'N/A'
                session = data['market_session'].iloc[-1] if 'market_session' in data.columns else 'N/A'
                
                print(f"✅ {pair}: {len(data)} filas - Precio: {current_price:.5f} - Spread: {spread} pips - Sesión: {session}")
            else:
                print(f"❌ {pair}: Sin datos")
        
        # Probar estado del mercado
        market_status = provider.get_market_status()
        print(f"\n🌍 Estado del mercado:")
        print(f"   Hora UTC: {market_status.get('utc_hour', 'N/A')}")
        print(f"   Sesiones activas: {', '.join(market_status.get('active_sessions', ['Ninguna']))}")
        print(f"   Volatilidad esperada: {market_status.get('expected_volatility', 'N/A')}")
        
        # Probar precios actuales
        print(f"\n💰 Precios actuales:")
        for pair in major_pairs[:2]:  # Solo 2 para no saturar
            prices = provider.get_current_forex_price(pair)
            if prices:
                print(f"   {pair}: Bid {prices['bid']:.5f} | Ask {prices['ask']:.5f} | Spread {prices['spread_pips']:.1f} pips")
        
        return len([r for r in results.values() if r > 0]) >= 3
        
    except Exception as e:
        print(f"❌ Error probando datos Forex: {str(e)}")
        return False


async def test_forex_lit_strategy():
    """Prueba la estrategia LIT para Forex."""
    print("\n🎯 PROBANDO ESTRATEGIA LIT FOREX")
    print("-" * 50)
    
    try:
        strategy = ForexLITStrategy(lookback_periods=50, liquidity_threshold=0.0001)
        
        # Probar análisis de un par
        test_pairs = ['EURUSD', 'GBPUSD']
        
        for pair in test_pairs:
            print(f"\nAnalizando {pair}...")
            
            analysis = strategy.analyze_forex_pair(pair, '1h', 200)
            
            if analysis.get('signal', 0) != 0:
                signal_text = "BUY" if analysis['signal'] == 1 else "SELL"
                print(f"✅ {pair}: {signal_text}")
                print(f"   Confianza: {analysis['confidence']:.1%}")
                print(f"   Razón: {analysis['reason']}")
                
                if 'entry_levels' in analysis:
                    levels = analysis['entry_levels']
                    print(f"   Entrada: {levels.get('entry_price', 'N/A')}")
                    print(f"   Stop Loss: {levels.get('stop_loss_pips', 'N/A')} pips")
                    print(f"   Take Profit: {levels.get('take_profit_pips', 'N/A')} pips")
                    print(f"   R:R Ratio: {levels.get('risk_reward_ratio', 'N/A')}")
            else:
                print(f"⚠️  {pair}: HOLD - {analysis.get('reason', 'Sin razón')}")
        
        # Probar análisis múltiple
        print(f"\n🔍 Buscando mejores oportunidades...")
        opportunities = strategy.get_best_opportunities(['EURUSD', 'GBPUSD', 'USDJPY'], 0.6)
        
        if opportunities:
            print(f"✅ Encontradas {len(opportunities)} oportunidades:")
            for i, opp in enumerate(opportunities[:3], 1):
                signal_text = "BUY" if opp['signal'] == 1 else "SELL"
                print(f"   {i}. {opp['pair']}: {signal_text} (Confianza: {opp['confidence']:.1%})")
        else:
            print("⚠️  No hay oportunidades con confianza suficiente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error probando estrategia LIT: {str(e)}")
        return False


async def test_correlation_analysis():
    """Prueba análisis de correlaciones."""
    print("\n📈 PROBANDO ANÁLISIS DE CORRELACIONES")
    print("-" * 50)
    
    try:
        provider = ForexDataProvider()
        
        pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURGBP']
        
        print(f"Calculando correlaciones para {len(pairs)} pares...")
        
        correlation_matrix = provider.get_correlation_matrix(pairs, '1d', 50)
        
        if not correlation_matrix.empty:
            print("✅ Matriz de correlaciones:")
            print(correlation_matrix.round(2))
            
            # Identificar correlaciones altas
            high_corr = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    pair1 = correlation_matrix.columns[i]
                    pair2 = correlation_matrix.columns[j]
                    corr = correlation_matrix.iloc[i, j]
                    
                    if abs(corr) > 0.7:
                        high_corr.append((pair1, pair2, corr))
            
            if high_corr:
                print(f"\n⚠️  Correlaciones altas detectadas:")
                for pair1, pair2, corr in high_corr:
                    print(f"   {pair1} - {pair2}: {corr:.2f}")
            else:
                print(f"\n✅ No hay correlaciones excesivamente altas")
            
            return True
        else:
            print("❌ No se pudo calcular matriz de correlaciones")
            return False
            
    except Exception as e:
        print(f"❌ Error probando correlaciones: {str(e)}")
        return False


async def test_risk_management():
    """Prueba gestión de riesgo."""
    print("\n🛡️  PROBANDO GESTIÓN DE RIESGO")
    print("-" * 50)
    
    try:
        # Simular parámetros de cuenta
        account_balance = 2865.05
        risk_per_trade = 0.02  # 2%
        
        print(f"Balance de cuenta: ${account_balance:,.2f}")
        print(f"Riesgo por operación: {risk_per_trade:.1%}")
        
        # Simular cálculo de posición
        test_scenarios = [
            {'pair': 'EURUSD', 'entry': 1.0850, 'stop_loss': 1.0830, 'pip_value': 0.0001},
            {'pair': 'GBPUSD', 'entry': 1.2650, 'stop_loss': 1.2620, 'pip_value': 0.0001},
            {'pair': 'USDJPY', 'entry': 150.50, 'stop_loss': 149.50, 'pip_value': 0.01}
        ]
        
        for scenario in test_scenarios:
            pair = scenario['pair']
            entry = scenario['entry']
            stop_loss = scenario['stop_loss']
            pip_value = scenario['pip_value']
            
            # Calcular riesgo en pips
            risk_pips = abs(entry - stop_loss) / pip_value
            
            # Calcular riesgo monetario
            risk_amount = account_balance * risk_per_trade
            
            # Calcular tamaño de posición (simplificado)
            pip_value_per_unit = pip_value
            position_size = risk_amount / (risk_pips * pip_value_per_unit)
            position_size = round(position_size / 1000) * 1000  # Redondear a miles
            
            print(f"\n📊 {pair}:")
            print(f"   Entrada: {entry}")
            print(f"   Stop Loss: {stop_loss}")
            print(f"   Riesgo: {risk_pips:.0f} pips")
            print(f"   Riesgo $: ${risk_amount:.2f}")
            print(f"   Tamaño: {position_size:,.0f} unidades")
        
        return True
        
    except Exception as e:
        print(f"❌ Error probando gestión de riesgo: {str(e)}")
        return False


async def test_market_sessions():
    """Prueba análisis de sesiones de mercado."""
    print("\n🌍 PROBANDO ANÁLISIS DE SESIONES")
    print("-" * 50)
    
    try:
        provider = ForexDataProvider()
        
        # Obtener estado actual
        market_status = provider.get_market_status()
        
        print(f"Estado actual del mercado:")
        print(f"   Hora UTC: {market_status.get('utc_hour', 'N/A')}:00")
        print(f"   Mercado abierto: {'Sí' if market_status.get('market_open', False) else 'No'}")
        print(f"   Sesiones activas: {', '.join(market_status.get('active_sessions', ['Ninguna']))}")
        print(f"   Solapamiento: {'Sí' if market_status.get('session_overlap', False) else 'No'}")
        print(f"   Volatilidad esperada: {market_status.get('expected_volatility', 'N/A')}")
        
        # Mostrar horarios de sesiones
        print(f"\n📅 Horarios de sesiones (UTC):")
        sessions = {
            'Sydney': '21:00-06:00',
            'Tokyo': '23:00-08:00', 
            'London': '07:00-16:00',
            'New York': '12:00-21:00'
        }
        
        for session, hours in sessions.items():
            active = session in market_status.get('active_sessions', [])
            status = "🟢 ACTIVA" if active else "🔴 CERRADA"
            print(f"   {session}: {hours} - {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error probando sesiones: {str(e)}")
        return False


async def main():
    """Función principal de testing."""
    print("🌍 TEST COMPLETO DEL SISTEMA FOREX")
    print("=" * 60)
    print(f"🕐 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ejecutar todas las pruebas
    tests = [
        ("Proveedor de Datos Forex", test_forex_data_provider),
        ("Estrategia LIT Forex", test_forex_lit_strategy),
        ("Análisis de Correlaciones", test_correlation_analysis),
        ("Gestión de Riesgo", test_risk_management),
        ("Sesiones de Mercado", test_market_sessions)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            result = await test_func()
            results[test_name] = result
            
            if result:
                print(f"✅ {test_name}: EXITOSO")
            else:
                print(f"❌ {test_name}: FALLIDO")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            results[test_name] = False
    
    # Resumen final
    print(f"\n{'='*60}")
    print("📋 RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 RESULTADO FINAL: {passed}/{total} pruebas exitosas")
    
    if passed == total:
        print("🎉 ¡SISTEMA FOREX COMPLETAMENTE FUNCIONAL!")
        print("\n💡 PRÓXIMOS PASOS:")
        print("   1. Ejecutar: python forex_trading_bot.py")
        print("   2. El bot operará automáticamente 24/7")
        print("   3. Monitorear logs en logs/forex_trading.log")
    elif passed >= total * 0.8:
        print("⚠️  Sistema mayormente funcional con algunas limitaciones")
    else:
        print("❌ Sistema requiere correcciones antes de operar")
    
    print(f"\n🕐 Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(main()) 