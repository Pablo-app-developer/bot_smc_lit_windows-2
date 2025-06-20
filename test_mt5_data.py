#!/usr/bin/env python3
"""
Script de prueba para MT5DataLoader.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.mt5_data_loader import MT5DataLoader

def test_mt5_data_loader():
    """Prueba el MT5DataLoader."""
    print("🔍 Probando MT5DataLoader...")
    
    loader = MT5DataLoader()
    
    try:
        # Conectar
        if not loader.connect():
            print("❌ Error conectando")
            return
        
        print("✅ Conectado a MT5")
        
        # Probar carga de datos
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        for symbol in symbols:
            print(f"\n📊 Probando {symbol}:")
            
            data = loader.load_data(symbol, '1h', 50)
            
            if len(data) > 0:
                print(f"   ✅ {len(data)} velas obtenidas")
                print(f"   💰 Último precio: {data['close'].iloc[-1]:.5f}")
                print(f"   📅 Desde: {data.index[0]}")
                print(f"   📅 Hasta: {data.index[-1]}")
            else:
                print(f"   ❌ Sin datos para {symbol}")
        
        # Probar precio actual
        print(f"\n💰 Precios actuales:")
        for symbol in symbols:
            price = loader.get_current_price(symbol)
            if price:
                print(f"   {symbol}: {price:.5f}")
        
        # Desconectar
        loader.disconnect()
        print("\n✅ Prueba completada")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_mt5_data_loader() 