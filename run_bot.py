#!/usr/bin/env python3
"""
Script de Ejecución Simplificado - Bot de Trading LIT + ML.

Este script proporciona una interfaz simplificada para ejecutar el bot.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd_list):
    """Ejecuta un comando y maneja errores."""
    try:
        print(f"Ejecutando: {' '.join(cmd_list)}")
        result = subprocess.run(cmd_list, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando comando: {e}")
        print(f"Salida de error: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error inesperado: {e}")
        return False


def validate_installation():
    """Valida que el sistema esté correctamente instalado."""
    print("=== VALIDANDO INSTALACIÓN ===")
    
    # Verificar archivos críticos
    critical_files = [
        "src/core/config.py",
        "src/data/data_loader.py", 
        "src/strategies/lit_detector.py",
        "src/models/predictor.py",
        "src/core/trade_executor.py",
        "requirements.txt",
        ".env.example"
    ]
    
    missing_files = []
    for file_path in critical_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Archivos faltantes:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("✅ Todos los archivos críticos están presentes")
    
    # Verificar que se pueden importar los módulos
    try:
        from src.core.config import config
        from src.data.data_loader import DataLoader
        print("✅ Módulos se pueden importar correctamente")
        return True
    except ImportError as e:
        print(f"❌ Error importando módulos: {e}")
        return False


def quick_setup():
    """Setup rápido del sistema."""
    print("=== SETUP RÁPIDO ===")
    
    # 1. Crear directorios necesarios
    dirs_to_create = ['data', 'models', 'logs', 'results']
    for dir_name in dirs_to_create:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Directorio creado: {dir_name}")
    
    # 2. Verificar .env
    if not Path('.env').exists():
        if Path('.env.example').exists():
            print("⚠️  Archivo .env no encontrado. Copiando desde .env.example...")
            subprocess.run(['cp', '.env.example', '.env'])
            print("✅ Archivo .env creado. EDÍTALO con tus configuraciones.")
        else:
            print("❌ Ni .env ni .env.example encontrados")
            return False
    
    # 3. Validar sistema
    return run_command(['python', 'main.py', 'validate'])


def run_quick_test():
    """Ejecuta una prueba rápida del sistema."""
    print("=== PRUEBA RÁPIDA ===")
    
    # 1. Verificar estado
    if not run_command(['python', 'main.py', 'status']):
        return False
    
    # 2. Entrenar modelo básico (datos simulados)
    print("Entrenando modelo de prueba...")
    if not run_command(['python', 'main.py', 'train', 'EURUSD', '500']):
        print("⚠️  Error entrenando modelo, continuando...")
    
    # 3. Hacer backtest corto
    print("Ejecutando backtest de prueba...")
    if not run_command(['python', 'main.py', 'backtest', 'EURUSD', '2024-01-01', '2024-01-31']):
        print("⚠️  Error en backtest, continuando...")
    
    print("✅ Prueba rápida completada")
    return True


def interactive_menu():
    """Menú interactivo para el usuario."""
    while True:
        print("\n" + "="*50)
        print("    BOT DE TRADING LIT + ML")
        print("="*50)
        print("1. 🚀 Ejecutar bot en vivo")
        print("2. 📊 Ejecutar backtest")
        print("3. 🤖 Entrenar modelo")
        print("4. ⚙️  Optimizar hiperparámetros")
        print("5. ✅ Validar sistema")
        print("6. 📋 Ver configuración")
        print("7. 🧪 Prueba rápida")
        print("8. 🔧 Setup inicial")
        print("9. ❌ Salir")
        print("="*50)
        
        try:
            choice = input("Selecciona una opción (1-9): ").strip()
            
            if choice == '1':
                print("Iniciando bot en vivo...")
                print("⚠️  IMPORTANTE: Asegúrate de haber probado en backtest primero")
                confirm = input("¿Estás seguro? (y/N): ").strip().lower()
                if confirm == 'y':
                    run_command(['python', 'main.py'])
                
            elif choice == '2':
                symbol = input("Símbolo (ej: EURUSD): ").strip() or "EURUSD"
                start_date = input("Fecha inicio (YYYY-MM-DD) [2024-01-01]: ").strip() or "2024-01-01"
                end_date = input("Fecha fin (YYYY-MM-DD) [2024-03-01]: ").strip() or "2024-03-01"
                run_command(['python', 'main.py', 'backtest', symbol, start_date, end_date])
                
            elif choice == '3':
                symbol = input("Símbolo (ej: EURUSD): ").strip() or "EURUSD"
                periods = input("Períodos de datos [2000]: ").strip() or "2000"
                run_command(['python', 'main.py', 'train', symbol, periods])
                
            elif choice == '4':
                symbol = input("Símbolo (ej: EURUSD): ").strip() or "EURUSD"
                periods = input("Períodos de datos [5000]: ").strip() or "5000"
                run_command(['python', 'main.py', 'optimize', symbol, periods])
                
            elif choice == '5':
                run_command(['python', 'main.py', 'validate'])
                
            elif choice == '6':
                run_command(['python', 'main.py', 'status'])
                
            elif choice == '7':
                run_quick_test()
                
            elif choice == '8':
                quick_setup()
                
            elif choice == '9':
                print("¡Adiós!")
                break
                
            else:
                print("❌ Opción inválida")
                
        except KeyboardInterrupt:
            print("\n\nProceso interrumpido por el usuario")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Script de Ejecución - Bot Trading LIT + ML")
    parser.add_argument('--quick-setup', action='store_true', help='Setup rápido del sistema')
    parser.add_argument('--validate', action='store_true', help='Validar instalación')
    parser.add_argument('--test', action='store_true', help='Ejecutar prueba rápida')
    parser.add_argument('--interactive', action='store_true', help='Modo interactivo')
    
    args = parser.parse_args()
    
    print("🤖 Bot de Trading LIT + ML - Script de Ejecución")
    print("=" * 60)
    
    # Validar instalación básica
    if not validate_installation():
        print("❌ La instalación no es válida. Revisa los archivos faltantes.")
        sys.exit(1)
    
    if args.quick_setup:
        success = quick_setup()
        sys.exit(0 if success else 1)
        
    elif args.validate:
        success = run_command(['python', 'main.py', 'validate'])
        sys.exit(0 if success else 1)
        
    elif args.test:
        success = run_quick_test()
        sys.exit(0 if success else 1)
        
    elif args.interactive or len(sys.argv) == 1:
        interactive_menu()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 