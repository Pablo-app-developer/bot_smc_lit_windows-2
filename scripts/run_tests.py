#!/usr/bin/env python3
"""
Script para ejecutar pruebas unitarias del sistema LIT + ML.

Permite ejecutar diferentes tipos de pruebas con configuraciones específicas.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Ejecutar comando y mostrar resultado."""
    if description:
        print(f"\n{'='*60}")
        print(f"🔍 {description}")
        print(f"{'='*60}")
    
    print(f"Ejecutando: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando comando: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ Comando no encontrado: {cmd[0]}")
        return False


def install_test_dependencies():
    """Instalar dependencias necesarias para las pruebas."""
    print("📦 Instalando dependencias de pruebas...")
    
    dependencies = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
        "pytest-xdist>=3.0.0",
        "pytest-timeout>=2.1.0",
        "pytest-html>=3.1.0"
    ]
    
    for dep in dependencies:
        cmd = [sys.executable, "-m", "pip", "install", dep]
        if not run_command(cmd, f"Instalando {dep}"):
            return False
    
    return True


def run_unit_tests(verbose=False, coverage=False):
    """Ejecutar pruebas unitarias."""
    cmd = [sys.executable, "-m", "pytest", "-m", "unit"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term-missing"])
    
    return run_command(cmd, "Ejecutando pruebas unitarias")


def run_integration_tests(verbose=False):
    """Ejecutar pruebas de integración."""
    cmd = [sys.executable, "-m", "pytest", "-m", "integration"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Ejecutando pruebas de integración")


def run_all_tests(verbose=False, coverage=False, parallel=False):
    """Ejecutar todas las pruebas."""
    cmd = [sys.executable, "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term-missing"])
    
    if parallel:
        cmd.extend(["-n", "auto"])
    
    return run_command(cmd, "Ejecutando todas las pruebas")


def run_specific_test(test_file, test_function=None, verbose=False):
    """Ejecutar una prueba específica."""
    cmd = [sys.executable, "-m", "pytest"]
    
    if test_function:
        cmd.append(f"{test_file}::{test_function}")
    else:
        cmd.append(test_file)
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, f"Ejecutando prueba específica: {test_file}")


def run_lit_detector_tests(verbose=False):
    """Ejecutar solo pruebas del detector LIT."""
    cmd = [sys.executable, "-m", "pytest", "tests/test_lit_detector.py"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Ejecutando pruebas del detector LIT")


def run_ml_model_tests(verbose=False):
    """Ejecutar solo pruebas del modelo ML."""
    cmd = [sys.executable, "-m", "pytest", "tests/test_ml_model.py"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Ejecutando pruebas del modelo ML")


def run_trade_executor_tests(verbose=False):
    """Ejecutar solo pruebas del trade executor."""
    cmd = [sys.executable, "-m", "pytest", "tests/test_trade_executor.py"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Ejecutando pruebas del trade executor")


def run_fast_tests(verbose=False):
    """Ejecutar solo pruebas rápidas (excluir lentas)."""
    cmd = [sys.executable, "-m", "pytest", "-m", "not slow"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Ejecutando pruebas rápidas")


def run_slow_tests(verbose=False):
    """Ejecutar solo pruebas lentas."""
    cmd = [sys.executable, "-m", "pytest", "-m", "slow"]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, "Ejecutando pruebas lentas")


def generate_test_report():
    """Generar reporte HTML de las pruebas."""
    cmd = [
        sys.executable, "-m", "pytest",
        "--html=tests/report.html",
        "--self-contained-html",
        "--cov=src",
        "--cov-report=html:tests/htmlcov"
    ]
    
    return run_command(cmd, "Generando reporte HTML de pruebas")


def check_test_environment():
    """Verificar que el entorno de pruebas esté configurado correctamente."""
    print("🔧 Verificando entorno de pruebas...")
    
    # Verificar Python
    print(f"Python: {sys.version}")
    
    # Verificar pytest
    try:
        import pytest
        print(f"pytest: {pytest.__version__}")
    except ImportError:
        print("❌ pytest no está instalado")
        return False
    
    # Verificar estructura de directorios
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"
    src_dir = project_root / "src"
    
    if not tests_dir.exists():
        print(f"❌ Directorio de pruebas no encontrado: {tests_dir}")
        return False
    
    if not src_dir.exists():
        print(f"❌ Directorio de código fuente no encontrado: {src_dir}")
        return False
    
    # Verificar archivos de prueba
    test_files = list(tests_dir.glob("test_*.py"))
    print(f"📁 Archivos de prueba encontrados: {len(test_files)}")
    
    for test_file in test_files:
        print(f"  - {test_file.name}")
    
    print("✅ Entorno de pruebas verificado correctamente")
    return True


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Script para ejecutar pruebas del sistema LIT + ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python scripts/run_tests.py --all                    # Todas las pruebas
  python scripts/run_tests.py --unit --coverage        # Pruebas unitarias con cobertura
  python scripts/run_tests.py --integration -v         # Pruebas de integración verbose
  python scripts/run_tests.py --lit-detector           # Solo pruebas LIT
  python scripts/run_tests.py --ml-model               # Solo pruebas ML
  python scripts/run_tests.py --trade-executor         # Solo pruebas executor
  python scripts/run_tests.py --fast                   # Solo pruebas rápidas
  python scripts/run_tests.py --specific tests/test_lit_detector.py
  python scripts/run_tests.py --install-deps           # Instalar dependencias
  python scripts/run_tests.py --check-env              # Verificar entorno
  python scripts/run_tests.py --report                 # Generar reporte HTML
        """
    )
    
    # Tipos de pruebas
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--all", action="store_true", help="Ejecutar todas las pruebas")
    test_group.add_argument("--unit", action="store_true", help="Ejecutar solo pruebas unitarias")
    test_group.add_argument("--integration", action="store_true", help="Ejecutar solo pruebas de integración")
    test_group.add_argument("--lit-detector", action="store_true", help="Ejecutar solo pruebas del detector LIT")
    test_group.add_argument("--ml-model", action="store_true", help="Ejecutar solo pruebas del modelo ML")
    test_group.add_argument("--trade-executor", action="store_true", help="Ejecutar solo pruebas del trade executor")
    test_group.add_argument("--fast", action="store_true", help="Ejecutar solo pruebas rápidas")
    test_group.add_argument("--slow", action="store_true", help="Ejecutar solo pruebas lentas")
    test_group.add_argument("--specific", type=str, help="Ejecutar prueba específica (archivo o archivo::función)")
    
    # Opciones
    parser.add_argument("-v", "--verbose", action="store_true", help="Salida detallada")
    parser.add_argument("--coverage", action="store_true", help="Generar reporte de cobertura")
    parser.add_argument("--parallel", action="store_true", help="Ejecutar pruebas en paralelo")
    
    # Utilidades
    parser.add_argument("--install-deps", action="store_true", help="Instalar dependencias de pruebas")
    parser.add_argument("--check-env", action="store_true", help="Verificar entorno de pruebas")
    parser.add_argument("--report", action="store_true", help="Generar reporte HTML")
    
    args = parser.parse_args()
    
    # Cambiar al directorio del proyecto
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    success = True
    
    # Instalar dependencias
    if args.install_deps:
        success = install_test_dependencies()
        if not success:
            sys.exit(1)
        return
    
    # Verificar entorno
    if args.check_env:
        success = check_test_environment()
        if not success:
            sys.exit(1)
        return
    
    # Generar reporte
    if args.report:
        success = generate_test_report()
        if success:
            print("\n📊 Reporte generado en tests/report.html")
            print("📊 Cobertura generada en tests/htmlcov/index.html")
        sys.exit(0 if success else 1)
    
    # Ejecutar pruebas según la opción seleccionada
    if args.all:
        success = run_all_tests(args.verbose, args.coverage, args.parallel)
    elif args.unit:
        success = run_unit_tests(args.verbose, args.coverage)
    elif args.integration:
        success = run_integration_tests(args.verbose)
    elif args.lit_detector:
        success = run_lit_detector_tests(args.verbose)
    elif args.ml_model:
        success = run_ml_model_tests(args.verbose)
    elif args.trade_executor:
        success = run_trade_executor_tests(args.verbose)
    elif args.fast:
        success = run_fast_tests(args.verbose)
    elif args.slow:
        success = run_slow_tests(args.verbose)
    elif args.specific:
        # Separar archivo y función si se especifica
        if "::" in args.specific:
            test_file, test_function = args.specific.split("::", 1)
        else:
            test_file, test_function = args.specific, None
        success = run_specific_test(test_file, test_function, args.verbose)
    else:
        # Por defecto, ejecutar pruebas rápidas
        print("ℹ️  No se especificó tipo de prueba. Ejecutando pruebas rápidas...")
        success = run_fast_tests(args.verbose)
    
    # Mostrar resultado final
    if success:
        print("\n✅ Todas las pruebas completadas exitosamente")
        sys.exit(0)
    else:
        print("\n❌ Algunas pruebas fallaron")
        sys.exit(1)


if __name__ == "__main__":
    main() 