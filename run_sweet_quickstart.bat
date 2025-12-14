@echo off
REM Script para verificar y ejecutar sistema SWEET Federated
REM Usa splits 70/20/10 (train/val/test)

echo ========================================
echo SWEET Federated Learning - Quick Start
echo Distribucion: 70%% train, 20%% val, 10%% test
echo ========================================
echo.

:menu
echo Selecciona una opcion:
echo.
echo 1. Verificar configuracion (70/20/10)
echo 2. Entrenar modelo baseline desde selection1
echo 3. Preparar splits federados
echo 4. Ejecutar demo federado completo
echo 5. Ejecutar todo el flujo (2+3+4)
echo 0. Salir
echo.

set /p option="Opcion: "

if "%option%"=="0" goto end
if "%option%"=="1" goto verify
if "%option%"=="2" goto baseline
if "%option%"=="3" goto prepare
if "%option%"=="4" goto demo
if "%option%"=="5" goto full_flow

echo Opcion invalida
goto menu

:verify
echo.
echo [1/1] Verificando configuracion 70/20/10...
python scripts\test_sweet_federated.py
echo.
pause
goto menu

:baseline
echo.
echo [BASELINE] Entrenando modelo baseline desde selection1...
echo.
set /p data_dir="Directorio de datos (default: data/SWEET/selection1): "
if "%data_dir%"=="" set data_dir=data/SWEET/selection1

set /p epochs="Numero de epochs (default: 50): "
if "%epochs%"=="" set epochs=50

python scripts\prepare_sweet_baseline.py --data-dir %data_dir% --output-dir baseline_models\sweet --epochs %epochs%
echo.
pause
goto menu

:prepare
echo.
echo [PREPARE] Preparando splits federados (70/20/10)...
echo.
set /p config="Config file (default: configs/sweet_federated.example.yaml): "
if "%config%"=="" set config=configs/sweet_federated.example.yaml

python scripts\prepare_sweet_federated.py --config %config%
echo.
pause
goto menu

:demo
echo.
echo [DEMO] Ejecutando demo federado...
echo.
echo IMPORTANTE: Asegurate de que mosquitto este corriendo
echo   Docker: docker run -d -p 1883:1883 eclipse-mosquitto
echo   Local:  mosquitto -c mosquitto.conf
echo.
pause

set /p config="Config file (default: configs/sweet_federated.example.yaml): "
if "%config%"=="" set config=configs/sweet_federated.example.yaml

set /p rounds="Numero de rounds (default: 10): "
if "%rounds%"=="" set rounds=10

python scripts\run_sweet_federated_demo.py --config %config% --num-rounds %rounds%
echo.
pause
goto menu

:full_flow
echo.
echo ========================================
echo FLUJO COMPLETO - SWEET Federated Learning
echo ========================================
echo.
echo Este script ejecutara:
echo   1. Entrenar modelo baseline (selection1)
echo   2. Preparar splits federados (70/20/10)
echo   3. Ejecutar demo federado
echo.
pause

REM Paso 1: Baseline
echo.
echo [1/3] Entrenando modelo baseline...
python scripts\prepare_sweet_baseline.py --data-dir data\SWEET\selection1 --output-dir baseline_models\sweet --epochs 50
if errorlevel 1 (
    echo ERROR: Fallo al entrenar baseline
    pause
    goto menu
)

REM Paso 2: Preparar splits
echo.
echo [2/3] Preparando splits federados...
python scripts\prepare_sweet_federated.py --config configs\sweet_federated.example.yaml
if errorlevel 1 (
    echo ERROR: Fallo al preparar splits
    pause
    goto menu
)

REM Paso 3: Demo
echo.
echo [3/3] Ejecutando demo federado...
echo.
echo IMPORTANTE: Asegurate de que mosquitto este corriendo
echo   Docker: docker run -d -p 1883:1883 eclipse-mosquitto
echo   Local:  mosquitto -c mosquitto.conf
echo.
pause

python scripts\run_sweet_federated_demo.py --config configs\sweet_federated.example.yaml --num-rounds 10
echo.
echo ========================================
echo FLUJO COMPLETO TERMINADO
echo ========================================
pause
goto menu

:end
echo.
echo Saliendo...
