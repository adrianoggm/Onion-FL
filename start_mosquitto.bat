@echo off
REM Quick script to start mosquitto MQTT broker for SWEET federated learning

echo ================================================================================
echo Starting Mosquitto MQTT Broker
echo ================================================================================
echo.
echo Broker will run on: localhost:1883
echo Config file: mosquitto.conf
echo.
echo Press Ctrl+C to stop the broker
echo ================================================================================
echo.

REM Check if mosquitto is installed
where mosquitto >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: mosquitto not found in PATH
    echo.
    echo Please install mosquitto:
    echo   1. Download from: https://mosquitto.org/download/
    echo   2. Or use Docker: docker run -d -p 1883:1883 eclipse-mosquitto
    echo.
    pause
    exit /b 1
)

REM Check if config file exists
if not exist mosquitto.conf (
    echo WARNING: mosquitto.conf not found
    echo Starting with default configuration...
    echo.
    mosquitto -v
) else (
    echo Using configuration: mosquitto.conf
    echo.
    mosquitto -c mosquitto.conf -v
)
