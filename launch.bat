@echo off
title DeepPlate-Reader

echo ============================================
echo        DeepPlate-Reader - Iniciando...
echo ============================================
echo.

:: Verificar que Python este instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no encontrado.
    echo Descarga Python desde: https://www.python.org/downloads/
    echo Asegurate de marcar "Add Python to PATH" al instalar.
    pause
    exit /b 1
)

echo [1/3] Instalando dependencias...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Fallo la instalacion de dependencias.
    pause
    exit /b 1
)
echo       Dependencias OK.
echo.

:: Entrenar modelo si no existe
if not exist "models\ocr\ocr_svm.pkl" (
    echo [2/3] Entrenando modelo SVM por primera vez...
    echo       Esto tarda aproximadamente 30 segundos.
    python scripts/train.py --samples 500
    if %errorlevel% neq 0 (
        echo [ERROR] Fallo el entrenamiento del modelo.
        pause
        exit /b 1
    )
    echo       Modelo entrenado OK.
) else (
    echo [2/3] Modelo ya entrenado, omitiendo...
)
echo.

echo [3/3] Lanzando aplicacion web...
echo       Abre tu navegador en: http://localhost:8501
echo.
echo       Para cerrar la aplicacion presiona Ctrl+C en esta ventana.
echo.
streamlit run app.py

pause
