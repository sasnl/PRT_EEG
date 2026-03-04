@echo off
REM ============================================
REM  PRT EEG Experiment Launcher
REM  Double-click this file to start
REM ============================================

REM Activate conda - try common install locations
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
) else if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    call "C:\ProgramData\anaconda3\Scripts\activate.bat"
) else if exist "C:\anaconda3\Scripts\activate.bat" (
    call "C:\anaconda3\Scripts\activate.bat"
) else (
    echo ERROR: Could not find Anaconda installation.
    echo Please open Anaconda Prompt manually and run the scripts from there.
    pause
    exit /b 1
)

call conda activate ABR

REM Change to the directory where this batch file lives
cd /d "%~dp0"

:menu
echo.
echo ========================================
echo   PRT EEG Experiment
echo ========================================
echo.
echo   1. Click Presentation (run first)
echo   2. Story Presentation
echo   3. Post-Session Calibration
echo   4. Exit
echo.
set /p choice="Select option (1-4): "

if "%choice%"=="1" goto click
if "%choice%"=="2" goto story
if "%choice%"=="3" goto calibration
if "%choice%"=="4" goto end

echo Invalid choice. Please enter 1, 2, 3, or 4.
goto menu

:click
echo.
echo Starting Click Presentation...
echo.
python code\experiment\prt_click_presentation.py
goto menu

:story
echo.
echo Starting Story Presentation...
echo.
python code\experiment\prt_story_presentation.py
goto menu

:calibration
echo.
echo Starting Post-Session Calibration...
echo.
python code\experiment\sound_calibration.py
goto menu

:end
echo.
echo Goodbye!
pause
