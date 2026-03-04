@echo off
REM ============================================
REM  Click ABR Quality Control Launcher
REM  Double-click this file to run QC
REM ============================================

REM Activate conda - try common install locations
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
) else if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    call "C:\ProgramData\anaconda3\Scripts\activate.bat"
) else if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    call "C:\ProgramData\miniconda3\Scripts\activate.bat"
) else if exist "C:\anaconda3\Scripts\activate.bat" (
    call "C:\anaconda3\Scripts\activate.bat"
) else if exist "C:\miniconda3\Scripts\activate.bat" (
    call "C:\miniconda3\Scripts\activate.bat"
) else (
    echo ERROR: Could not find Anaconda/Miniconda installation.
    echo Please open Anaconda Prompt manually and run the script from there.
    pause
    exit /b 1
)

call conda activate ABR

REM Change to the project root (two levels up from this batch file)
cd /d "%~dp0..\.."

echo.
echo ========================================
echo   Click ABR Quality Control
echo ========================================
echo.
set /p vhdr="Enter path to .vhdr file: "
echo.
echo Running QC on: %vhdr%
echo.
python code\click_QC\click_qc.py "%vhdr%"
echo.
pause
