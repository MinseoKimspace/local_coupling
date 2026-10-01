@echo off
rem Open an interactive PSF training shell; never starts training automatically.
setlocal
cd /d "%~dp0"
if errorlevel 1 goto failed

set "PSF_VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"
if not exist "%PSF_VCVARS%" (
    echo Visual Studio setup not found: %PSF_VCVARS%
    goto failed
)
call "%PSF_VCVARS%" amd64 -vcvars_ver=14.44
if errorlevel 1 goto failed

where conda >nul 2>&1
if not errorlevel 1 (
    call conda activate local_coupling
) else (
    set "PSF_CONDA="
    for %%R in (anaconda3 miniconda3 miniforge3) do if exist "%USERPROFILE%\%%R\condabin\conda.bat" set "PSF_CONDA=%USERPROFILE%\%%R\condabin\conda.bat"
    call :activate_conda
)
if errorlevel 1 goto failed

rem Keep caches beside this checkout, on the same drive as the launcher.
for %%D in ("%~dp0..") do set "PSF_PARENT=%%~fD"
set "TORCH_EXTENSIONS_DIR=%PSF_PARENT%\torch_extensions"
set "TEMP=%PSF_PARENT%\tmp"
set "TMP=%TEMP%"
if not exist "%TORCH_EXTENSIONS_DIR%" mkdir "%TORCH_EXTENSIONS_DIR%"
if not exist "%TORCH_EXTENSIONS_DIR%" goto failed
if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%TEMP%" goto failed
set "MAX_JOBS=2"
set "DISTUTILS_USE_SDK=1"
set "TORCH_CUDA_ARCH_LIST="
where cl
if errorlevel 1 goto failed
where nvcc
if errorlevel 1 goto failed
python -c "import sys,torch; print('Python:',sys.executable); print('Torch:',torch.__version__); print('CUDA available:',torch.cuda.is_available())"
if errorlevel 1 goto failed
echo.
echo Project: %CD%
echo CUDA build cache: %TORCH_EXTENSIONS_DIR%
echo Ready. Enter your training or evaluation command below.
echo This is CMD: use %%VARIABLE%%, not PowerShell $VARIABLE.
cmd.exe /k
exit /b 0

:activate_conda
if not defined PSF_CONDA (
    echo Conda not found. Run this launcher from an Anaconda Prompt.
    exit /b 1
)
call "%PSF_CONDA%" activate local_coupling
exit /b %errorlevel%

:failed
echo.
echo Setup failed. Training has NOT been started.
pause
exit /b 1
