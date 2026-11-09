rem Windows one-click batch that locates `FigTab.py` (next to this .bat,
rem in the parent / grandparent directory, in the cwd, or in one level of
rem subfolders; you can also drag `FigTab.py` onto the .bat), and runs it
rem with the active Python interpreter. Produces `FIGsTABs/` with every
rem paper figure and table.

rem Copyright (c) Peng Yue, Jinghui Zhang, Jiangyong Du
rem All rights reserved.

@echo off
setlocal
cd /d "%~dp0"

rem Optional: drag FigTab.py (or a folder containing it) onto this .bat.
set "FIGTAB="
if not "%~1"=="" (
  if /i "%~x1"==".py" set "FIGTAB=%~1"
  if exist "%~1\FigTab.py" set "FIGTAB=%~1\FigTab.py"
)

rem 1) Same folder as this .bat.
if not defined FIGTAB if exist "%~dp0FigTab.py" set "FIGTAB=%~dp0FigTab.py"

rem 2) Parent and grandparent.
if not defined FIGTAB if exist "%~dp0..\FigTab.py" set "FIGTAB=%~dp0..\FigTab.py"
if not defined FIGTAB if exist "%~dp0..\..\FigTab.py" set "FIGTAB=%~dp0..\..\FigTab.py"

rem 3) Current working directory.
if not defined FIGTAB if exist "%CD%\FigTab.py" set "FIGTAB=%CD%\FigTab.py"

rem 4) One level of subfolders, next to this .bat.
if not defined FIGTAB (
  for /d %%D in ("%~dp0*") do (
    if not defined FIGTAB if exist "%%~fD\FigTab.py" set "FIGTAB=%%~fD\FigTab.py"
  )
)

rem 5) One level of subfolders, one level above this .bat.
if not defined FIGTAB (
  for /d %%D in ("%~dp0..\*") do (
    if not defined FIGTAB if exist "%%~fD\FigTab.py" set "FIGTAB=%%~fD\FigTab.py"
  )
)

rem 6) Last resort: ask for the path.
if not defined FIGTAB (
  echo.
  echo Could not find FigTab.py automatically.
  echo Drag FigTab.py onto this .bat, or paste its full path below.
  echo.
  set /p "FIGTAB=Path to FigTab.py: "
)

if defined FIGTAB set "FIGTAB=%FIGTAB:"=%"

if not exist "%FIGTAB%" (
  echo.
  echo ERROR: FigTab.py not found:
  echo   %FIGTAB%
  pause
  exit /b 1
)

rem Run from FigTab.py's own folder, so its HERE is correct.
for %%F in ("%FIGTAB%") do (
  set "FIGTAB=%%~fF"
  cd /d "%%~dpF"
)

rem Use the Python that has numpy/scipy/numba/matplotlib.
rem If "python" is not that interpreter, set the full path:
rem set "PY=D:\Python\python.exe"
set "PY=python"

set "RUN_ARGS="
if /i "%~1"=="--only" set "RUN_ARGS=%*"

echo ============================================================
echo  FigTab: figures and tables
echo  FigTab.py = %FIGTAB%
echo  Folder    = %CD%
echo  Data      = %CD%\..\from_zero
echo  Output    = %CD%\FIGsTABs
echo  This takes about 20-25 minutes. Do not close this window.
echo ============================================================

"%PY%" "%FIGTAB%" %RUN_ARGS%
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" (
  echo ============================================================
  echo  FAILED, exit code %RC%.
  echo  See FIGsTABs\*.log for the failing step.
  echo ============================================================
) else (
  echo ============================================================
  echo  DONE. Figures/tables/logs are in:
  echo  %CD%\FIGsTABs
  echo ============================================================
)

pause
exit /b %RC%
