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

rem Use the Python that has numpy/scipy/numba/matplotlib.
rem If "python" is not that interpreter, set the full path:
rem set "PY=D:\Python\python.exe"
set "PY=python"

if not exist rerun.py (
  echo.
  echo ERROR: rerun.py not found. Put this .bat next to Yue_Du.py / rerun.py.
  pause
  exit 1
)
if not exist seed_configs (
  echo.
  echo ERROR: seed_configs\ not found.
  echo Unzip the code package completely, then run this file again.
  pause
  exit 1
)
if not exist from_zero mkdir from_zero

echo ============================================================
echo  Build ALL raw solutions from zero
echo  No precomputed .npz is used.
echo  python = %PY%
echo  Output: %CD%\from_zero
echo  This takes about 50-60 minutes. Do not close this window.
echo ============================================================

rem ---------- Phase 1: independent runs (from initial data) ----------
call :run3 seed_configs\eqzoom_v2 base_coarse from_zero\eqzoom_v2
call :run3 seed_configs\eqzoom_v2 pert_coarse from_zero\eqzoom_v2
call :run3 seed_configs\scan_eps-1_v2 pert_coarse from_zero\scan_eps-1_v2
call :run3 seed_configs\scan_eps0.5_v2 pert_coarse from_zero\scan_eps0.5_v2
call :run3 seed_configs\scan_eps2_v2 pert_coarse from_zero\scan_eps2_v2
call :run3 seed_configs\full_small_v2 base_coarse from_zero\full_small_v2
call :run3 seed_configs\full_small_v2_h005 base_coarse from_zero\full_small_v2_h005
call :run3 seed_configs\family_bump_eps0.5_v2 base_coarse from_zero\family_bump_eps0.5_v2

rem ---------- Phase 2: restart runs from the NEW parents ----------
call :run6 seed_configs\zoom_base_h1e3_v2 single from_zero\zoom_base_h1e3_v2 "grid.restart_file=from_zero\eqzoom_v2" "grid.restart_tag=base_coarse" "grid.restart_u=1.28"
call :run6 seed_configs\zoom_pert_h1e3_v2 single from_zero\zoom_pert_h1e3_v2 "grid.restart_file=from_zero\eqzoom_v2" "grid.restart_tag=pert_coarse" "grid.restart_u=1.28"
call :run6 seed_configs\zoom_bump_eps0.5_base_v3 single from_zero\zoom_bump_eps0.5_base_v3 "grid.restart_file=from_zero\family_bump_eps0.5_v2" "grid.restart_tag=base_coarse" "grid.restart_u=1.28"
call :run6 seed_configs\zoom_eps-1_pert_v3 single from_zero\zoom_eps-1_pert_v3 "grid.restart_file=from_zero\scan_eps-1_v2" "grid.restart_tag=pert_coarse" "grid.restart_u=1.28"
call :run6 seed_configs\zoom_eps0.5_pert_v3 single from_zero\zoom_eps0.5_pert_v3 "grid.restart_file=from_zero\scan_eps0.5_v2" "grid.restart_tag=pert_coarse" "grid.restart_u=1.28"
call :run6 seed_configs\zoom_eps2_pert_v3 single from_zero\zoom_eps2_pert_v3 "grid.restart_file=from_zero\scan_eps2_v2" "grid.restart_tag=pert_coarse" "grid.restart_u=1.28"

call :run6 seed_configs\zoom_base_h5e4b_v2 single from_zero\zoom_base_h5e4b_v2 "grid.restart_file=from_zero\eqzoom_v2" "grid.restart_tag=base_coarse" "grid.restart_u=1.28"
call :run6 seed_configs\zoom_pert_h5e4b_v2 single from_zero\zoom_pert_h5e4b_v2 "grid.restart_file=from_zero\eqzoom_v2" "grid.restart_tag=pert_coarse" "grid.restart_u=1.28"
call :run6 seed_configs\zoom_bump_eps0.5_h5e4_v2 single from_zero\zoom_bump_eps0.5_h5e4_v2 "grid.restart_file=from_zero\family_bump_eps0.5_v2" "grid.restart_tag=base_coarse" "grid.restart_u=1.28"
call :run6 seed_configs\z5_eps-1_v2 single from_zero\z5_eps-1_v2 "grid.restart_file=from_zero\scan_eps-1_v2" "grid.restart_tag=pert_coarse" "grid.restart_u=1.28"
call :run6 seed_configs\z5_eps0.5_v2 single from_zero\z5_eps0.5_v2 "grid.restart_file=from_zero\scan_eps0.5_v2" "grid.restart_tag=pert_coarse" "grid.restart_u=1.28"
call :run6 seed_configs\z5_eps2_v2 single from_zero\z5_eps2_v2 "grid.restart_file=from_zero\scan_eps2_v2" "grid.restart_tag=pert_coarse" "grid.restart_u=1.28"

call :run6 seed_configs\h4_v5_x2 single from_zero\h4_v5_x2 "grid.restart_file=from_zero\eqzoom_v2" "grid.restart_tag=base_coarse" "grid.restart_u=1.28"
call :run6 seed_configs\h4_v5_x2_pert single from_zero\h4_v5_x2_pert "grid.restart_file=from_zero\eqzoom_v2" "grid.restart_tag=pert_coarse" "grid.restart_u=1.28"
call :run6 seed_configs\zoom_bump_eps0.5_h25e5_v2 single from_zero\zoom_bump_eps0.5_h25e5_v2 "grid.restart_file=from_zero\family_bump_eps0.5_v2" "grid.restart_tag=base_coarse" "grid.restart_u=1.28"
call :run6 seed_configs\h25_eps-1 single from_zero\h25_eps-1 "grid.restart_file=from_zero\scan_eps-1_v2" "grid.restart_tag=pert_coarse" "grid.restart_u=1.28"
call :run6 seed_configs\h25_eps0.5 single from_zero\h25_eps0.5 "grid.restart_file=from_zero\scan_eps0.5_v2" "grid.restart_tag=pert_coarse" "grid.restart_u=1.28"
call :run6 seed_configs\h25_eps2 single from_zero\h25_eps2 "grid.restart_file=from_zero\scan_eps2_v2" "grid.restart_tag=pert_coarse" "grid.restart_u=1.28"

echo.
echo ============================================================
echo  ALL 26 RUNS DONE
echo  Output: %CD%\from_zero
echo ============================================================
pause
exit /b 0

:run3
echo.
echo === %1  %2  TO  %3 ===
"%PY%" rerun.py %1 %2 %3
if errorlevel 1 (
  echo.
  echo FAILED: %1 %2
  goto :fail
)
exit /b 0

:run6
echo.
echo === %1  %2  TO  %3 ===
"%PY%" rerun.py %1 %2 %3 %4 %5 %6
if errorlevel 1 (
  echo.
  echo FAILED: %1 %2
  goto :fail
)
exit /b 0

:fail
echo.
echo ============================================================
echo  ABORTED. Fix the error above, then run this script again.
echo ============================================================
pause
exit 1