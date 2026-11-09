#1. What This Folder Contains

	1. `Yue_Du.py`: solver and analysis library.
	2. `core_numba.py`: numba-accelerated computational kernels.
	3. `rerun.py`: a driver script that reads a pre-stored reference configuration (JSON), sets up the grid / initial data / analysis parameters, invokes the solver in `Yue_Du`, and writes the result to a specified output directory.
	4. `FigTab/`: all scripts that generate the paper's figures and tables from the computed results.
	5. `run_from_zero.bat`: one-click batch driver that computes all 26 solutions from scratch.
	6. `requirements.txt`: Python package requirements for the computation.
	7. `check.py`: a self-check script that verifies the installed Python environment satisfies `requirements.txt`.
	8. `seed_configs/`: 26 `*_provenance.json` parameter recipes (configuration only; no solution data).
	9. `README.md`: this guide.
	10. `run_case_list.txt`: names and argument strings of the 26 configurations.
	11. `run_case_list.pdf`: a table with the full parameter values of all 26 configurations.

#2. How to Produce the Results and the Figures

	**Step 1. Run `check.py` and verify the environment.**

		Open a command prompt (Win+R, type `cmd`, press Enter):

		1. Run `where python` and locate the Python interpreter on which `numba`, `scipy`, etc. are installed; usually the first entry. Copy its path.
		2. Open a new cmd window and `cd /d` into the directory containing `check.py`, e.g.
   `C:\Users\Think> cd /d F:\DU1108New\ALL_NEW`
		3. Invoke that Python interpreter on `check.py`, e.g.
   `C:\Users\Think\AppData\Local\Programs\Python\Python312\python.exe check.py`
		4. If the output ends with "all checks passed; you can run rerun.py / run_from_zero.bat", proceed to the next step. Otherwise, install the missing or upgraded packages as prompted and re-run.

	**Step 2. Run the computation.**

		The simplest option is to double-click `run_from_zero.bat`. To run inside PyCharm instead:

		1. Open `rerun.py` in PyCharm, click the run-configuration dropdown in the top-right, and choose "Edit Configurations".
		2. Click "+" in the top-left and select "Shell Script".
		3. Fill in:
   			- Name: any label, e.g. `run_all_from_zero.bat`
   			- Execute: choose "Script file"
  			 - Script path: select `run_from_zero.bat`
   			- Script arguments: leave empty
   			- Working directory: the folder containing the `.bat` (this must match `%~dp0` inside the batch file; critical)
   			- Interpreter path (Windows): `cmd.exe`
   			- Check "Run in terminal" (otherwise the trailing `pause` will hang with no visible output)
		4. Click Apply → OK. Pressing Run now is equivalent to double-clicking the `.bat`.

	**Step 3. Wait for the computation to finish.**

		This takes about 50–60 minutes. The results are written to `from_zero/`, roughly 8.6 GB.

	**Step 4. Generate the figures and tables.**

	Once the computation is complete, double-click `run_FigTab.bat` inside the `FigTab/` folder (or open and run `FigTab/FigTab.py` in PyCharm). All figures and tables are written to `FigTab/FIGsTABs/`.

#3. Running a Single Configuration Instead of All 26

	**Step 1.** Run `check.py` first and confirm the environment passes.

	**Step 2. Add a Python run configuration in PyCharm.**

		1. Open `rerun.py`, click the run-configuration dropdown, and choose "Edit Configurations".
		2. Click "+" in the top-left and select "Python" (do NOT choose "Shell Script").
		3. Fill in one configuration, using `P1_01` as an example:
   			- Name: a memorable label, e.g. `P1_01_eqzoom_base`
   			- Script path: select `rerun.py` (this path is identical for all 26 configurations)
   			- Parameters: paste the corresponding argument string, e.g. `seed_configs\eqzoom_v2 base_coarse from_zero\eqzoom_v2`
   			- Working directory: the project root that contains `seed_configs/`, `rerun.py`, etc. (identical for all 26 configurations)
  			 - Python interpreter: the interpreter that has `numba` and the other dependencies
   			- Leave "Environment variables", "Redirect input", and "Add content roots" at their defaults.
   			Click Apply; do not close the window yet.
		4. Build the remaining 25 configurations by duplicating this one. Select `P1_01_eqzoom_base` in the left panel, click the duplicate icon (two small squares) at the top of the window. The new configuration inherits script path, working directory, and interpreter automatically; only two fields need to be changed:
   			- Name (e.g. `P1_02_eqzoom_pert`)
   			- Parameters (paste the next argument string from `run_case_list.txt`)
   			Click Apply.
		5. Repeat until all 26 configurations are created: 8 Phase-1 entries (`P1_01`–`P1_08`) and 18 Phase-2 entries (`P2_01`–`P2_18`). Click OK to close the dialog.

	**Step 3. Run order is strict and must not be permuted.**

		- Run the 8 Phase-1 configurations sequentially. Wait for each one to finish, confirm that the corresponding folder and result files appear under `from_zero/`, before launching the next.
		- Only after all 8 Phase-1 runs are done, run the 18 Phase-2 configurations. Every Phase-2 run restarts from a `.npz` / `.json` produced by Phase-1; running it early will fail with a missing restart file.
		- If any run errors, stop immediately and fix the path / configuration issue before proceeding.

	**Step 4. To switch configurations, pick the next one in the dropdown and press Run.**


#4. Additional Notes

	1. `seed_configs/` contains only parameters (grid, initial data, analysis); no solution data is stored there.
	2. The program first computes 8 base solutions and then 18 zoom-in solutions.
	3. Do not delete or modify any files inside the folder beyond the operations described above.
