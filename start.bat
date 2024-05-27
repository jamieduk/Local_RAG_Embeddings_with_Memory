@echo off
python -m venv myenv
venv\Scripts\activate

rem Run the Python script
python localragV8.py

rem Deactivate the virtual environment after script execution (optional)
rem  deactivate

