@echo off
cd /d "%~dp0"
echo Starting Flask application...
poetry run python app.py
pause
