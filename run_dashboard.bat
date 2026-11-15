@echo off
echo ========================================================
echo Starting MultiSentinel-G Research Dashboard...
echo ========================================================
cd /d "%~dp0"
if exist .venv\Scripts\streamlit.exe (
    .venv\Scripts\streamlit.exe run app.py
) else (
    uv run streamlit run app.py
)
pause
