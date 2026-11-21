import sys
import runpy
from pathlib import Path

# Add workspace root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute Streamlit App entrypoint in __main__ scope to prevent import lock deadlocks
app_path = ROOT_DIR / "ui" / "streamlit_app.py"
runpy.run_path(str(app_path), run_name="__main__")
