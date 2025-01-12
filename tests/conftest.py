import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
