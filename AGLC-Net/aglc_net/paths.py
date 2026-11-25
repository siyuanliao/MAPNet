"""Default paths for the standalone network project."""
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR.parent / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "runs" / "aglc_net"
DEFAULT_CHECKPOINT = DEFAULT_OUTPUT_DIR / "best.pt"
