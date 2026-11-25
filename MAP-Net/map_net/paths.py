"""Default paths for the standalone network project."""
from pathlib import Path

# 由本文件的位置反推工程目录，而不是使用当前工作目录（os.getcwd）。
PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = PROJECT_DIR.parent / "dataset"

DEFAULT_OUTPUT_DIR = PROJECT_DIR / "runs" / "map_net"
DEFAULT_CHECKPOINT = DEFAULT_OUTPUT_DIR / "best.pt"
