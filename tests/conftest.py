# Always add the project root (folder that contains `src/`) to sys.path
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional sanity check during collection:
assert (ROOT / "src").exists(), f"'src' not found at {ROOT}"

