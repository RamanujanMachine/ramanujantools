import sys
import warnings
from packaging.version import parse

current = sys.argv[1].lstrip("v")
latest = sys.argv[2].lstrip("v")

curr = parse(current)
last = parse(latest)

if curr < last:
    sys.exit(f"Tag v{current} is older than latest tag v{latest}")

if curr == last:
    warnings.warn(
        f"Tag v{current} is the same as latest tag v{latest}. "
        "Deployment will fail if this version has already been deployed."
    )
