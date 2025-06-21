import sys
from packaging.version import parse

current = sys.argv[1].lstrip("v")
latest = sys.argv[2].lstrip("v")

curr = parse(current)
last = parse(latest)

if curr <= last:
    sys.exit(f"Tag v{current} is not newer than latest tag v{latest}")
