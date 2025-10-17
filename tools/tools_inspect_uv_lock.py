import tomllib
from pathlib import Path

p = Path("uv.lock")
with p.open("rb") as f:
    data = tomllib.load(f)

packages = data.get("package", [])
print(f"total packages: {len(packages)}")
missing_version = []
for i, pkg in enumerate(packages):
    name = pkg.get("name")
    version = pkg.get("version")
    if version is None:
        missing_version.append((i, name, pkg))

if missing_version:
    print("Packages missing version:")
    for idx, name, _pkg in missing_version:
        print(idx, name)
else:
    print("All packages have version field")
