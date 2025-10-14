from pathlib import Path

p = Path("routetools/plot.py")
text = p.read_text().splitlines()
for i, line in enumerate(text, start=1):
    print(f"{i:4d}: {line}")
