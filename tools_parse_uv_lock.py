import pathlib
import tomllib

p = pathlib.Path("uv.lock")
if not p.exists():
    print("uv.lock not found")
    raise SystemExit(1) from None

try:
    with p.open("rb") as f:
        tomllib.load(f)
    print("PARSED_OK")
except tomllib.TOMLDecodeError as e:
    print("TOML_ERROR")
    print(str(e))
    # attributes: lineno, colno
    lineno = getattr(e, "lineno", None)
    colno = getattr(e, "colno", None)
    print("lineno:", lineno, "colno:", colno)
    text = p.read_text().splitlines()
    if lineno is not None:
        idx = lineno - 1
        start = max(0, idx - 5)
        end = min(len(text), idx + 5)
        print("\n--- surrounding lines ---")
        for i in range(start, end):
            print(f"{i + 1:5d}: {text[i]}")
    raise SystemExit(2) from None
