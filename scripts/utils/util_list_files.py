import os
from pathlib import Path

# List all files in /kaggle/working/ and /kaggle/input/
for base in ["/kaggle/working", "/kaggle/input"]:
    p = Path(base)
    if p.exists():
        print(f"\n{'='*60}")
        print(f"  {base}")
        print(f"{'='*60}")
        for f in sorted(p.rglob("*")):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"  {str(f):<70}  {size_kb:>8.1f} KB")
    else:
        print(f"\n{base} does not exist")
