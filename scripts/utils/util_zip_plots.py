"""
ZIP PLOTS FOR DOWNLOAD
Zips the plots/ folder (with main_paper/ and supplementary/ subfolders)
and saves the zip to /kaggle/working/ so you can download it directly
from the Kaggle output panel.
"""

import os
import zipfile
from datetime import datetime

PLOTS_DIR  = '/kaggle/working/plots'
OUTPUT_ZIP = '/kaggle/working/NLPTS_plots.zip'

def zip_plots():
    if not os.path.exists(PLOTS_DIR):
        print(f"❌ Plots folder not found: {PLOTS_DIR}")
        print("   Run NLPTS_PUBLICATION_PLOTS_v2.py first to generate the figures.")
        return

    # Collect all files
    all_files = []
    for root, dirs, files in os.walk(PLOTS_DIR):
        for f in files:
            if f.endswith('.png'):
                all_files.append(os.path.join(root, f))

    if not all_files:
        print("❌ No PNG files found in plots/")
        return

    print(f"Found {len(all_files)} PNG files. Zipping...")

    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(all_files):
            # Archive name preserves subfolder structure: plots/main_paper/Fig1...
            arcname = os.path.relpath(fpath, '/kaggle/working')
            zf.write(fpath, arcname)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  + {arcname}  ({size_kb:.0f} KB)")

    zip_size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print(f"\n✅ Done: {OUTPUT_ZIP}")
    print(f"   Total files : {len(all_files)}")
    print(f"   Zip size    : {zip_size_mb:.1f} MB")
    print(f"\n📥 Download from Kaggle:")
    print(f"   Output panel → NLPTS_plots.zip → Download")

if __name__ == '__main__':
    zip_plots()