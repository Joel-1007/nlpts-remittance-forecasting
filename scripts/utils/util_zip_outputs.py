import zipfile
import os
import shutil

output_dir = '/kaggle/working'
zip_path = '/kaggle/working/NLPTS_outputs.zip'

# File extensions to include
include_ext = {'.csv', '.png', '.json', '.keras'}

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in sorted(os.listdir(output_dir)):
        if any(fname.endswith(ext) for ext in include_ext):
            full_path = os.path.join(output_dir, fname)
            zf.write(full_path, fname)
            print(f'  ✓ {fname}')

size_mb = os.path.getsize(zip_path) / (1e6)
print(f'\nZip saved: {zip_path}')
print(f'Size: {size_mb:.1f} MB')