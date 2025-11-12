import zipfile
import os
from pathlib import Path

def create_layer_zip():
    zip_name = 'python-multipart-layer.zip'
    
    print(f"Creating {zip_name}...")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the python directory
        for root, dirs, files in os.walk('python'):
            for file in files:
                file_path = os.path.join(root, file)
                # Add to zip with relative path
                arcname = file_path
                zipf.write(file_path, arcname)
                
    print(f"✅ {zip_name} created successfully!")
    print(f"Size: {os.path.getsize(zip_name) / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    create_layer_zip()