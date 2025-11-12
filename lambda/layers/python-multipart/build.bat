@echo off

echo Cleaning previous build...
if exist python rmdir /s /q python
if exist *.zip del /q *.zip

echo Creating directory structure...
mkdir python\lib\python3.12\site-packages

echo Installing dependencies...
pip install -r requirements.txt -t python\lib\python3.12\site-packages

echo Creating ZIP file with Python...
python create_zip.py

echo.
echo ✅ Layer created: python-multipart-layer.zip
echo Upload this file to AWS Lambda Layers
pause