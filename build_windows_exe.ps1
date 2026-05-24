# Build one-file Windows executable for non-developer users.
# Run this script in PowerShell on Windows.

python -m pip install --upgrade pip
pip install pyinstaller ultralytics opencv-python numpy

pyinstaller --onefile --windowed --name FishCounter fish_counter_exe_launcher.py

Write-Host "Build complete: dist/FishCounter.exe"
Write-Host "Distribute with your model file at: training_data/best.pt"
