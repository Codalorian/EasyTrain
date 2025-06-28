@echo off
setlocal enabledelayedexpansion

:: Check for PowerShell
where powershell >nul 2>&1
if %errorlevel% neq 0 (
    echo PowerShell is required to run this script
    exit /b 1
)

:: GPU Detection
echo Detecting NVIDIA GPU...
powershell.exe -Command "(Get-CimInstance -ClassName Win32_VideoController).Name" | findstr -i "NVIDIA"
if %errorlevel% neq 0 (
    echo No NVIDIA GPU detected. Exiting.
    exit /b 1
)
echo NVIDIA GPU detected

:: Package Manager Detection
set "PKG_MGR="
for /f "tokens=1,2 delims= " %%a in ('where choco 2^>nul') do if not defined PKG_MGR set "PKG_MGR=choco"
for /f "tokens=1,2 delims= " %%a in ('where winget 2^>nul') do if not defined PKG_MGR set "PKG_MGR=winget"
for /f "tokens=1,2 delims= " %%a in ('where scoop 2^>nul') do if not defined PKG_MGR set "PKG_MGR=scoop"

if not defined PKG_MGR (
    echo No supported package manager found
    echo Please install one of the following first:
    echo 1. Chocolatey (https://chocolatey.org/install)
    echo 2. Winget (https://learn.microsoft.com/en-us/windows/package-manager/winget/)
    echo 3. Scoop (https://scoop.sh/)
    exit /b 1
)
echo Using package manager: %PKG_MGR%

:: Driver and CUDA Installation
echo.
echo Installing NVIDIA drivers and CUDA toolkit...
echo This may take several minutes...

if "%PKG_MGR%"=="choco" (
    echo Installing with Chocolatey...
    choco install nvidia-display-driver -y --ignore-checksums
    choco install cuda --version=12.1.1 --yes --ignore-checksums
) else if "%PKG_MGR%"=="winget" (
    echo Installing with Winget...
    winget install --id=NVIDIA.CUDA --version=12.1.1.1
    echo NOTE: NVIDIA drivers may need to be installed manually from Device Manager
) else if "%PKG_MGR%"=="scoop" (
    echo Installing with Scoop...
    scoop bucket add nvidia https://github.com/ScoopInstaller/NirSoftBucket
    echo ERROR: CUDA toolkit not available in Scoop repositories
    echo Manual installation from NVIDIA website is recommended
    exit /b 1
)

:: Environment Setup
echo.
echo Setting up environment variables...
setx PATH "%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
setx CUDA_PATH_V12_1 "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"

:: Verification
echo.
echo Verifying installation...
nvidia-smi
nvcc --version

echo.
echo Installation complete! Please reboot your system.
echo For full functionality, consider installing:
echo 1. Visual Studio Build Tools (for CUDA compilation)
echo 2. CUDA samples from NVIDIA website

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes deepspeed
