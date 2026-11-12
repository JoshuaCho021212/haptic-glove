@echo off
setlocal
set PI_USER=joshuacho
set PI_IP=10.37.87.186
set PI_NAME=raspberrypi

set PI_HOST=%PI_IP%
if not "%PI_NAME%"=="" (
  ping -n 1 -w 1500 %PI_NAME%.local >nul 2>&1 && set PI_HOST=%PI_NAME%.local
)

echo Connecting to Haptic Glove at %PI_HOST% ...
start "Haptic Glove connection - keep this window open" "C:\Windows\System32\OpenSSH\ssh.exe" -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ConnectTimeout=8 -L 5005:127.0.0.1:5005 %PI_USER%@%PI_HOST%
timeout /t 4 /nobreak >nul
start "" http://localhost:5005
endlocal
