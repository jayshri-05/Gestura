@echo off
setlocal
cd /d "%~dp0frontend"
if not exist node_modules (
  echo Installing frontend dependencies. This may take a few minutes...
  call npm install
  if errorlevel 1 (
    echo Dependency installation failed. Check that Node.js and npm are installed.
    pause
    exit /b 1
  )
)
echo.
echo GESTURA preview is starting. Open the Local URL printed below in your browser.
echo Keep this window open while you review the site.
echo.
call npm run dev -- --host 127.0.0.1
pause
