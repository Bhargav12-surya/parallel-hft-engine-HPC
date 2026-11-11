@echo off
setlocal
cd /d "%~dp0.."
cmake --build build --config Release --parallel
if errorlevel 1 exit /b 1
ctest --test-dir build -C Release --output-on-failure
if errorlevel 1 exit /b 1
cpack --config build\CPackConfig.cmake -C Release
