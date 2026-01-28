@echo off
REM Build the spcpp runner for Windows
cd /d "%~dp0.."

if not exist bin mkdir bin

cl.exe /O2 /EHsc /std:c++17 src\spcpp_runner.cpp /Fe:bin\spcpp.exe

echo Built: bin\spcpp.exe
