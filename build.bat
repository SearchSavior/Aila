@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
cd /d e:\RiderProjects\Aila\build
cmake --build . --config Release