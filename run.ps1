cmd /c '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && set' | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }; Write-Host ':: oneAPI environment initialized ::' -ForegroundColor Green
cd e:\RiderProjects\Aila\build
"Introduce yourself", "What is 1 + 2?", "Explain AI in few sentences", "/quit" | .\Aila.exe | Tee-Object -FilePath "e:\RiderProjects\Aila\run_log.txt"
cd e:\RiderProjects\Aila