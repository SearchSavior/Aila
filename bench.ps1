cmd /c '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && set' | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }; Write-Host ':: oneAPI environment initialized ::' -ForegroundColor Green
cd e:\RiderProjects\Aila\build
.\Aila.exe -m ..\Qwen3-0.6B --bench --bench-pp 512 --bench-tg 512 | Tee-Object -FilePath "e:\RiderProjects\Aila\bench_log.txt"
cd e:\RiderProjects\Aila