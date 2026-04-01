# Aila, a pure LLM inference engine built with SYCL + oneDNN
## Model Support
Currently it just support Qwen3-0.6B BF16
## Verified Devices
GPU:
>Intel A770 16G
## Build
Use build.bat to build on windows, it automatically initialize all required environment variables for OneAPI(cmake needs those headers to build the project).
## Run
Use run.bat to build on windows, it automatically initialize all required environment variables for OneAPI(the exectuable needs those OneAPI dlls to run).