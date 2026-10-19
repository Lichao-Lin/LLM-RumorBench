# 本地部署 DeepSeek

示例模型为 **DeepSeek-R1-Distill-Qwen-1.5B**，使用 Ollama 标签
`deepseek-r1:1.5b`（量化权重约 1.1 GB）。这是基于 Qwen 的蒸馏模型，
适合演示；与完整版 DeepSeek-R1、网页端或论文中的模型不能直接等同。
推理还需要额外内存，实际占用随上下文和并发变化；CPU 可运行，速度取决于硬件。

## 启动与请求

先安装 Docker Engine + Compose 或 Docker Desktop（Linux 容器模式），并启动 Docker。
以下命令都在**仓库根目录**执行：

```bash
# 1. 启动本地模型服务，等待服务健康检查通过。
docker compose -f deploy/local-llm/compose.yaml up -d --wait

# 2. 第一次需要联网下载模型；模型保存在 Docker 卷内。
docker compose -f deploy/local-llm/compose.yaml exec -T ollama ollama pull deepseek-r1:1.5b

# 3. 用 Python 请求本机服务；可选保存原始请求和完整响应。
python deploy/local-llm/chat.py "请解释谣言识别为什么需要核对来源。" --output outputs/local-answer.json

# 4. 查看运行状态和已下载模型；研究时保留模型 ID 与服务版本。
docker compose -f deploy/local-llm/compose.yaml exec -T ollama ollama list
docker compose -f deploy/local-llm/compose.yaml exec -T ollama ollama --version

# 5. 停止服务，保留下载的模型。
docker compose -f deploy/local-llm/compose.yaml down
```

服务地址为 `http://127.0.0.1:11434`，默认只允许本机访问。另一台电脑访问服务器时，
可先建立 SSH 隧道：`ssh -N -L 11434:127.0.0.1:11434 user@server`，
再在自己的电脑运行同一个 Python 请求命令。

响应太慢可加 `--timeout 600`；输出被截断可加 `--max-tokens 4096`。
输出文件已存在时换一个文件名。端口被占用时，修改 Compose 映射左侧端口，
再用 `--base-url http://127.0.0.1:新端口` 调用。

## 可选 NVIDIA GPU

先配置 NVIDIA 驱动和容器 GPU 支持，再替换启动命令：

```bash
docker compose -f deploy/local-llm/compose.yaml -f deploy/local-llm/compose.gpu.yaml up -d --wait
```

Windows 需要支持 GPU 的 Docker Desktop/WSL2；Linux 需要 NVIDIA Container Toolkit。
后续停止也使用同样两个 `-f` 参数。未配置 GPU 时使用默认 CPU 文件即可。

## 与研究评估的关系

`chat.py` 只演示一次请求，不自动实施搜索/思考四条件，也不自动把回答变成标签。
正式实验应固定提示词、模型 ID 与工具设置，保存完整响应，按事先约定的规则提取标签，
再整理为主 README 的六列数据后运行 `python -m rumorbench`。
模型自己的思考过程不能当作“思考开关”的受控实验。

DeepSeek 已公开该模型权重；使用前阅读其模型卡和许可说明。
Ollama 的量化包由 Ollama 模型库提供。模型标签可能更新，研究记录需要保留实际模型 ID。

来源：[DeepSeek 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)、
[Ollama 模型说明](https://ollama.com/library/deepseek-r1:1.5b)、
[Docker 部署](https://docs.ollama.com/docker)、
[兼容接口](https://docs.ollama.com/api/openai-compatibility)。
