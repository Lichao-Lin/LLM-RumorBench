"""调用本地 DeepSeek 服务；只用 Python 标准库，无需安装额外 SDK。"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def chat(prompt, *, base_url="http://127.0.0.1:11434",
         model="deepseek-r1:1.5b", timeout=300, max_tokens=1024):
    """返回原始 API 响应。它可能含推理过程，需要按实验规则另行解析标签。"""
    if not prompt.strip():
        raise ValueError("问题不能为空")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout 必须是大于 0 的有限数值")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens 必须是大于 0 的整数")
    # 使用 OpenAI 兼容格式；这里访问的是本地 Ollama，不是云端收费 API。
    payload = {
        "model": model,
        # R1 示例将任务说明放在用户消息中，不额外设置 system 提示。
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,       # 一次返回完整 JSON，初学者不必处理流式分片。
        "temperature": 0.6,    # 采样温度影响输出随机性；不是实验条件的开关。
        "top_p": 0.95,
        "max_tokens": max_tokens,  # 包含推理所需的生成预算，过小可能被截断。
    }
    request = Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        # CPU 首次加载模型可能较慢；超时可在命令行调整。
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"本地服务返回 HTTP {error.code}：{detail}；请检查模型是否已下载") from error
    except (URLError, TimeoutError, socket.timeout) as error:
        raise RuntimeError(
            f"连接本地模型失败：{error}；请检查服务、端口或增加 --timeout") from error
    # 不只检查字段存在，还检查类型，让异常响应得到中文提示而不是堆栈报错。
    choices = result.get("choices") if isinstance(result, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("服务响应的 choices 必须是非空对象列表")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("服务响应缺少有效的 message 对象")
    if message.get("content") is not None and not isinstance(message["content"], str):
        raise ValueError("服务响应的 content 必须是文本或 null")
    return {
        "requested_model": model,
        "base_url": base_url,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "request": payload,    # 保留原始提示词与参数，方便以后核对实验。
        "response": result,    # 保留完整响应、token 用量与截断原因。
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="给模型的问题；含空格时使用引号")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="deepseek-r1:1.5b")
    parser.add_argument("--timeout", type=float, default=300, help="等待响应的秒数")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path, help="可选：保存原始请求和响应到新 JSON 文件")
    args = parser.parse_args()
    try:
        # 提前检查，避免推理结束后才发现目标文件不能保存。
        if args.output:
            if args.output.exists():
                raise ValueError("输出路径已存在，请换一个文件名")
            args.output.parent.mkdir(parents=True, exist_ok=True)
        record = chat(args.prompt, base_url=args.base_url, model=args.model,
                      timeout=args.timeout, max_tokens=args.max_tokens)
        # content 不一定是最终标签；length 表示生成可能被预算截断。
        choice = record["response"]["choices"][0]
        content = choice["message"].get("content")
        print(content or "提示：模型未返回最终正文；原始响应可能仅含推理内容，请查看保存的记录。")
        if choice.get("finish_reason") == "length":
            print("提示：输出达到 token 上限，可增大 --max-tokens 再试。")
        if args.output:
            # x 模式拒绝覆盖已有实验记录。
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"错误：{error}\n")


if __name__ == "__main__":
    main()
