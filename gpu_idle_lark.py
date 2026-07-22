#!/usr/bin/env python3
"""检测空闲 GPU，并通过飞书自定义机器人发送通知。

参考: https://docs.swanlab.cn/plugin/notification-lark.html
改好下方默认参数后直接运行: python gpu_idle_lark.py
"""

from __future__ import annotations

# ======================== 默认参数（直接改这里）========================
LARK_WEBHOOK_URL = ""
LARK_SECRET = ""  # 未开启签名校验可留空 ""
INTERVAL = 10.0  # 检测间隔（秒）
UTIL_THRESHOLD = 10  # 空闲判定：GPU 利用率上限（%）
MEM_THRESHOLD = 1024  # 空闲判定：已用显存上限（MiB）
CONFIRM_CYCLES = 2  # 连续多少个周期状态相同才确认空闲/忙碌
DRY_RUN = False  # True: 只打印消息，不真正发送飞书
# ================================================
======================

import argparse
import base64
import hashlib
import hmac
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set, Tuple

import requests

try:
    import pynvml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 pynvml: pip install nvidia-ml-py") from exc


@dataclass(frozen=True)
class GpuStatus:
    index: int
    name: str
    util: int
    mem_used_mib: int
    mem_total_mib: int

    @property
    def mem_free_mib(self) -> int:
        return self.mem_total_mib - self.mem_used_mib

    def is_idle(self, util_threshold: int, mem_threshold: int) -> bool:
        return self.util <= util_threshold and self.mem_used_mib <= mem_threshold

    def summary(self) -> str:
        return (
            f"GPU {self.index} ({self.name}): "
            f"利用率 {self.util}%, "
            f"显存 {self.mem_used_mib}/{self.mem_total_mib} MiB "
            f"(空闲 {self.mem_free_mib} MiB)"
        )


class LarkBot:
    """飞书自定义机器人，签名逻辑与 SwanLab LarkCallback 一致。"""

    def __init__(self, webhook_url: str, secret: Optional[str] = None):
        self.webhook_url = webhook_url
        self.secret = secret

    def _gen_sign(self, timestamp: int) -> str:
        if not self.secret:
            raise ValueError("secret is required for signing")
        string_to_sign = f"{timestamp}\n{self.secret}"
        hmac_code = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
        return base64.b64encode(hmac_code).decode("utf-8")

    def send_msg(self, content: str, timeout: float = 10.0) -> None:
        timestamp = int(time.time())
        payload = {
            "timestamp": timestamp,
            "msg_type": "text",
            "content": {"text": content},
        }
        if self.secret:
            payload["sign"] = self._gen_sign(timestamp)

        resp = requests.post(self.webhook_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()
        code = result.get("code", result.get("StatusCode", 0))
        if code not in (0, None):
            raise RuntimeError(f"飞书发送失败: {result}")


class StateConfirmer:
    """连续 N 个周期观测到相同状态后，才更新确认状态。"""

    def __init__(self, confirm_cycles: int = 2):
        self.confirm_cycles = confirm_cycles
        # index -> (当前连续观测状态, 连续次数)
        self._streak: Dict[int, Tuple[bool, int]] = {}
        # index -> 已确认是否空闲；未出现过的 GPU 视为未知
        self._confirmed: Dict[int, bool] = {}

    def update(self, raw_idle_by_index: Dict[int, bool]) -> Tuple[Set[int], Set[int], Set[int]]:
        """根据本周期原始观测更新确认状态。

        Returns:
            (confirmed_idle, became_idle, became_busy)
        """
        prev_idle = {idx for idx, idle in self._confirmed.items() if idle}

        for idx, raw_idle in raw_idle_by_index.items():
            prev = self._streak.get(idx)
            if prev is not None and prev[0] == raw_idle:
                streak = prev[1] + 1
            else:
                streak = 1
            self._streak[idx] = (raw_idle, streak)
            if streak >= self.confirm_cycles:
                self._confirmed[idx] = raw_idle

        confirmed_idle = {idx for idx, idle in self._confirmed.items() if idle}
        became_idle = confirmed_idle - prev_idle
        became_busy = prev_idle - confirmed_idle
        return confirmed_idle, became_idle, became_busy


def query_gpus() -> List[GpuStatus]:
    pynvml.nvmlInit()
    try:
        count = pynvml.nvmlDeviceGetCount()
        gpus: List[GpuStatus] = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                GpuStatus(
                    index=i,
                    name=name,
                    util=int(util),
                    mem_used_mib=int(mem.used / 1024 / 1024),
                    mem_total_mib=int(mem.total / 1024 / 1024),
                )
            )
        return gpus
    finally:
        pynvml.nvmlShutdown()


def _header(title: str) -> List[str]:
    return [
        title,
        f"主机: {socket.gethostname()}",
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]


def format_idle_message(
    became_idle: Sequence[GpuStatus],
    confirmed_idle: Sequence[GpuStatus],
    all_gpus: Sequence[GpuStatus],
) -> str:
    lines = _header("🟢 GPU 空闲通知")
    lines.append(f"新空闲: {', '.join(f'GPU {g.index}' for g in became_idle)}")
    lines.append(f"当前空闲卡数: {len(confirmed_idle)}/{len(all_gpus)}")
    lines.append("")
    lines.append("新空闲 GPU:")
    for gpu in became_idle:
        lines.append(f"  - {gpu.summary()}")
    if confirmed_idle:
        lines.append("")
        lines.append("当前全部空闲 GPU:")
        for gpu in confirmed_idle:
            lines.append(f"  - {gpu.summary()}")
    busy = [g for g in all_gpus if g.index not in {x.index for x in confirmed_idle}]
    if busy:
        lines.append("")
        lines.append("忙碌 GPU:")
        for gpu in busy:
            lines.append(f"  - {gpu.summary()}")
    return "\n".join(lines)


def format_busy_message(
    became_busy: Sequence[GpuStatus],
    confirmed_idle: Sequence[GpuStatus],
    all_gpus: Sequence[GpuStatus],
) -> str:
    lines = _header("🔴 GPU 占用通知")
    lines.append(f"新占用: {', '.join(f'GPU {g.index}' for g in became_busy)}")
    lines.append(f"当前空闲卡数: {len(confirmed_idle)}/{len(all_gpus)}")
    lines.append("")
    lines.append("新占用 GPU:")
    for gpu in became_busy:
        lines.append(f"  - {gpu.summary()}")
    lines.append("")
    lines.append("当前仍空闲 GPU:")
    if confirmed_idle:
        for gpu in confirmed_idle:
            lines.append(f"  - {gpu.summary()}")
    else:
        lines.append("  - 无")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检测空闲 GPU 并发送飞书通知")
    parser.add_argument("--webhook-url", default=LARK_WEBHOOK_URL, help="飞书机器人 Webhook")
    parser.add_argument("--secret", default=LARK_SECRET, help="飞书机器人签名")
    parser.add_argument("--interval", type=float, default=INTERVAL, help="检测间隔秒数")
    parser.add_argument(
        "--util-threshold",
        type=int,
        default=UTIL_THRESHOLD,
        help="判定空闲的 GPU 利用率上限（%%）",
    )
    parser.add_argument(
        "--mem-threshold",
        type=int,
        default=MEM_THRESHOLD,
        help="判定空闲的已用显存上限（MiB）",
    )
    parser.add_argument(
        "--confirm-cycles",
        type=int,
        default=CONFIRM_CYCLES,
        help="连续多少个周期状态相同才确认",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=DRY_RUN,
        help="只打印检测结果，不发送飞书消息",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run and (not args.webhook_url or "xxxx" in args.webhook_url):
        print("请在文件开头设置 LARK_WEBHOOK_URL（或通过 --webhook-url 传入）。", file=sys.stderr)
        return 1
    if args.confirm_cycles < 1:
        print("--confirm-cycles 必须 >= 1", file=sys.stderr)
        return 1

    secret = args.secret if args.secret and args.secret != "xxxx" else None
    bot = None if args.dry_run else LarkBot(args.webhook_url, secret)
    confirmer = StateConfirmer(confirm_cycles=args.confirm_cycles)

    def log(text: str = "", *, err: bool = False) -> None:
        print(text, file=sys.stderr if err else sys.stdout, flush=True)

    def notify(msg: str) -> None:
        if args.dry_run:
            log("--- dry-run 消息 ---")
            log(msg)
            log("--------------------")
        else:
            assert bot is not None
            bot.send_msg(msg)
            log("飞书通知已发送")

    log(
        f"开始监控 GPU | 间隔 {args.interval}s | "
        f"确认周期 {args.confirm_cycles} | "
        f"空闲条件: util<={args.util_threshold}% 且 mem_used<={args.mem_threshold}MiB"
    )

    while True:
        try:
            gpus = query_gpus()
            gpu_map = {g.index: g for g in gpus}
            raw_idle = {
                g.index: g.is_idle(args.util_threshold, args.mem_threshold) for g in gpus
            }
            confirmed_idle_ids, became_idle_ids, became_busy_ids = confirmer.update(raw_idle)
            stamp = datetime.now().strftime("%H:%M:%S")

            raw_idle_ids = sorted(i for i, idle in raw_idle.items() if idle)
            log(
                f"[{stamp}] 原始空闲={raw_idle_ids} | "
                f"确认空闲={sorted(confirmed_idle_ids)} | "
                f"新空闲={sorted(became_idle_ids)} | "
                f"新占用={sorted(became_busy_ids)}"
            )

            confirmed_idle = [gpu_map[i] for i in sorted(confirmed_idle_ids) if i in gpu_map]
            if became_idle_ids:
                became_idle = [gpu_map[i] for i in sorted(became_idle_ids) if i in gpu_map]
                notify(format_idle_message(became_idle, confirmed_idle, gpus))
            if became_busy_ids:
                became_busy = [gpu_map[i] for i in sorted(became_busy_ids) if i in gpu_map]
                notify(format_busy_message(became_busy, confirmed_idle, gpus))
        except KeyboardInterrupt:
            log("\n已停止监控")
            return 0
        except Exception as exc:
            log(f"[{datetime.now().strftime('%H:%M:%S')}] 检测/发送失败: {exc}", err=True)

        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            log("\n已停止监控")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
