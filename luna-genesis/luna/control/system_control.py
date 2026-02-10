"""Basic system control introspection commands."""

from __future__ import annotations

import platform
import shutil


def handle_system_command(user_input: str) -> str:
    text = (user_input or "").lower()
    if "status" in text:
        total, used, free = shutil.disk_usage("/")
        return (
            f"System: {platform.system()} {platform.release()} | "
            f"Disk Total: {total // (1024**3)} GB, Used: {used // (1024**3)} GB, "
            f"Free: {free // (1024**3)} GB"
        )
    if "restart" in text or "shutdown" in text:
        return "Safety policy: refusing destructive system command from assistant layer."
    return "System control supports: status, restart, shutdown (restricted)."
