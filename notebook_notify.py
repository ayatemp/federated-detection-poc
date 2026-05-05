"""Small Discord notification helper for long-running notebooks.

The helper uses Discord Incoming Webhooks, so it does not need a long-running
bot process. Store the webhook URL in DISCORD_WEBHOOK_URL and call
notify_discord(...) from the final notebook cell.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union
from urllib import error, parse, request


DEFAULT_WEBHOOK_ENV_VARS = (
    "DISCORD_WEBHOOK_URL",
    "DISCORD_NOTEBOOK_WEBHOOK_URL",
)
DOTENV_FILENAMES = (".env",)
DISCORD_CONTENT_LIMIT = 2000
SAFE_CHUNK_LIMIT = 1900


@dataclass(frozen=True)
class DiscordNotifyResult:
    """Result returned by notify_discord."""

    ok: bool
    chunks_sent: int
    status_codes: Sequence[int]
    dry_run: bool = False
    error: Optional[str] = None


def notify_discord(
    message: object,
    *,
    title: Optional[str] = "Notebook finished",
    webhook_url: Optional[str] = None,
    username: Optional[str] = "Notebook notifier",
    avatar_url: Optional[str] = None,
    context: Optional[Mapping[str, object]] = None,
    include_default_context: bool = True,
    allow_mentions: bool = False,
    fail_silently: bool = False,
    dry_run: bool = False,
    timeout: float = 15.0,
) -> DiscordNotifyResult:
    """Send a message to Discord from a notebook or Python script.

    Parameters
    ----------
    message:
        Text to post. Non-string values are converted with str(...).
    title:
        Optional bold heading shown before the message.
    webhook_url:
        Discord Incoming Webhook URL. If omitted, environment variables and a
        local .env file are checked.
    username:
        Optional per-message webhook display name.
    avatar_url:
        Optional per-message webhook avatar URL.
    context:
        Extra key/value lines to append, such as {"run": "dqa08"}.
    include_default_context:
        Append finished_at, host, and cwd metadata.
    allow_mentions:
        When False, Discord mentions are disabled for this message.
    fail_silently:
        Return an error result instead of raising if the notification fails.
    dry_run:
        Format and split the message without sending it.
    timeout:
        HTTP timeout in seconds.
    """

    formatted = _format_message(
        message,
        title=title,
        context=context,
        include_default_context=include_default_context,
    )
    chunks = tuple(_split_content(formatted, SAFE_CHUNK_LIMIT))

    if dry_run:
        return DiscordNotifyResult(
            ok=True,
            chunks_sent=len(chunks),
            status_codes=(),
            dry_run=True,
        )

    try:
        url = _resolve_webhook_url(webhook_url)
        status_codes = [
            _post_webhook(
                url,
                _with_part_label(chunk, index, len(chunks)),
                username=username,
                avatar_url=avatar_url,
                allow_mentions=allow_mentions,
                timeout=timeout,
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
    except Exception as exc:  # noqa: BLE001 - notebook users need one clear error
        error_message = str(exc)
        if fail_silently:
            return DiscordNotifyResult(
                ok=False,
                chunks_sent=0,
                status_codes=(),
                error=error_message,
            )
        raise RuntimeError(f"Discord notification failed: {error_message}") from exc

    ok = all(200 <= status < 300 for status in status_codes)
    return DiscordNotifyResult(
        ok=ok,
        chunks_sent=len(status_codes),
        status_codes=tuple(status_codes),
    )


def save_discord_webhook_url(
    webhook_url: str,
    *,
    env_path: Optional[Union[Path, str]] = None,
    key: str = "DISCORD_WEBHOOK_URL",
    set_environment: bool = True,
) -> Path:
    """Save a Discord webhook URL to a local .env file.

    Existing .env content is preserved. If the key already exists, only that
    line is replaced. The saved .env file is intended to be local-only and is
    ignored by this repository's .gitignore.
    """

    url = _validate_webhook_url(webhook_url)
    path = Path(env_path) if env_path is not None else _default_dotenv_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    updated_lines = []
    replaced = False
    for raw_line in lines:
        normalized = raw_line.strip()
        if normalized.startswith("export "):
            normalized = normalized.removeprefix("export ").strip()

        line_key = normalized.split("=", 1)[0].strip() if "=" in normalized else ""
        if line_key == key:
            updated_lines.append(f"{key}={url}")
            replaced = True
        else:
            updated_lines.append(raw_line)

    if not replaced:
        updated_lines.append(f"{key}={url}")

    path.write_text("\n".join(updated_lines).rstrip() + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass

    if set_environment:
        os.environ[key] = url
    return path


def _format_message(
    message: object,
    *,
    title: Optional[str],
    context: Optional[Mapping[str, object]],
    include_default_context: bool,
) -> str:
    lines = []
    normalized_title = _clean_text(title)
    normalized_message = _clean_text(message) or "(no message)"

    if normalized_title:
        lines.append(f"**{normalized_title}**")
    lines.append(normalized_message)

    context_lines = []
    if include_default_context:
        finished_at = datetime.now(timezone.utc).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        )
        context_lines.extend(
            (
                ("finished_at", finished_at),
                ("host", socket.gethostname()),
                ("cwd", str(Path.cwd())),
            )
        )
    if context:
        context_lines.extend((str(key), value) for key, value in context.items())

    if context_lines:
        lines.append("")
        for key, value in context_lines:
            if value is None:
                continue
            lines.append(f"`{key}`: {_clean_text(value)}")

    return "\n".join(lines)


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _resolve_webhook_url(webhook_url: Optional[str]) -> str:
    url = _clean_text(webhook_url)
    if not url:
        for env_var in DEFAULT_WEBHOOK_ENV_VARS:
            url = _clean_text(os.environ.get(env_var))
            if url:
                break
    if not url:
        url = _clean_text(_read_dotenv_value(DEFAULT_WEBHOOK_ENV_VARS))

    if not url:
        env_names = " or ".join(DEFAULT_WEBHOOK_ENV_VARS)
        raise ValueError(f"Set {env_names}, add it to .env, or pass webhook_url=...")

    return _validate_webhook_url(url)


def _validate_webhook_url(webhook_url: object) -> str:
    url = _clean_text(webhook_url)
    parsed = parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Webhook URL must be an https URL")
    if "/api/webhooks/" not in parsed.path:
        raise ValueError("Webhook URL does not look like a Discord webhook URL")
    return url


def _default_dotenv_path() -> Path:
    return Path(__file__).resolve().parent / ".env"


def _read_dotenv_value(keys: Sequence[str]) -> str:
    for directory in (Path.cwd(), *Path.cwd().parents):
        for filename in DOTENV_FILENAMES:
            path = directory / filename
            if not path.is_file():
                continue
            value = _read_env_file(path, keys)
            if value:
                return value
    return ""


def _read_env_file(path: Path, keys: Sequence[str]) -> str:
    wanted = set(keys)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if key not in wanted:
            continue
        value = value.strip().strip("'\"")
        if value:
            return value
    return ""


def _split_content(text: str, limit: int) -> Sequence[str]:
    if limit > DISCORD_CONTENT_LIMIT:
        raise ValueError("Chunk limit cannot exceed Discord's content limit")
    if len(text) <= limit:
        return (text,)

    chunks = []
    current = []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if len(line) > limit:
            if current:
                chunks.append("".join(current).rstrip())
                current = []
                current_len = 0
            for start in range(0, len(line), limit):
                chunks.append(line[start : start + limit].rstrip())
            continue

        if current and current_len + len(line) > limit:
            chunks.append("".join(current).rstrip())
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line)

    if current:
        chunks.append("".join(current).rstrip())
    return tuple(chunk for chunk in chunks if chunk)


def _with_part_label(chunk: str, index: int, total: int) -> str:
    if total <= 1:
        return chunk
    label = f"(part {index}/{total})\n"
    if len(label) + len(chunk) <= DISCORD_CONTENT_LIMIT:
        return f"{label}{chunk}"
    return f"{label}{chunk[: DISCORD_CONTENT_LIMIT - len(label)]}"


def _post_webhook(
    url: str,
    content: str,
    *,
    username: Optional[str],
    avatar_url: Optional[str],
    allow_mentions: bool,
    timeout: float,
) -> int:
    payload = {"content": content}
    if username:
        payload["username"] = username
    if avatar_url:
        payload["avatar_url"] = avatar_url
    if not allow_mentions:
        payload["allowed_mentions"] = {"parse": []}

    data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "object-detection-notebook-notify/1.0",
        },
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            return int(response.status)
    except error.HTTPError as exc:
        body = exc.read(500).decode("utf-8", errors="replace")
        detail = f"HTTP {exc.code}"
        if body:
            detail = f"{detail}: {body}"
        raise RuntimeError(detail) from exc


__all__ = ["DiscordNotifyResult", "notify_discord", "save_discord_webhook_url"]
