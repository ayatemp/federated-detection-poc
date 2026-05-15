# Notebook Discord Notification Guide

This guide explains how to use `notebook_notify.py` to send Discord messages
from notebooks or long-running experiment scripts.

## What It Does

`notebook_notify.py` sends a message to Discord through a Discord Incoming
Webhook.

It is not a Discord bot. There is no background process to start, no bot token,
and no extra Python package to install. The helper simply sends an HTTPS request
to Discord when `notify_discord(...)` is called.

Typical uses:

- send a message when a notebook finishes
- send a start/end message from a long experiment script
- include run metadata such as host, working directory, workspace path, or error

## What You Need On Another Computer

To use the same notification setup on another computer, you need:

1. `notebook_notify.py`
2. a Discord webhook URL
3. a way for Python to import `notebook_notify.py`

The webhook URL should not be committed to git. Treat it like a password for
posting into the target Discord channel.

## Create A Discord Webhook

In Discord:

1. Open the target server and channel.
2. Open channel settings.
3. Go to Integrations.
4. Create or open an Incoming Webhook.
5. Copy the webhook URL.

The URL usually looks like this:

```text
https://discord.com/api/webhooks/...
```

## Set The Webhook URL

Recommended option:

```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

If the notebook kernel is already running, restart the kernel after setting the
environment variable.

Local `.env` option:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

The `.env` file is local-only and should stay out of git.

You can also save the URL from inside a notebook without showing it in notebook
outputs:

```python
from getpass import getpass
from notebook_notify import save_discord_webhook_url

save_discord_webhook_url(getpass("Discord webhook URL: "))
```

## Use It In A Notebook

Add this near the end of a notebook:

```python
from notebook_notify import notify_discord

notify_discord(
    "Notebook finished.",
    title="Experiment complete",
)
```

With extra context:

```python
from notebook_notify import notify_discord

notify_discord(
    "DQA run finished. Check the output directory for metrics.",
    title="DQA complete",
    context={
        "workspace": "output/01_1_dqa_diagnostic_sweep",
        "rounds": 3,
        "status": "ok",
    },
)
```

By default, the helper also adds:

- `finished_at`
- `host`
- `cwd`

## If Import Fails

If the notebook is running from a subdirectory, Python may not find
`notebook_notify.py`.

Add this before importing it:

```python
from pathlib import Path
import sys

root = next(
    path for path in [Path.cwd(), *Path.cwd().parents]
    if (path / "notebook_notify.py").exists()
)
sys.path.insert(0, str(root))

from notebook_notify import notify_discord
```

## Use It From Experiment Scripts

Some scripts in this repository already support notification flags such as:

```bash
--notify
```

or:

```bash
--notify-start
--notify-end
```

Those scripts call `notify_discord(...)` internally. On a new computer, the
flags will work as long as `notebook_notify.py` is importable and
`DISCORD_WEBHOOK_URL` is set.

Example:

```bash
python scripts/run_scene_daynight_dqa_01_1.py \
  --workspace-root output/01_1_dqa_diagnostic_sweep \
  --notify
```

## Test Without Sending

Use `dry_run=True` to check formatting without posting to Discord:

```python
from notebook_notify import notify_discord

result = notify_discord(
    "Dry-run test. This will not send to Discord.",
    title="Notification dry run",
    dry_run=True,
)

print(result)
```

## Send A Real Test Message

After setting `DISCORD_WEBHOOK_URL`, run:

```python
from notebook_notify import notify_discord

result = notify_discord(
    "Test message from notebook_notify.py.",
    title="Discord notifier test",
)

print(result)
```

Success usually looks like:

```text
DiscordNotifyResult(ok=True, chunks_sent=1, status_codes=(204,), dry_run=False, error=None)
```

Discord returns status code `204` when the webhook post succeeds.

## Troubleshooting

### `Set DISCORD_WEBHOOK_URL...`

Python could not find a webhook URL.

Fix one of these:

- export `DISCORD_WEBHOOK_URL`
- create a local `.env`
- pass `webhook_url="..."` directly to `notify_discord(...)`

### `Webhook URL must be an https URL`

The value is not a valid HTTPS URL. Re-copy the webhook URL from Discord.

### `Webhook URL does not look like a Discord webhook URL`

The URL does not contain `/api/webhooks/`. Make sure you copied the Incoming
Webhook URL, not a normal Discord channel link.

### `ModuleNotFoundError: No module named 'notebook_notify'`

Python cannot import `notebook_notify.py`.

Fix it by running the notebook from the repository root, copying
`notebook_notify.py` next to the notebook, or adding the repository root to
`sys.path`.

### No message appears in Discord

Check these:

- the webhook URL is for the channel you are watching
- the notebook kernel was restarted after setting the environment variable
- the machine has internet access
- the webhook still exists in Discord

## Security Notes

Do not commit webhook URLs to git.

Anyone with the webhook URL can post messages to that Discord channel. If the
URL is accidentally shared, delete or regenerate the webhook in Discord.

