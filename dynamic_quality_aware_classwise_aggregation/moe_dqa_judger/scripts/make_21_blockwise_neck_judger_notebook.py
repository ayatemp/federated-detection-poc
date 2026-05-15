#!/usr/bin/env python3
"""Create the notebook for experiment 21."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "21_blockwise_neck_judger.ipynb"


def main() -> None:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# 21 Blockwise Neck Judger\n\n"
            "20で見えた neck BN の微小改善を、C1/C2/C3/C4 の block-wise mixing に分解する。"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "ROOT = Path('/app/Object_Detection')\n"
            "SCRIPT = ROOT / 'dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/scripts/run_21_blockwise_neck_judger.py'\n"
            "WORKSPACE = ROOT / 'dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/21_blockwise_neck_judger'\n"
            "print(SCRIPT)\n"
            "print(WORKSPACE)"
        ),
        nbf.v4.new_code_cell(
            "import subprocess, sys\n"
            "cmd = [\n"
            "    sys.executable, str(SCRIPT),\n"
            "    '--workspace-root', str(WORKSPACE),\n"
            "    '--imgsz', '960',\n"
            "    '--val-batch-size', '16',\n"
            "    '--full-eval-topk', '8',\n"
            "    '--notify-discord',\n"
            "]\n"
            "print(' '.join(cmd))\n"
            "subprocess.run(cmd, check=True)"
        ),
        nbf.v4.new_code_cell(
            "score_path = WORKSPACE / 'stats/21_scorecard.json'\n"
            "metrics_path = WORKSPACE / 'stats/21_full_metrics.csv'\n"
            "print(score_path.read_text())\n"
            "print(metrics_path.read_text())"
        ),
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(nb, NOTEBOOK)
    print(f"wrote {NOTEBOOK}")


if __name__ == "__main__":
    main()
