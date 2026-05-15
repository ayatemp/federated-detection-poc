#!/usr/bin/env python3
"""Create the notebook for experiment 20."""

from __future__ import annotations

import nbformat as nbf
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "20_neck_alpha_judger.ipynb"


def main() -> None:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# 20 Neck Alpha Judger\n\n"
            "19で見えた neck-only BN の微小改善を、alpha sweepで詰める。"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "ROOT = Path('/app/Object_Detection')\n"
            "SCRIPT = ROOT / 'dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/scripts/run_20_neck_alpha_judger.py'\n"
            "WORKSPACE = ROOT / 'dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/20_neck_alpha_judger'\n"
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
            "score_path = WORKSPACE / 'stats/20_scorecard.json'\n"
            "metrics_path = WORKSPACE / 'stats/20_full_metrics.csv'\n"
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
