#!/usr/bin/env python3
"""Create the notebook for experiment 19."""

from __future__ import annotations

import nbformat as nbf
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "19_vector_bn_delta_judger.ipynb"


def main() -> None:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# 19 Vector BN Delta Judger\n\n"
            "18で得た各clientのBN差分を、FedAWA/L-DAWA風にclient vectorとして扱い、"
            "backbone/neck別に混ぜ方をjudgeする実験。外部teacherは使わず、warmupと自己生成client checkpointだけから候補を作る。"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "ROOT = Path('/app/Object_Detection')\n"
            "SCRIPT = ROOT / 'dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/scripts/run_19_vector_bn_delta_judger.py'\n"
            "WORKSPACE = ROOT / 'dynamic_quality_aware_classwise_aggregation/moe_dqa_judger/output/19_vector_bn_delta_judger'\n"
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
            "    '--full-eval-topk', '6',\n"
            "    '--notify-discord',\n"
            "]\n"
            "print(' '.join(cmd))\n"
            "subprocess.run(cmd, check=True)"
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "score_path = WORKSPACE / 'stats/19_scorecard.json'\n"
            "metrics_path = WORKSPACE / 'stats/19_full_metrics.csv'\n"
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
