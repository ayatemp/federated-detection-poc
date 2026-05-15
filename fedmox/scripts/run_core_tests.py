"""Run FedMox core checks without requiring pytest."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    test_path = Path(__file__).resolve().parents[1] / "tests" / "test_fedmox_core.py"
    spec = importlib.util.spec_from_file_location("test_fedmox_core", test_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {test_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in sorted(item for item in dir(module) if item.startswith("test_")):
        getattr(module, name)()
        print(f"{name}: ok")
    print("FedMox core checks passed.")


if __name__ == "__main__":
    main()
