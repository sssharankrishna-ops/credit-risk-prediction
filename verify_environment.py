"""
Phase 1 - environment verification.

Run AFTER activating the virtual environment:

    python verify_environment.py

Exit code 0 = every required package is present at a sufficient version.
Exit code 1 = one or more packages missing/outdated (see output for fix).

Teaching note: an *import name* (what you type after `import`) is not always
the same as the *distribution name* (what you type after `pip install`):
    import sklearn            <- pip install scikit-learn
    import dotenv            <- pip install python-dotenv
    import langchain_community <- pip install langchain-community
so this script checks distribution metadata first, module metadata second.
"""
from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version as dist_version
import platform
import sys

# import name -> minimum version (floors chosen for Python 3.13 support)
REQUIRED = {
    "numpy": "2.1",
    "pandas": "2.2",
    "sklearn": "1.6",                # import name of scikit-learn
    "matplotlib": "3.9",
    "seaborn": "0.13",
    "torch": "2.6",
    "mlflow": "3.0",
    "fastapi": "0.116",
    "uvicorn": "0.34",
    "streamlit": "1.39",
    "transformers": "4.46",
    "langchain": "1.0",
    "langchain_community": "0.3",
    "sqlalchemy": "2.0",
    "pydantic": "2.11",
    "joblib": "1.4",
    "dotenv": "1.0",                 # import name of python-dotenv
    "pytest": "8.0",
}

# import names that differ from their pip distribution names
DIST_NAME = {
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
    "langchain_community": "langchain-community",
}


def parse_version(version: str) -> tuple[int, ...]:
    """'2.8.0+cu128' -> (2, 8, 0). Stops at the first non-numeric part."""
    parts: list[int] = []
    for token in version.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def installed_version(module_name: str) -> str | None:
    """Return the installed version string, or None if not installed."""
    try:  # 1) ask pip's package metadata (reliable, no import side effects)
        return dist_version(DIST_NAME.get(module_name, module_name))
    except PackageNotFoundError:
        pass
    try:  # 2) fall back to the module's own __version__ attribute
        module = importlib.import_module(module_name)
        return str(getattr(module, "__version__", "") or None)
    except ImportError:
        return None


def main() -> int:
    print(f"Python : {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
    print("-" * 64)

    missing: list[str] = []
    outdated: list[str] = []

    for module_name, min_version in REQUIRED.items():
        version = installed_version(module_name)
        if version is None:
            missing.append(module_name)
            print(f"[FAIL] {module_name:22s} NOT INSTALLED  (need >= {min_version})")
            continue
        ok = parse_version(version) >= parse_version(min_version)
        status = "PASS" if ok else "OLD "
        if not ok:
            outdated.append(module_name)
        print(f"[{status}] {module_name:22s} {version:14s} (need >= {min_version})")

    print("-" * 64)
    try:
        import torch
        print(f"PyTorch device check: CUDA available = {torch.cuda.is_available()} "
              f"(CPU-only is expected and sufficient for this project)")
    except ImportError:
        pass

    if missing or outdated:
        print(f"\nRESULT: FAIL -> missing={missing or 'none'} | outdated={outdated or 'none'}")
        print("Fix: pip install -r requirements.txt   (then re-run this script)")
        return 1

    print("\nRESULT: PASS - all required packages installed. Phase 1 environment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
