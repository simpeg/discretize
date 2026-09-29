"""Verify version pins are consistent between pyproject.toml and .pre-commit-config.yaml.

Exit 1 if any mismatch is found, 0 otherwise.
"""

import re
import sys
import tomllib
import pathlib

try:
    import yaml
except ImportError:
    print(
        "ERROR: pyyaml is required. Install it with: pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(2)

ROOT = pathlib.Path(__file__).parent.parent
PYPROJECT = ROOT / "pyproject.toml"
PRE_COMMIT_CONFIG = ROOT / ".pre-commit-config.yaml"

# Maps a substring of the pre-commit repo URL to the corresponding pip package name.
# Only repos whose rev tracks the package version should be listed here.
PRECOMMIT_REPO_PACKAGES = {
    "psf/black": "black",
    "pycqa/flake8": "flake8",
}


def normalize_name(name: str) -> str:
    """Normalize a package name to a canonical form for comparison."""
    return re.sub(r"[-_.]+", "_", name).lower()


def parse_pip_req(req: str):
    """Return (normalized_name, op, version) for a pip requirement string, or None."""
    m = re.match(
        r"^([A-Za-z0-9][A-Za-z0-9._\-]*)\s*(==|>=|<=|!=|~=)\s*(\S+)$", req.strip()
    )
    if m:
        return normalize_name(m.group(1)), m.group(2), m.group(3)
    return None


def load_pyproject():
    """Return (optional_deps, tool_black_version).

    optional_deps: {group: {normalized_name: (op, version)}}
    tool_black_version: plain version string from [tool.black] required-version, or None
    """
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)

    optional_deps: dict[str, dict[str, tuple[str, str]]] = {}
    for group, reqs in data["project"]["optional-dependencies"].items():
        parsed = {}
        for req in reqs:
            result = parse_pip_req(req)
            if result:
                name, op, ver = result
                parsed[name] = (op, ver)
        optional_deps[group] = parsed

    tool_black_version: str | None = (
        data.get("tool", {}).get("black", {}).get("required-version")
    )

    return optional_deps, tool_black_version


def load_precommit():
    """Return {normalized_name: (op, version)} from .pre-commit-config.yaml."""
    with open(PRE_COMMIT_CONFIG) as f:
        config = yaml.safe_load(f)

    result: dict[str, tuple[str, str]] = {}
    for repo_cfg in config.get("repos", []):
        repo_url = repo_cfg.get("repo", "")
        rev = repo_cfg.get("rev", "")

        for url_fragment, pkg_name in PRECOMMIT_REPO_PACKAGES.items():
            if url_fragment.lower() in repo_url.lower():
                result[normalize_name(pkg_name)] = ("==", rev)

        for hook in repo_cfg.get("hooks", []):
            for dep in hook.get("additional_dependencies", []):
                parsed = parse_pip_req(dep)
                if parsed:
                    name, op, ver = parsed
                    result[name] = (op, ver)

    return result


def main() -> int:
    optional_deps, tool_black_version = load_pyproject()
    precommit_versions = load_precommit()
    style_deps = optional_deps.get("style", {})

    errors: list[str] = []

    # Check A: pyproject.toml [style] ↔ .pre-commit-config.yaml
    for pkg, (op, ver) in style_deps.items():
        if pkg in precommit_versions:
            pre_op, pre_ver = precommit_versions[pkg]
            if (pre_op, pre_ver) != (op, ver):
                errors.append(
                    f"  {pkg}: pyproject.toml [style] has {op}{ver},"
                    f" .pre-commit-config.yaml has {pre_op}{pre_ver}"
                )
        else:
            errors.append(
                f"  {pkg}: in pyproject.toml [style] but not found"
                f" in .pre-commit-config.yaml"
            )
    if errors:
        print(
            "FAIL: pyproject.toml [style] and .pre-commit-config.yaml are out of sync:"
        )
        for e in errors:
            print(e)
        errors.clear()
        style_check_failed = True
    else:
        style_check_failed = False

    # Check A2: [tool.black] required-version ↔ black in [style]
    black_key = normalize_name("black")
    if tool_black_version is not None and black_key in style_deps:
        style_op, style_ver = style_deps[black_key]
        if style_ver != tool_black_version:
            print(
                f"FAIL: black version mismatch within pyproject.toml:\n"
                f"  [style] has black{style_op}{style_ver},"
                f" [tool.black] required-version = '{tool_black_version}'"
            )
            style_check_failed = True

    return 1 if style_check_failed else 0


if __name__ == "__main__":
    sys.exit(main())
