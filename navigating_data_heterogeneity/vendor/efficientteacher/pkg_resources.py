from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata

try:
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.version import parse as parse_version
except Exception:  # pragma: no cover - fallback for very small environments
    Requirement = None
    SpecifierSet = None

    def parse_version(value: str):
        from distutils.version import LooseVersion

        return LooseVersion(str(value))


class DistributionNotFound(Exception):
    pass


class VersionConflict(Exception):
    pass


@dataclass
class ParsedRequirement:
    name: str
    specifier: object


def parse_requirements(lines):
    for raw in lines:
        line = str(raw).strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if Requirement is not None:
            req = Requirement(line)
            yield ParsedRequirement(req.name, req.specifier)
        else:
            name = line
            for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
                if sep in line:
                    name = line.split(sep, 1)[0].strip()
                    break
            yield ParsedRequirement(name, "")


def require(requirement):
    requirements = [requirement] if isinstance(requirement, str) else list(requirement)
    for item in requirements:
        if Requirement is not None:
            req = Requirement(str(item))
            name = req.name
            specifier = req.specifier
        else:
            name = str(item)
            specifier = None
            for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
                if sep in name:
                    name = name.split(sep, 1)[0].strip()
                    break
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise DistributionNotFound(str(item)) from exc
        if specifier and SpecifierSet is not None and not specifier.contains(version, prereleases=True):
            raise VersionConflict(f"{name} {version} does not satisfy {specifier}")
    return []
