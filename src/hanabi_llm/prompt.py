"""Loads a versioned system prompt from YAML and fills in a few variables.

Deliberately small. The prompt is a static text with ``${placeholders}`` for
run settings only (the seat's name, the player list, the per-turn budgets).
Nothing is read from the ontology or the game state, so what a model is told
is exactly what is in the file, and the file is what gets versioned.

Each file declares its ``variables``; the loader checks they are exactly the
placeholders used in the text, so a typo fails at load time, not mid-game.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib import resources
from string import Template
from typing import Any

import yaml

_FILE = "{kind}.v{version}.yaml"
KINDS = ("system", "rerank")  # system: the chat prompt; rerank: the query for a rerank model


@dataclass(frozen=True, slots=True)
class PromptSpec:
    version: int
    description: str
    variables: tuple[str, ...]
    template: str
    sha256: str  # of the raw file, logged with every game
    kind: str = "system"

    def render(self, **values: object) -> str:
        """Fill the placeholders. The keys must be exactly the declared variables."""
        if set(values) != set(self.variables):
            raise ValueError(f"prompt v{self.version} needs exactly {sorted(self.variables)}, got {sorted(values)}")
        return Template(self.template).substitute({k: str(v) for k, v in values.items()})


def available_versions(kind: str = "system") -> list[int]:
    """Versions of one kind of prompt present on disk, ascending."""
    names = (p.name for p in resources.files("hanabi_llm").joinpath("prompts").iterdir())
    found = (re.fullmatch(rf"{re.escape(kind)}\.v(\d+)\.yaml", n) for n in names)
    return sorted(int(m.group(1)) for m in found if m)


def load_prompt(version: int | str | None = None, kind: str = "system") -> PromptSpec:
    """Load ``<kind>.v<version>.yaml`` (the latest if ``version`` is omitted). Accepts 2 or "v2"."""
    if kind not in KINDS:
        raise ValueError(f"unknown prompt kind {kind!r}; kinds: {list(KINDS)}")
    if version is None:
        version = available_versions(kind)[-1]
    number = int(str(version).lower().removeprefix("v"))
    path = resources.files("hanabi_llm").joinpath("prompts", _FILE.format(kind=kind, version=number))
    if not path.is_file():
        raise ValueError(f"no {kind} prompt version {number}; available: {available_versions(kind)}")
    return parse_prompt(path.read_bytes(), number, kind)


def parse_prompt(raw: bytes, expected_version: int, kind: str = "system") -> PromptSpec:
    """Validate and build a ``PromptSpec`` from the bytes of a prompt file."""
    number = expected_version
    data: Any = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"prompt v{number}: top level must be a mapping")
    for key in ("version", "description", "variables", "system"):
        if key not in data:
            raise ValueError(f"prompt v{number}: missing key {key!r}")
    if data["version"] != number:
        raise ValueError(f"prompt file v{number} declares version {data['version']!r}")
    template = str(data["system"]).strip() + "\n"
    declared = tuple(str(v) for v in data["variables"])
    used = set(Template(template).get_identifiers())
    if used != set(declared):
        raise ValueError(f"prompt v{number}: declared variables {sorted(declared)} != placeholders used {sorted(used)}")
    return PromptSpec(number, str(data["description"]).strip(), declared, template, hashlib.sha256(raw).hexdigest(), kind)
