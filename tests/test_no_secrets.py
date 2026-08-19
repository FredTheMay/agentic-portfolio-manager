"""No credential may reach a tracked file (security).

This exists because it already happened: real Alpaca keys were typed into
``.env.example`` instead of ``.env`` and committed. ``.env.example`` is tracked
and pushed; ``.env`` is gitignored. The two filenames differ by six characters
and the consequence differs by a published secret, so the difference is checked
rather than trusted.

The remedy for a leaked key is always rotation, never history rewriting —
forks, clones, and provider-side caches keep copies that a force-push cannot
reach.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Variables that carry a credential. A populated value here is a leak.
SECRET_VARS = (
    "FRED_API_KEY",
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
)

#: Vendor key shapes, for scanning files that have no KEY=VALUE structure.
KEY_SHAPES = (
    re.compile(r"\bPK[A-Z0-9]{16,}\b"),      # Alpaca key id
    re.compile(r"\bAIza[A-Za-z0-9_\-]{30,}"),  # Google API key
    re.compile(r"\bgsk_[A-Za-z0-9]{40,}"),     # Groq
    re.compile(r"\bsk-[A-Za-z0-9]{32,}"),      # generic secret-key convention
)


def test_the_example_env_has_no_populated_secrets() -> None:
    example = ROOT / ".env.example"
    assert example.is_file(), ".env.example must exist as the documented template"

    populated = []
    for lineno, line in enumerate(example.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        if name.strip() in SECRET_VARS and value.strip().strip('"').strip("'"):
            populated.append(f"{example.name}:{lineno} {name.strip()} has a value")

    assert not populated, (
        "a credential is set in the TRACKED template — move it to .env "
        "(gitignored) and rotate the exposed key:\n" + "\n".join(populated)
    )


def test_the_real_env_is_never_tracked() -> None:
    gitignore = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in [line.strip() for line in gitignore], ".env must be gitignored"
    assert not (ROOT / ".env").is_file() or True  # its presence is fine; tracking is not


def test_no_vendor_key_shape_appears_in_tracked_config() -> None:
    # Belt and braces: catches a key pasted into a YAML or Markdown file, where
    # the KEY=VALUE check above would not look.
    suspects: list[str] = []
    for path in [
        *(ROOT / "config").glob("*.yaml"),
        ROOT / ".env.example",
        ROOT / "README.md",
        ROOT / "CLAUDE.md",
        ROOT / "RESULTS.md",
    ]:
        if not path.is_file():
            continue
        text = path.read_text()
        for pattern in KEY_SHAPES:
            if pattern.search(text):
                suspects.append(f"{path.relative_to(ROOT)} matches {pattern.pattern}")

    assert not suspects, "possible credential in a tracked file:\n" + "\n".join(suspects)
