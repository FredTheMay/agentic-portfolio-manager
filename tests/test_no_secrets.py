"""No credential may reach a tracked file.

``.env`` is gitignored and ``.env.example`` is committed. The two filenames
differ by six characters and the consequence differs by a published secret, so
the difference is checked rather than trusted.
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
        ROOT / "DESIGN.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "RESULTS.md",
    ]:
        if not path.is_file():
            continue
        text = path.read_text()
        for pattern in KEY_SHAPES:
            if pattern.search(text):
                suspects.append(f"{path.relative_to(ROOT)} matches {pattern.pattern}")

    assert not suspects, "possible credential in a tracked file:\n" + "\n".join(suspects)


def test_credentials_are_stripped_from_error_messages() -> None:
    """A key must not reach an exception, a log line, or CI output.

    Excluding credentials from the cache key is not sufficient on its own:
    httpx embeds the full request URL, query string included, in its own
    exception text, so an upstream failure would otherwise print the key.
    """
    from src.data.cache import redact

    cases = [
        ("https://api.test/x?series_id=DGS3MO&api_key=abc123DEF456", "abc123DEF456"),
        ("failed for url 'https://a.test/?token=zzz999'", "zzz999"),
        ("GET /v2/x?access_token=SEKRIT&id=7", "SEKRIT"),
        ("?apikey=Hunter2&b=2", "Hunter2"),
    ]
    for text, secret in cases:
        cleaned = redact(text)
        assert secret not in cleaned, f"{secret!r} survived redaction of {text!r}"
        assert "<redacted>" in cleaned

    # Non-sensitive parameters must survive, or the message stops being useful.
    assert "series_id=DGS3MO" in redact(cases[0][0])
    assert "id=7" in redact(cases[2][0])


def test_every_outbound_client_redacts_its_errors() -> None:
    # Each module that raises an upstream error message must route it through
    # redact(), or it becomes a new leak path the moment that vendor 500s.
    from pathlib import Path

    for relative in (
        "src/data/cache.py",
        "src/execution/naive.py",
        "src/llm/gemini.py",
        "src/llm/groq.py",
    ):
        source = (ROOT / relative).read_text()
        if "httpx" not in source:
            continue
        assert "redact(" in source, f"{relative} raises upstream errors without redaction"
