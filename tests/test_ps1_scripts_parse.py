"""Every PowerShell script in scripts/ must actually parse.

Two of the four scripts in this repo could not run at all, and nothing noticed
because no test had ever executed them:

    run-autonomous.ps1 -> line 172: Missing closing '}' in statement block
    sync-to-hatch.ps1  -> line 59:  The string is missing the terminator: "

``run-autonomous.ps1`` is the entry point the README Quickstart tells a stranger
to run and the one ``bb rtx results`` names when there is no data yet
("No results yet -- run in Alienware: .\\scripts\\run-autonomous.ps1"). It was a
syntax error from top to bottom.

The cause was encoding, not a typo. The files are UTF-8 with no BOM, and Windows
PowerShell 5.1 -- the only PowerShell on the target box, there is no pwsh --
decodes a BOM-less script using the system ANSI codepage. An em dash (U+2014,
bytes ``E2 80 94``) therefore arrives as ``â€”``, and that last
character, U+201D RIGHT DOUBLE QUOTATION MARK, is one of the characters
PowerShell's tokenizer accepts as a double quote. So::

    Write-Host "Watching every $IntervalSec sec - Ctrl+C to stop"
                                              ^ an em dash here closes the
                                                string early; the real closing
                                                quote then opens a new one that
                                                never terminates

The fix was to keep the scripts ASCII-only rather than to add a BOM. A BOM is
invisible state that any editor or agent rewriting the file can silently drop,
which is precisely how this got in; ASCII source cannot be mis-decoded by any
codepage in the first place. ``test_scripts_are_ascii_only`` is what enforces
that, and it is the primary gate here because it is deterministic and runs on
every platform.

``test_script_parses`` is the weaker of the two and is deliberately secondary:
under pwsh 7, which defaults to UTF-8, it would have happily passed on the exact
files that were broken. It exists to catch ordinary syntax errors on the shell
that actually matters, not to catch encoding faults.

Both tests parse only. Nothing here executes a script -- run-autonomous.ps1
would launch a training run.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

PS1_SCRIPTS = sorted(SCRIPTS_DIR.glob("*.ps1"))
PS1_IDS = [p.name for p in PS1_SCRIPTS]

# Characters that survive an ANSI round-trip as PowerShell quote delimiters.
# U+2014 em dash -> U+201D, U+2192 arrow -> U+2019, and so on: any multi-byte
# UTF-8 character can land on one of these, so the gate below bans all of them
# rather than trying to enumerate the dangerous ones.
_SMART_QUOTES = "“”‘’"


def test_scripts_directory_is_not_empty():
    """A glob that finds nothing would make every parametrised test vacuous.

    Without this, renaming scripts/ or changing the extension turns the whole
    file into zero collected tests, which reads as green.
    """
    assert PS1_SCRIPTS, (
        f"no .ps1 files found under {SCRIPTS_DIR} -- this suite would silently "
        "cover nothing"
    )


@pytest.mark.parametrize("script", PS1_SCRIPTS, ids=PS1_IDS)
def test_scripts_are_ascii_only(script):
    """Non-ASCII in a BOM-less .ps1 is decoded as ANSI by PowerShell 5.1.

    Byte-level and platform-independent, so this runs in any environment.
    """
    raw = script.read_bytes()
    offenders = []
    for lineno, line in enumerate(raw.split(b"\n"), 1):
        if any(byte > 127 for byte in line):
            decoded = line.decode("utf-8", "replace").rstrip()
            mangled = line.decode("cp1252", "replace")
            offenders.append(
                f"  line {lineno}: {decoded}\n"
                f"    PowerShell 5.1 reads this as: {mangled.rstrip()}"
                + (
                    "\n    ^ contains a character PowerShell treats as a quote"
                    if any(ch in _SMART_QUOTES for ch in mangled)
                    else ""
                )
            )

    assert not offenders, (
        f"{script.name} contains non-ASCII bytes. Windows PowerShell 5.1 decodes "
        "a BOM-less script with the system ANSI codepage, so these do not arrive "
        "as written and can terminate a string literal early:\n"
        + "\n".join(offenders)
        + "\n\nUse ASCII (e.g. '-' for an em dash). Do not fix this with a BOM: "
        "a BOM is invisible and the next tool to rewrite the file will drop it."
    )


@pytest.mark.skipif(
    os.name != "nt", reason="PowerShell parser check requires Windows"
)
@pytest.mark.parametrize("script", PS1_SCRIPTS, ids=PS1_IDS)
def test_script_parses(script):
    """Parse the script with the real PowerShell parser. Does not execute it."""
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        # Fail rather than skip: on Windows a missing PowerShell means these
        # scripts were never checked at all, and a skip is exit 0.
        pytest.fail(
            "neither 'pwsh' nor 'powershell' is on PATH, so no .ps1 in this "
            "repo can be syntax-checked. Environment failure, not a defect in "
            f"{script.name}.",
            pytrace=False,
        )

    # ParseFile only builds an AST; it runs nothing.
    check = (
        "$t=$null; $e=$null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{script}',[ref]$t,[ref]$e); "
        "if ($e.Count -gt 0) { "
        "$e | ForEach-Object { 'line ' + $_.Extent.StartLineNumber + ': ' + $_.Message }; "
        "exit 1 } else { exit 0 }"
    )
    proc = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", check],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 0, (
        f"{script.name} is not valid PowerShell, so it cannot run at all:\n"
        f"{proc.stdout}{proc.stderr}"
    )
