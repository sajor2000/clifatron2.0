"""U19: every command in docs/plans/l40-runbook.md resolves to an existing entry point.

Each `uv run python -m <module> ...` and `uv run torchrun ... -m <module> ...` line inside
a fenced code block names a module that exists, and every `--flag` on the line appears in
that module's own `--help` (run in a subprocess, with the subcommand when the line has
one). The timing run's torchrun line is the experiment matrix's own launch command. No
command line starts a bare `torchrun`: outside `uv run` it is whichever torchrun is first
on PATH, not the locked environment's.
"""

import importlib.util
import re
import shlex
import subprocess
import sys
import unittest
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs/plans/l40-runbook.md"
# Every fenced block, with its language tag; only shell blocks hold commands (a ```toml
# block must still be consumed, or the fences after it pair up the wrong way).
FENCE = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)
SHELL = ("", "bash", "sh")


TORCHRUN = "uv run torchrun "


def fenced_lines() -> list[str]:
    """Every non-comment line of the runbook's fenced blocks."""
    lines = []
    for language, block in FENCE.findall(RUNBOOK.read_text()):
        if language not in SHELL:
            continue
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def runbook_commands() -> list[list[str]]:
    """Token lists of the module launches in the runbook's fenced blocks."""
    commands = []
    for line in fenced_lines():
        if line.startswith("uv run python -m ") or (
                line.startswith(TORCHRUN) and " -m " in line):
            commands.append(shlex.split(line, comments=True))
    return commands


def is_torchrun(tokens: list[str]) -> bool:
    return tokens[:3] == ["uv", "run", "torchrun"]


def module_and_subcommand(tokens: list[str]) -> tuple[str, list[str], list[str]]:
    """(module, subcommand tokens, flags) of one launch line."""
    index = tokens.index("-m")
    module = tokens[index + 1]
    rest = tokens[index + 2:]
    sub = []
    for token in rest:
        if token.startswith("-"):
            break
        sub.append(token)
    flags = [t.split("=", 1)[0] for t in rest if t.startswith("--")]
    return module, sub, flags


@cache
def module_help(module: str, sub: tuple[str, ...]) -> str:
    out = subprocess.run([sys.executable, "-m", module, *sub, "--help"], cwd=ROOT,
                         capture_output=True, text=True, timeout=300, check=False)
    if out.returncode != 0:
        raise AssertionError(f"`python -m {module} {' '.join(sub)} --help` exited "
                             f"{out.returncode}: {out.stderr[-500:]}")
    return out.stdout


class RunbookCommandsTest(unittest.TestCase):
    def test_no_bare_torchrun(self):
        bare = [line for line in fenced_lines()
                if re.match(r"^(\S+=\S+\s+)*torchrun\b", line)]
        self.assertEqual(bare, [], "launch torchrun through `uv run torchrun`")
        self.assertTrue(any(is_torchrun(t) for t in runbook_commands()))

    def test_runbook_has_the_pipeline_commands(self):
        modules = {module_and_subcommand(t)[0] for t in runbook_commands()}
        for expected in ("src.data.cohort", "src.data.tokenize", "src.data.value_stats",
                         "src.data.tokenize_continuous", "src.data.extubation_cohort",
                         "src.eval.extubation_labeler", "src.eval.extubation_audit",
                         "src.train.preflight", "src.train.run_matrix",
                         "src.train.run_tokenization_ablation", "src.eval.threshold_eval",
                         "src.eval.claims_report", "src.eval.baselines"):
            self.assertIn(expected, modules)

    def test_every_module_exists_and_accepts_every_flag(self):
        commands = runbook_commands()
        self.assertGreater(len(commands), 20)
        for tokens in commands:
            module, sub, flags = module_and_subcommand(tokens)
            with self.subTest(command=" ".join(tokens[:8])):
                self.assertIsNotNone(importlib.util.find_spec(module),
                                     f"no module {module}")
                text = module_help(module, tuple(sub))
                for flag in flags:
                    self.assertRegex(text, rf"(?<![\w-]){re.escape(flag)}(?![\w-])",
                                     f"{module} {' '.join(sub)} does not accept {flag}")

    def test_unknown_flag_is_detected(self):
        """The check itself can fail: a flag the module does not define is not in its
        help."""
        text = module_help("src.data.cohort", ())
        self.assertIsNone(re.search(r"(?<![\w-])--no-such-flag(?![\w-])", text))
        self.assertIsNotNone(re.search(r"(?<![\w-])--data(?![\w-])", text))

    def test_timing_run_is_the_matrix_launch_command(self):
        from src.train.run_matrix import expand_matrix, launch_commands, load_matrix

        matrix = load_matrix()
        run = next(r for r in expand_matrix(matrix)
                   if r["run_id"] == "clinical_soft.full.30m.time.s1.screening")
        expected = shlex.split(launch_commands(run, matrix)["torchrun"])
        torchruns = [t for t in runbook_commands() if is_torchrun(t)]
        self.assertIn(expected, torchruns)


if __name__ == "__main__":
    unittest.main()
