"""A test module's `if __name__ == "__main__"` block is its last statement.

Placed earlier, `unittest.main()` runs when `python tests/test_x.py` reaches it, before the
classes defined below it exist, so the direct run silently skips them (pytest is not
affected). Found in four modules during PR review.
"""

import ast
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def is_main_guard(node: ast.stmt) -> bool:
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")


class MainGuardIsLastTest(unittest.TestCase):
    def test_nothing_follows_the_main_guard(self):
        offenders = []
        for path in sorted(TESTS.glob("test_*.py")):
            body = ast.parse(path.read_text()).body
            for index, node in enumerate(body):
                if is_main_guard(node) and index != len(body) - 1:
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
