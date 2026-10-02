"""asl.prod must stay deployable without asl.lab (the API image does not ship it)."""

import ast
from pathlib import Path

PROD = Path(__file__).resolve().parents[1] / "asl" / "prod"


def imported_modules(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_prod_never_imports_lab():
    offenders = {
        str(path.relative_to(PROD)): module
        for path in PROD.rglob("*.py")
        for module in imported_modules(path)
        if module == "asl.lab" or module.startswith("asl.lab.")
    }
    assert offenders == {}
