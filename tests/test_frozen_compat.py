"""Tests for PyInstaller frozen-build compatibility.

These tests catch issues that only manifest in --noconsole PyInstaller builds:
- Relative imports in __main__.py (no parent package context)
- APIs that assume real file descriptors (faulthandler, fileno)
- Modules that must be importable via absolute paths
- Dynamic imports must be listed in dictator.spec hiddenimports
"""

import ast
import io
import os
import re
import sys
import unittest
from pathlib import Path

# Root of the dictator package
_DICTATOR_PKG = Path(__file__).resolve().parent.parent / "dictator"
_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestNoRelativeImportsInMain(unittest.TestCase):
    """__main__.py must use absolute imports for PyInstaller compatibility."""

    def test_no_relative_imports(self):
        source = (_DICTATOR_PKG / "__main__.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="__main__.py")

        relative_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                relative_imports.append(
                    f"line {node.lineno}: from {'.' * node.level}{node.module or ''} import ..."
                )

        self.assertEqual(
            relative_imports,
            [],
            f"__main__.py must not use relative imports (breaks PyInstaller):\n"
            + "\n".join(relative_imports),
        )


class TestFaulthandlerWithStringIO(unittest.TestCase):
    """faulthandler.enable() must be guarded for --noconsole builds."""

    def test_faulthandler_tolerates_stringio_stderr(self):
        import faulthandler

        original_stderr = sys.stderr
        try:
            sys.stderr = io.StringIO()
            try:
                faulthandler.enable()
            except io.UnsupportedOperation:
                pass
        finally:
            sys.stderr = original_stderr

    def test_main_guards_faulthandler(self):
        source = (_DICTATOR_PKG / "__main__.py").read_text(encoding="utf-8")
        self.assertIn("io.UnsupportedOperation", source,
                       "faulthandler.enable() must be guarded with "
                       "except io.UnsupportedOperation")


class TestStdioSafetyPatches(unittest.TestCase):
    """__main__.py must patch None stdout/stderr for --noconsole builds."""

    def test_stdout_none_guard_exists(self):
        source = (_DICTATOR_PKG / "__main__.py").read_text(encoding="utf-8")
        self.assertIn("sys.stdout is None", source)

    def test_stderr_none_guard_exists(self):
        source = (_DICTATOR_PKG / "__main__.py").read_text(encoding="utf-8")
        self.assertIn("sys.stderr is None", source)


class TestAllModulesImportable(unittest.TestCase):
    """Every .py file in dictator/ must be importable via absolute paths."""

    _SKIP_MODULES = frozenset({
        "dictator.engine.cohere_transcribe",
        "dictator.engine.granite_speech",
    })

    def test_import_all_modules(self):
        failures = []
        for py_file in sorted(_DICTATOR_PKG.rglob("*.py")):
            rel = py_file.relative_to(_DICTATOR_PKG.parent)
            module_name = str(rel.with_suffix("")).replace("\\", ".").replace("/", ".")

            if module_name in self._SKIP_MODULES:
                continue
            if "__pycache__" in module_name:
                continue

            try:
                __import__(module_name)
            except Exception as exc:
                failures.append(f"{module_name}: {type(exc).__name__}: {exc}")

        self.assertEqual(
            failures,
            [],
            f"Failed to import the following modules:\n" + "\n".join(failures),
        )


class TestRelativeImportsInSubpackages(unittest.TestCase):

    def test_engine_subpackage_imports(self):
        from dictator.engine import ENGINES
        self.assertIsInstance(ENGINES, dict)

    def test_engine_base_imports(self):
        from dictator.engine.base import SpeechEngine
        self.assertTrue(callable(SpeechEngine))


class TestHiddenImportsInSpec(unittest.TestCase):
    """Dynamic imports in __main__.py must be listed in dictator.spec hiddenimports."""

    _INTERNAL_PREFIXES = ("dictator.",)

    _STDLIB = frozenset({
        "argparse", "ctypes", "faulthandler", "io", "json", "logging",
        "logging.handlers", "os", "sys", "re", "pathlib", "tempfile",
        "time", "subprocess", "unittest", "importlib", "threading",
        "collections", "functools", "typing", "traceback", "copy",
        "shutil", "signal", "struct", "abc", "dataclasses", "enum",
    })

    def _parse_hidden_imports(self) -> set[str]:
        spec_path = _REPO_ROOT / "dictator.spec"
        spec_text = spec_path.read_text(encoding="utf-8")
        match = re.search(
            r"hiddenimports\s*=\s*\[(.*?)\]", spec_text, re.DOTALL
        )
        self.assertIsNotNone(match, "Could not find hiddenimports in dictator.spec")
        entries = re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))
        return set(entries)

    def _collect_deferred_imports(self, filepath: Path) -> list[tuple[int, str]]:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filepath.name)

        deferred: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Import):
                    for alias in child.names:
                        deferred.append((child.lineno, alias.name.split(".")[0]))
                elif isinstance(child, ast.ImportFrom):
                    if child.level == 0 and child.module:
                        deferred.append((child.lineno, child.module.split(".")[0]))
        return deferred

    def test_dynamic_imports_in_hiddenimports(self):
        hidden = self._parse_hidden_imports()
        deferred = self._collect_deferred_imports(_DICTATOR_PKG / "__main__.py")

        missing = []
        for lineno, top_module in deferred:
            if top_module in self._STDLIB:
                continue
            if any(top_module.startswith(p.rstrip(".")) for p in self._INTERNAL_PREFIXES):
                continue
            if not any(h == top_module or h.startswith(top_module + ".") for h in hidden):
                missing.append(f"line {lineno}: {top_module}")

        self.assertEqual(
            missing,
            [],
            "Dynamic imports in __main__.py not listed in dictator.spec hiddenimports:\n"
            + "\n".join(missing)
            + "\nAdd them to hiddenimports in dictator.spec.",
        )

    def test_dynamic_imports_in_main_window(self):
        hidden = self._parse_hidden_imports()
        deferred = self._collect_deferred_imports(_DICTATOR_PKG / "main_window.py")

        missing = []
        for lineno, top_module in deferred:
            if top_module in self._STDLIB:
                continue
            if any(top_module.startswith(p.rstrip(".")) for p in self._INTERNAL_PREFIXES):
                continue
            if not any(h == top_module or h.startswith(top_module + ".") for h in hidden):
                missing.append(f"line {lineno}: {top_module}")

        self.assertEqual(
            missing,
            [],
            "Dynamic imports in main_window.py not listed in dictator.spec hiddenimports:\n"
            + "\n".join(missing)
            + "\nAdd them to hiddenimports in dictator.spec.",
        )


class TestTransitiveDependenciesInSpec(unittest.TestCase):
    """Transitive dependencies used at runtime must be bundled in the spec."""

    def _read_spec(self) -> str:
        return (_REPO_ROOT / "dictator.spec").read_text(encoding="utf-8")

    def _parse_hidden_imports(self) -> set[str]:
        spec_text = self._read_spec()
        match = re.search(
            r"hiddenimports\s*=\s*\[(.*?)\]", spec_text, re.DOTALL
        )
        assert match, "Could not find hiddenimports in dictator.spec"
        return set(re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)))

    def _parse_excludes(self) -> set[str]:
        spec_text = self._read_spec()
        match = re.search(
            r"excludes\s*=\s*\[(.*?)\]", spec_text, re.DOTALL
        )
        assert match, "Could not find excludes in dictator.spec"
        return set(re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)))

    def test_transformers_in_hiddenimports(self):
        hidden = self._parse_hidden_imports()
        self.assertIn("transformers", hidden)

    def test_torch_in_hiddenimports(self):
        hidden = self._parse_hidden_imports()
        self.assertIn("torch", hidden)

    def test_transformers_data_files_collected(self):
        spec_text = self._read_spec()
        self.assertIn(
            "collect_data_files('transformers')",
            spec_text,
            "dictator.spec must call collect_data_files('transformers')",
        )
