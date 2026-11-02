"""The user checkout and installed runtime work without replay maintenance tools."""

import ast
from pathlib import Path
import unittest

from opennpu_quant.cli import build_parser
from opennpu_quant.configuration import load_config
from opennpu_quant.models.spec import model_names


class UserDistributionTests(unittest.TestCase):
    def test_user_commands_and_model_sets(self):
        parser = build_parser()
        for config in (
            "core10",
            "cnn30",
            "bundle/configs/core10.yaml",
            "bundle/configs/cnn30.yaml",
        ):
            self.assertEqual(
                config, parser.parse_args(["run", "-c", config, "--dry-run"]).config
            )
        core, cnn = (load_config(name) for name in ("core10", "cnn30"))
        self.assertEqual(core["models"], cnn["models"][:10])
        self.assertEqual(70, len(core["models"]) * len(core["conditions"]))
        self.assertEqual(210, len(cnn["models"]) * len(cnn["conditions"]))
        self.assertEqual(set(model_names()), set(cnn["models"]))

    def test_user_tree_does_not_bundle_archived_tools_or_reference_data(self):
        root = Path(__file__).resolve().parents[1]
        for path in (
            "tools/maintainer",
            "tools/reproduce.py",
            "tools/check_export.py",
            "tools/validation",
        ):
            self.assertFalse((root / path).exists(), path)
        import opennpu_quant

        package = Path(opennpu_quant.__file__).parent
        self.assertFalse(list(package.rglob("selected.json")))
        self.assertFalse(list(package.rglob("lfw-input-reference.json")))
        self.assertFalse(list(package.rglob("export-contracts.json")))
        for path in package.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else []
                )
                self.assertFalse(
                    any(
                        name.startswith("tools") or "reproduction" in name.split(".")
                        for name in names
                    ),
                    path,
                )
