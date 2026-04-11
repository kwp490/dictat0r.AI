"""Tests for engine loading, model file detection, and registry."""

import os
import tempfile
import unittest
from pathlib import Path

from dictator.engine import ENGINES, _model_files_exist, get_available_engines


class TestEngineRegistry(unittest.TestCase):
    """Engine registry must contain Granite and Cohere."""

    def test_granite_registered(self):
        self.assertIn("granite", ENGINES)

    def test_cohere_registered(self):
        self.assertIn("cohere", ENGINES)

    def test_granite_engine_name(self):
        engine = ENGINES["granite"]()
        self.assertEqual(engine.name, "granite")

    def test_cohere_engine_name(self):
        engine = ENGINES["cohere"]()
        self.assertEqual(engine.name, "cohere")

    def test_granite_vram_estimate(self):
        engine = ENGINES["granite"]()
        self.assertGreater(engine.vram_estimate_gb, 0)

    def test_cohere_vram_estimate(self):
        engine = ENGINES["cohere"]()
        self.assertGreater(engine.vram_estimate_gb, 0)


class TestModelFileDetection(unittest.TestCase):
    """Model file detection must correctly identify present/absent models."""

    def test_granite_with_config(self):
        with tempfile.TemporaryDirectory() as d:
            granite_dir = os.path.join(d, "granite")
            os.makedirs(granite_dir)
            with open(os.path.join(granite_dir, "config.json"), "w") as f:
                f.write("{}")
            self.assertTrue(_model_files_exist("granite", d))

    def test_granite_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            granite_dir = os.path.join(d, "granite")
            os.makedirs(granite_dir)
            self.assertFalse(_model_files_exist("granite", d))

    def test_granite_no_directory(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(_model_files_exist("granite", d))

    def test_cohere_with_config(self):
        with tempfile.TemporaryDirectory() as d:
            cohere_dir = os.path.join(d, "cohere")
            os.makedirs(cohere_dir)
            with open(os.path.join(cohere_dir, "config.json"), "w") as f:
                f.write("{}")
            self.assertTrue(_model_files_exist("cohere", d))

    def test_cohere_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            cohere_dir = os.path.join(d, "cohere")
            os.makedirs(cohere_dir)
            self.assertFalse(_model_files_exist("cohere", d))

    def test_unknown_engine(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(_model_files_exist("nonexistent", d))


class TestGetAvailableEngines(unittest.TestCase):
    """get_available_engines must only return engines with model files."""

    def test_both_present(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("granite", "cohere"):
                engine_dir = os.path.join(d, name)
                os.makedirs(engine_dir)
                with open(os.path.join(engine_dir, "config.json"), "w") as f:
                    f.write("{}")
            available = get_available_engines(d)
            self.assertIn("granite", available)
            self.assertIn("cohere", available)

    def test_none_present(self):
        with tempfile.TemporaryDirectory() as d:
            available = get_available_engines(d)
            self.assertEqual(available, [])

    def test_only_granite_present(self):
        with tempfile.TemporaryDirectory() as d:
            granite_dir = os.path.join(d, "granite")
            os.makedirs(granite_dir)
            with open(os.path.join(granite_dir, "config.json"), "w") as f:
                f.write("{}")
            available = get_available_engines(d)
            self.assertIn("granite", available)
            self.assertNotIn("cohere", available)
