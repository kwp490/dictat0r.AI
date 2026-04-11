"""Tests for the model downloader module."""

import os
import tempfile
import unittest

from dictator.model_downloader import (
    COHERE_REPO_ID,
    GRANITE_REPO_ID,
    _ENGINE_REPO_MAP,
    model_ready,
)


class TestModelConstants(unittest.TestCase):
    """Model constants must be consistent and non-empty."""

    def test_repo_ids_are_valid(self):
        for repo_id in (COHERE_REPO_ID, GRANITE_REPO_ID):
            self.assertIn("/", repo_id)
            self.assertFalse(repo_id.startswith("http"))

    def test_engine_repo_map(self):
        self.assertIn("granite", _ENGINE_REPO_MAP)
        self.assertIn("cohere", _ENGINE_REPO_MAP)
        self.assertEqual(_ENGINE_REPO_MAP["granite"], GRANITE_REPO_ID)
        self.assertEqual(_ENGINE_REPO_MAP["cohere"], COHERE_REPO_ID)


class TestModelReady(unittest.TestCase):
    """model_ready must correctly detect present/absent models."""

    def test_ready_when_config_exists(self):
        with tempfile.TemporaryDirectory() as d:
            engine_dir = os.path.join(d, "granite")
            os.makedirs(engine_dir)
            with open(os.path.join(engine_dir, "config.json"), "w") as f:
                f.write("{}")
            self.assertTrue(model_ready("granite", d))

    def test_not_ready_when_no_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(model_ready("granite", d))

    def test_not_ready_when_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "cohere"))
            self.assertFalse(model_ready("cohere", d))

    def test_ready_for_cohere(self):
        with tempfile.TemporaryDirectory() as d:
            engine_dir = os.path.join(d, "cohere")
            os.makedirs(engine_dir)
            with open(os.path.join(engine_dir, "config.json"), "w") as f:
                f.write("{}")
            self.assertTrue(model_ready("cohere", d))
