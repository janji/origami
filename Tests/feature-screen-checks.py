#!/usr/bin/env python3
"""CPU-only harness checks: no server, model, GPU or host-memory observation is used."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Tools" / "bench"))
from family_features import arithmetic_class, normalize_variants, validate_prompt_isolation


class FeatureIsolationChecks(unittest.TestCase):
    def test_qwen_decode_uses_equal_isolated_scopes(self):
        requested = {"control": {}, "keepalive": {"gpu_keepalive": False}}
        actual = normalize_variants("qwen", requested)
        self.assertEqual(actual, {"control": {"qwen_prefill_scope": 8192},
                                  "keepalive": {"qwen_prefill_scope": 8192, "gpu_keepalive": False}})
        self.assertEqual(requested, {"control": {}, "keepalive": {"gpu_keepalive": False}})
        self.assertTrue(arithmetic_class("qwen", actual).startswith("BITWISE"))

    def test_qwen_prefill_keeps_distinct_scopes_and_numeric_label(self):
        requested = {"control": {"qwen_prefill_scope": 256}, "scope_2048": {"qwen_prefill_scope": 2048}}
        actual = normalize_variants("qwen", requested)
        self.assertEqual(actual, requested)
        self.assertTrue(arithmetic_class("qwen", actual).startswith("NUMERIC"))

    def test_equal_explicit_scopes_do_not_change_arithmetic_class(self):
        actual = normalize_variants("qwen", {"control": {"qwen_prefill_scope": 256},
                                             "keepalive": {"qwen_prefill_scope": 256, "gpu_keepalive": False}})
        self.assertTrue(arithmetic_class("qwen", actual).startswith("BITWISE"))

    def test_family_variants_are_unchanged(self):
        requested = {"control": {}, "prefetch": {"family_prefetch": True}}
        for family in ["glm", "deepseek"]:
            self.assertEqual(normalize_variants(family, requested), requested)
            self.assertTrue(arithmetic_class(family, requested).startswith("BITWISE"))

    def test_cached_or_unobserved_prefix_is_rejected(self):
        encoded = {"prompt_tokens": 41}
        validate_prompt_isolation({"prompt_tokens": 41, "prompt_tokens_details": {"cached_tokens": 0}}, encoded)
        for details in [{"cached_tokens": 41}, {"cached_tokens": 1}, {}]:
            with self.assertRaisesRegex(ValueError, "prefix reuse"):
                validate_prompt_isolation({"prompt_tokens": 41, "prompt_tokens_details": details}, encoded)
        with self.assertRaisesRegex(ValueError, "prefix reuse"):
            validate_prompt_isolation({"prompt_tokens": 41}, encoded)
        with self.assertRaisesRegex(ValueError, "token count"):
            validate_prompt_isolation({"prompt_tokens": 40, "prompt_tokens_details": {"cached_tokens": 0}}, encoded)


if __name__ == "__main__":
    unittest.main()
