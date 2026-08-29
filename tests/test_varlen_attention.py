"""U13: variable-length, document-isolated attention over packed rows.

Data-free: a tiny GPT2 on CPU with synthetic packs. The load-bearing invariant is
document isolation — a token in one packed document must never influence another
document's hidden states or its anchor. The CPU fallback realizes this by running
each document as its own forward pass (no dense [B,H,L,L] mask), so isolation is
structural; these tests hold the PUBLIC contract so a future "optimize into one
flattened forward" refactor that leaks across documents fails here.
"""

import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel


def _tiny_backbone(vocab: int = 40, seed: int = 0):
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=vocab, n_positions=64, n_embd=16, n_layer=2,
                     n_head=2, bos_token_id=0, eos_token_id=0)
    return GPT2LMHeadModel(cfg).eval()


class VarlenIsolationTest(unittest.TestCase):
    def test_changing_one_document_does_not_move_another(self):
        from src.model.varlen_attention import document_hidden_states

        backbone = _tiny_backbone()
        doc_a, doc_b = [3, 4, 5, 6], [7, 8, 9]
        cu = torch.tensor([0, 4, 7], dtype=torch.int32)

        h1 = document_hidden_states(
            backbone, torch.tensor(doc_a + doc_b), cu, force_fallback=True)
        h2 = document_hidden_states(
            backbone, torch.tensor([10, 11, 12, 13] + doc_b), cu, force_fallback=True)

        # Document B (flattened positions 4:7) is untouched by mutating document A.
        self.assertTrue(torch.equal(h1[4:7], h2[4:7]),
                        "document B leaked information from document A")
        # Document A itself did change (sanity: the mutation was real).
        self.assertFalse(torch.allclose(h1[0:4], h2[0:4]))

    def test_causality_holds_within_a_document(self):
        """A token cannot attend to a later token in its own document."""
        from src.model.varlen_attention import document_hidden_states

        backbone = _tiny_backbone()
        cu = torch.tensor([0, 4], dtype=torch.int32)
        h1 = document_hidden_states(
            backbone, torch.tensor([3, 4, 5, 6]), cu, force_fallback=True)
        # change only the LAST token; the first token's state must not move (causal)
        h2 = document_hidden_states(
            backbone, torch.tensor([3, 4, 5, 39]), cu, force_fallback=True)
        self.assertTrue(torch.equal(h1[0], h2[0]))
        self.assertFalse(torch.equal(h1[3], h2[3]))

    def test_single_document_matches_the_dense_path(self):
        """Byte/numerical equivalence: one document == the existing dense forward."""
        from src.model.head_adapter import CLIFATRONHeads
        from src.model.varlen_attention import document_hidden_states

        backbone = _tiny_backbone()
        doc = [3, 4, 5, 6, 7]
        cu = torch.tensor([0, len(doc)], dtype=torch.int32)
        flat = document_hidden_states(
            backbone, torch.tensor(doc), cu, force_fallback=True)

        model = CLIFATRONHeads(backbone, n_targets=4, freeze_backbone=True)
        dense = model.hidden_states(
            torch.tensor(doc).unsqueeze(0),
            torch.ones(1, len(doc), dtype=torch.long))[0]
        self.assertTrue(torch.allclose(flat, dense, atol=1e-5))

    def test_anchor_gather_selects_each_documents_anchor(self):
        from src.model.varlen_attention import (
            document_hidden_states,
            gather_anchor_states,
        )

        backbone = _tiny_backbone()
        cu = torch.tensor([0, 4, 7], dtype=torch.int32)
        flat = document_hidden_states(
            backbone, torch.tensor([3, 4, 5, 6, 7, 8, 9]), cu, force_fallback=True)
        # anchor doc A at its last token (idx 3), doc B at its last token (idx 6)
        anchors = gather_anchor_states(flat, torch.tensor([3, 6]))
        self.assertEqual(tuple(anchors.shape), (2, 16))
        self.assertTrue(torch.equal(anchors[0], flat[3]))
        self.assertTrue(torch.equal(anchors[1], flat[6]))

    def test_a_length_one_document_and_a_full_row_document_gather_correctly(self):
        from src.model.varlen_attention import (
            document_hidden_states,
            gather_anchor_states,
        )

        backbone = _tiny_backbone()
        cu = torch.tensor([0, 1, 6], dtype=torch.int32)  # doc A len 1, doc B len 5
        flat = document_hidden_states(
            backbone, torch.tensor([3, 4, 5, 6, 7, 8]), cu, force_fallback=True)
        self.assertEqual(flat.size(0), 6)
        anchors = gather_anchor_states(flat, torch.tensor([0, 5]))
        self.assertTrue(torch.equal(anchors[0], flat[0]))
        self.assertTrue(torch.equal(anchors[1], flat[5]))

    def test_an_out_of_range_anchor_fails_closed(self):
        from src.model.varlen_attention import gather_anchor_states

        flat = torch.zeros((7, 16))
        with self.assertRaises(ValueError):
            gather_anchor_states(flat, torch.tensor([3, 7]))  # 7 == total, out of range

    def test_an_anchor_in_the_wrong_document_fails_closed(self):
        """In-global-range but wrong-document anchor must not gather a neighbour's row."""
        from src.model.varlen_attention import gather_anchor_states

        flat = torch.arange(21, dtype=torch.float32).reshape(7, 3)
        boundaries = [0, 4, 7]  # doc A = [0,4), doc B = [4,7)
        # Two anchors both inside doc A (indices 1 and 3) — not strictly increasing docs.
        with self.assertRaisesRegex(ValueError, "document"):
            gather_anchor_states(flat, torch.tensor([1, 3]), boundaries)
        # Backwards: anchor in doc B then doc A.
        with self.assertRaisesRegex(ValueError, "document"):
            gather_anchor_states(flat, torch.tensor([5, 1]), boundaries)
        # In-order, one per document: accepted.
        ok = gather_anchor_states(flat, torch.tensor([3, 5]), boundaries)
        self.assertEqual(tuple(ok.shape), (2, 3))

    def test_zero_anchors_returns_an_empty_gather(self):
        from src.model.varlen_attention import gather_anchor_states

        flat = torch.zeros((5, 3))
        out = gather_anchor_states(flat, torch.tensor([], dtype=torch.long))
        self.assertEqual(tuple(out.shape), (0, 3))

    def test_a_three_document_pack_isolates_and_gathers_all_three(self):
        from src.model.varlen_attention import (
            document_hidden_states,
            gather_anchor_states,
        )

        backbone = _tiny_backbone()
        cu = torch.tensor([0, 3, 5, 9], dtype=torch.int32)  # docs of len 3, 2, 4
        ids = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11])
        flat = document_hidden_states(backbone, ids, cu, force_fallback=True)
        self.assertEqual(flat.size(0), 9)
        # Mutating the MIDDLE document leaves both neighbours bit-identical.
        ids2 = torch.tensor([3, 4, 5, 20, 21, 8, 9, 10, 11])
        flat2 = document_hidden_states(backbone, ids2, cu, force_fallback=True)
        self.assertTrue(torch.equal(flat[0:3], flat2[0:3]), "doc 0 leaked")
        self.assertTrue(torch.equal(flat[5:9], flat2[5:9]), "doc 2 leaked")
        self.assertFalse(torch.allclose(flat[3:5], flat2[3:5]))
        anchors = gather_anchor_states(flat, torch.tensor([2, 4, 8]),
                                       [0, 3, 5, 9])
        self.assertEqual(tuple(anchors.shape), (3, 16))

    def test_a_malformed_pack_fails_closed_before_the_model_runs(self):
        from src.model.varlen_attention import validate_pack

        # cu_seqlens must start at 0 and be non-decreasing.
        with self.assertRaises(ValueError):
            validate_pack(torch.tensor([1, 4, 7], dtype=torch.int32), total=7)
        with self.assertRaises(ValueError):
            validate_pack(torch.tensor([0, 4, 3], dtype=torch.int32), total=3)
        # final boundary must equal the flattened length.
        with self.assertRaises(ValueError):
            validate_pack(torch.tensor([0, 4], dtype=torch.int32), total=7)

    def test_heads_anchor_states_from_pack_returns_per_document_rows(self):
        from src.model.head_adapter import CLIFATRONHeads

        backbone = _tiny_backbone()
        model = CLIFATRONHeads(backbone, n_targets=4, freeze_backbone=True)
        batch = {
            "flash_input_ids": torch.tensor([3, 4, 5, 6, 7, 8, 9]),
            "cu_seqlens": torch.tensor([0, 4, 7], dtype=torch.int32),
            "flash_anchor_idx": torch.tensor([3, 6]),
        }
        anchors = model.anchor_states_from_pack(batch)
        self.assertEqual(tuple(anchors.shape), (2, 16))


class TrainingIsolationGateTest(unittest.TestCase):
    """The condition the multi-doc training reject enforces — data-free, both branches."""

    def test_eager_backbone_is_not_isolation_active(self):
        from src.model.varlen_attention import training_isolation_active

        backbone = _tiny_backbone()  # GPT2, eager attention, no FA2
        self.assertFalse(training_isolation_active(backbone))

    def test_a_config_flipped_to_fa2_over_eager_layers_is_not_trusted(self):
        """The config string can lie: eager layers under a flipped config must fail closed."""
        from src.model.varlen_attention import _uses_flash_attention_2

        class _Cfg:
            _attn_implementation = "flash_attention_2"

        class _Layer(torch.nn.Module):
            def __init__(self, impl):
                super().__init__()
                self._attn_implementation = impl

        class _Model(torch.nn.Module):
            def __init__(self, layer_impl):
                super().__init__()
                self.config = _Cfg()
                self.attn = _Layer(layer_impl)

        # Config says fa2 but the actual layer is eager -> not trusted.
        self.assertFalse(_uses_flash_attention_2(_Model("eager")))
        # Config and the layer agree on fa2 -> trusted.
        self.assertTrue(_uses_flash_attention_2(_Model("flash_attention_2")))

    def test_an_unsupported_architecture_never_takes_the_position_id_path(self):
        """Qwen3-Next reports flash_attention_2 but needs explicit boundary args."""
        from src.model.varlen_attention import (
            _architecture_isolates_from_position_ids,
        )

        class _Cfg:
            def __init__(self, mt):
                self.model_type = mt

        class _M:
            def __init__(self, mt):
                self.config = _Cfg(mt)

        self.assertTrue(_architecture_isolates_from_position_ids(_M("qwen2")))
        self.assertTrue(_architecture_isolates_from_position_ids(_M("qwen3")))
        self.assertFalse(_architecture_isolates_from_position_ids(_M("qwen3_next")))
        self.assertFalse(_architecture_isolates_from_position_ids(_M("gpt2")))

    def test_fa2_config_without_a_gpu_is_still_not_active(self):
        """FA2 in config is necessary but not sufficient — the hardware must be there."""
        from src.model.varlen_attention import (
            flash_attention_available,
            training_isolation_active,
        )

        backbone = _tiny_backbone()
        backbone.config._attn_implementation = "flash_attention_2"
        # On a CPU CI box flash_attention_available() is False, so the gate stays closed.
        self.assertEqual(training_isolation_active(backbone),
                         flash_attention_available())


@unittest.skipUnless(
    torch.cuda.is_available(), "FlashAttention-2 path requires a CUDA device")
class VarlenFlashAttentionTest(unittest.TestCase):
    def test_fa2_and_fallback_agree_on_anchors_when_available(self):
        from src.model.varlen_attention import (
            document_hidden_states,
            flash_attention_available,
        )

        if not flash_attention_available():
            self.skipTest("flash-attn not installed")
        backbone = _tiny_backbone().cuda()
        ids = torch.tensor([3, 4, 5, 6, 7, 8, 9])
        cu = torch.tensor([0, 4, 7], dtype=torch.int32)
        fallback = document_hidden_states(backbone, ids, cu, force_fallback=True)
        # Reload with FA2 to exercise the GPU path.
        backbone.config._attn_implementation = "flash_attention_2"
        fa2 = document_hidden_states(backbone, ids, cu)
        self.assertTrue(torch.allclose(fallback, fa2.cpu(), atol=1e-2))


if __name__ == "__main__":
    unittest.main()
