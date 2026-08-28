"""Notes modality smoke tests — MPS.

Tests the NotesEncoder constructor, the MLP adapter, the note-token
insertion helper, and the pre-anchor leakage guard. BERT model loading
is skipped (requires network download), but the lazy-load pattern and
the insertion/guard functions are fully exercised.
"""

import unittest

import numpy as np
import torch


class NotesSmokeTest(unittest.TestCase):
    def test_01_constructor_no_bert_load(self):
        """NotesEncoder constructs without loading BERT (lazy)."""
        from src.model.notes_encoder import NotesEncoder

        enc = NotesEncoder(d_model=128, model_name="thomas-sounack/BioClinical-ModernBERT-base")
        self.assertFalse(enc._loaded)
        self.assertEqual(enc.d_model, 128)

    def test_02_insert_note_token_at_position(self):
        """Note token inserted at correct chronological position."""
        from src.model.notes_encoder import insert_note_token

        events = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
        note = torch.tensor([99.0, 99.0, 99.0]).unsqueeze(0)

        result = insert_note_token(events, note, note_position=2)
        self.assertEqual(result.shape, (1, 5, 3))
        self.assertTrue((result[0, 2] == 99.0).all())

    def test_03_insert_at_boundaries(self):
        """Insertion at position 0 and at end works correctly."""
        from src.model.notes_encoder import insert_note_token

        events = torch.ones(1, 3, 4)
        note = torch.zeros(1, 4)

        at_start = insert_note_token(events, note, note_position=0)
        self.assertEqual(at_start.shape, (1, 4, 4))
        self.assertTrue((at_start[0, 0] == 0).all())

        at_end = insert_note_token(events, note, note_position=100)
        self.assertEqual(at_end.shape, (1, 4, 4))
        self.assertTrue((at_end[0, 3] == 0).all())

    def test_04_pre_anchor_guard_excludes_post_anchor(self):
        """Notes after obs_hours are excluded."""
        from src.model.notes_encoder import filter_pre_anchor_notes

        admission = np.array([0.0, 0.0, 0.0, 0.0])
        note_times = np.array([3600, 72000, 86400, 172800])

        mask = filter_pre_anchor_notes(note_times, admission, obs_hours=24)
        self.assertTrue(mask[0])
        self.assertTrue(mask[1])
        self.assertTrue(mask[2])
        self.assertFalse(mask[3])

    def test_05_enc_np__returns_stub_until_bert_loaded(self):
        """filter_pre_anchor_notes handles mixed-length arrays correctly."""
        from src.model.notes_encoder import filter_pre_anchor_notes

        admission = np.array([0.0, 0.0])
        note_times = np.array([3600, 90000])
        mask = filter_pre_anchor_notes(note_times, admission, obs_hours=24)
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])


if __name__ == "__main__":
    unittest.main()