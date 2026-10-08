"""Generic patchwise pairing and deterministic source-graph invariants."""

import unittest

import numpy as np

import tg_pairing


class TGPairingTests(unittest.TestCase):
    def test_source_graph_has_unique_sorted_edges_and_deterministic_ties(self):
        source = np.array([[0., 0.], [1., 0.], [-1., 0.], [8., 0.]])
        before = source.copy()
        edges = tg_pairing.build_source_edges(source, 1)
        np.testing.assert_array_equal(edges, np.array([[0, 1], [0, 2], [1, 3]]))
        self.assertEqual(edges.dtype, np.dtype("int64"))
        np.testing.assert_array_equal(edges, tg_pairing.build_source_edges(source, 1))
        np.testing.assert_array_equal(source, before)

    def test_source_graph_keeps_neighbors_across_patch_membership(self):
        source = np.array([[0., 0.], [.01, 0.], [10., 0.], [10.01, 0.]])
        labels = np.array([0, 1, 0, 1])
        edges = tg_pairing.build_source_edges(source, 1)
        np.testing.assert_array_equal(edges, np.array([[0, 1], [2, 3]]))
        self.assertTrue(np.all(labels[edges[:, 0]] != labels[edges[:, 1]]))

    def test_patchwise_random_bijections_are_valid_and_deterministic(self):
        source_labels = np.array([0, 0, 0, 1, 1, 1])
        target_labels = np.array([0, 1, 0, 1, 0, 1])
        originals = source_labels.copy(), target_labels.copy()
        a_rng, b_rng = np.random.default_rng(19), np.random.default_rng(19)
        samples = []
        for _ in range(8):
            first = tg_pairing.random_valid_permutation(source_labels, target_labels, 2, a_rng)
            second = tg_pairing.random_valid_permutation(source_labels, target_labels, 2, b_rng)
            self.assertEqual(first.dtype, np.dtype("int32"))
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(np.sort(first), np.arange(6))
            np.testing.assert_array_equal(target_labels[first], source_labels)
            samples.append(tuple(first))
        self.assertGreater(len(set(samples)), 1)
        np.testing.assert_array_equal(source_labels, originals[0])
        np.testing.assert_array_equal(target_labels, originals[1])

    def test_helpers_do_not_mutate_numpy_global_rng(self):
        before = np.random.get_state()
        tg_pairing.build_source_edges(np.array([[0., 0.], [1., 0.], [2., 0.]]), 1)
        labels = np.array([0, 0, 1, 1])
        tg_pairing.random_valid_permutation(labels, labels, 2, np.random.default_rng(7))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_source_graph_rejects_invalid_points_and_neighbor_counts(self):
        invalid_sources = ([], [1., 2.], np.empty((0, 2)), [[np.nan, 0.], [1., 0.]],
                           [[np.inf, 0.], [1., 0.]], [[1j, 0.], [1., 0.]],
                           [[1e308, 0.], [-1e308, 0.]])
        for source in invalid_sources:
            with self.subTest(source=repr(source)), self.assertRaises(ValueError):
                tg_pairing.build_source_edges(source, 1)
        source = np.array([[0., 0.], [1., 0.], [2., 0.]])
        for neighbors in (0, -1, 3, 4, True, 1.5, "1"):
            with self.subTest(neighbors=neighbors), self.assertRaises(ValueError):
                tg_pairing.build_source_edges(source, neighbors)

    def test_patchwise_pairing_rejects_bad_labels_counts_and_patch_count(self):
        labels = np.array([0, 0, 1, 1])
        invalid = (
            (labels, np.array([0, 1, 1, 1]), 2),
            (labels, np.array([0, 0, 1]), 2),
            (labels.astype(float), labels, 2),
            (labels, np.array([0, 0, 1, -1]), 2),
            (labels, np.array([0, 0, 1, 2]), 2),
            (labels.reshape(2, 2), labels, 2),
            (np.array([], dtype=int), np.array([], dtype=int), 1),
            (labels, labels, 0),
            (labels, labels, 5),
            (labels, labels, True),
            (labels, labels, 2.5),
        )
        for source, target, k in invalid:
            with self.subTest(source=source, target=target, k=k), self.assertRaises(ValueError):
                tg_pairing.random_valid_permutation(source, target, k, np.random.default_rng(7))


if __name__ == "__main__":
    unittest.main()
