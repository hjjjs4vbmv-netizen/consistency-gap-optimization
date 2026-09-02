import argparse
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from scripts import run_imagenet64_feature_queue as queue


class ImageNet64FeatureQueueTests(unittest.TestCase):
    def test_three_shards_are_disjoint_and_complete(self):
        matrix = queue.jobs()
        shards = [queue.shard_jobs(matrix, index, 3) for index in range(3)]
        self.assertEqual([len(shard) for shard in shards], [40, 40, 40])
        self.assertEqual(set().union(*map(set, shards)), set(matrix))
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertTrue(set(shards[left]).isdisjoint(shards[right]))

    def test_one_to_three_distinct_gpus_are_supported(self):
        self.assertEqual(queue.parse_gpus("6"), ("6",))
        self.assertEqual(queue.parse_gpus("0,1"), ("0", "1"))
        self.assertEqual(queue.parse_gpus("0,1,2"), ("0", "1", "2"))
        for value in ("", "0,0", "0,1,2,3"):
            with self.subTest(value=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                queue.parse_gpus(value)

    def test_shards_use_disjoint_default_worker_ports(self):
        self.assertEqual(
            [queue.worker_port(29610, shard, 0) for shard in range(3)],
            [29610, 29613, 29616],
        )

    def test_existing_feature_header_is_checked_without_loading_payload(self):
        job = queue.jobs()[0]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            queue, "FEATURE_SHAPE", (3, 8)
        ):
            path = Path(directory) / "features.final.npy"
            np.save(path, np.zeros((3, 8), dtype=np.float32))
            queue.validate_feature_file(path, job)

            np.save(path, np.zeros((3, 7), dtype=np.float32))
            with self.assertRaisesRegex(RuntimeError, "existing feature header"):
                queue.validate_feature_file(path, job)

            np.save(path, np.zeros((3, 8), dtype=np.float64))
            with self.assertRaisesRegex(RuntimeError, "existing feature header"):
                queue.validate_feature_file(path, job)

            path.write_bytes(b"not a numpy file")
            with self.assertRaisesRegex(RuntimeError, "invalid existing feature"):
                queue.validate_feature_file(path, job)


if __name__ == "__main__":
    unittest.main()
