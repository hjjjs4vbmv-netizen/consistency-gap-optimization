from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.check_evaluation_dataset_equivalence import compare_entries


class DatasetEquivalenceTests(unittest.TestCase):
    def test_different_containers_same_entries_are_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = (Path(temp) / name for name in ('a.zip', 'b.zip'))
            for path, compression in ((left, zipfile.ZIP_STORED), (right, zipfile.ZIP_DEFLATED)):
                with zipfile.ZipFile(path, 'w', compression=compression) as z:
                    z.writestr('image.png', b'original-image-bytes')
                    z.writestr('dataset.json', b'{"labels":null}')
            self.assertNotEqual(left.read_bytes(), right.read_bytes())
            self.assertEqual(compare_entries(left, right, 1)['status'], 'ALL_ENTRY_BYTES_MATCH')

    def test_changed_image_or_labels_are_rejected(self):
        for changed in ('image.png', 'dataset.json'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temp:
                paths = [Path(temp) / name for name in ('a.zip', 'b.zip')]
                for index, path in enumerate(paths):
                    with zipfile.ZipFile(path, 'w') as z:
                        for name in ('image.png', 'dataset.json'):
                            z.writestr(name, b'changed' if index and name == changed else b'original')
                with self.assertRaisesRegex(ValueError, 'entry differs'):
                    compare_entries(*paths, expected_images=1)


if __name__ == '__main__':
    unittest.main()
