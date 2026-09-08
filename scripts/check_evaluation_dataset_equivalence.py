"""Compare dataset entries when ZIP container hashes differ."""
import argparse
import json
from pathlib import Path
import zipfile

from scripts.build_m1_evaluation_slots import sha256_file


def compare_entries(left, right, expected_images=50000):
    with zipfile.ZipFile(left) as a, zipfile.ZipFile(right) as b:
        names = sorted(a.namelist())
        if names != sorted(b.namelist()) or len(names) != len(set(names)):
            raise ValueError('dataset entry names differ or contain duplicates')
        images = sum(name.endswith('.png') for name in names)
        if images != expected_images or 'dataset.json' not in names:
            raise ValueError('unexpected image count or absent dataset metadata')
        for name in names:
            if a.read(name) != b.read(name):
                raise ValueError('uncompressed dataset entry differs: ' + name)
    return dict(status='ALL_ENTRY_BYTES_MATCH', images=images, entries=len(names))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare_entries(args.original, args.evaluation)
    result.update(original_path=str(args.original.resolve()),
                  original_sha256=sha256_file(args.original),
                  evaluation_path=str(args.evaluation.resolve()),
                  evaluation_sha256=sha256_file(args.evaluation))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
