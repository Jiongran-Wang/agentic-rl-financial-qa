#!/usr/bin/env python3
"""Verify and unpack the public experiment input bundle; never overwrite changes."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def unpack():
    manifest = json.loads((ROOT / 'artifacts/input-manifest.json').read_text())
    archive = ROOT / 'artifacts' / manifest['archive']
    sha = lambda b: hashlib.sha256(b).hexdigest()
    if sha(archive.read_bytes()) != manifest['sha256']:
        raise ValueError('Input archive checksum mismatch')
    contents = {}
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                    or path.parts[0] not in {'data', 'results'}
                    or member.name in contents):
                raise ValueError('Unexpected archive member')
            data = tf.extractfile(member).read()
            if sha(data) != manifest['file_hashes'].get(member.name):
                raise ValueError('Input member checksum mismatch: ' + member.name)
            target = ROOT / member.name
            if target.exists() and target.read_bytes() != data:
                raise FileExistsError('Refusing to overwrite changed input: ' + member.name)
            contents[member.name] = data
    if set(contents) != set(manifest['file_hashes']):
        raise ValueError('Incomplete input archive')
    for name, data in contents.items():
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(f'Verified and unpacked {len(contents)} input files.')


if __name__ == '__main__':
    unpack()
