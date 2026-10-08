"""Download the exact upstream pretrained generator and its example input sheet."""
import hashlib
from pathlib import Path
import urllib.request
import zipfile
import gdown

REVISION='f1231fcc581133c9a7d8451e26a8bcd293fe03d5'
ARCHIVE_HASH='c3d8d35c123e9b994148fa8488df0ec4f7a3e7febfdd1a324259f5a888575b5f'
GRAPH_HASH='e595d1602e9ff3267c138342c6dfb530ef65842f1452d1ca6322760f0208a60e'
GRID_HASH='361d5130d8c96b11bb6ca6628b69fc82a9d61cd86074893b48181ff214d7ad91'


def verify(path,expected):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
        raise ValueError(f'Checksum mismatch: {path}')


def main():
    root=Path('artifacts/upstream'); root.mkdir(parents=True,exist_ok=True)
    graph=root/'AttGAN_128/generator.pb'
    if not graph.exists():
        archive=root/'AttGAN_128_generator_only.zip'
        if not archive.exists():
            partial=archive.with_suffix('.partial')
            if not gdown.download(id='1lcQ-ijNrGD4919eJ5Dv-7ja5rsx5p0Tp',output=str(partial),quiet=False):
                raise RuntimeError('Upstream model unavailable; retry the official link later.')
            verify(partial,ARCHIVE_HASH); partial.replace(archive)
        verify(archive,ARCHIVE_HASH)
        with zipfile.ZipFile(archive) as z:
            # Extract only the two expected files; do not trust arbitrary archive paths.
            for name in ['AttGAN_128/generator.pb','AttGAN_128/settings.yml']:
                target=root/name; target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(z.read(name))
    verify(graph,GRAPH_HASH)
    grid=root/'sample_validation.jpg'
    if not grid.exists():
        url=f'https://raw.githubusercontent.com/LynnHo/AttGAN-Tensorflow/{REVISION}/pics/sample_validation.jpg'
        with urllib.request.urlopen(url,timeout=60) as response:
            contents=response.read(10*1024*1024)
        if hashlib.sha256(contents).hexdigest()!=GRID_HASH:
            raise ValueError('Example image checksum mismatch')
        grid.write_bytes(contents)
    verify(grid,GRID_HASH)
    print(f'Verified {graph} and {grid}')


if __name__=='__main__':
    main()
