"""Download pinned weights and benchmark archives into a local directory."""

import argparse
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

RTDETR = (
    "https://huggingface.co/PekingU/rtdetr_r18vd/resolve/ac77a11ff0170a41b771c03264987f8ce2b0d753/"
)
ASSETS = {
    "dawn-fog": (
        "DAWN-Fog.zip",
        "https://data.mendeley.com/public-files/datasets/766ygrbt8y/files/"
        "c286e791-88c9-4bec-b43d-2ed276dd11ba/file_downloaded",
        "09f4f8987e030b8da9e76eb73ab00b64fb628cd43df42e460bc26b9a0912c7c2",
    ),
    "gram": (
        "GRAM-RTMv4.zip",
        "https://gram.web.uah.es/data/datasets/rtm/GRAM-RTMv4.zip",
        "acf7d560f0eb27e45325c35feea3e64d0ad1ba324db62ca0656c8ecf56bffb00",
    ),
    "rtdetr-config": (
        "rtdetr-r18/config.json",
        RTDETR + "config.json",
        "8493be71f51a1c0a741f8f71ec151039227579379de7bcba047c0470d9320c3c",
    ),
    "rtdetr-processor": (
        "rtdetr-r18/preprocessor_config.json",
        RTDETR + "preprocessor_config.json",
        "ffb4b9461a1dad746be8f0f9c8330ed7743a1ba5fba4f75c232cd281b3d4c64a",
    ),
    "rtdetr-weights": (
        "rtdetr-r18/model.safetensors",
        RTDETR + "model.safetensors",
        "fe87a5a30f5daf298d10794c7682a63b6107986f97d6a770ba948d89e4340093",
    ),
}


def sha256(path: Path) -> str:
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def fetch(asset: str, root: Path) -> Path:
    name, url, digest = ASSETS[asset]
    path = root / name
    if path.exists():
        if sha256(path) != digest:
            raise ValueError(f"checksum mismatch for {path}; move the file aside before retrying")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with urlopen(url, timeout=45) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if sha256(temporary) != digest:
            raise ValueError(f"downloaded {name} with an unexpected SHA-256")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("assets", choices=tuple(ASSETS), nargs="+")
    args = parser.parse_args()
    try:
        for asset in args.assets:
            print(fetch(asset, args.directory), flush=True)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
