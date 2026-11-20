"""Semantic Sensemaking piece: Sensemaking: draft lexicon.

Runs draft_lexicon from helpfultangent/Semantic-Sensemaking at commit
6f586c7f2a81ab9039cdd6d1c13196e1519ef508.
Its inputs arrive one file each (SOURCE_<position>) and it reports where each
of its outputs is. Dependencies are installed at exact versions, each checked
against its hash; the library is loaded from the commit's archive, checked
against a recorded SHA-256. Nothing else is installed.
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

_PIECE = "draft_lexicon"
_REF = "6f586c7f2a81ab9039cdd6d1c13196e1519ef508"
_ARCHIVE = "https://github.com/helpfultangent/Semantic-Sensemaking/archive/" + _REF + ".tar.gz"
_ARCHIVE_SHA256 = "1dcea12d281310e9cd03f1c397fa593a5a0c272796803f341c1b4511fda7777b"
# input position -> the library's name for that input
_INPUTS = {'0': 'CORPUS_URI'}
# output position -> where the library's result says that output is
_OUTPUTS = {'0': 'lexicon_uri', '1': 'taxonomy_uri'}
_LOCK = """\
annotated-types==0.8.0 --hash=sha256:f072f4d804ea359e4eaf198b1af7a8b0943881a87f31bb764f8bf219bb9419e0
anyio==4.16.0 --hash=sha256:c66ae634e0b8228396b801741beb3294326d4b6102f7bba4041c0dfb58c29026
certifi==2026.10.23 --hash=sha256:e9b54de76e530bb6cf0ccd036787103acb58fd42ba930ea8bf948ddbaa2f61d1
h11==0.16.0 --hash=sha256:63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86
httpcore==1.0.9 --hash=sha256:2d400746a40668fc9dec9810239072b40b4484b640a8c38fd654a024c7a1bf55
httpx==0.28.1 --hash=sha256:d909fcccc110f8c7faf814ca82a9a4d816bc5a6dbfea25d6591d6985b8ba59ad
idna==3.20 --hash=sha256:ab7ae7122974553370f0bdb919e1a960b2cd1bc1ef0276416d896db81c14582c
lxml==6.1.3 --hash=sha256:49fbc2682a9306135b7ec49e93f97f9c26689b9b7f96ed2742d8d6497e994d13 --hash=sha256:527195c188d7d0af748cd48d220ab8cdc5cb99be3d49ac4d9be7324d8abf9bc0
numpy==2.4.6 --hash=sha256:89cd468399cfd2504718f0ba50e410dca55a170b61a02ad92bb18c8a65186e93
pandas==3.1.1 --hash=sha256:4f1602f614bb83f443381dbcbb94c4ac764b2e50c8d6c965614c5cd83e328c47
pydantic==2.14.4 --hash=sha256:653a2ad9f6d8858529c77c72717d0ea991d6c6837fe683246b2f4b2bbfc3fadc
pydantic-core==2.50.5 --hash=sha256:488319076ec92908b91fc6265ce761f1d78748f69f85518564fd16bf333e1b70
pydantic-settings==2.16.3 --hash=sha256:d1db16a2d34006323a1b42c0640898da58adfbad7ec30baaa9e2b18aafc08999
pypdf==7.1.1 --hash=sha256:04c5680d4b4cfa04ab505b09393fc949860c34a4dad6565284d11a2f5c3ce8b1
python-dateutil==4.0.0 --hash=sha256:a5e246f79eb80475e691bf93862cead98decd8a498cf53ba0516fcdc0f50dc1c
python-docx==1.2.0 --hash=sha256:3fd478f3250fbbbfd3b94fe1e985955737c145627498896a8a6bf81f4baf66c7
python-dotenv==1.2.4 --hash=sha256:42269a8a5b3fd54ffa6f3d84b18abed50064717576b4ecf03dc4a55d8aa04fdc
pyyaml==6.0.3 --hash=sha256:b8bb0864c5a28024fac8a632c443c87c5aa6f215c0b126c449ae1a150412f31d
tabulate==0.10.0 --hash=sha256:f0b0622e567335c8fabaaa659f1b33bcb6ddfe2e496071b743aa113f8774f2d3
typing-extensions==4.16.0 --hash=sha256:481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8
typing-inspection==0.4.5 --hash=sha256:bc210535442b39e1bf2805e0c7a17dfd6a4ace760c84c7f7d16245d23a6b1c39
tzdata==2026.6 --hash=sha256:0a10eec6e13ef16b627fabed9a5f2ba61a62b1c67f327305f4e7e412a70b4aec
"""


def _load_library():
    work = tempfile.mkdtemp(prefix="ssm-")
    lock = os.path.join(work, "lock.txt")
    with open(lock, "w", encoding="utf-8") as handle:
        handle.write(_LOCK)
    subprocess.check_call([
        sys.executable, "-m", "pip", "--isolated", "install", "--quiet", "--disable-pip-version-check",
        "--no-input", "--require-hashes", "--no-deps", "--only-binary=:all:",
        "--index-url", "https://pypi.org/simple", "-r", lock,
    ])
    with urllib.request.urlopen(_ARCHIVE, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != _ARCHIVE_SHA256:
        raise RuntimeError("the library archive is not the one this piece was approved with")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        if hasattr(tarfile, "data_filter"):
            archive.extractall(work, filter="data")
        else:  # an older Python: the archive's hash was checked above, and nothing odd is let through
            for member in archive.getmembers():
                if member.name.startswith("/") or ".." in member.name.split("/") or member.issym() or member.islnk():
                    raise RuntimeError("the library archive holds a path this piece will not extract")
            archive.extractall(work)
    sys.path.insert(0, os.path.join(work, "Semantic-Sensemaking-" + _REF, "src"))


def transform(inputs):
    given = {str(k): v for k, v in (inputs or {}).items() if v is not None and str(v) != ""}
    # Each input is linked by itself; nothing is carried over from an earlier piece.
    given.pop("SOURCE_RESULT", None)
    for position, name in _INPUTS.items():
        value = given.pop("SOURCE_" + position, None)
        if value:
            given[name] = value
    missing = sorted(name for name in _INPUTS.values() if name not in given)
    if missing:
        raise ValueError(_PIECE + ": not linked to an input: " + ", ".join(missing))
    _load_library()
    from semantic_sensemaking.pieces import run_piece

    result = json.loads(run_piece(_PIECE, given))
    return {"svo_flow": {"outputs": {position: result[key] for position, key in _OUTPUTS.items()}}}
