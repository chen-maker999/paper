"""Fetch the BPE tokenizer used for all token counts (the legacy Claude BPE that
shipped inside the MIT-licensed `anthropic` Python SDK, version 0.28.0).

usage: python experiments/get_tokenizer.py data/tokenizer.json
then:  export SKC_TOKENIZER=data/tokenizer.json
"""
import subprocess
import sys
import tempfile
import zipfile
import glob
import os

out = sys.argv[1] if len(sys.argv) > 1 else "data/tokenizer.json"
with tempfile.TemporaryDirectory() as d:
    subprocess.check_call([sys.executable, "-m", "pip", "download", "-q", "--no-deps", "anthropic==0.28.0", "-d", d])
    whl = glob.glob(os.path.join(d, "anthropic-0.28.0-*.whl"))[0]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with zipfile.ZipFile(whl) as z, open(out, "wb") as f:
        f.write(z.read("anthropic/tokenizer.json"))
print("wrote", out)
