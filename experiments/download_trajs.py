"""Download public SWE-agent trajectories (SWE-bench Lite submissions) from the
public, unsigned S3 bucket used by github.com/SWE-bench/experiments.

usage: python experiments/download_trajs.py data/trajs
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore import UNSIGNED
from botocore.config import Config

sys.path.insert(0, os.path.dirname(__file__))
from corpus import SUBMISSIONS  # noqa: E402

BUCKET = "swe-bench-submissions"


def main(out_root):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, max_pool_connections=32))
    for sub in SUBMISSIONS:
        keys = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"lite/{sub}/trajs/"):
            keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".traj")]
        d = os.path.join(out_root, sub)
        os.makedirs(d, exist_ok=True)

        def get(k):
            fn = os.path.join(d, os.path.basename(k))
            if not os.path.exists(fn):
                s3.download_file(BUCKET, k, fn)

        with ThreadPoolExecutor(16) as ex:
            list(ex.map(get, keys))
        print(sub, len(keys), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/trajs")
