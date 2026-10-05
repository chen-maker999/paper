"""Download public agent trajectories from the public, unsigned S3 bucket used by
github.com/SWE-bench/experiments.

usage: python experiments/download_trajs.py data/trajs            # SWE-agent on SWE-bench Lite (~6 GB)
       python experiments/download_trajs.py data/trajs_v --verified # OpenHands etc. on Verified (~4 GB)
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore import UNSIGNED
from botocore.config import Config

sys.path.insert(0, os.path.dirname(__file__))
from corpus import SUBMISSIONS, VERIFIED  # noqa: E402

BUCKET = "swe-bench-submissions"


def main(out_root, verified=False):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, max_pool_connections=32))
    split, subs = ("verified", VERIFIED) if verified else ("lite", SUBMISSIONS)
    for sub in subs:
        keys = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"{split}/{sub}/trajs/"):
            keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith((".traj", ".json"))]
        d = os.path.join(out_root, sub)
        os.makedirs(d, exist_ok=True)

        def get(k):
            base = os.path.basename(k)
            if base == "trajectory.json":  # one directory per instance
                base = k.split("/")[-2] + ".json"
            fn = os.path.join(d, base)
            if not os.path.exists(fn):
                s3.download_file(BUCKET, k, fn)

        with ThreadPoolExecutor(16) as ex:
            list(ex.map(get, keys))
        print(sub, len(keys), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/trajs", verified="--verified" in sys.argv)
