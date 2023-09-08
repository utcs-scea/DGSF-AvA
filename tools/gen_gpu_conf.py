#!/usr/bin/python3

import argparse
import subprocess
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", default="gpu.conf", type=str)
    args = parser.parse_args()

    with subprocess.Popen(["nvidia-smi", "-L"], stdout=subprocess.PIPE) as proc:
        outs, errs = proc.communicate()
    outs = outs.decode("utf-8")
    out = outs.splitlines()
    pattern = re.compile(r"GPU-[0-9a-f\-]*", re.I)

    with open(args.o, "w") as f:
        for l in out:
            ret = re.search(pattern, l)
            uuid = ret.group(0)
            f.write("CUDA_VISIBLE_DEVICES={}\n".format(uuid))
