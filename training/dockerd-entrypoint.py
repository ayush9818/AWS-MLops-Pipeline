import os
import shlex
import subprocess
import sys
from subprocess import CalledProcessError


def main():
    if sys.argv[1] == "train":
        subprocess.call(['python3','/opt/ml/code/train.py'])
    else:
        subprocess.check_call(shlex.split(" ".join(sys.argv[1:])))

main()