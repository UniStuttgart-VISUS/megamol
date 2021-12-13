from PIL import Image
import argparse
import re
import os

parser = argparse.ArgumentParser(usage="%(prog)s <FILE>", description="try to reconstruct a MegaMol for running a screenshot")
parser.add_argument('file')
args = parser.parse_args()
if not args.file:
    print ("need one input file")
    exit(1)
try:
    im = Image.open(args.file)
    im.load()
    soft = im.info['Software']
    branch = "master"
    url = im.info['RemoteURL']
    m = re.search(r'/(.*?)$', im.info['RemoteBranch'])
    if m:
        branch = m.group(1)
    # MegaMol 1.3.16a73a1d7523-dirty
    m = re.search(r'MegaMol \d+\.\d+\.([^-]+)(-dirty)?', soft)
    if m:
        myhash = m.group(1)
        print(f"This is a MegaMol screenshot. Using hash '{myhash}' on branch '{branch}' at '{url}'.")
        if m.group(2):
            print("Warning! Screenshot taken using MegaMol with uncommitted changes!")
            print("Press [Enter] if you want to continue:")
            input()

        os.system(f"git clone {url} repro-megamol --depth 1 --branch {branch}")
        os.chdir('repro-megamol')
        os.system(f'git checkout {myhash}')
        os.mkdir('build')
        os.chdir('build')
        out = open("CMakeCache.txt", "w")
        out.write(im.info['CMakeCache'])
        out.close()
        m = re.search(r'CMAKE_CACHEFILE_DIR=(\S+)', im.info['CMakeCache'])
        if m:
            print("\n=========\n")
            print("\nI have tried checking out the correct source into repro-megamol.")
            print("I have created a build/ directory there and placed the correct CMakeCache.txt inside.")
            print(f"However, the build was originally located at {m.group(1)}. Once you move it there, you should be able to recreate a suitable binary.")
    else:
        print("This is not a MegaMol screenshot, exiting")
except (FileNotFoundError, IsADirectoryError) as err:
    print(f"{args.file}: {err.strerror}")
