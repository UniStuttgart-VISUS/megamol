import argparse
import os
import os.path
import glob
from PIL import Image
import subprocess
import re

parser = argparse.ArgumentParser(usage="%(prog)s <DIRECTORY>", description="execute test scripts in DIRECTORY")
parser.add_argument('directories', nargs="*")
parser.add_argument('--generate-reference', action='count', help='generate reference pngs instead of testing against them')
parser.add_argument('--generate-neutral-test', action='count', help='generate a first basic test (.1) for all found projects')
args = parser.parse_args()

resultname = 'result.png'
istest = re.compile(r'.*\.test\.\d+\.lua')

if not args.directories:
    print("need at least one input directory")
    exit(1)
for dir in args.directories:
    for subdir, dirs, files in os.walk(dir, topdown=True):
        for file in files:
            entry = os.path.join(subdir, file)
            if (not istest.match(entry)):
                #print(f'found {entry}')
                testname, _ = os.path.splitext(entry)
                #print(f'using {testname}')
                if args.generate_neutral_test:
                    tfname = f'{testname}.test.1.lua'
                    if not os.path.isfile(tfname):
                        with open(tfname, "w") as outfile:
                            print(f'generating neutral test {tfname}')
                            outfile.write('mmRenderNextFrame()\nmmScreenshot("result.png")\nmmQuit()\n')
                    continue

                tests = list(glob.iglob(f'{testname}.test.*.lua'))
                if len(tests) > 0:
                    print(f'found tests for {file}')
                    for testfile in tests:
                        print(f"running test {testfile}")
                        refname = testfile + ".png"
                        with subprocess.Popen(f'megamol.exe --nogui {entry} {testfile}') as proc:
                            proc.wait()
                            if args.generate_reference:
                                try:
                                    os.rename(resultname, refname)
                                except:
                                    print(f'could not move {resultname} to {refname}')
                            else:
                                try:
                                    reference_image = Image.open(refname)
                                    print(f'testing {entry}')
                                except:
                                    print(f'could not open reference image {testname + ".png"}')

