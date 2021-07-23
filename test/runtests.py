import argparse
import os
import os.path
import glob
from PIL import Image
from SSIM_PIL import compare_ssim
import subprocess
import re

parser = argparse.ArgumentParser(usage="%(prog)s <DIRECTORY>", description="execute test scripts in DIRECTORY")
parser.add_argument('directories', nargs="*")
parser.add_argument('--generate-reference', action='count', help='generate reference pngs instead of testing against them')
parser.add_argument('--generate-neutral-test', action='count', help='generate a first basic test (.1) for all found projects')
args = parser.parse_args()

resultname = 'result.png'
istest = re.compile(r'.*\.test\.\d+\.lua')
testresults = []

ssim_threshold = 0.95
class TestResult:
    testfile: str
    passed: bool
    result: str

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
                        stdoutfile = f'{testfile}.stdout'
                        stderrfile = f'{testfile}.stderr'
                        if os.path.isfile(resultname):
                            os.remove(resultname)
                        if os.path.isfile(stdoutfile):
                            os.remove(stdoutfile)
                        # if os.path.isfile(stderrfile):
                        #     os.remove(stderrfile)
                        print(f"running test {testfile}... ", end='')
                        tr = TestResult()
                        tr.testfile=testfile
                        tr.passed=True
                        refname = testfile + ".png"
                        compl = subprocess.run(f'megamol.exe --nogui {entry} {testfile}', capture_output=True)
                        if args.generate_reference:
                            try:
                                os.rename(resultname, refname)
                            except:
                                print(f'could not move {resultname} to {refname}')
                        else:
                            if not os.path.isfile(resultname):
                                print(f'failed')
                                tr.passed = False
                                tr.result = "no output generated"
                                testresults.append(tr)
                                continue
                            if not os.path.isfile(refname):
                                print(f'failed')
                                tr.passed = False
                                tr.result = "missing reference image"
                                testresults.append(tr)
                                continue
                            try:
                                reference_image = Image.open(refname)
                                result_image = Image.open(resultname)
                                ssim = compare_ssim(reference_image, result_image, GPU=False)
                                if ssim > ssim_threshold:
                                    print(f'passed ({ssim})')
                                else:
                                    print(f'failed ({ssim})')
                                    tr.passed = False
                                    with open(stdoutfile, "wb") as outfile:
                                        outfile.write(compl.stdout)
                                    # with open(stderrfile, "wb") as outfile:
                                    #     outfile.write(compl.stderr)

                                tr.result = f'SSIM = {ssim}'
                                testresults.append(tr)
                            except Exception as e:
                                print(f'unexpected exception: {e}')

for tr in testresults:
    print(f'{tr.testfile}: {"passed" if tr.passed else "failed"} {tr.result}')