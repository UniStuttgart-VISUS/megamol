import sys
import subprocess
import tempfile
import os
import re
import string
# pip install jsondiff --user
import jsondiff 
import json

import argparse

parser = argparse.ArgumentParser(description='checks OpenGL states when entering/leaving Render() and during module creation (only Renderer[23]DModule)')
parser.add_argument('args', nargs=argparse.REMAINDER, help='The args you want to pass to MegaMol. Separate from the rest using \'--\' to allow for dashes inside those.')
parser.add_argument('--apitrace', action='store', help='the full path of apitrace.exe', default='T:\\Utilities\\apitrace-msvc\\x64\\bin\\apitrace.exe')
parser.add_argument('--exe', action='store', help='the frontend you want to use (defaults to megamol.exe)', default='megamol.exe')
parser.add_argument('--keeptrace', action='count', help='do not delete the trace file')

parseResult = parser.parse_args()
the_apitrace = parseResult.apitrace
the_exe = parseResult.exe
the_args = parseResult.args
the_keeptrace = parseResult.keeptrace
# the_apitrace = 'C:\\Utilities\\apitrace-msvc\\x64\\bin\\apitrace.exe'
# the_exe = 'mmconsole.exe'
# the_args = ['-p', '..\\examples\\testspheres.lua']
# the_keeptrace = 1

if not os.path.isfile(the_apitrace):
    sys.exit("cannot find apitrace executable ({}), please use --apitrace to specify".format(the_apitrace))

if not os.path.isfile(the_exe):
    sys.exit("cannot find the executable ({}), please use --exe to specify".format(the_exe))

groupstack = []

def safeString(text):
    t = re.sub(r'[\n\r]', '', text, 0, flags=re.DOTALL)
    t = re.sub(r'[\t]', '    ', t, 0, flags=re.DOTALL)
    t = re.sub(r'\"__data__\":.*?\".*?\"', '\"__data__\":\"omitted\"', t, 0, flags = re.DOTALL)
    return t

def findDebugGroups(startnum, endnum, mmtracefile):
    # dump that frame
    #c:\utilities\apitrace-8.0.20190414-win64\bin\apitrace.exe dump mmconsole.trace --calls=23165-23562
    args = [the_apitrace, 'dump', '--calls=' + startnum + '-' + endnum, '--color=never', mmtracefile]
    proc = subprocess.run(args, capture_output=True)
    res = proc.stdout.decode("utf-8") 

    #find interesting debuggroups
    # 30079 glPushDebugGroup(source = GL_DEBUG_SOURCE_APPLICATION, id = 1234, length = -1, message = "SphereRenderer::Render")
    commands = res.split(os.linesep)
    done = {}
    groupstart = 0
    groupend = 0
    currgroup = ""
    for c in commands:
        m = re.search(r'^(\d+)\s+glPopDebugGroup', c)
        if m:
            groupend = m.group(1)
            if len(groupstack) == 0:
                print("found pop without push: " + c)
            else:
                groupstart, currgroup = groupstack.pop()
                print("found end of debug group " + currgroup + " " + groupstart + "-" + groupend)
                if int(groupend) - int(groupstart) == 1:
                    print("no content")
                else:
                    # dump both states
                    # c:\utilities\apitrace-msvc\x64\bin\apitrace.exe replay -D 167273 mmconsole.1.trace > before.json
                    args = [the_apitrace, 'replay', '-D', groupstart, mmtracefile]
                    proc = subprocess.run(args, capture_output=True)
                    text = safeString(proc.stdout.decode("ascii"))
                    before = json.loads(text)

                    args = [the_apitrace, 'replay', '-D', groupend, mmtracefile]
                    proc = subprocess.run(args, capture_output=True)
                    text = safeString(proc.stdout.decode("ascii"))
                    after = json.loads(text)

                    diffstr = jsondiff.diff(before, after, marshal=True)

                    print("found differences:")
                    json.dump(diffstr, sys.stdout, indent=2)
                    print("")
        m = re.search(r'^(\d+)\s+glPushDebugGroup', c)
        if m:
            thestart = m.group(1)
            m = re.search(r',\s+id\s+=\s+(\d+)', c)
            if m:
                m = re.search(r',\s+message\s+=\s+"(.*?)"', c)
                if m:
                    thingy = m.group(1)
                    if thingy in done:
                        print("already looked at " + thingy)
                    else:
                        print("looking at " + thingy)
                        # do not look twice. the first frame seems to be
                        # different though, so I deactivated this
                        # done[thingy] = 1
                        groupstack.append([thestart, thingy])
                        #groupstart = thestart
                        #currgroup = thingy
                        #ingroup = True


# run megamol using the parameters from cmdline
# megamol must be compiled with the rigged Call that generates
# DebugGroups for CallRenderXDs (MEGAMOL_USE_OPENGL_DEBUGGROUPS)

mmtracefile = next(tempfile._get_candidate_names()) + ".trace"
args = sys.argv
args = [the_apitrace, "trace", "-o", mmtracefile, the_exe] + the_args

sep = " "
sepnewline = "\n"
print("running " + sep.join(args))

proc = subprocess.run(args, capture_output=True)
#mmtracefile = "t0zmtikm.trace"

# find frame delimiters
args = [the_apitrace, 'dump', '--grep=SwapBuffers', '--color=never', mmtracefile]
proc = subprocess.run(args, capture_output=True)
res = proc.stdout.decode("utf-8")
frames = res.split(os.linesep + os.linesep)
the_re = re.compile(r"^\s*//")
the_check = lambda x : not the_re.match(x)
frames = list(filter(the_check, list(filter(None, frames))))
# print("found frame delimiters at commands: " + sepnewline.join(frames))

startf = frames[0]
endf = frames[1]
print("using commands between frame 0 and 1: " + startf + " - " + endf)
startnum = re.search(r'^(\d+)', startf).group(1)
endnum = re.search(r'^(\d+)', endf).group(1)
print("that is events " + startnum + "-" + endnum)
findDebugGroups(startnum, endnum, mmtracefile)

startf = frames[1]
endf = frames[2]
print("using commands between frame 1 and 2: " + startf + " - " + endf)
startnum = re.search(r'^(\d+)', startf).group(1)
endnum = re.search(r'^(\d+)', endf).group(1)
print("that is events " + startnum + "-" + endnum)
findDebugGroups(startnum, endnum, mmtracefile)

# cleanup
if (the_keeptrace > 0):
    print("keeping the trace file ({}) around".format(mmtracefile))
else:
    os.remove(mmtracefile)
