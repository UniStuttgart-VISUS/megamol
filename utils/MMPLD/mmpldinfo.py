# MMPLDinfo - MegaMol Particle List Data File Information Utility
# Copyright 2014, 2016 (C) by
# S. Grottel, TU Dresden, Germany and G. Reina, University of Stuttgart
# All rights reserved. Alle Rechte vorbehalten.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of S. Grottel, TU Dresden, nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY S. GROTTEL AS IS AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL S. GROTTEL BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function

import sys
import os
import struct
import argparse

if sys.version_info.major < 3:
    raise Exception("This script requires Python 3")

def getUShort(file):
    stuff = file.read(2)
    return struct.unpack("<H", stuff)[0]

def getUInt(file):
    stuff = file.read(4)
    return struct.unpack("<I", stuff)[0]

def getFloat(file):
    stuff = file.read(4)
    return struct.unpack("<f", stuff)[0]

def getDouble(file):
    stuff = file.read(8)
    return struct.unpack("<d", stuff)[0]

def getUInt64(file):
    stuff = file.read(8)
    return struct.unpack("<Q", stuff)[0]

def getByte(file):
    return file.read(1)[0]

def pluralTuple(thingy):
    return thingy, "s" if (thingy != 1) else ""

def listFramedata(parseResult, frame):
    return (parseResult.v and (parseResult.v > 1) or (parseResult.v == 1 and frame == 0))

# returns timeStamp, numLists
def readFrameHeader(file):
    mem = 0
    timestamp = 0.0
    if (version >= 102):
        timestamp = getFloat(f)
        mem += 4
    numLists = getUInt(f)
    mem +=4
    return timestamp, numLists, mem

# returns vertType, colType, stride, globalRad, globalCol, intensityRange, listNumParts, listBBox
def readListHeader(file):
    frameMem = 0
    vertType = getByte(f)
    if (vertType >= len(vertexNames)):
        vertType = 0
    colType = getByte(f)
    if (colType >= len(colorNames)):
        colType = 0
    if (vertType == 0):
        colType = 0
    frameMem += 2

    listFramedata(parseResult, fi) and print("    #%u: %s, %s" % (li, vertexNames[vertType], colorNames[colType]))
    stride = vertexSizes[vertType] + colorSizes[colType]
    listFramedata(parseResult, fi) and print("        %u byte%s per particle" % pluralTuple(stride))

    globalRad = 0.05
    globalCol = (0.0, 0.0, 0.0, 0.0)
    intensityRange = (0.0, 0.0)
    listBBox = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if (vertType == 1 or vertType == 3 or vertType == 4):
        globalRad = getFloat(f)
        frameMem += 4
        listFramedata(parseResult, fi) and print("        global radius: %f" % (globalRad))

    if (colType == 0):
        globalCol = [getByte(f) for x in range(4)]
        frameMem += 4
        listFramedata(parseResult, fi) and print("        global color: (%u, %u, %u, %u)" % (tuple(globalCol)))
    elif (colType == 3 or colType == 7):
        intensityRange = [getFloat(f) for x in range(2)]
        frameMem += 8
        listFramedata(parseResult, fi) and print("        intensity color range: [%f, %f]" % (tuple(intensityRange)))

    listNumParts = getUInt64(f)
    frameMem += 8
    #listFramedata and print("        %Q particle%s" % countPlural(listNumParts))
    listFramedata(parseResult, fi) and print("        {0} particle{1}".format(*pluralTuple(listNumParts)))
    frameMem += listNumParts * stride

    if (version >= 103):
        listBBox = [getFloat(f) for x in range(6)]
        frameMem += 6 * 4
        listFramedata(parseResult, fi) and print("        list bounding box: (%f, %f, %f) - (%f, %f, %f)" % (tuple(box)))
    return vertType, colType, stride, globalRad, globalCol, intensityRange, listNumParts, listBBox, frameMem

def readParticles(number, vertType, colType, file, listIndex):
    mins = [sys.float_info.max, sys.float_info.max, sys.float_info.max]
    maxs = [-sys.float_info.max, -sys.float_info.max, -sys.float_info.max]
    consoleSilent = parseResult.bboxonly or parseResult.dumpxyz or parseResult.dumpvpd
    for p in range(number):
        if (vertType == 0):
            consoleSilent or print("        no position", end ='')
        elif (vertType == 1):
            pos = [getFloat(file) for x in range(3)]
            consoleSilent or print("        pos = (%f, %f, %f)" % (tuple(pos)), end ='')
        elif (vertType == 2):
            pos = [getFloat(file) for x in range(4)]
            consoleSilent or print("        pos = (%f, %f, %f), rad = %f" % (tuple(pos)), end ='')
        elif (vertType == 3):            
            pos = [getUShort(file) for x in range(3)]
            consoleSilent or print("        pos = (%f, %f, %f)" % (tuple(pos)), end ='')
        elif (vertType == 4):
            pos = [getDouble(file) for x in range(3)]
            consoleSilent or print("        pos = (%f, %f, %f)" % (tuple(pos)), end ='')
        for x in range(3):
            if (pos[x] < mins[x]):
                mins[x] = pos[x]
            if (pos[x] > maxs[x]):
                maxs[x] = pos[x]

        if (parseResult.dumpxyz):
            outFile.write("%s %f %f %f\n" % (chr(listIndex + 65), *pos))
        if (parseResult.dumpvpd):
            outFile.write(struct.pack('<f', pos[0]))
            outFile.write(struct.pack('<f', pos[1]))
            outFile.write(struct.pack('<f', pos[2]))
            if vertType == 2:
                outFile.write(struct.pack('<f', pos[3]))
            elif vertType == 1:
                outFile.write(struct.pack('<f', globalRad))
            else:
                sys.exit("unsupported particle format for dumping as binary")

        if (colType == 0):
            consoleSilent or print(", no color")
        elif (colType == 1):
            col = [getByte(file) for x in range(3)]
            consoleSilent or print(", col = (%d, %d, %d)" % (tuple(col)))
        elif (colType == 2):
            col = [getByte(file) for x in range(4)]
            consoleSilent or print(", col = (%d, %d, %d, %d)" % (tuple(col)))
        elif (colType == 3):
            col = [getFloat(file) for x in range(1)]
            consoleSilent or print(", col = (%f)" % (tuple(col)))
        elif (colType == 4):
            col = [getFloat(file) for x in range(3)]
            consoleSilent or print(", col = (%f, %f, %f)" % (tuple(col)))
        elif (colType == 5):
            col = [getFloat(file) for x in range(4)]
            consoleSilent or print(", col = (%f, %f, %f, %f)" % (tuple(col)))
        elif (colType == 6):
            col = [getUShort(file) for x in range(4)]
            consoleSilent or print(", col = (%d, %d, %d, %d)" % (tuple(col)))
        elif (colType == 7):
            col = [getDouble(file) for x in range(1)]
            consoleSilent or print(", col = (%f)" % (tuple(col)))
    if (number > 0):
        consoleSilent or print("        bounding box of these particles: (%f, %f, %f) - (%f, %f, %f)" % tuple(mins + maxs))

vertexSizes = [0, 12, 16, 6, 24]
vertexNames = ["VERTDATA_NONE", "VERTDATA_FLOAT_XYZ", "VERTDATA_FLOAT_XYZR", "VERTDATA_SHORT_XYZ", "VERTDATA_DOUBLE_XYZ"]
colorSizes = [0, 3, 4, 4, 12, 16, 8, 8]
colorNames = ["COLDATA_NONE", "COLDATA_UINT8_RGB", "COLDATA_UINT8_RGBA", "COLDATA_FLOAT_I", "COLDATA_FLOAT_RGB", "COLDATA_FLOAT_RGBA", "COLDATA_USHORT_RGBA", "COLDATA_DOUBLE_I"]

parser = argparse.ArgumentParser(description='display information about .mmpld file(s)')
parser.add_argument('inputfiles', metavar='file', nargs='+', help='file to get info about')
parser.add_argument('-v', action='count', help='show verbose frame info (add another v for more verbosity)')
parser.add_argument('--head', action='store', help='show that many particles at the start of each shown frame (or "all"))')
parser.add_argument('--tail', action='store', help='show that many particles at the end of each shown frame (or "all"))')
parser.add_argument('--bboxonly', action='count', help='only print the bbox of head/tail, not the particles)')
parser.add_argument('--versiononly', action='count', help='show only file version and exit')
parser.add_argument('--dumpxyz', action='store', help='dump/convert mmpld to xyz files <DUMPXYZ>_<frame>.xyz')
parser.add_argument('--dumpvpd', action='store', help='dump/convert mmpld to vpd files <DUMPVPD>_<frame>.vpd')
parser.add_argument('--dumpft', action='count', help='dump frame table')
parser.add_argument('--try-recovery', action='count', help='try interpreting trailing dead data as a dangling frame')
parseResult = parser.parse_args()

specific_only = parseResult.bboxonly or parseResult.versiononly
hideVersion = parseResult.bboxonly
fakeFrame = False

if (not specific_only):
    print("")
    print("MMPLDinfo (v.: 1.3.1) - MegaMol Particle List Data File Information Utility")
    print("Copyright 2014-2018 (C) by")
    print("S. Grottel, TU Dresden, Germany and G. Reina, University of Stuttgart")
    print("All rights reserved. Alle Rechte vorbehalten.")
    print("")

for filename in parseResult.inputfiles:
    if (not specific_only):
        print("")
        print("File: " + (filename))
        print("-" * (6 + len(filename)))

    if not os.path.isfile(filename):
        print("error: cannot open file.")
        continue
    
    with open(filename, "rb") as f:
        magic = f.read(6)
        if (magic != b'MMPLD\x00'):
            print("this is not an mmpld file!")
            exit(1)
        version = getUShort(f)
        if (version == 100):
            hideVersion or print("mmpld version 1.0")
        elif (version == 101):
            hideVersion or print("mmpld version 1.1")
        elif (version == 102):
            hideVersion or print("mmpld version 1.2")
        elif (version == 103):
            hideVersion or print("mmpld version 1.3")
        else:
            print("unsupported mmpld version " + str(version / 100) + "." + str(version % 100))
            exit(1)
        
        if (parseResult.versiononly):
            exit(0)

        frameCount = getUInt(f)
        specific_only or print("Number of frames: " + str(frameCount))

        box = [getFloat(f) for x in range(6)]
        print("Bounding box: (%f, %f, %f) - (%f, %f, %f)" % (tuple(box)))
        box = [getFloat(f) for x in range(6)]
        print("Clipping box: (%f, %f, %f) - (%f, %f, %f)" % (tuple(box)))

        if (parseResult.bboxonly):
            exit(0)

        if (frameCount <= 0):
            print("out of data")
            exit(1)
        
        frameTable = []
        refFrameOffset = 60
        for x in range(frameCount + 1):
            frameOffset = getUInt64(f)
            if frameOffset < refFrameOffset:
                print(f"error: frame table entry {x} is invalid: not monotonically increasing or too small first offset: ({frameOffset}) < ({refFrameOffset})")
            refFrameOffset = frameOffset
            frameTable.append(frameOffset)

        if (f.tell() != frameTable[0]):
            print(f"warning: dead data trailing header: position after reading frame table ({f.tell()}) is not the start of the first frame ({frameTable[0]})")
            if parseResult.v:
                print("dead data:")
                for i in range(frameTable[0] - f.tell()):
                    print(hex(getByte(f)), end=' ')
                print()

        if os.path.getsize(filename) != frameTable[frameCount]:
            print(f"error: file end pointer (frameTable[{frameCount}]) in frame table inconsistent with file size ({os.path.getsize(filename)}): ", end='')
            if os.path.getsize(filename) < frameTable[frameCount]:
                print("Data truncated.")
            if os.path.getsize(filename) > frameTable[frameCount]:
                print("Trailing garbage.")
                if parseResult.try_recovery:
                    print("--> Appending hypothetical frame until the end of the file <--")
                    fakeFrame = True
                    frameCount += 1
                    frameTable.append(os.path.getsize(filename))

        if parseResult.dumpft:
            print(f"Frame Table{' (recovered)' if fakeFrame else ''}:")
            print("index                offset                 hex")
            sum = 0
            for i in range(len(frameTable)):
                print(f"{i:5}: {frameTable[i]:20}  {frameTable[i]:#018x}")
                if i < frameCount:
                    sum = sum + frameTable[i+1] - frameTable[i]
                # else:
                #     print(f"ignoring entry {i}, should be the end pointer")
            print(f"average frame size: {sum / frameCount}")
        
        minNumLists = 0
        maxNumLists = 0
        minNumParts = 0
        maxNumParts = 0
        frameNumParts = 0
        listNumParts = 0
        accumulatedParts = 0

        for fi in range(frameCount):
            f.seek(frameTable[fi], os.SEEK_SET)
            expectedFrameSize = frameTable[fi+1] - frameTable[fi]
            if parseResult.v:
                print(f"jumping to frame {fi}, expecting a frame of size {expectedFrameSize}")
            timeStamp, numLists, frameTotalMem = readFrameHeader(f)
            timeStampString = ""
            if (version >= 102):
                timeStampString = "(%f)" % timeStamp
            listFramedata(parseResult, fi) and (sys.stdout.write("Frame # %u (offset %lu = %s) %s" % (fi, frameTable[fi], hex(frameTable[fi]), timeStampString)), print("- %u list%s" % pluralTuple(numLists)))
            if (fi == 0):
                minNumLists = maxNumLists = numLists
            else:
                if (minNumLists > numLists):
                    minNumLists = numLists
                if (maxNumLists < numLists):
                    maxNumLists = numLists
            frameNumParts = 0
            for li in range(numLists):
                vertType, colType, stride, globalRad, globalCol, intensityRange, listNumParts, listBBox, frameMem = readListHeader(f)
                frameNumParts += listNumParts
                frameTotalMem += frameMem
                
                if (not parseResult.dumpxyz and not parseResult.dumpvpd):
                    if (listFramedata(parseResult, fi)):
                        if (parseResult.head):
                            if (parseResult.head == "all"):
                                numHead = listNumParts
                            else:
                                numHead = min(int(parseResult.head), listNumParts)
                        else:
                            numHead = 0
                        if (parseResult.tail):
                            if (parseResult.tail == "all"):
                                numTail = listNumParts
                            else:
                                numTail = min(int(parseResult.tail), listNumParts)
                        else:
                            numTail = 0
                        if (numHead > 0):
                            print("        list head (%d particles):" % numHead)
                        readParticles(numHead, vertType, colType, f, li)
                        if (numTail + numHead > listNumParts):
                            print("        not enough particles to also list tail")
                            numTail = 0
                        f.seek((listNumParts - numHead - numTail) * stride, os.SEEK_CUR)
                        if (numTail > 0):
                            print("        list tail (%d particles):" % numTail)
                        readParticles(numTail, vertType, colType, f, li)
                    else:
                        f.seek(listNumParts * stride, os.SEEK_CUR)
                else:
                    f.seek(listNumParts * stride, os.SEEK_CUR)

            if frameTotalMem != expectedFrameSize:
                print(f"frame size {frameTotalMem} differs from expected size {expectedFrameSize} (from frameTable)")
            # if (f.tell() != frameTable[fi + 1]):
            #     print("warning: trailing data after frame %u or frame table corrupted" % (fi))
            if (fi == 0):
                minNumParts = maxNumParts = frameNumParts
            else:
                if (minNumParts > frameNumParts):
                    minNumParts = frameNumParts
                if (maxNumParts < frameNumParts):
                    maxNumParts = frameNumParts

            # now that we know the number of particles in the frame we can actually read them for dumping
            if (parseResult.dumpxyz):
                outName = "%s_%04u.xyz" % (parseResult.dumpxyz, fi)
                print("dumping frame into %s" % outName)
                outFile = open(outName, "x")
                outFile.write("%lu\n" % frameNumParts)
                outFile.write("%s frame %u\n" % (filename, fi))
                f.seek(frameTable[fi], os.SEEK_SET)

                timeStamp, numLists, _ = readFrameHeader(f)
                for li in range(numLists):
                    vertType, colType, stride, globalRad, globalCol, intensityRange, listNumParts, listBBox, particleMem = readListHeader(f)
                    readParticles(listNumParts, vertType, colType, f, li)

            if (parseResult.dumpvpd):
                outName = "%s_%04u.vpd" % (parseResult.dumpvpd, fi)
                print("dumping frame into %s" % outName)
                outFile = open(outName, mode="wb")
                outFile.write(struct.pack('<Q', frameNumParts))
                f.seek(frameTable[fi], os.SEEK_SET)

                timeStamp, numLists, _ = readFrameHeader(f)
                for li in range(numLists):
                    vertType, colType, stride, globalRad, globalCol, intensityRange, listNumParts, listBBox, particleMem = readListHeader(f)
                    readParticles(listNumParts, vertType, colType, f, li)

            accumulatedParts += frameNumParts
            if (parseResult.dumpxyz or parseResult.dumpvpd):
                outFile.close()

        accumulatedParts /= frameCount

        print("Data Summary")
        print("    %u time frame%s" % pluralTuple(frameCount))
        if (minNumLists == maxNumLists):
            print("    %u particle list%s per frame" % pluralTuple(minNumLists))
        else:
            sys.stdout.write("    %u" % minNumLists)
            print(" .. %u particle list%s per frame" % pluralTuple(maxNumLists))

        if (minNumParts == maxNumParts):
            print("    {0} particle{1} per frame".format(*pluralTuple(minNumParts)))
        else:
            sys.stdout.write("    {0}".format(minNumParts))
            print(" .. {0} particle{1} per frame".format(*pluralTuple(maxNumParts)))
            print("    {0} particles on average".format(int(accumulatedParts)))

#input()
