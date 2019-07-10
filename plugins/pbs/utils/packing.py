from os import listdir, path, stat
from os.path import isfile, join, commonprefix, splitext
import collections
import functools
import operator

#thePath = "\\\\etude\\Archiv\\Daten\\Partikel\\FARO\\dischingen_raw_final\\"
thePath = "\\\\paganmetal\\u$\\Dischingen\\"
numNodes = 20
capMax = 999 * 1024 * 1024 * 1024

allFiles = [f for f in listdir(thePath) if isfile(join(thePath, f))]
filteredFiles = []

for f in allFiles:
    found = False
    for f2 in allFiles:
        pf = commonprefix([f, f2])
        # print("common prefix of %s and %s : %s" % (f, f2, pf))
        if pf == splitext(f)[0]:
            found = True
            break

    if not splitext(f)[1] == ".mmpkd":
        found = True
    
    if not found and path.getsize(thePath + f) > 48:
        # print("%s is a leaf" % f)
        filteredFiles.append(thePath + f)

nodeFiles = [ [] for i in range(numNodes) ]
nodeCapacity = [0] * numNodes
sortedFiles = sorted(filteredFiles, key=operator.itemgetter(2), reverse=True)

items = []
for f in sortedFiles:
    items.append((1, f, path.getsize(f)))

print("-- got " + str(len(items)) + " leaves")
# for i in items:
#     print(str(i[0]) + " " + i[1] + " " + str(i[2]) )

for i in items:
    smallestCap = capMax
    smallestIdx = 0
    for j in range(0, numNodes):
        if nodeCapacity[j] < smallestCap:
            smallestCap = nodeCapacity[j]
            smallestIdx = j
        
    nodeFiles[smallestIdx].append(i[1])
    nodeCapacity[smallestIdx] += i[2]

for i in range(numNodes):
    print("-- node " + str(i) + " has capacity " + str(nodeCapacity[i]) + ":")

print("nodeFiles = {")
for i in range(numNodes):
    print("{", end='')
    for j in range(len(nodeFiles[i])):
        if (j != 0):
            print(",", end='')
        ding = nodeFiles[i][j]
        print('"' + ding.replace('\\', '\\\\') + '"', end='')
    print("},")
print("}")

print("return _ENV")