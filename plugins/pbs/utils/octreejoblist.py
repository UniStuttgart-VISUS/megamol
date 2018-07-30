from os import listdir, stat
from os.path import isfile, join, commonprefix, splitext

thePath = "\\\\etude\\Archiv\\Daten\\Partikel\\FARO\\"
theScript = "U:\\home\\reina\\src\\megamol-dev\\plugins\\pbs\\utils\\cpe2mmpld.lua"

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

    if not splitext(f)[1] == ".raw":
        found = True
    
    size = stat(thePath + f)
    if not found and size.st_size > 48:
        # print("%s is a leaf" % f)
        filteredFiles.append(f)

for f in filteredFiles:
    print("mmconsole -p %s -o file %s" % (theScript, splitext(f)[0]))
