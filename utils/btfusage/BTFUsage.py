import argparse
import os
import btfparser
import cppparser

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def parseArguments():
    descr = """This script creates a human-readable report on the usage of shaders residing in 
    BTF-Files. It only needs the path to to the plugin as input. It should also work with the
    core itself. The report is printed to the console and should ideally be piped into a file."""
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument("path", type=dir_path, help="Path to the plugin folder")
    args = parser.parse_args()
    return args.path

def listSourceFiles(path):
    sourcepath = os.path.join(path, 'src')
    result = []
    for root, subdirs, files in os.walk(sourcepath):
        for file in files:
            fname = os.fsdecode(file)
            ext = os.path.splitext(file)[-1].lower()
            if ext == ".cpp":
                result.append(os.path.join(root, file))
            elif ext == ".c":
                result.append(os.path.join(root, file))
            elif ext == ".hpp":
                result.append(os.path.join(root, file))
            elif ext == ".h":
                result.append(os.path.join(root, file))
            elif ext == ".cu":
                result.append(os.path.join(root, file))
            elif ext == ".cuh":
                result.append(os.path.join(root, file))
            elif ext == ".cxx":
                result.append(os.path.join(root, file))
            elif ext == ".cc":
                result.append(os.path.join(root, file))
    return result

def listBtfFiles(path):
    btfpath = os.path.join(path, 'shaders')
    result = []
    for root, subdirs, files in os.walk(btfpath):
        for file in files:
            fname = os.fsdecode(file)
            ext = os.path.splitext(file)[-1].lower()
            if ext == ".btf":
                result.append(os.path.join(root, file))
    return result

def parsebtfs(btflist):
    res = []
    for file in btflist:
        res = res + btfparser.parseBTF(file)
    return res

def parsesources(sourcelist):
    res = []
    for file in sourcelist:
        res.append(cppparser.parseCPP(file))
    #create dictionary that maps shader names to files
    dictionary = {}
    for e in res:
        if len(e) > 0:
            for sname in e[1]: #access all shader names
                if sname in dictionary:
                    dictionary[sname] = dictionary[sname] + e[0]
                else:
                    dictionary[sname] = [e[0]]
    return dictionary

def createReport(path, btfresult, srcresult):
    divisor = "------------------------------------------------------------------"
    divisorsmall = "-----"
    print("MegaMol BTF Shader Usage Report for \"" + path + "\":")
    name = ""
    for shader in btfresult:
        if name != shader[0]: # new file
            print(divisor)
            name = shader[0]
            print("SHADER FILE = \"" + name + "\"")
            print(divisorsmall)
        print("NAME = \"" + shader[1] + "\", LINES: " + str(shader[2]) + " to " + str(shader[3]))
        print("    used in")
        dictres = []
        if shader[1] in srcresult:
            dictres = srcresult[shader[1]]
        if len(dictres) == 0:
            print("        NOWHERE")
        else:
            for e in dictres:
                print("        " + e)

    print(divisor)
    print("WARNING: There is NO GUARANTEE that alle usages are listed, as shader names can be constructed programmatically!")

def main():
    path = parseArguments()
    btflist = listBtfFiles(path)
    srclist = listSourceFiles(path)
    btfres = parsebtfs(btflist)
    srcres = parsesources(srclist)
    createReport(path, btfres, srcres)

if __name__ == "__main__":
    main()
