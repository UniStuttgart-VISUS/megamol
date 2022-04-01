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
    parser = argparse.ArgumentParser()
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
    for file in btflist:
        res = btfparser.parseBTF(file)
        break # intentional right now for testing

def main():
    path = parseArguments()
    btflist = listBtfFiles(path)
    srclist = listSourceFiles(path)
    parsebtfs(btflist)

if __name__ == "__main__":
    main()
