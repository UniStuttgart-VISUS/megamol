import re

def stripSnippetName(name):
    # special case for complete shaders treated as snippet
    if name.count(':') < 4: 
        return name
    lastindex = name.rfind("::")
    return name[:lastindex]

def parseCPP(path):
    pattern = 'MakeShaderSource\(\"(?P<sname>[A-Za-z0-9.:_]*)\"'
    pattern2 = 'MakeShaderSnippet\(\"(?P<sname>[A-Za-z0-9.:_]*)\"'
    result = ()
    with open(path) as f:
        text = f.read()
        res = re.findall(pattern, text, re.DOTALL)
        if len(res) > 0:
            result = (path, res)
        res2 = re.findall(pattern2, text, re.DOTALL)
        lst = []
        for e in res2:
            lst.append(stripSnippetName(e))
        if len(lst) > 0:
            if len(result) > 0:
                result = (path, result[1] + lst)
            else:
                result = (path, lst)
        if len(result) > 0:
            # remove possible duplications
            res = list(dict.fromkeys(result[1]))
            result = (path, res)
    return result
