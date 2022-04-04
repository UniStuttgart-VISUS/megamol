import sys
sys.modules['_elementtree'] = None
import xml.etree.ElementTree as et

class LineNumberingParser(et.XMLParser):
    def _start(self, *args, **kwargs):
        element = super(self.__class__, self)._start(*args, **kwargs)
        element._start_line_number = self.parser.CurrentLineNumber
        element._start_column_number = self.parser.CurrentColumnNumber
        element._start_byte_index = self.parser.CurrentByteIndex
        return element
    
    def _end(self, *args, **kwargs):
        element = super(self.__class__, self)._end(*args, **kwargs)
        element._end_line_number = self.parser.CurrentLineNumber
        element._end_column_number = self.parser.CurrentColumnNumber
        element._end_byte_index = self.parser.CurrentByteIndex
        return element

def appendName(src, tgt):
    if len(tgt) == 0:
        return src
    return tgt + "::" + src 

def parseNode(node, curname, curobj, filepath):
    name = curname
    struc = curobj
    if 'name' in node.attrib:
        name = appendName(node.attrib['name'], name)
    elif 'namespace' in node.attrib:
        name = appendName(node.attrib['namespace'], name)
    for n in node:
        parseNode(n, name, struc, filepath)
    if node.tag == 'shader':
        #print(name)
        struc.append((filepath, name, node._start_line_number, node._end_line_number))
    return struc

def parseBTF(path):
    tree = et.parse(path, parser=LineNumberingParser())
    root = tree.getroot()
    return parseNode(root, '', [], path)