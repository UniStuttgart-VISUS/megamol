import os
import xml.etree.ElementTree as et

def parseBTF(path):
    tree = et.parse(path)
    root = tree.getroot()
    print(root.attrib)
