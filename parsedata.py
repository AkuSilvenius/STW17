from __future__ import print_function
from lxml import etree
import xml.etree.ElementTree as ET

parser = etree.XMLParser(recover=True)

tree = ET.parse('./data/MovieDiC_V2.xml', parser=parser)
root = tree.getroot()

i = tree.getiterator()

for m in i:
    print(m.tag)
    for di in m.findall('dialogue'):
        print("dialogue id: " + di.get('id'))
        for u in di.findall('speaker'):
            print(u.text + ": " + u.get('utterance'))
        print("dialogue changes")
