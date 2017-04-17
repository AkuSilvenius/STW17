#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from lxml import etree
import xml.etree.ElementTree as ET
import io

parser = etree.XMLParser(recover=True)

tree = ET.parse('./data/MovieDiC_V2.xml', parser=parser)
root = tree.getroot()
file = io.open('cleanedData.txt','w',encoding='utf8')

i = tree.getiterator()

print('Cleanin up the data...')
for m in i:
    #print(m.tag)
    for di in m.findall('dialogue'):
        #print("dialogue id: " + di.get('id'))
        for u in di.findall('utterance'):
            utterance = u.text
            file.write(u.text + "\n")
    #print("dialogue changes")
file.close()
print('Clean up complete!')
