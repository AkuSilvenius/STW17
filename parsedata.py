#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from lxml import etree
import xml.etree.ElementTree as ET
import io

parser = etree.XMLParser(recover=True)

tree = ET.parse('./data/MovieDiC_V2.xml', parser=parser)
root = tree.getroot()
file = io.open('cleanedData.txt', 'w', encoding='utf8')

i = tree.getiterator()

print('Cleaning up the data...')

for m in i:
    for di in m.findall('dialogue'):
        for u in di.findall('utterance'):
            file.write(u.text + "\n")

file.close()
print('Cleanup complete!')
