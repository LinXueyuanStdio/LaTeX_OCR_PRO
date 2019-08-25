import os

def check(filename):
  with open(filename) as f:
    lines = f.readlines()
    linenumber = 0
    for line in lines:
      if len(line) <= 1:
        print(linenumber)
      linenumber += 1

check('../data/fullhand/formulas/formulas.norm.txt_text')