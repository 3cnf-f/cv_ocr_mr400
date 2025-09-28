import sys
from math import sqrt
argv = sys.argv

side_a=int(argv[1])
side_b=int(argv[2])
hypothenuse=sqrt(side_a**2+side_b**2)
print(hypothenuse)
