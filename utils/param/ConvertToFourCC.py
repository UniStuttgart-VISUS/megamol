import struct
import sys

l = len(sys.argv)
if l != 2:
    print("usage: " + sys.argv[0] + " <fourcc code>")
    exit

a = bytes(sys.argv[1] + "\x00\x00", "utf-8")

print(struct.unpack('Q', a)[0])
