from PIL import Image
import argparse

parser = argparse.ArgumentParser(usage="%(prog)s <FILE> [[<field>] <field> ...]", description="show metadata in a MegaMol screenshot")
parser.add_argument('file')
parser.add_argument('field', nargs='*')
args = parser.parse_args()
if not args.file:
    print ("need one input file")
    exit(1)
try:
    im = Image.open(args.file)
    im.load()
    if not args.field:
        print("found these metadata fields:")
        for pair in im.info:
            print(pair)
    else:
        for f in args.field:
            print(f"{f} = {im.info[f]}")

except (FileNotFoundError, IsADirectoryError) as err:
    print(f"{args.file}: {err.strerror}")
