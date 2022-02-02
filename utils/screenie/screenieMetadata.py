import argparse
from datetime import datetime
import json
import os
from PIL import Image
from PIL.PngImagePlugin import PngImageFile, PngInfo

MY_EXT = "rmeta"

def get_dumpfile(orig):
    base = os.path.splitext(orig)[0]
    return f"{base}.{MY_EXT}"

def dump_meta(filename, metadata, fields):
    outname = get_dumpfile(filename)
    if fields and fields[0] != '*':
        print(f"dumping metadata[{fields}] to {outname}")
        newmeta = {}
        for f in fields:
            newmeta[f] = metadata[f]
    else:
        print(f"dumping metadata to {outname}")
        newmeta = metadata
    with open(outname, 'x') as outfile:
        json.dump(newmeta, outfile, indent=4)

def replace_meta(filename, fields):
    metaname = get_dumpfile(filename)
    with open(metaname) as json_file:
        meta = json.load(json_file)
    if fields and fields[0] != '*':
        print(f"overwriting metadata[{fields}] in {filename} from {metaname}")
        newmeta = {}
        for f in fields:
            newmeta[f] = meta[f]
    else:
        print(f"overwriting metadata in {filename} from {metaname}")
        newmeta = meta

    newmeta['Metadata Modification Time'] = f"{datetime.now()}"
    img = PngImageFile(filename)
    metadata = PngInfo()
    for f in newmeta:
        metadata.add_text(f, newmeta[f])
    img.save(filename, pnginfo=metadata)

def print_meta(metadata, fields):
    if fields and fields[0] != '*':
        for f in fields:
            data = im.info[f]
            info = (data[:65] + '...') if len(data) > 65 else data
            print(f"{f} = {info if args.trunc else data}")
    elif fields and fields[0] == '*':
        for f in metadata:
            data = im.info[f]
            info = (data[:65] + '...') if len(data) > 65 else data
            print(f"{f} = {info if args.trunc else data}")
    else:
        print("found these metadata fields:")
        for f in metadata:
            print(f)

parser = argparse.ArgumentParser(usage="%(prog)s <FILE> [[<field>] <field> ...] [additional args]",
    description="show metadata in a MegaMol screenshot")
parser.add_argument('file', help='the input file')
parser.add_argument('field', nargs='*', help='any number of metadata fields')
parser.add_argument('--trunc', action='store_true', help='shorten values if printing')
parser.add_argument('--dump', action='store_true', help=f'dump metadata in sidecar {MY_EXT}')
parser.add_argument('--overwrite', action='store_true',
    help=f'replace metadata with sidecar {MY_EXT} data')

args = parser.parse_args()
if not args.file:
    print ("need one input file")
    exit(1)
try:
    im = Image.open(args.file)
    im.load()
    if args.dump:
        dump_meta(args.file, im.info, args.field)
    elif args.overwrite:
        replace_meta(args.file, args.field)
    else:
        print_meta(im.info, args.field)

except (FileNotFoundError, IsADirectoryError) as err:
    print(f"{args.file}: {err.strerror}")
