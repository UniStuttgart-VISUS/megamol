#! /bin/bash
# useVISglut.sh
#
# Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

# Greeting
echo ""
echo "    VISlib Use-VISglut/freeGlut shell script"
echo "    Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten."
echo ""

# visglut
echo -n "Enter full unix path to the VISglut/freeGlut directory: "
read visglutLinPath
echo -n "Enter full windows path to the VISglut/freeGlut directory: "
read visglutWinPath

# configure the file
echo ""
echo "Creating file:"

cat glutInclude.visglut.h \
| sed -e "s_%visglutLinPath%_""$visglutLinPath""_g" \
| sed -e "s_%visglutWinPath%_""$visglutWinPath""_g" \
> glutInclude.h

cat glutInclude.visglut.mk \
| sed -e "s_%visglutLinPath%_""$visglutLinPath""_g" \
| sed -e "s_%visglutWinPath%_""$visglutWinPath""_g" \
> glutInclude.mk

echo "    Done."

