#!/bin/bash
#
# MegaMol startup script
# Copyright 2015, http://go.visus.uni-stuttgart.de/megamol
#

# Edit this if you want to use the debug version
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:@cfg_LIB_PATH@:@cfg_EXTERNAL_LIB_PATH@ @cfg_MEGAMOLCON@ "$@"

