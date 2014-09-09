#
# MegaMol(TM) Solution Makefile
# Copyright 2009, by VISUS (Universitaet Stuttgart)
# Alle Rechte vorbehalten.
#

# Applications:
MAKE = make
SHELL = /bin/bash

include ./common.mk
include ./ExtLibs.mk

# Project directories to make:
ProjectDirs = Viewer Glut Console

################################################################################

all:
	@for dir in $(ProjectDirs); do $(MAKE) -C $$dir $@ || exit 1; done
	cp $(vislibpath)/gl/include/glload/build/libglload.so $(outbin)

sweep:
	@for dir in $(ProjectDirs); do $(MAKE) -C $$dir $@ || exit 1; done
	
clean:	
	@for dir in $(ProjectDirs); do $(MAKE) -C $$dir $@ || exit 1; done
	
rebuild: clean all

.PHONY: all clean sweep rebuild VersionInfo