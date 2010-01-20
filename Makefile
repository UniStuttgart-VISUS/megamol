#
# MegaMol(TM) Solution Makefile
# Copyright 2009, by VISUS (Universitaet Stuttgart)
# Alle Rechte vorbehalten.
#

# Applications:
MAKE = make
SHELL = /bin/bash

include ./common.mk

# Project directories to make:
ProjectDirs = Viewer Glut Console

################################################################################

all: VersionInfo
	@for dir in $(ProjectDirs); do $(MAKE) -C $$dir $@ || exit 1; done

sweep:
	@for dir in $(ProjectDirs); do $(MAKE) -C $$dir $@ || exit 1; done
	
clean:	
	@for dir in $(ProjectDirs); do $(MAKE) -C $$dir $@ || exit 1; done
	
rebuild: clean all

VersionInfo:
	@mkdir -p $(dir $@)
	@echo -e '\E[1;32;40m'"GEN "'\E[0;32;40m'"consoleversion.gen.h: "
	@tput sgr0
	$(Q)perl VersionInfo.pl .

.PHONY: all clean sweep rebuild VersionInfo