#
# Makefile
# Viewer (MegaMol)
#
# Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
# Alle Rechte vorbehalten.
#

include ../common.mk


# Target name
TargetName := MegaMolViewer
OutDir := lib
# subdirectories below $(InputRootDir)
InputRootDir := $(InputDir)
InputDir := .
IncludeDir := .


all: $(InputDir)/include/MegaMolViewerDynamic.h


# Rules for generated files:
$(InputDir)/include/MegaMolViewerDynamic.h: $(InputDir)/genDynIncl.pl $(InputDir)/include/MegaMolViewerStatic.h Makefile
	@mkdir -p $(dir $@)
	@echo -e '\E[1;32;40m'"GEN "'\E[0;32;40m'"$@: "
	@tput sgr0
	$(Q)perl $(InputDir)/genDynIncl.pl $(InputDir)/include/MegaMolViewerStatic.h


# Cleanup rules:
clean:
	rm -f $(InputDir)/include/MegaMolViewerDynamic.h


rebuild: clean all


.PHONY: clean rebuild tags
