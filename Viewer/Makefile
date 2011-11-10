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


all: VersionInfo $(InputDir)/include/MegaMolViewerDynamic.h


# Rules for generated files:
$(InputDir)/include/MegaMolViewerDynamic.h: VersionInfo genDynIncl.pl $(InputDir)/include/MegaMolViewerStatic.h Makefile
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"GEN "$(COLORINFO)"$@: "
	@tput sgr0
	$(Q)perl genDynIncl.pl $(InputDir)/include/MegaMolViewerStatic.h

VersionInfo: version.template.h VersionInfo.pl Makefile
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"GEN "$(COLORINFO)"include/MegaMolViewer.version.gen.h: "
	@tput sgr0
	$(Q)perl VersionInfo.pl .


# Cleanup rules:
clean:
	rm -f $(InputDir)/include/MegaMolViewerDynamic.h


rebuild: clean all


.PHONY: clean rebuild tags VersionInfo
