#
# Makefile  13.09.2006 (mueller)
#
# Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VISUS).
# Alle Rechte vorbehalten.
#

#
# This is the default VISlib makefile. Do not change it. Any per project 
# configuration must be done in the separate file project.mk. project.mk must
# be located in the same directory as this makefile.
#
# At least the TargetName variable must be set in project.mk.
#


# Include project default configuration.
-include ../config.mk


# user settings
TargetName := vislibgl
ProjectPrerequisites := ./include/glload/build/libglload.so

IncludeDir += ../base/include ../math/include ../sys/include ../graphics/include ../glutInclude


#ifneq ($(MAKECMDGOALS), clean)
#ifneq ($(MAKECMDGOALS), sweep)
#include ../glutInclude/vislibGlutInclude.mk
#endif
#endif

# Include common build rules.
-include ../librules.mk


# build target for glload
./include/glload/build/libglload.so:
	$(shell include/glload/build.sh)

