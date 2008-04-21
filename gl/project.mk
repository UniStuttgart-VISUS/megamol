#
# project.mk
#
# Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

TargetName := vislibgl

IncludeDir += ../base/include ../math/include ../sys/include ../graphics/include

# forward declaration to make it the default target
all: $(TargetName)d $(TargetName)

# add dependancy to glh
$(TargetName)d: ./include/glh/glh_genext.h
$(TargetName): ./include/glh/glh_genext.h

# build target for glh
./include/glh/glh_genext.h:
	cd ./include/glh && make glh_genext.h
