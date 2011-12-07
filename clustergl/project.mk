#
# project.mk
#
# Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

TargetName := vislibclustergl

IncludeDir += ../base/include ../math/include ../sys/include ../net/include ../graphics/include ../gl/include ../cluster/include ../glutInclude

ifneq ($(MAKECMDGOALS), clean)
ifneq ($(MAKECMDGOALS), sweep)
include ../glutInclude/vislibGlutInclude.mk
endif
endif

