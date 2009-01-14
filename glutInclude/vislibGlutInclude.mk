#
# vislibGlutInclude.mk
#
# Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
# Alle Rechte vorbehalten.
#

ifneq ($(MAKECMDGOALS), clean)
ifneq ($(MAKECMDGOALS), sweep)
include ../glutInclude/glutInclude.lin.mk
endif
endif
