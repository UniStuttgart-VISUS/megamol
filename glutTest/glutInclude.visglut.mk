#
# glutInclude.mk
#
# Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#
# Include this file for the visglut library.
#

ifeq ($(BITS), 64)
	LDFLAGS += -L%visglutLinPath%/freeglut/lib/lin64d -lfreeglut
else
	LDFLAGS += -L%visglutLinPath%/freeglut/lib/lin32d -lfreeglut
endif
