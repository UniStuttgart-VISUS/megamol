#
# glutInclude.mk
#
# Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#
# Include this file for the visglut library.
#

ifeq ($(PLATFORM), x86_64)
	LDFLAGS += -l%visglutLinPath%/freeglut/lib/lin64d/libfreeglut.a
else
	LDFLAGS += -l%visglutLinPath%/freeglut/lib/lin32d/libfreeglut.a
endif
