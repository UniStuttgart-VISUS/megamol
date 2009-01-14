#
# Makefile
# glutInclude (VISLIB)
#
# Copyright (C) 2006 - 2009 by VISUS (Universitaet Stuttgart).
# Alle Rechte vorbehalten.
#

include ../config.mk

all:
	$(Q)cp -T -u glutInclude.system.h glutInclude.lin.h
	$(Q)cp -T -u glutInclude.system.mk glutInclude.lin.mk

clean:
# we do not want to delete these, because this is configuration result, not build result.
#	rm -f glutInclude.lin.h
#	rm -f glutInclude.lin.mk


rebuild: clean all


.PHONY: clean rebuild tags
