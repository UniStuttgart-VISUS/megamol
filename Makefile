# Applications:
MAKE = make
SHELL = /bin/bash

# Project directories to make:
ProjectDirs = base sys math graphics gl net cluster clusterGL test glutTest

################################################################################

all:
	@for dir in $(ProjectDirs); do $(MAKE) -C $$dir $@ || exit 1; done

sweep:
	@for dir in $(ProjectDirs); do $(MAKE) -C $$dir $@ || exit 1; done
	
clean:	
	@for dir in $(ProjectDirs); do $(MAKE) -C $$dir $@ || exit 1; done
	
rebuild: clean all

#base:
#	$(MAKE) -C $@
#	
#sys: base
#	$(MAKE) -C $@
#	
#math: base
#	$(MAKE) -C $@
#	
#net: base sys
#	$(MAKE) -C $@
#	
#gl: base graphics math sys
#	$(MAKE) -C $@
#	
#graphics: base math
#	$(MAKE) -C $@
	
.PHONY: all clean sweep rebuild