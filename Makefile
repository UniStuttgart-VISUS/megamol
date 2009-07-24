# Applications:
MAKE = make
SHELL = /bin/bash

# Project directories to make:
ProjectDirs = base sys math graphics glutInclude gl net cluster test glutTest clustergl clusterTest

################################################################################

all: VersionInfo
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

VersionInfo:
	@mkdir -p $(dir $@)
	@echo -e '\E[1;32;40m'"GEN "'\E[0;32;40m'"base/include/vislib/vislibversion.h: "
	@tput sgr0
	$(Q)cd base && $(Q)perl makevislibversion.pl .. vislibversion.template.h include/vislib/vislibversion.h

.PHONY: all clean sweep rebuild VersionInfo
