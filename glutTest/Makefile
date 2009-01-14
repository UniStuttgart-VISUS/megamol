#
# Makefile  13.09.2006 (mueller)
#
# Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

include ../config.mk

TargetName := vislibgluttest

IncludeDir := . ../base/include ../math/include ../graphics/include ../gl/include ../sys/include ../glutInclude
InputDir := ./


# Name of the ouput file:
OutName := vislibgluttest


# C++ compiler flags:
CPPFLAGS := -DDEBUG -DUNIX -D_GNU_SOURCE -DGLX_GLXEXT_LEGACY -Wall -ansi -O3 -ggdb -Wno-missing-braces
# removed: -pedantic 

# Linker flags:
LDFLAGS := -L/usr/X11R6/lib -L../lib -lm -lpthread -lGLU -lGL

ifneq ($(MAKECMDGOALS), clean)
ifneq ($(MAKECMDGOALS), sweep)
include ../glutInclude/vislibGlutInclude.mk
endif
endif

###############################################################################

IntDir := Lin$(BITSEX)
ifeq ($(PLATFORM), x64)
	OutDir := ../Lin64
	CPPFLAGS += -D_LIN64
	LDFLAGS += -lvislibgl64 -lvislibgraphics64 -lvislibmath64 -lvislibsys64 -lvislibbase64
else
	OutDir := ../Lin32
	LDFLAGS += -lvislibgl32 -lvislibgraphics32 -lvislibmath32 -lvislibsys32 -lvislibbase32
endif

# Append the include directories to the flags:
CPPFLAGS += $(addprefix -I, $(IncludeDir)) $(addprefix -isystem, $(SystemIncludeDir))

# Collect all sources from source directory:
CPP_SRCS := $(filter-out $(ExcludeFromBuild), $(wildcard $(InputDir)/$(dir)/*.cpp))

# Create object files in intermediate directory:
CPP_OBJS = $(addprefix $(IntDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
CPP_DEPS = $(patsubst %.o, %.d, $(CPP_OBJS))


all: $(OutDir)/$(OutName)

$(OutDir)/$(OutName): $(IntDir)/$(OutName)
	@mkdir -p $(OutDir)
	@cp $(IntDir)/$(OutName) $(OutDir)/
	
$(IntDir)/$(OutName): $(CPP_OBJS) 
	@echo -e '\E[1;32;40m'"LNK "'\E[0;32;40m'" $@: "
	@tput sgr0
	$(Q)$(LINK) -o $@ $^ $(LDFLAGS)

$(IntDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(IntDir)
	@echo -e '\E[1;32;40m'"DEP "'\E[0;32;40m'"$@: "
	@tput sgr0
	$(Q)$(CPP) -MM $(CPPFLAGS) $< | sed -e 's/..*\.o\s*[:]/$(IntDir)\/\0/g' > $@


ifneq ($(MAKECMDGOALS), clean)
ifneq ($(MAKECMDGOALS), sweep)
-include $(CPP_DEPS)
endif
endif
	
	
$(IntDir)/%.o:
	@mkdir -p $(IntDir)
	@echo -e '\E[1;32;40m'"CPP "'\E[0;32;40m'"$@: "
	@tput sgr0
	$(Q)$(CPP) -c $(CPPFLAGS) -o $@ $<


#tags:
#	ctags *.c *.cpp *.h

clean: sweep
	rm -f $(OutDir)/$(OutName)

sweep:
	rm -f $(IntDir)/*

rebuild: clean all

.PHONY: clean sweep rebuild tags
