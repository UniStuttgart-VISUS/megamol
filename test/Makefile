#
# Makefile  13.09.2006 (mueller)
#
# Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

-include ../common.mk

TargetName := vislibtest

IncludeDir := ./include ../base/include ../sys/include ../math/include ../graphics/include
InputDir := ./

# VISlib libraries to link (in that order):
VISlibDir := ../lib
VISlibs := graphics math net sys base


# Name of the ouput file:
OutName := vislibtest


# C++ compiler flags:
CPPFLAGS := -DDEBUG -DUNIX -D_GNU_SOURCE -Wall -ansi -pedantic -O3 -ggdb

# Linker flags:
LDFLAGS := -L/usr/X11R6/lib -L$(VISlibDir) -lm -lpthread -lpam -lpam_misc -ldl 


###############################################################################

# Determine the current platform:
PLATFORM := $(shell uname -p)

ifeq ($(PLATFORM), x86_64)
	IntDir := Lin64
	OutDir := ../Lin64
	CPPFLAGS += -D_LIN64
	VISlibs := $(addsuffix 64d, $(addprefix vislib, $(VISlibs)))
else
	IntDir := Lin32
	OutDir := ../Lin32
	VISlibs := $(addsuffix 32d, $(addprefix vislib, $(VISlibs)))
endif

# Append the VISlib libraries to the linker flags:
LDFLAGS += $(addprefix -l, $(VISlibs))

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
	
$(IntDir)/$(OutName): $(CPP_OBJS) $(addsuffix .a, $(addprefix $(VISlibDir)/lib, $(VISlibs)))
	@echo -e '\E[1;32;40m'"LNK "'\E[0;32;40m'" $@: "
	@tput sgr0
	$(LINK) -o $@ $^ $(LDFLAGS)

$(IntDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(IntDir)
	@echo -e '\E[1;32;40m'"DEP "'\E[0;32;40m'"$@: "
	@tput sgr0
	$(CPP) -MM $(CPPFLAGS) $^ | sed -e 's/..*\.o\s*[:]/$(IntDir)\/\0/g' > $@


ifneq ($(MAKECMDGOALS), clean)
ifneq ($(MAKECMDGOALS), sweep)
-include $(CPP_DEPS)
endif
endif

	
$(IntDir)/%.o:
	@mkdir -p $(IntDir)
	@echo -e '\E[1;32;40m'"CPP "'\E[0;32;40m'"$@: "
	@tput sgr0
	$(CPP) -c $(CPPFLAGS) -o $@ $<
	
#tags:
#	ctags *.c *.cpp *.h

clean: sweep
	rm -f $(OutDir)/$(OutName)

sweep:
	rm -f $(IntDir)/*

rebuild: clean all

.PHONY: clean sweep rebuild tags
