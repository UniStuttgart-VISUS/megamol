#
# Makefile  13.09.2006 (mueller)
#
# Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

-include ../common.mk

TargetName := vislibtest

IncludeDir := ./include ../base/include ../sys/include ../math/include
InputDir := ./


# Name of the ouput file:
OutName := vislibtest


# C++ compiler flags:
CPPFLAGS := -DUNIX -D_GNU_SOURCE -Wall -ansi -pedantic -O3 -ggdb

# Linker flags:
LDFLAGS := -L/usr/X11R6/lib -L../lib -lm -lpthread -lvislibsys32d -lvislibmath32d -lvislibbase32d

###############################################################################

# Determine the current platform:
PLATFORM := $(shell uname -p)

ifeq ($(PLATFORM), x86_64)
	IntDir := Lin64
	OutDir := ../Lin64
	CPPFLAGS += -D_LIN64
else
	IntDir := Lin32
	OutDir := ../Lin32
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
	$(LINK) -o $@ $^ $(LDFLAGS)

$(IntDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(IntDir)
	$(CPP) -MM $(CPPFLAGS) $^ | sed -e 's/..*\.o\s*[:]/$(IntDir)\/\0/g' > $@

-include $(CPP_DEPS)
	
$(IntDir)/%.o:
	@mkdir -p $(IntDir)
	$(CPP) -c $(CPPFLAGS) -o $@ $<
	
#tags:
#	ctags *.c *.cpp *.h

clean: sweep
	rm -f $(OutDir)/$(OutName)

sweep:
	rm -f $(IntDir)/*

rebuild: clean all

.PHONY: clean sweep rebuild tags
