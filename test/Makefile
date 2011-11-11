#
# Makefile  13.09.2006 (mueller)
#
# Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

-include ../config.mk

TargetName := vislibtest

IncludeDir := ./include ../base/include ../sys/include ../math/include ../graphics/include ../net/include
InputDir := ./

# VISlib libraries to link (in that order):
VISlibDir := ../lib
VISlibs := graphics math net sys base


# C++ compiler flags:
CPPFLAGS := -DDEBUG -DUNIX -D_GNU_SOURCE -Wall -ansi -pedantic -O0 -ggdb

# Linker flags:
#LDFLAGS := -L$(VISlibDir) -lX11 -lXext -lm -lpthread -lpam -lpam_misc -ldl -lncurses 
LDFLAGS := -L$(VISlibDir) -lX11 -lXext -lm -lpthread -ldl -lncurses  -lrt -lcrypt


###############################################################################

IntDir := Lin$(BITSEX)
ifeq ($(PLATFORM), x64)
	OutDir := ../Lin64
	CPPFLAGS += -D_LIN64
#	LDFLAGS += -L/usr/X11R6/lib64
	VISlibs := $(addsuffix 64d, $(addprefix vislib, $(VISlibs)))
else
	OutDir := ../Lin32
#	LDFLAGS += -L/usr/X11R6/lib
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


all: $(OutDir)/$(TargetName)

$(OutDir)/$(TargetName): $(IntDir)/$(TargetName)
	@mkdir -p $(OutDir)
	@cp $^ $(OutDir)/
	
$(IntDir)/$(TargetName): $(CPP_OBJS) $(addsuffix .a, $(addprefix $(VISlibDir)/lib, $(VISlibs)))
	@echo -e $(COLORACTION)"LNK "$(COLORINFO)" $@: "
	@$(CLEARTERMCMD)
	$(Q)$(LINK) -o $@ $^ $(LDFLAGS)

$(IntDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(IntDir)
	@echo -e $(COLORACTION)"DEP "$(COLORINFO)"$@: "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -MM $(CPPFLAGS) $^ | sed -e 's/..*\.o\s*[:]/$(IntDir)\/\0/g' > $@


ifneq ($(MAKECMDGOALS), clean)
ifneq ($(MAKECMDGOALS), sweep)
-include $(CPP_DEPS)
endif
endif

	
$(IntDir)/%.o:
	@mkdir -p $(IntDir)
	@echo -e $(COLORACTION)"CPP "$(COLORINFO)"$@: "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -c $(CPPFLAGS) -o $@ $<
	
#tags:
#	ctags *.c *.cpp *.h

clean: sweep
	rm -f $(OutDir)/$(TargetName)

sweep:
	rm -f $(IntDir)/*

rebuild: clean all

.PHONY: clean sweep rebuild tags $(OutDir)/$(TargetName)
