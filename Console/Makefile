#
# Makefile
# Console (MegaMol)
#
# Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
# Alle Rechte vorbehalten.
#

include ../common.mk
include ../ExtLibs.mk

# Target name
TargetName := MegaMolCon
IncludeDir := $(IncludeDir) ../Viewer/include $(visglutpath)/include $(mmcorepath)/api
VISlibs := sys math base graphics gl


# Additional compiler flags
CompilerFlags := $(CompilerFlags)

# Additional linker flags
LinkerFlags := $(LinkerFlags)

# Libraries
LIBS := $(LIBS) m pthread pam pam_misc dl ncurses GL


# Collect Files
CPP_SRCS := $(filter-out $(ExcludeFromBuild), $(wildcard $(InputDir)/*.cpp))
CPP_DEPS := $(addprefix $(IntDir)/$(DebugDir)/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS)))) \
	$(addprefix $(IntDir)/$(ReleaseDir)/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))

IncludeDir := $(IncludeDir) $(addprefix $(vislibpath)/,$(addsuffix /include,$(VISlibs)))
DebugLinkerFlags := $(DebugLinkerFlags) $(addprefix -lvislib,$(addsuffix $(BITS)d,$(VISlibs)))
ReleaseLinkerFlags := $(ReleaseLinkerFlags) $(addprefix -lvislib,$(addsuffix $(BITS),$(VISlibs)))

CPPFLAGS := $(CompilerFlags) $(addprefix -I, $(IncludeDir)) $(addprefix -isystem, $(SystemIncludeDir))
LDFLAGS := $(LinkerFlags) -L$(mmcorepath)/lib -L$(vislibpath)/lib


all: $(TargetName)Debug $(TargetName)Release


# Rules for binary in $(OutDir):
$(TargetName)Debug: $(IntDir)/$(DebugDir)/$(TargetName)
	@mkdir -p $(outbin)
	cp $< $(outbin)/$(TargetName)$(BITS)d

$(TargetName)Release: $(IntDir)/$(ReleaseDir)/$(TargetName)
	@mkdir -p $(outbin)
	cp $< $(outbin)/$(TargetName)$(BITS)


# Rules for intermediate binary:	
$(IntDir)/$(DebugDir)/$(TargetName): $(addprefix $(IntDir)/$(DebugDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS)))) 
	@echo -e '\E[1;32;40m'"LNK "'\E[0;32;40m'"$(IntDir)/$(DebugDir)/$(TargetName): "
	@tput sgr0
	$(Q)$(LINK) $(LDFLAGS) $(addprefix $(IntDir)/$(DebugDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS)))) \
	$(DebugLinkerFlags) -lMegaMolCore$(BITS)d \
	$(addprefix -l,$(LIBS)) -o $(IntDir)/$(DebugDir)/$(TargetName)

$(IntDir)/$(ReleaseDir)/$(TargetName): $(addprefix $(IntDir)/$(ReleaseDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
	@echo -e '\E[1;32;40m'"LNK "'\E[0;32;40m'"$(IntDir)/$(ReleaseDir)/$(TargetName): "
	@tput sgr0
	$(Q)$(LINK) $(LDFLAGS) $(addprefix $(IntDir)/$(ReleaseDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS)))) \
	$(ReleaseLinkerFlags) -lMegaMolCore$(BITS) \
	$(addprefix -l,$(LIBS)) -o $(IntDir)/$(ReleaseDir)/$(TargetName)


# Rules for dependencies:
$(IntDir)/$(DebugDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	@echo -e '\E[1;32;40m'"DEP "'\E[0;32;40m'"$@: "
	@tput sgr0
	$(Q)$(CPP) -MM $(CPPFLAGS) $(DebugCompilerFlags) $^ | sed -e 's/..*\.o\s*[:]/$(IntDir)\/$(DebugDir)\/\0/g' > $@

$(IntDir)/$(ReleaseDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	@echo -e '\E[1;32;40m'"DEP "'\E[0;32;40m'"$@: "
	@tput sgr0
	$(Q)$(CPP) -MM $(CPPFLAGS) $(ReleaseCompilerFlags) $^ | sed -e 's/..*\.o\s*[:]/$(IntDir)\/$(ReleaseDir)\/\0/g' > $@


ifneq ($(MAKECMDGOALS), clean)
ifneq ($(MAKECMDGOALS), sweep)
-include $(CPP_DEPS)
endif
endif


# Rules for object files: 
$(IntDir)/$(DebugDir)/%.o:
	@mkdir -p $(dir $@)
	@echo -e '\E[1;32;40m'"CPP "'\E[0;32;40m'"$@: "
	@tput sgr0
	$(Q)$(CPP) -c $(CPPFLAGS) $(DebugCompilerFlags) -o $@ $<	

$(IntDir)/$(ReleaseDir)/%.o:
	@mkdir -p $(dir $@)
	@echo -e '\E[1;32;40m'"CPP "'\E[0;32;40m'"$@: "
	@tput sgr0
	$(Q)$(CPP) -c $(CPPFLAGS) $(ReleaseCompilerFlags) -o $@ $<	


# Cleanup rules:	
clean: sweep
	rm -f $(SolOutputDir)/$(TargetName)$(BITS) $(SolOutputDir)/$(TargetName)$(BITS)d


sweep:
	rm -f $(CPP_DEPS)
	rm -rf $(IntDir)/*


rebuild: clean all


.PHONY: clean sweep rebuild tags
