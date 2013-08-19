#
# Makefile
# Protein (MegaMol)
#
# Copyright (C) 2008-2010 by VISUS (Universitaet Stuttgart).
# Alle Rechte vorbehalten.
#

include common.mk
include ExtLibs.mk

# Target name
# TODO: Change the name "Template" to the name of your plugin
TargetName := Protein
# subdirectories below $(InputRootDir)
InputRootDir := $(InputDir)
InputDirs := .
IncludeDir := $(IncludeDir) $(mmcorepath)
VISlibs := net gl graphics sys math base


# Additional compiler flags
CompilerFlags := $(CompilerFlags) -fPIC -std=c++0x
ExcludeFromBuild += ./dllmain.cpp


# Libraries
LIBS := $(LIBS) m pthread pam pam_misc dl ncurses uuid GL GLU

# Additional linker flags
LinkerFlags := $(LinkerFlags) -shared -Wl,-Bsymbolic 

# Collect Files
# WARNING: $(InputDirs) MUST be relative paths!
CPP_SRCS := $(filter-out $(ExcludeFromBuild), $(foreach InputDir, $(InputDirs), $(wildcard $(InputDir)/*.cpp)))
CPP_DEPS := $(addprefix $(IntDir)/$(DebugDir)/, $(patsubst %.cpp, %.d, $(CPP_SRCS)))\
	$(addprefix $(IntDir)/$(ReleaseDir)/, $(patsubst %.cpp, %.d, $(CPP_SRCS)))
CPP_D_OBJS := $(addprefix $(IntDir)/$(DebugDir)/, $(patsubst %.cpp, %.o, $(CPP_SRCS)))
CPP_R_OBJS := $(addprefix $(IntDir)/$(ReleaseDir)/, $(patsubst %.cpp, %.o, $(CPP_SRCS)))

IncludeDir := $(IncludeDir) $(addprefix $(vislibpath)/,$(addsuffix /include,$(VISlibs)))

DebugLinkerFlags := $(DebugLinkerFlags) $(addprefix -lvislib,$(addsuffix $(BITS)d,$(VISlibs)))
ReleaseLinkerFlags := $(ReleaseLinkerFlags) $(addprefix -lvislib,$(addsuffix $(BITS),$(VISlibs)))

CPPFLAGS := $(CompilerFlags) $(addprefix -I, $(IncludeDir)) $(addprefix -isystem, $(SystemIncludeDir))
LDFLAGS := $(LinkerFlags) -L$(vislibpath)/lib -L$(expatpath)/lib

all: $(TargetName)d $(TargetName)


# Rules for plugins in $(SolOutputDir):
$(TargetName)d: $(IntDir)/$(DebugDir)/$(TargetName)$(BITS)d.lin$(BITS).mmplg
	@mkdir -p $(outbin)
	cp $< $(outbin)/$(TargetName)$(BITS)d.lin$(BITS).mmplg

$(TargetName): $(IntDir)/$(ReleaseDir)/$(TargetName)$(BITS).lin$(BITS).mmplg
	@mkdir -p $(outbin)
	cp $< $(outbin)/$(TargetName)$(BITS).lin$(BITS).mmplg

# Rules for intermediate plugins:

$(IntDir)/$(DebugDir)/$(TargetName)$(BITS)d.lin$(BITS).mmplg: Makefile $(addprefix $(IntDir)/$(DebugDir)/, $(patsubst %.cpp, %.o, $(CPP_SRCS)))\
	$(addprefix $(IntDir)/$(CudaDebugDir)/, $(patsubst %.cu, %.o, $(CU_SOURCES)))
	@echo -e '\E[1;32;40m'"LNK "'\E[0;32;40m'"$(IntDir)/$(DebugDir)/$(TargetName)$(BITS)d.lin$(BITS).mmplg: "
	@tput sgr0
	$(Q)$(LINK) $(LDFLAGS) $(CPP_D_OBJS) $(CU_D_OBS) $(addprefix -l,$(LIBS)) $(DebugLinkerFlags) \
	-o $(IntDir)/$(DebugDir)/$(TargetName)$(BITS)d.lin$(BITS).mmplg

$(IntDir)/$(ReleaseDir)/$(TargetName)$(BITS).lin$(BITS).mmplg: Makefile $(addprefix $(IntDir)/$(ReleaseDir)/, $(patsubst %.cpp, %.o, $(CPP_SRCS)))\
	$(addprefix $(IntDir)/$(CudaReleaseDir)/, $(patsubst %.cu, %.o, $(CU_SOURCES)))
	@echo -e '\E[1;32;40m'"LNK "'\E[0;32;40m'"$(IntDir)/$(ReleaseDir)/$(TargetName)$(BITS).lin$(BITS).mmplg: "
	@tput sgr0
	$(Q)$(LINK) $(LDFLAGS) $(CPP_R_OBJS) $(CU_R_OBS) $(addprefix -l,$(LIBS)) $(ReleaseLinkerFlags) \
	-o $(IntDir)/$(ReleaseDir)/$(TargetName)$(BITS).lin$(BITS).mmplg

# Rules for dependencies:

$(IntDir)/$(DebugDir)/%.d: $(InputDir)/%.cpp Makefile
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"DEP "$(COLORINFO)"$@: "
	@$(CLEARTERMCMD)
	@echo -n $(dir $@) > $@
	$(Q)$(CPP) -MM $(CPPFLAGS) $(DebugCompilerFlags) $< >> $@

$(IntDir)/$(ReleaseDir)/%.d: $(InputDir)/%.cpp Makefile
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"DEP "$(COLORINFO)"$@: "
	@$(CLEARTERMCMD)
	@echo -n $(dir $@) > $@
	$(Q)$(CPP) -MM $(CPPFLAGS) $(ReleaseCompilerFlags) $< >> $@ 


ifneq ($(MAKECMDGOALS), clean)
ifneq ($(MAKECMDGOALS), sweep)
-include $(CPP_DEPS)
-include $(CU_DEPS)
endif
endif

# Rules for object files:

$(IntDir)/$(DebugDir)/%.o:
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"CPP "$(COLORINFO)"$@: "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -c $(CPPFLAGS) $(DebugCompilerFlags) -o $@ $<

$(IntDir)/$(ReleaseDir)/%.o:
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"CPP "$(COLORINFO)"$@: "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -c $(CPPFLAGS) $(ReleaseCompilerFlags) -o $@ $<

# Cleanup rules:

clean: sweep
	rm -f $(outbin)/$(TargetName)$(BITS)d.lin$(BITS).mmplg \
	$(outbin)/$(TargetName)$(BITS).lin$(BITS).mmplg

sweep:
	rm -f $(CPP_DEPS)
	rm -rf $(IntDir)/*

rebuild: clean all


.PHONY: clean sweep rebuild tags
