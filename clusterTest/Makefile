# Makefile 12.4.2008
#
# Copyright (C) 2008 by Visualisierungsinstitut der Universität Stuttgart.
# Alle Rechte vorbehalten.
# Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
#

# Include global VISlib configuration.
include ../config.mk


# Include VISlib properties
include ExtLibs.mk


# Target name
TargetName := clusterTest

# Output directory
OutDir := ../$(VISlibPlatform)

#Intermediate directory
IntDir := ./Lin$(BITSEX)
 #$(OutDir)

# Output subdirectory name for Release files
ReleaseDir := Release

# Output subdirectory for Debug files
DebugDir := Debug


# Additional include directories
IncludeDirs := .

# Additional system include directories
SystemIncludeDirs := /usr/include/g++ /usr/include/g++/bits /usr/include/g++/ext

# Additional compiler flags
CompilerFlags := -fPIC -DVISLIB_CLUSTER_WITH_OPENGL -D_GNU_SOURCE -DGLX_GLXEXT_LEGACY
CompilerFlagsDebug := -ggdb
CompilerFlagsRelease := -O3 -g0

# Additional linker flags
LinkerFlags := -lX11 -lXext -lXxf86vm -lm -lpthread -lGLU -lGL -lncurses
LinkerFlagsDebug :=
LinkerFlagsRelease :=


# Input directory
InputDir := .

# Files to be excluded from build
ExcludeFromBuild :=


#glutInclude
IncludeDirs += ../glutInclude

ifneq ($(MAKECMDGOALS), clean)
ifneq ($(MAKECMDGOALS), sweep)
include ../glutInclude/vislibGlutInclude.mk
# why a different naming convention than in '../config.mk' ?
IncludeDirs += $(IncludeDir)
LinkerFlags += $(LDFLAGS)
endif
endif


#
# USE VARIABLES ABOVE TO MODIFY COMPILER AND LINKER FLAGS AS WELL AS IN- AND OUTPUT!
#
CPPFLAGS := $(CompilerFlags) $(addprefix -I, $(IncludeDirs)) $(addprefix -I, $(VISlibIncludeDirs)) $(addprefix -isystem, $(SystemIncludeDirs))
CPPFLAGS_DEBUG := $(CPPFLAGS) $(CompilerFlagsDebug) -DDEBUG -D_DEBUG -DLIN$(VISlibBits)
CPPFLAGS_RELEASE := $(CPPFLAGS) $(CompilerFlagsRelease) -DNDEBUG -D_NDEBUG -DLIN$(VISlibBits)

LDFLAGS := $(LinkerFlags) -L$(VISlibLibDir)
LDFLAGS_DEBUG := $(addprefix -l, $(VISlibsDebug)) $(LDFLAGS) $(LinkerFlagsDebug)
LDFLAGS_RELEASE := $(addprefix -l, $(VISlibsRelease)) $(LDFLAGS) $(LinkerFlagsRelease)

CPPSRCS := $(filter-out $(ExcludeFromBuild), $(wildcard $(InputDir)/*.cpp))

CPPOBJS := $(patsubst %.cpp, %.o, $(notdir $(CPPSRCS)))
CPPOBJS_DEBUG := $(addprefix $(IntDir)/$(DebugDir)/, $(notdir $(CPPOBJS)))
CPPOBJS_RELEASE := $(addprefix $(IntDir)/$(ReleaseDir)/, $(notdir $(CPPOBJS)))
CPPOBJS_ALL := $(CPPOBJS_DEBUG) $(CPPOBJS_RELEASE)

CPPDEPS := $(notdir $(patsubst %.cpp, %.d, $(CPPSRCS)))
CPPDEPS_DEBUG := $(addprefix $(IntDir)/$(DebugDir)/, $(CPPDEPS))
CPPDEPS_RELEASE := $(addprefix $(IntDir)/$(ReleaseDir)/, $(CPPDEPS))
CPPDEPS_ALL := $(CPPDEPS_DEBUG) $(CPPDEPS_RELEASE)

#
# RULES
#
all: $(TargetName)Debug $(TargetName)Release


# Rules for final binaries:
$(TargetName)Debug: $(OutDir)/$(DebugDir)/$(TargetName)
$(OutDir)/$(DebugDir)/$(TargetName): $(IntDir)/$(DebugDir)/$(TargetName)
	@mkdir -p $(dir $@)
	if [ ! $@ -ef $< ]; then cp $< $@; fi

$(TargetName)Release: $(OutDir)/$(ReleaseDir)/$(TargetName)
$(OutDir)/$(ReleaseDir)/$(TargetName): $(IntDir)/$(ReleaseDir)/$(TargetName)
	@mkdir -p $(dir $@)
	if [ ! $@ -ef $< ]; then cp $< $@; fi


# Rules for intermediate binaries:
$(IntDir)/$(DebugDir)/$(TargetName): $(CPPOBJS_DEBUG)
	@echo -e $(COLORACTION)"LNK "$(COLORINFO)" $@: "
	@$(CLEARTERMCMD)
	$(Q)$(LINK) $^ $(LDFLAGS_DEBUG) -o $@

$(IntDir)/$(ReleaseDir)/$(TargetName): $(CPPOBJS_RELEASE)
	@echo -e $(COLORACTION)"LNK "$(COLORINFO)" $@: "
	@$(CLEARTERMCMD)
	$(Q)$(LINK) $^ $(LDFLAGS_RELEASE) -o $@


# Rules for dependencies:
$(IntDir)/$(DebugDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"DEP "$(COLORINFO)"$@: "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -MM $(CPPFLAGS_DEBUG) $^ | sed -e 's/..*\.o\s*[:]/$(subst .,\.,$(subst /,\/,$(dir $@)))\0/g' > $@

$(IntDir)/$(ReleaseDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"DEP "$(COLORINFO)"$@: "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -MM $(CPPFLAGS_RELEASE) $^ | sed -e 's/..*\.o\s*[:]/$(subst .,\.,$(subst /,\/,$(dir $@)))\0/g' > $@

ifneq ($(MAKECMDGOALS), clean)
ifneq ($(MAKECMDGOALS), sweep)
-include $(CPPDEPS_ALL)
endif
endif


# Rules for object files:
$(IntDir)/$(DebugDir)/%.o:
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"CPP "$(COLORINFO)"$@: "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -c $(CPPFLAGS_DEBUG) -o $@ $<

$(IntDir)/$(ReleaseDir)/%.o:
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"CPP "$(COLORINFO)"$@: "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -c $(CPPFLAGS_RELEASE) -o $@ $<


# Cleanup rules:
clean: sweep
	rm -rf $(IntDir)/*
	rm -f $(OutDir)/$(DebugDir)/$(TargetName)
	rm -f $(OutDir)/$(ReleaseDir)/$(TargetName)


sweep:
	rm -f $(CPPDEPS_ALL)
	rm -f $(CPPOBJS_ALL)


rebuild: clean all


.PHONY: clean sweep rebuild tags
