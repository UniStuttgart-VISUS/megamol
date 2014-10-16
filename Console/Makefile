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
VISlibs := gl graphics sys math base


# Additional compiler flags
CompilerFlags := $(CompilerFlags)

# Additional linker flags
LinkerFlags := $(LinkerFlags)

# Libraries
LIBS := $(LIBS) m pthread pam pam_misc uuid dl ncurses GL rt X11 glload


# Collect Files
CPP_SRCS := $(filter-out $(ExcludeFromBuild), $(wildcard $(InputDir)/*.cpp))
CPP_DEPS := $(addprefix $(IntDir)/$(DebugDir)/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS)))) \
	$(addprefix $(IntDir)/$(ReleaseDir)/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))

IncludeDir := $(IncludeDir) $(addprefix $(vislibpath)/,$(addsuffix /include,$(VISlibs)))
DebugLinkerFlags := $(DebugLinkerFlags) $(addprefix -lvislib,$(addsuffix $(BITS)d,$(VISlibs)))
ReleaseLinkerFlags := $(ReleaseLinkerFlags) $(addprefix -lvislib,$(addsuffix $(BITS),$(VISlibs)))

CPPFLAGS := $(CompilerFlags) $(addprefix -I, $(IncludeDir)) $(addprefix -isystem, $(SystemIncludeDir))
LDFLAGS := $(LinkerFlags) -L$(mmcorepath)/lib -L$(vislibpath)/lib -L$(vislibpath)/gl/include/glload/build


all: VersionInfo $(TargetName)Debug $(TargetName)Release


# Rules for binary in $(OutDir):
$(TargetName)Debug: $(IntDir)/$(DebugDir)/$(TargetName)
	@mkdir -p $(outbin)
	cp $< $(outbin)/$(TargetName)$(BITS)d

$(TargetName)Release: $(IntDir)/$(ReleaseDir)/$(TargetName)
	@mkdir -p $(outbin)
	cp $< $(outbin)/$(TargetName)$(BITS)


# Rules for intermediate binary:
$(IntDir)/$(DebugDir)/$(TargetName): Makefile version.gen.h $(addprefix $(IntDir)/$(DebugDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
	@echo -e $(COLORACTION)"LNK "$(COLORINFO)"$(IntDir)/$(DebugDir)/$(TargetName): "
	@$(CLEARTERMCMD)
	$(Q)$(LINK) $(LDFLAGS) $(addprefix $(IntDir)/$(DebugDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS)))) \
	$(DebugLinkerFlags) -lMegaMolCore$(BITS)d \
	$(addprefix -l,$(LIBS)) -o $(IntDir)/$(DebugDir)/$(TargetName)

$(IntDir)/$(ReleaseDir)/$(TargetName): Makefile version.gen.h $(addprefix $(IntDir)/$(ReleaseDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
	@echo -e $(COLORACTION)"LNK "$(COLORINFO)"$(IntDir)/$(ReleaseDir)/$(TargetName): "
	@$(CLEARTERMCMD)
	$(Q)$(LINK) $(LDFLAGS) $(addprefix $(IntDir)/$(ReleaseDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS)))) \
	$(ReleaseLinkerFlags) -lMegaMolCore$(BITS) \
	$(addprefix -l,$(LIBS)) -o $(IntDir)/$(ReleaseDir)/$(TargetName)


# Rules for dependencies:
$(IntDir)/$(DebugDir)/%.d: $(InputDir)/%.cpp Makefile version.gen.h
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"DEP "$(COLORINFO)"$@: "
	@$(CLEARTERMCMD)
	@echo -n $(dir $@) > $@
	$(Q)$(CPP) -MM $(CPPFLAGS) $(DebugCompilerFlags) $< >> $@

$(IntDir)/$(ReleaseDir)/%.d: $(InputDir)/%.cpp Makefile version.gen.h
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"DEP "$(COLORINFO)"$@: "
	@$(CLEARTERMCMD)
	@echo -n $(dir $@) > $@
	$(Q)$(CPP) -MM $(CPPFLAGS) $(ReleaseCompilerFlags) $< >> $@


ifneq ($(MAKECMDGOALS), clean)
ifneq ($(MAKECMDGOALS), sweep)
-include $(CPP_DEPS)
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


# Rules for generated files:
VersionInfo: version.template.h VersionInfo.pl Makefile
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"GEN "$(COLORINFO)"version.gen.h: "
	@$(CLEARTERMCMD)
	$(Q)perl VersionInfo.pl .

version.gen.h: version.template.h VersionInfo.pl Makefile
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"GEN "$(COLORINFO)"version.gen.h: "
	@$(CLEARTERMCMD)
	$(Q)perl VersionInfo.pl .


# Cleanup rules:	
clean: sweep
	rm -f $(SolOutputDir)/$(TargetName)$(BITS) $(SolOutputDir)/$(TargetName)$(BITS)d


sweep:
	rm -f $(CPP_DEPS)
	rm -rf $(IntDir)/*


rebuild: clean all


.PHONY: clean sweep rebuild tags
