#
# librules.mk  13.09.2006 (mueller)
#
# Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

CPP_SRCS := $(filter-out $(ExcludeFromBuild), $(wildcard $(InputDir)/*.cpp))
CPP_DEPS_DEBUG := $(addprefix $(IntDir)/$(DebugDir)/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))
CPP_DEPS_RELEASE :=	$(addprefix $(IntDir)/$(ReleaseDir)/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))
CPP_DEPS := $(CPP_DEPS_DEBUG) $(CPP_DEPS_RELEASE)

OBJS_DEBUG := $(addprefix $(IntDir)/$(DebugDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
OBJS_RELEASE := $(addprefix $(IntDir)/$(ReleaseDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))

CPPFLAGS := $(CompilerFlags) $(addprefix -I, $(IncludeDir)) $(addprefix -isystem, $(SystemIncludeDir))
LDFLAGS := $(LinkerFlags)


all: $(TargetName)d $(TargetName)


# Rules for archives in $(OutDir):
$(TargetName)d: $(IntDir)/$(DebugDir)/lib$(TargetName).a
	@mkdir -p $(OutDir)
	cp $< $(OutDir)/lib$(TargetName)$(BITS)d.a
	
$(TargetName): $(IntDir)/$(ReleaseDir)/lib$(TargetName).a 
	@mkdir -p $(OutDir)
	cp $< $(OutDir)/lib$(TargetName)$(BITS).a
	

# Rules for intermediate archives:
$(IntDir)/$(DebugDir)/lib$(TargetName).a: $(OBJS_DEBUG)
	@echo -e '\E[1;32;40m'"AR "'\E[0;32;40m'"$@ "
	@tput sgr0
	$(Q)$(AR) $(ARFLAGS) $@ $^
	
$(IntDir)/$(ReleaseDir)/lib$(TargetName).a: $(OBJS_RELEASE)
	@echo -e '\E[1;32;40m'"AR "'\E[0;32;40m'"$@ "
	@tput sgr0
	$(Q)$(AR) $(ARFLAGS) $@ $^
	

# Rules for dependencies:
$(IntDir)/$(DebugDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	@echo -e '\E[1;32;40m'"DEP "'\E[0;32;40m'"$@ "
	@tput sgr0
	$(Q)$(CPP) -MM $(CPPFLAGS) $(DebugCompilerFlags) $(filter %.cpp, $^) | sed -e 's/\(..*\)\.o\s*\:/$(IntDir)\/$(DebugDir)\/\1.d $(IntDir)\/$(DebugDir)\/\1.o:/g' > $@
	
$(IntDir)/$(ReleaseDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	@echo -e '\E[1;32;40m'"DEP "'\E[0;32;40m'"$@ "
	@tput sgr0
	$(Q)$(CPP) -MM $(CPPFLAGS) $(ReleaseCompilerFlags) $(filter %.cpp, $^) | sed -e 's/\(..*\)\.o\s*\:/$(IntDir)\/$(ReleaseDir)\/\1.d $(IntDir)\/$(ReleaseDir)\/\1.o:/g' > $@


ifneq ($(MAKECMDGOALS), clean)
ifneq ($(MAKECMDGOALS), sweep)
-include $(CPP_DEPS)
endif
endif


# Rules for object files:	
$(IntDir)/$(DebugDir)/%.o:
	@mkdir -p $(dir $@)
	@echo -e '\E[1;32;40m'"CPP "'\E[0;32;40m'"$@ "
	@tput sgr0
	$(Q)$(CPP) -c $(CPPFLAGS) $(DebugCompilerFlags) -o $@ $<	
	
$(IntDir)/$(ReleaseDir)/%.o:
	@mkdir -p $(dir $@)
	@echo -e '\E[1;32;40m'"CPP "'\E[0;32;40m'"$@ "
	@tput sgr0
	$(Q)$(CPP) -c $(CPPFLAGS) $(ReleaseCompilerFlags) -o $@ $<	
	
	
# Cleanup rules:	
clean: sweep
	rm -f $(OutDir)/lib$(TargetName)$(BITS).a $(OutDir)/lib$(TargetName)$(BITS)d.a

sweep:
	rm -rf $(IntDir)/*

rebuild: 
	@$(MAKE) clean
	@$(MAKE) all

.PHONY: clean sweep rebuild tags