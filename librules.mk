#
# librules.mk  13.09.2006 (mueller)
#
# Copyright (C) 2006-2014 by Universitaet Stuttgart (VIS) and TU Dresden (CGV).
# Alle Rechte vorbehalten.
#

CPP_SRCS := $(filter-out $(ExcludeFromBuild), $(wildcard $(InputDir)/*.cpp))
CPP_DEPS_DEBUG := $(addprefix $(IntDir)/$(DebugDir)/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))
CPP_DEPS_RELEASE :=	$(addprefix $(IntDir)/$(ReleaseDir)/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))
CPP_DEPS := $(CPP_DEPS_DEBUG) $(CPP_DEPS_RELEASE) $(CPP_DEPS_DEBUG_EXPORT) $(CPP_DEPS_RELEASE_EXPORT) $(CPP_DEPS_DEBUG_IMPORT) $(CPP_DEPS_RELEASE_IMPORT)

OBJS_DEBUG := $(addprefix $(IntDir)/$(DebugDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
OBJS_RELEASE := $(addprefix $(IntDir)/$(ReleaseDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))

CPPFLAGS := $(CompilerFlags) $(addprefix -I, $(IncludeDir)) $(addprefix -isystem, $(SystemIncludeDir))
LDFLAGS := $(LinkerFlags)


all: $(TargetName)d $(TargetName)


# Rules for archives in $(OutDir):
$(TargetName)d: $(IntDir)/$(DebugDir)/lib$(TargetName).a
	@mkdir -p $(OutDir)
	cp $< $(OutDir)/lib$(TargetName)$(BITS)d.a
	cp $< $(OutDir)/lib$(TargetName)$(BITSEX)d.a

$(TargetName): $(IntDir)/$(ReleaseDir)/lib$(TargetName).a 
	@mkdir -p $(OutDir)
	cp $< $(OutDir)/lib$(TargetName)$(BITS).a
	cp $< $(OutDir)/lib$(TargetName)$(BITSEX).a


# Rules for intermediate archives:
$(IntDir)/$(DebugDir)/lib$(TargetName).a: $(OBJS_DEBUG)
	@echo -e $(COLORACTION)"AR "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(AR) $(ARFLAGS) $@ $^

$(IntDir)/$(ReleaseDir)/lib$(TargetName).a: $(OBJS_RELEASE)
	@echo -e $(COLORACTION)"AR "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(AR) $(ARFLAGS) $@ $^


# Rules for dependencies:
$(IntDir)/$(DebugDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"DEP "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -MM $(CPPFLAGS) $(DebugCompilerFlags) $(filter %.cpp, $^) | sed -e 's/\(..*\)\.o\s*\:/$(IntDir)\/$(DebugDir)\/\1.d $(IntDir)\/$(DebugDir)\/\1.o:/g' > $@

$(IntDir)/$(ReleaseDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"DEP "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -MM $(CPPFLAGS) $(ReleaseCompilerFlags) $(filter %.cpp, $^) | sed -e 's/\(..*\)\.o\s*\:/$(IntDir)\/$(ReleaseDir)\/\1.d $(IntDir)\/$(ReleaseDir)\/\1.o:/g' > $@


ifneq ($(MAKECMDGOALS), clean)
ifneq ($(MAKECMDGOALS), sweep)
-include $(CPP_DEPS)
endif
endif


# Rules for object files:
$(IntDir)/$(DebugDir)/%.o:
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"CPP "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -c $(CPPFLAGS) $(DebugCompilerFlags) -o $@ $<

$(IntDir)/$(ReleaseDir)/%.o:
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"CPP "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -c $(CPPFLAGS) $(ReleaseCompilerFlags) -o $@ $<


# Cleanup rules:
clean: sweep
	rm -f $(OutDir)/lib$(TargetName)$(BITS).a $(OutDir)/lib$(TargetName)$(BITS)d.a \
		$(OutDir)/lib$(TargetName)$(BITSEX).a $(OutDir)/lib$(TargetName)$(BITSEX)d.a

sweep:
	rm -rf $(IntDir)/*

rebuild: 
	@$(MAKE) clean
	@$(MAKE) all

.PHONY: clean sweep rebuild tags
