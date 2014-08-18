#
# librules.mk  13.09.2006 (mueller)
#
# Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

CPP_SRCS := $(filter-out $(ExcludeFromBuild), $(wildcard $(InputDir)/*.cpp))
CPP_DEPS_DEBUG := $(addprefix $(IntDir)/$(DebugDir)/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))
CPP_DEPS_DEBUG_EXPORT := $(addprefix $(IntDir)/$(DebugDir)Export/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))
CPP_DEPS_DEBUG_IMPORT := $(addprefix $(IntDir)/$(DebugDir)Import/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))
CPP_DEPS_RELEASE :=	$(addprefix $(IntDir)/$(ReleaseDir)/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))
CPP_DEPS_RELEASE_EXPORT :=	$(addprefix $(IntDir)/$(ReleaseDir)Export/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))
CPP_DEPS_RELEASE_IMPORT :=	$(addprefix $(IntDir)/$(ReleaseDir)Import/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))
CPP_DEPS := $(CPP_DEPS_DEBUG) $(CPP_DEPS_RELEASE) $(CPP_DEPS_DEBUG_EXPORT) $(CPP_DEPS_RELEASE_EXPORT) $(CPP_DEPS_DEBUG_IMPORT) $(CPP_DEPS_RELEASE_IMPORT)

OBJS_DEBUG := $(addprefix $(IntDir)/$(DebugDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
OBJS_DEBUG_EXPORT := $(addprefix $(IntDir)/$(DebugDir)Export/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
OBJS_DEBUG_IMPORT := $(addprefix $(IntDir)/$(DebugDir)Import/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
OBJS_RELEASE := $(addprefix $(IntDir)/$(ReleaseDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
OBJS_RELEASE_EXPORT := $(addprefix $(IntDir)/$(ReleaseDir)Export/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
OBJS_RELEASE_IMPORT := $(addprefix $(IntDir)/$(ReleaseDir)Import/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))

CPPFLAGS := $(CompilerFlags) $(addprefix -I, $(IncludeDir)) $(addprefix -isystem, $(SystemIncludeDir))
LDFLAGS := $(LinkerFlags)

EXPORTCOMPILEFLAGS := -DVISLIB_SYMBOL_EXPORT
IMPORTCOMPILEFLAGS := -DVISLIB_SYMBOL_IMPORT


all: $(TargetName)d $(TargetName) $(TargetName)id $(TargetName)i $(TargetName)ed $(TargetName)e


# Rules for archives in $(OutDir):
$(TargetName)d: $(IntDir)/$(DebugDir)/lib$(TargetName).a
	@mkdir -p $(OutDir)
	cp $< $(OutDir)/lib$(TargetName)$(BITS)d.a
	cp $< $(OutDir)/lib$(TargetName)$(BITSEX)d.a

$(TargetName): $(IntDir)/$(ReleaseDir)/lib$(TargetName).a 
	@mkdir -p $(OutDir)
	cp $< $(OutDir)/lib$(TargetName)$(BITS).a
	cp $< $(OutDir)/lib$(TargetName)$(BITSEX).a

$(TargetName)id: $(IntDir)/$(DebugDir)Import/lib$(TargetName).a
	@mkdir -p $(OutDir)
	cp $< $(OutDir)/lib$(TargetName)$(BITS)id.a
	cp $< $(OutDir)/lib$(TargetName)$(BITSEX)id.a

$(TargetName)i: $(IntDir)/$(ReleaseDir)Import/lib$(TargetName).a 
	@mkdir -p $(OutDir)
	cp $< $(OutDir)/lib$(TargetName)$(BITS)i.a
	cp $< $(OutDir)/lib$(TargetName)$(BITSEX)i.a

$(TargetName)ed: $(IntDir)/$(DebugDir)Export/lib$(TargetName).a
	@mkdir -p $(OutDir)
	cp $< $(OutDir)/lib$(TargetName)$(BITS)ed.a
	cp $< $(OutDir)/lib$(TargetName)$(BITSEX)ed.a

$(TargetName)e: $(IntDir)/$(ReleaseDir)Export/lib$(TargetName).a 
	@mkdir -p $(OutDir)
	cp $< $(OutDir)/lib$(TargetName)$(BITS)e.a
	cp $< $(OutDir)/lib$(TargetName)$(BITSEX)e.a


# Rules for intermediate archives:
$(IntDir)/$(DebugDir)/lib$(TargetName).a: $(OBJS_DEBUG)
	@echo -e $(COLORACTION)"AR "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(AR) $(ARFLAGS) $@ $^

$(IntDir)/$(ReleaseDir)/lib$(TargetName).a: $(OBJS_RELEASE)
	@echo -e $(COLORACTION)"AR "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(AR) $(ARFLAGS) $@ $^

$(IntDir)/$(DebugDir)Import/lib$(TargetName).a: $(OBJS_DEBUG_IMPORT)
	@echo -e $(COLORACTION)"AR "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(AR) $(ARFLAGS) $@ $^

$(IntDir)/$(ReleaseDir)Import/lib$(TargetName).a: $(OBJS_RELEASE_IMPORT)
	@echo -e $(COLORACTION)"AR "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(AR) $(ARFLAGS) $@ $^

$(IntDir)/$(DebugDir)Export/lib$(TargetName).a: $(OBJS_DEBUG_EXPORT)
	@echo -e $(COLORACTION)"AR "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(AR) $(ARFLAGS) $@ $^

$(IntDir)/$(ReleaseDir)Export/lib$(TargetName).a: $(OBJS_RELEASE_EXPORT)
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

$(IntDir)/$(DebugDir)Export/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"DEP "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -MM $(CPPFLAGS) $(DebugCompilerFlags) $(EXPORTCOMPILEFLAGS) $(filter %.cpp, $^) | sed -e 's/\(..*\)\.o\s*\:/$(IntDir)\/$(DebugDir)Export\/\1.d $(IntDir)\/$(DebugDir)Export\/\1.o:/g' > $@

$(IntDir)/$(ReleaseDir)Export/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"DEP "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -MM $(CPPFLAGS) $(ReleaseCompilerFlags) $(EXPORTCOMPILEFLAGS) $(filter %.cpp, $^) | sed -e 's/\(..*\)\.o\s*\:/$(IntDir)\/$(ReleaseDir)Export\/\1.d $(IntDir)\/$(ReleaseDir)Export\/\1.o:/g' > $@

$(IntDir)/$(DebugDir)Import/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"DEP "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -MM $(CPPFLAGS) $(DebugCompilerFlags) $(IMPORTCOMPILEFLAGS) $(filter %.cpp, $^) | sed -e 's/\(..*\)\.o\s*\:/$(IntDir)\/$(DebugDir)Import\/\1.d $(IntDir)\/$(DebugDir)Import\/\1.o:/g' > $@

$(IntDir)/$(ReleaseDir)Import/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"DEP "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -MM $(CPPFLAGS) $(ReleaseCompilerFlags) $(IMPORTCOMPILEFLAGS) $(filter %.cpp, $^) | sed -e 's/\(..*\)\.o\s*\:/$(IntDir)\/$(ReleaseDir)Import\/\1.d $(IntDir)\/$(ReleaseDir)Import\/\1.o:/g' > $@


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

$(IntDir)/$(DebugDir)Export/%.o:
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"CPP "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -c $(CPPFLAGS) $(DebugCompilerFlags) $(EXPORTCOMPILEFLAGS) -o $@ $<

$(IntDir)/$(ReleaseDir)Export/%.o:
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"CPP "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -c $(CPPFLAGS) $(ReleaseCompilerFlags) $(EXPORTCOMPILEFLAGS) -o $@ $<

$(IntDir)/$(DebugDir)Import/%.o:
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"CPP "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -c $(CPPFLAGS) $(DebugCompilerFlags) $(IMPORTCOMPILEFLAGS) -o $@ $<

$(IntDir)/$(ReleaseDir)Import/%.o:
	@mkdir -p $(dir $@)
	@echo -e $(COLORACTION)"CPP "$(COLORINFO)"$@ "
	@$(CLEARTERMCMD)
	$(Q)$(CPP) -c $(CPPFLAGS) $(ReleaseCompilerFlags) $(IMPORTCOMPILEFLAGS) -o $@ $<


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