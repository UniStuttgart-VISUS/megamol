#
# librules.mk  13.09.2006 (mueller)
#
# Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

CPP_SRCS := $(filter-out $(ExcludeFromBuild), $(wildcard $(InputDir)/*.cpp))
CPP_DEPS := $(addprefix $(IntDir)/$(DebugDir)/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))\
	$(addprefix $(IntDir)/$(ReleaseDir)/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))

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
$(IntDir)/$(DebugDir)/lib$(TargetName).a: $(addprefix $(IntDir)/$(DebugDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
	$(AR) -cvq $@ $^
	
$(IntDir)/$(ReleaseDir)/lib$(TargetName).a: $(addprefix $(IntDir)/$(ReleaseDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
	$(AR) -cvq $@ $^	
	

# Rules for dependencies:
$(IntDir)/$(DebugDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	$(CPP) -MM $(CPPFLAGS) $(DebugCompilerFlags) $^ | sed -e 's/..*\.o\s*[:]/$(IntDir)\/$(DebugDir)\/\0/g' > $@
	
$(IntDir)/$(ReleaseDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	$(CPP) -MM $(CPPFLAGS) $(ReleaseCompilerFlags) $^ | sed -e 's/..*\.o\s*[:]/$(IntDir)\/$(ReleaseDir)\/\0/g' > $@
	
	
-include $(CPP_DEPS)
	
# Rules for object files:	
$(IntDir)/$(DebugDir)/%.o:
	@mkdir -p $(dir $@)
	$(CPP) -c $(CPPFLAGS) $(DebugCompilerFlags) -o $@ $<	
	
$(IntDir)/$(ReleaseDir)/%.o:
	@mkdir -p $(dir $@)
	$(CPP) -c $(CPPFLAGS) $(ReleaseCompilerFlags) -o $@ $<	
	
	
# Cleanup rules:	
clean: sweep
	rm -f $(OutDir)/lib$(TargetName)$(BITS).a $(OutDir)/lib$(TargetName)$(BITS)d.a

sweep:
	rm -f $(CPP_DEPS)
	rm -rf $(IntDir)/*

rebuild: clean all

.PHONY: clean sweep rebuild tags