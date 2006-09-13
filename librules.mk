#
# librules.mk  13.09.2006 (mueller)
#
# Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

CPP_SRCS := $(filter-out $(ExcludeFromBuild), $(wildcard $(InputDir)/*.cpp))
CPP_DEPS := $(addprefix $(IntDir)/$(DebugANSIDir)/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))\
	$(addprefix $(IntDir)/$(DebugUnicodeDir)/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))\
	$(addprefix $(IntDir)/$(ReleaseANSIDir)/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))\
	$(addprefix $(IntDir)/$(ReleaseUnicodeDir)/, $(notdir $(patsubst %.cpp, %.d, $(CPP_SRCS))))

CPPFLAGS := $(CompilerFlags) $(addprefix -I, $(IncludeDir)) $(addprefix -isystem, $(SystemIncludeDir))
LDFLAGS := $(LinkerFlags)

all: $(TargetName)ad $(TargetName)wd $(TargetName)a $(TargetName)w


# Rules for archives in $(OutDir):
$(TargetName)ad: $(IntDir)/$(DebugANSIDir)/lib$(TargetName).a
	@mkdir -p $(OutDir)
	cp $< $(OutDir)/lib$(TargetName)$(BITS)ad.a
	
$(TargetName)wd: $(IntDir)/$(DebugUnicodeDir)/lib$(TargetName).a
	@mkdir -p $(OutDir)
	cp $< $(OutDir)/lib$(TargetName)$(BITS)wd.a
	
$(TargetName)a: $(IntDir)/$(ReleaseANSIDir)/lib$(TargetName).a
	@mkdir -p $(OutDir)
	cp $< $(OutDir)/lib$(TargetName)$(BITS)a.a	
	
$(TargetName)w: $(IntDir)/$(ReleaseUnicodeDir)/lib$(TargetName).a
	@mkdir -p $(OutDir)
	cp $< $(OutDir)/lib$(TargetName)$(BITS)w.a		
	

# Rules for intermediate archives:	
$(IntDir)/$(DebugANSIDir)/lib$(TargetName).a: $(addprefix $(IntDir)/$(DebugANSIDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
	$(AR) -cvq $@ $^
	
$(IntDir)/$(DebugUnicodeDir)/lib$(TargetName).a: $(addprefix $(IntDir)/$(DebugUnicodeDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
	$(AR) -cvq $@ $^
	
$(IntDir)/$(ReleaseANSIDir)/lib$(TargetName).a: $(addprefix $(IntDir)/$(ReleaseANSIDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
	$(AR) -cvq $@ $^
	
$(IntDir)/$(ReleaseUnicodeDir)/lib$(TargetName).a: $(addprefix $(IntDir)/$(ReleaseUnicodeDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
	$(AR) -cvq $@ $^			
	

# Rules for dependencies:
$(IntDir)/$(DebugANSIDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	$(CPP) -MM $(CPPFLAGS) $(DebugANSICompilerFlags) $^ | sed -e 's/..*\.o\s*[:]/$(IntDir)\/$(DebugANSIDir)\/\0/g' > $@
	
$(IntDir)/$(DebugUnicodeDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	$(CPP) -MM $(CPPFLAGS) $(DebugUnicodeCompilerFlags) $^ | sed -e 's/..*\.o\s*[:]/$(IntDir)\/$(DebugUnicodeDir)\/\0/g' > $@
	
$(IntDir)/$(ReleaseANSIDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	$(CPP) -MM $(CPPFLAGS) $(ReleaseANSICompilerFlags) $^ | sed -e 's/..*\.o\s*[:]/$(IntDir)\/$(ReleaseANSIDir)\/\0/g' > $@
	
$(IntDir)/$(ReleaseUnicodeDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(dir $@)
	$(CPP) -MM $(CPPFLAGS) $(ReleaseUnicodeCompilerFlags) $^ | sed -e 's/..*\.o\s*[:]/$(IntDir)\/$(ReleaseUnicodeDir)\/\0/g' > $@		
	
	
-include $(CPP_DEPS)
	
# Rules for object files:	
$(IntDir)/$(DebugANSIDir)/%.o:
	@mkdir -p $(dir $@)
	$(CPP) -c $(CPPFLAGS) $(DebugANSICompilerFlags) -o $@ $<	
	
$(IntDir)/$(DebugUnicodeDir)/%.o:
	@mkdir -p $(dir $@)
	$(CPP) -c $(CPPFLAGS) $(DebugUnicodeCompilerFlags) -o $@ $<	
	
$(IntDir)/$(ReleaseANSIDir)/%.o:
	@mkdir -p $(dir $@)
	$(CPP) -c $(CPPFLAGS) $(ReleaseANSICompilerFlags) -o $@ $<	
	
$(IntDir)/$(ReleaseUnicodeDir)/%.o:
	@mkdir -p $(dir $@)
	$(CPP) -c $(CPPFLAGS) $(ReleaseUnicodeCompilerFlags) -o $@ $<				
	
	
# Cleanup rules:	
clean: sweep
	rm -f $(OutDir)/*.a

sweep:
	rm -f $(CPP_DEPS)
	rm -rf $(IntDir)/*

rebuild: clean all

.PHONY: clean sweep rebuild tags