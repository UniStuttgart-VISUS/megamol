#
# Makefile
# Glut (MegaMol)
#
# Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
# Alle Rechte vorbehalten.
#

include ../common.mk
include ../ExtLibs.mk

# Target name
TargetName := MegaMolGlut
OutDir := lib
# subdirectories below $(InputRootDir)
InputRootDir := $(InputDir)
InputDir := .
IncludeDir := . ../Viewer/include
IncludeDir := $(IncludeDir) $(vislibpath)/base/include $(vislibpath)/math/include \
	$(vislibpath)/sys/include $(vislibpath)/graphics/include $(vislibpath)/gl/include \
	$(visglutpath)/include


# Additional compiler flags
CompilerFlags := $(CompilerFlags) -fPIC -DMEGAMOLVIEWER_EXPORTS -DMEGAMOLVIEWER_USESTATIC
ExcludeFromBuild += ./DllEntry.c


# Libraries
LIBS := $(LIBS) m pthread pam pam_misc dl ncurses GL GLU


# Additional linker flags
LinkerFlags := $(LinkerFlags) -shared
#	-lc -Wl,-e,mmCoreMain


# Collect Files
# WARNING: $(InputDirs) MUST be relative paths!
CPP_SRCS := $(filter-out $(ExcludeFromBuild), $(wildcard $(InputDir)/*.cpp))
CPP_DEPS := $(addprefix $(IntDir)/$(DebugDir)/, $(patsubst %.cpp, %.d, $(CPP_SRCS)))\
	$(addprefix $(IntDir)/$(ReleaseDir)/, $(patsubst %.cpp, %.d, $(CPP_SRCS)))
CPP_D_OBJS := $(addprefix $(IntDir)/$(DebugDir)/, $(patsubst %.cpp, %.o, $(CPP_SRCS)))
CPP_R_OBJS := $(addprefix $(IntDir)/$(ReleaseDir)/, $(patsubst %.cpp, %.o, $(CPP_SRCS)))

CPPFLAGS := $(CompilerFlags) $(addprefix -I, $(IncludeDir)) $(addprefix -isystem, $(SystemIncludeDir))
LDFLAGS := $(LinkerFlags) -L$(vislibpath)/lib -L$(visglutpath)/lib


all: $(TargetName)d $(TargetName)


# Rules for shared objects in $(OutDir):
$(TargetName)d: $(IntDir)/$(DebugDir)/lib$(TargetName)$(BITS)d.so
	@mkdir -p $(OutDir)
	@mkdir -p $(SolOutputDir)
	cp $< $(OutDir)/lib$(TargetName)$(BITS)d.so
	cp $< $(SolOutputDir)/lib$(TargetName)$(BITS)d.so

$(TargetName): $(IntDir)/$(ReleaseDir)/lib$(TargetName)$(BITS).so
	@mkdir -p $(OutDir)
	@mkdir -p $(SolOutputDir)
	cp $< $(OutDir)/lib$(TargetName)$(BITS).so
	cp $< $(SolOutputDir)/lib$(TargetName)$(BITS).so


# Rules for intermediate shared objects:
$(IntDir)/$(DebugDir)/lib$(TargetName)$(BITS)d.so: Makefile $(addprefix $(IntDir)/$(DebugDir)/, $(patsubst %.cpp, %.o, $(CPP_SRCS)))
	@echo -e '\E[1;32;40m'"LNK "'\E[0;32;40m'"$(IntDir)/$(DebugDir)/lib$(TargetName).so: "
	@tput sgr0
	$(Q)$(LINK) $(LDFLAGS) $(CPP_D_OBJS) $(addprefix -l,$(LIBS)) \
	-lvislibgl$(BITS) -lvislibgraphics$(BITS)d -lvislibsys$(BITS)d -lvislibmath$(BITS)d -lvislibbase$(BITS)d -lvisglut$(BITS)d \
	-o $(IntDir)/$(DebugDir)/lib$(TargetName)$(BITS)d.so

#	gcc -Wall -W -DPIC -fPIC -shared -DUNIX -D_GNU_SOURCE -D_LIN64 DllEntry.c -lc -Wl,-e,mmCoreMain \
#	-o $(IntDir)/$(DebugDir)/lib$(TargetName)$(BITS)d.so

$(IntDir)/$(ReleaseDir)/lib$(TargetName)$(BITS).so: Makefile $(addprefix $(IntDir)/$(ReleaseDir)/, $(patsubst %.cpp, %.o, $(CPP_SRCS)))
	@echo -e '\E[1;32;40m'"LNK "'\E[0;32;40m'"$(IntDir)/$(ReleaseDir)/lib$(TargetName).so: "
	@tput sgr0
	$(Q)$(LINK) $(LDFLAGS) $(CPP_R_OBJS) $(addprefix -l,$(LIBS)) \
	-lvislibgl$(BITS) -lvislibgraphics$(BITS) -lvislibsys$(BITS) -lvislibmath$(BITS) -lvislibbase$(BITS) -lvisglut$(BITS) \
	-o $(IntDir)/$(ReleaseDir)/lib$(TargetName)$(BITS).so

#	gcc -Wall -W -DPIC -fPIC -shared -DUNIX -D_GNU_SOURCE -D_LIN64 DllEntry.c -lc -Wl,-e,mmCoreMain \
#	-o $(IntDir)/$(ReleaseDir)/lib$(TargetName)$(BITS).so


# Rules for dependencies:
$(IntDir)/$(DebugDir)/%.d: $(InputDir)/%.cpp Makefile
	@mkdir -p $(dir $@)
	@echo -e '\E[1;32;40m'"DEP "'\E[0;32;40m'"$@: "
	@tput sgr0
	@echo -n $(dir $@) > $@
	$(Q)$(CPP) -MM $(CPPFLAGS) $(DebugCompilerFlags) $< >> $@

$(IntDir)/$(ReleaseDir)/%.d: $(InputDir)/%.cpp Makefile
	@mkdir -p $(dir $@)
	@echo -e '\E[1;32;40m'"DEP "'\E[0;32;40m'"$@: "
	@tput sgr0
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
	rm -f $(OutDir)/lib$(TargetName)$(BITS).so $(OutDir)/lib$(TargetName)$(BITS)d.so \
	$(SolOutputDir)/lib$(TargetName)$(BITS).so $(SolOutputDir)/lib$(TargetName)$(BITS)d.so


sweep:
	rm -f $(CPP_DEPS)
	rm -rf $(IntDir)/*


rebuild: clean all


.PHONY: clean sweep rebuild tags
