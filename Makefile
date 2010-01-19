#
# Makefile
# Core (MegaMol)
#
# Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
# Alle Rechte vorbehalten.
#

include ../common.mk
include ../ExtLibs.mk
include ../ExtLibs_core.mk

# Target name
TargetName := MegaMolCore
OutDir := lib
# subdirectories below $(InputRootDir)
InputRootDir := $(InputDir)
InputDirs := . api utility utility/xml param modnet view special job protein misc vismol2 moldyn
IncludeDir := $(IncludeDir) $(vislibpath)/base/include $(vislibpath)/math/include \
	$(vislibpath)/sys/include $(vislibpath)/graphics/include $(vislibpath)/gl/include \
	$(vislibpath)/net/include $(vislibpath)/cluster/include \
	$(expatpath)/include 


# Additional compiler flags
CompilerFlags := $(CompilerFlags) -fPIC
ExcludeFromBuild += ./DllEntry.c protein/RMSRenderer.cpp vismol2/Mol20DataCall.cpp \
	vismol2/Mol20DataSource.cpp vismol2/Mol20Renderer.cpp vismol2/Mol2Data.cpp


# Libraries
LIBS := $(LIBS) m pthread pam pam_misc dl ncurses uuid GL GLU expat$(BITS) png z


# Additional linker flags
LinkerFlags := $(LinkerFlags) -shared
#	-lc -Wl,-e,mmCoreMain


# Collect Files
# WARNING: $(InputDirs) MUST be relative paths!
CPP_SRCS := $(filter-out $(ExcludeFromBuild), $(foreach InputDir, $(InputDirs), $(wildcard $(InputDir)/*.cpp)))
CPP_DEPS := $(addprefix $(IntDir)/$(DebugDir)/, $(patsubst %.cpp, %.d, $(CPP_SRCS)))\
	$(addprefix $(IntDir)/$(ReleaseDir)/, $(patsubst %.cpp, %.d, $(CPP_SRCS)))
CPP_D_OBJS := $(addprefix $(IntDir)/$(DebugDir)/, $(patsubst %.cpp, %.o, $(CPP_SRCS)))
CPP_R_OBJS := $(addprefix $(IntDir)/$(ReleaseDir)/, $(patsubst %.cpp, %.o, $(CPP_SRCS)))

CPPFLAGS := $(CompilerFlags) $(addprefix -I, $(IncludeDir)) $(addprefix -isystem, $(SystemIncludeDir))
LDFLAGS := $(LinkerFlags) -L$(vislibpath)/lib -L$(expatpath)/lib


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


# Rules for special intermediate files
generatedFiles: api/MegaMolCore.inl


api/MegaMolCore.inl: api/MegaMolCore.h api/geninl.pl
	@echo -e '\E[1;32;40m'"GEN "'\E[0;32;40m'"api/MegaMolCore.inl: "
	@tput sgr0
	$(Q)perl api/geninl.pl api/MegaMolCore.h api/MegaMolCore.inl


# Rules for intermediate shared objects:
$(IntDir)/$(DebugDir)/lib$(TargetName)$(BITS)d.so: Makefile generatedFiles $(addprefix $(IntDir)/$(DebugDir)/, $(patsubst %.cpp, %.o, $(CPP_SRCS)))
	@echo -e '\E[1;32;40m'"LNK "'\E[0;32;40m'"$(IntDir)/$(DebugDir)/lib$(TargetName).so: "
	@tput sgr0
	$(Q)$(LINK) $(LDFLAGS) $(CPP_D_OBJS) $(addprefix -l,$(LIBS)) $(DebugLinkerFlags) \
	-lvislibgl$(BITS)d -lvislibgraphics$(BITS)d -lvislibcluster$(BITS)d -lvislibnet$(BITS)d -lvislibsys$(BITS)d -lvislibmath$(BITS)d -lvislibbase$(BITS)d \
	-o $(IntDir)/$(DebugDir)/lib$(TargetName)$(BITS)d.so

#	gcc -Wall -W -DPIC -fPIC -shared -DUNIX -D_GNU_SOURCE -D_LIN64 DllEntry.c -lc -Wl,-e,mmCoreMain \
#	-o $(IntDir)/$(DebugDir)/lib$(TargetName)$(BITS)d.so

$(IntDir)/$(ReleaseDir)/lib$(TargetName)$(BITS).so: Makefile generatedFiles $(addprefix $(IntDir)/$(ReleaseDir)/, $(patsubst %.cpp, %.o, $(CPP_SRCS)))
	@echo -e '\E[1;32;40m'"LNK "'\E[0;32;40m'"$(IntDir)/$(ReleaseDir)/lib$(TargetName).so: "
	@tput sgr0
	$(Q)$(LINK) $(LDFLAGS) $(CPP_R_OBJS) $(addprefix -l,$(LIBS)) $(ReleaseLinkerFlags) \
	-lvislibgl$(BITS) -lvislibgraphics$(BITS) -lvislibcluster$(BITS) -lvislibnet$(BITS) -lvislibsys$(BITS) -lvislibmath$(BITS) -lvislibbase$(BITS) \
	-o $(IntDir)/$(ReleaseDir)/lib$(TargetName)$(BITS).so

#	gcc -Wall -W -DPIC -fPIC -shared -DUNIX -D_GNU_SOURCE -D_LIN64 DllEntry.c -lc -Wl,-e,mmCoreMain \
#	-o $(IntDir)/$(ReleaseDir)/lib$(TargetName)$(BITS).so


# Rules for dependencies:
$(IntDir)/$(DebugDir)/%.d: $(InputDir)/%.cpp Makefile api/MegaMolCore.inl
	@mkdir -p $(dir $@)
	@echo -e '\E[1;32;40m'"DEP "'\E[0;32;40m'"$@: "
	@tput sgr0
	@echo -n $(dir $@) > $@
	$(Q)$(CPP) -MM $(CPPFLAGS) -I$(<D) $(DebugCompilerFlags) $< >> $@

$(IntDir)/$(ReleaseDir)/%.d: $(InputDir)/%.cpp Makefile api/MegaMolCore.inl
	@mkdir -p $(dir $@)
	@echo -e '\E[1;32;40m'"DEP "'\E[0;32;40m'"$@: "
	@tput sgr0
	@echo -n $(dir $@) > $@
	$(Q)$(CPP) -MM $(CPPFLAGS) -I$(<D) $(ReleaseCompilerFlags) $< >> $@


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
	$(Q)$(CPP) -c $(CPPFLAGS) -I$(<D) $(DebugCompilerFlags) -o $@ $<

$(IntDir)/$(ReleaseDir)/%.o:
	@mkdir -p $(dir $@)
	@echo -e '\E[1;32;40m'"CPP "'\E[0;32;40m'"$@: "
	@tput sgr0
	$(Q)$(CPP) -c $(CPPFLAGS) -I$(<D) $(ReleaseCompilerFlags) -o $@ $<


# Cleanup rules:	
clean: sweep
	rm -f $(OutDir)/lib$(TargetName)$(BITS).so $(OutDir)/lib$(TargetName)$(BITS)d.so \
	$(SolOutputDir)/lib$(TargetName)$(BITS).so $(SolOutputDir)/lib$(TargetName)$(BITS)d.so


sweep:
	rm -f $(CPP_DEPS)
	rm -rf $(IntDir)/*


rebuild: clean all


.PHONY: clean sweep rebuild tags
