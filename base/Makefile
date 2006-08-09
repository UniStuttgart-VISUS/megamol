# Applications:
CPP = g++
LINK = g++
AR = ar

# Name of the ouput file:
OutName := libBase.a

# This directory contains the source files:
InputDir := ./src

# List of include directories:
IncludeDir := ./include

# List of system include directories:
SystemIncludeDir := /usr/include/g++ /usr/include/g++/bits /usr/include/g++/ext

# List of files to be excluded from build:
ExcludeFromBuild := 

# Intermediate directory:
IntDir := Lin32

# Output directory:
OutDir := ../Lin32

# C++ compiler flags:
CPPFLAGS := -DUNIX -D_GNU_SOURCE -Wall -ansi -pedantic -O3 -ggdb

# Linker flags:
LDFLAGS := -L/usr/X11R6/lib -lm -lpthread

###############################################################################

# Append the include directories to the flags:
CPPFLAGS += $(addprefix -I, $(IncludeDir)) $(addprefix -isystem, $(SystemIncludeDir))

# Collect all sources from source directory:
CPP_SRCS := $(filter-out $(ExcludeFromBuild), $(wildcard $(InputDir)/$(dir)/*.cpp))

# Create object files in intermediate directory:
CPP_OBJS = $(addprefix $(IntDir)/, $(notdir $(patsubst %.cpp, %.o, $(CPP_SRCS))))
CPP_DEPS = $(patsubst %.o, %.d, $(CPP_OBJS))


all: $(OutDir)/$(OutName)

$(OutDir)/$(OutName): $(IntDir)/$(OutName)
	@mkdir -p $(OutDir)
	@cp $(IntDir)/$(OutName) $(OutDir)/
	
$(IntDir)/$(OutName): $(CPP_OBJS) 
	$(AR) -cvq $(IntDir)/$(OutName) $(CPP_OBJS)

$(IntDir)/%.d: $(InputDir)/%.cpp
	@mkdir -p $(IntDir)
	$(CPP) -MM $(CPPFLAGS) $^ | sed -e 's/..*\.o\s*[:]/$(IntDir)\/\0/g' > $@

-include $(CPP_DEPS)
	
$(IntDir)/%.o:
	@mkdir -p $(IntDir)
	$(CPP) -c $(CPPFLAGS) -o $@ $<
	
#tags:
#	ctags *.c *.cpp *.h

clean: sweep
	rm -f $(OutDir)/$(ProjectName).a

sweep:
	rm -f $(IntDir)/*

rebuild: clean all

.PHONY: clean sweep rebuild tags
