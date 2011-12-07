#
# common.mk  13.09.2006 (mueller)
#
# Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

# Applications
AR = ar
CPP = g++ 
LINK = g++
MAKE = make
SHELL = /bin/bash

# Experimental support for ICC
ifndef VISLIB_ICC
	VISLIB_ICC = 0
endif
ifneq ($(VISLIB_ICC), 0)
CPP = icpc
LINK = icpc
endif


# Set Verbosity
ifndef VISLIB_VERBOSE
    VISLIB_VERBOSE = 0
endif
ifeq ($(VISLIB_VERBOSE), 1)
    Q =
    ARFLAGS = -cvr
else
    Q = @
    ARFLAGS = -cr
endif


# Determine the plaform architecture
ifeq ($(shell uname -m), x86_64)
	PLATFORM := x64
	BITS := 64
else
	PLATFORM := x86
	BITS := 32
endif


# check whether we have the terminal for color outout
ifeq ($(TERM), xterm)
        COLORACTION = '\E[1;32;40m'
        COLORINFO =  '\E[0;32;40m'
        CLEARTERMCMD = tput sgr0
else
        COLORACTION =
        COLORINFO =
        CLEARTERMCMD = true
endif


# Add clib version to the bits field for binary compatibility
CLIBVER := $(shell /lib/libc.so.6 | sed -n 's/^.*C.*Library.*version[[:space:]]\+\([[:digit:]\.]\+\)[[:space:]].*$ /\1/I p' | sed -n 's/\./_/g p')
BITSEX := $(BITS)_clib$(CLIBVER)


# The default input directory
InputDir := ./src

# The default include directories:
IncludeDir := ./include

# List of system include directories:
SystemIncludeDir := /usr/include/g++ /usr/include/g++/bits /usr/include/g++/ext

# The default output directory
OutDir := ../lib

# Default source exclude patterns:
ExcludeFromBuild := 


# The intermediate directory
IntDir := Lin$(BITSEX)


# The configuration intermetidate directories
DebugDir := Debug
ReleaseDir := Release


# Common compiler flags
CompilerFlags := -DUNIX -D_GNU_SOURCE -D_LIN$(BITS) -Wall -ansi -pedantic -fPIC

CPPVersionInfo := $(shell $(CPP) --version | tr "[:upper:]" "[:lower:]")
ifneq (,$(findstring gcc,$(CPPVersionInfo)))
ifneq (,$(findstring 4.3,$(CPPVersionInfo)))
	# Add -fpermissive to gcc 4.3.* flags because they fail at name resolution
	CompilerFlags := $(CompilerFlags) -fpermissive
endif
ifneq (,$(findstring 4.,$(CPPVersionInfo)))
	# Add -Wno-variadic-macros to gcc 4.* flags 
	CompilerFlags := $(CompilerFlags) -Wno-variadic-macros
endif
endif

ifneq ($VISLIB_ICC, 0)
	# ICC does not support "-pedantic"
	CompilerFlags := $(filter-out -pedantic, $(CompilerFlags))
endif

# Additional compiler flags for special configurations
DebugCompilerFlags := -DDEBUG -D_DEBUG -ggdb
ReleaseCompilerFlags := -DNDEBUG -D_NDEBUG -O3 -g0


# Common linker flags
LinkerFlags := -lX11 -lXext -lXxf86vm -lm 
ifneq ($(VISLIB_ICC), 0)
	# Katrin says this is required ...
	LinkerFlags += -lstdc++
endif

# Additional linker flags for special configurations
DebugLinkerFlags :=
ReleaseLinkerFlags :=
