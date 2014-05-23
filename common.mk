#
# common.mk
# MegaMol
#
# Copyright (C) 2008-2010 by VISUS (Universitaet Stuttgart).
# Alle Rechte vorbehalten.
#

# Applications
AR = ar
CPP = g++
LINK = g++
MAKE = make
SHELL = /bin/bash


# Set Verbosity
ifndef MEGAMOL_VERBOSE
	MEGAMOL_VERBOSE = 0
endif
ifeq ($(MEGAMOL_VERBOSE),1)
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


ifeq ($(TERM), xterm)
	COLORACTION = '\E[1;32;40m'
	COLORINFO =  '\E[0;32;40m'
	CLEARTERMCMD = tput sgr0
else
	COLORACTION =
	COLORINFO =
	CLEARTERMCMD = true
endif


# The default input directory
InputDir := .


# The default include directories:
IncludeDir := . ./datraw


# List of system include directories:
SystemIncludeDir := /usr/include/g++ /usr/include/g++/bits /usr/include/g++/ext


# Default source exclude patterns:
ExcludeFromBuild := 


# The intermediate directory
ifeq ($(PLATFORM), x64)
	IntDir := Lin64
else
	IntDir := Lin32
endif


# The configuration intermetidate directories
DebugDir := Debug
ReleaseDir := Release


# Common compiler flags
CompilerFlags := -DUNIX -D_GNU_SOURCE -DGLX_GLXEXT_LEGACY -D_LIN$(BITS) -DMEGAMOLCORE_EXPORTS -Wall -ansi -pedantic -fopenmp

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


# Additional compiler flags for special configurations
DebugCompilerFlags := -DDEBUG -D_DEBUG -O0 -ggdb
ReleaseCompilerFlags := -DNDEBUG -D_NDEBUG -O3 -g0


# Common linker flags
LinkerFlags := -lX11 -lXext -lgomp -fopenmp


# Additional linker flags for special configurations
DebugLinkerFlags := 
ReleaseLinkerFlags := 
