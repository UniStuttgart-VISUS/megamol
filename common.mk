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


# Determine the plaform architecture
ifeq ($(shell uname -p), x86_64)
	PLATFORM := x64
	BITS := 64
else
	PLATFORM := x86
	BITS = 32
endif


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
ifeq ($(PLATFORM), x64)
	IntDir := Lin64
else
	IntDir := Lin32
endif	


# The configuration intermetidate directories
DebugANSIDir := Debug_ANSI
DebugUnicodeDir := Debug_UNICODE
ReleaseANSIDir := Release_ANSI
ReleaseUnicodeDir := Release_UNICODE


# Common compiler flags
CompilerFlags := -DUNIX -D_GNU_SOURCE -Wall -ansi -pedantic

# Additional compiler flags for special configurations
DebugANSILinkerFlags := -DDEBUG -D_DEBUG -ggdb
DebugUnicodeLinkerFlags := -DDEBUG -D_DEBUG -ggdb -DUNICODE -D_UNICODE
ReleaseANSILinkerFlags := -DNDEBUG -D_NDEBUG -O3 -g0
ReleaseUnicodeLinkerFlags := -DNDEBUG -D_NDEBUG -O3 -g0


# Common linker flags
LinkerFlags := -L/usr/X11R6/lib -lm

# Additional linker flags for special configurations
DebugANSILinkerFlags :=
DebugUnicodeLinkerFlags :=
ReleaseANSILinkerFlags :=
ReleaseUnicodeLinkerFlags :=
