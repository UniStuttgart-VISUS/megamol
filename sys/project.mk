#
# project.mk
#
# Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

TargetName := vislibsys

IncludeDir += ../base/include ../math/include

ExcludeFromBuild += ./src/DynamicFunctionPointer.cpp ./src/PAMException.cpp ./src/LinuxDaemon.cpp