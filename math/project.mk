#
# project.mk
#
# Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

TargetName := vislibmath

IncludeDir += ../base/include

ExcludeFromBuild += $(InputDir)/Cuboid.cpp $(InputDir)/Dimension.cpp $(InputDir)/Dimension2D.cpp $(InputDir)/Dimension3D.cpp $(InputDir)/Matrix4x4.cpp $(InputDir)/Quaternion.cpp $(InputDir)/Rectangle.cpp $(InputDir)/Plane.cpp
