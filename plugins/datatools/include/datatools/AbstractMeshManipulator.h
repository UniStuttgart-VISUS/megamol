/*
 * AbstractMeshManipulator.h
 *
 * Copyright (C) 2019 MegaMol Dev Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "AbstractManipulator.h"
#include "geometry_calls_gl/CallTriMeshDataGL.h"

namespace megamol::datatools {

using AbstractMeshManipulator = AbstractManipulator<geocalls_gl::CallTriMeshDataGL>;

} // namespace megamol::datatools
