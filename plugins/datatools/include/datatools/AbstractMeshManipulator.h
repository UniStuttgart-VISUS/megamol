/*
 * AbstractMeshManipulator.h
 *
 * Copyright (C) 2019 MegaMol Dev Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "AbstractManipulator.h"
#include "geometry_calls_gl/CallTriMeshDataGL.h"

namespace megamol {
namespace datatools {

using AbstractMeshManipulator = AbstractManipulator<geocalls_gl::CallTriMeshDataGL>;

} /* end namespace datatools */
} /* end namespace megamol */
