/*
 * AbstractMeshManipulator.h
 *
 * Copyright (C) 2019 MegaMol Dev Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "AbstractManipulator.h"
#include "geometry_calls/CallTriMeshData.h"

namespace megamol {
namespace datatools {

using AbstractMeshManipulator = AbstractManipulator<geocalls::CallTriMeshData>;

} /* end namespace datatools */
} /* end namespace megamol */
