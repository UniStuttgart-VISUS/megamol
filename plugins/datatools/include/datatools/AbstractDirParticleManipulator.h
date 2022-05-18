/*
 * AbstractDirParticleManipulator.h
 *
 * Copyright (C) 2019 MegaMol Dev Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "datatools/AbstractManipulator.h"
#include "geometry_calls/EllipsoidalDataCall.h"

namespace megamol {
namespace datatools {

using AbstractDirParticleManipulator = AbstractManipulator<geocalls::EllipsoidalParticleDataCall>;

} /* end namespace datatools */
} /* end namespace megamol */
