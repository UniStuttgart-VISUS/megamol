/*
 * AbstractDirParticleManipulator.h
 *
 * Copyright (C) 2019 MegaMol Dev Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "geometry_calls/EllipsoidalDataCall.h"
#include "datatools/AbstractManipulator.h"

namespace megamol {
namespace datatools {

using AbstractDirParticleManipulator = AbstractManipulator<geocalls::EllipsoidalParticleDataCall>;

} /* end namespace datatools */
} /* end namespace megamol */
