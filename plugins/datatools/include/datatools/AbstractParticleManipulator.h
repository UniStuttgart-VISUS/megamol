/*
 * AbstractParticleManipulator.h
 *
 * Copyright (C) 2019 MegaMol Dev Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "geometry_calls/MultiParticleDataCall.h"
#include "datatools/AbstractManipulator.h"

namespace megamol {
namespace datatools {

using AbstractParticleManipulator = AbstractManipulator<geocalls::MultiParticleDataCall>;

} /* end namespace datatools */
} /* end namespace megamol */
