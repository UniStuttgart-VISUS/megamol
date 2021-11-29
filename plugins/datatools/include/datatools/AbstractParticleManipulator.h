/*
 * AbstractParticleManipulator.h
 *
 * Copyright (C) 2019 MegaMol Dev Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "datatools/AbstractManipulator.h"
#include "geometry_calls/MultiParticleDataCall.h"

namespace megamol {
namespace datatools {

using AbstractParticleManipulator = AbstractManipulator<geocalls::MultiParticleDataCall>;

} /* end namespace datatools */
} /* end namespace megamol */
