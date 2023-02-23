/*
 * AbstractParticleManipulator.h
 *
 * Copyright (C) 2019 MegaMol Dev Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "datatools/AbstractManipulator.h"
#include "geometry_calls/MultiParticleDataCall.h"

namespace megamol::datatools {

using AbstractParticleManipulator = AbstractManipulator<geocalls::MultiParticleDataCall>;

} // namespace megamol::datatools
