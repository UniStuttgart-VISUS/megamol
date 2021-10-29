/*
 * AbstractDirParticleManipulator.h
 *
 * Copyright (C) 2019 MegaMol Dev Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "geometry_calls/EllipsoidalDataCall.h"
#include "mmstd_datatools/AbstractManipulator.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

using AbstractDirParticleManipulator = AbstractManipulator<geocalls::EllipsoidalParticleDataCall>;

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */
