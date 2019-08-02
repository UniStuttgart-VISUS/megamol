/*
 * AbstractParticleManipulator.h
 *
 * Copyright (C) 2019 MegaMol Dev Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmstd_datatools/AbstractManipulator.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

using AbstractParticleManipulator = AbstractManipulator<core::moldyn::MultiParticleDataCall>;

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */
