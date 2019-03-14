/*
 * AbstractDirParticleManipulator.h
 *
 * Copyright (C) 2019 MegaMol Dev Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/moldyn/DirectionalParticleDataCall.h"
#include "mmstd_datatools/AbstractManipulator.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

using AbstractDirParticleManipulator = AbstractManipulator<core::moldyn::DirectionalParticleDataCall>;

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */
