/*
 * CameraSerializer.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/nextgen/CameraSerializer.h"

#include "json.hpp"

using namespace megamol::core;
using namespace megamol::core::nextgen;

/*
 * CameraSerializer::CameraSerializer
 */
CameraSerializer::CameraSerializer(bool prettyMode) : prettyMode(prettyMode) {
    // intentionally empty
}

/*
 * CameraSerializer::~CameraSerializer
 */
CameraSerializer::~CameraSerializer(void) {
    // intentionally empty
}
