/*
 * cam_typedes.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CAM_TYPEDEFS_H_INCLUDED
#define MEGAMOLCORE_CAM_TYPEDEFS_H_INCLUDED

#include "mmcore/thecam/arcball_manipulator.h"
#include "mmcore/thecam/orbit_altitude_manipulator.h"
#include "mmcore/thecam/rotate_manipulator.h"
#include "mmcore/thecam/translate_manipulator.h"
#include "mmcore/thecam/turntable_manipulator.h"
#include "mmcore/view/Camera.h"

typedef megamol::core::view::Camera cam_type;
typedef megamol::core::thecam::arcball_manipulator<cam_type> arcball_type;
typedef megamol::core::thecam::translate_manipulator<cam_type> xlate_type;
typedef megamol::core::thecam::rotate_manipulator<cam_type> rotate_type;
typedef megamol::core::thecam::TurntableManipulator<cam_type> turntable_type;
typedef megamol::core::thecam::OrbitAltitudeManipulator<cam_type> orbit_altitude_type;

#endif // !MEGAMOLCORE_CAM_TYPEDEFS_H_INCLUDED
