/*
 * Camera_2.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_CAMERA_2_H_INCLUDED
#define MEGAMOLCORE_CAMERA_2_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/thecam/arcball_manipulator.h"
#include "mmcore/thecam/camera.h"
#include "mmcore/thecam/camera_maths.h"
#include "mmcore/thecam/turntable_manipulator.h"
#include "mmcore/thecam/orbit_altitude_manipulator.h"
#include "mmcore/thecam/rotate_manipulator.h"
#include "mmcore/thecam/translate_manipulator.h"

#include "mmcore/BoundingBoxes_2.h"

typedef megamol::core::thecam::glm_camera_maths<> cam_maths_type;
typedef megamol::core::thecam::camera<cam_maths_type> cam_type;
typedef megamol::core::thecam::arcball_manipulator<cam_type> arcball_type;
typedef megamol::core::thecam::translate_manipulator<cam_type> xlate_type;
typedef megamol::core::thecam::rotate_manipulator<cam_type> rotate_type;
typedef megamol::core::thecam::TurntableManipulator<cam_type> turntable_type;
typedef megamol::core::thecam::OrbitAltitudeManipulator<cam_type> orbit_altitude_type;

namespace megamol {
namespace core {
namespace view {
/*
 * Wrapper for the template-heavy camera class
 */
class MEGAMOLCORE_API Camera_2 : public cam_type {
public:
    /**
     * Constructor
     */
    Camera_2(void);

    /**
     * Constructor using a minimal state to construct the camera
     *
     * @param other The state that is used to construct the camera
     */
    Camera_2(const cam_type::minimal_state_type& other);

    /**
     * Destructor
     */
    virtual ~Camera_2(void);

    /**
     * Assign the camera's properties from a minimal state snapshot.
     *
     * @param rhs The minimal camera state to be applied.
     *
     * @return *this
     */
    Camera_2& operator=(const cam_type::minimal_state_type& rhs);

    /**
     * Calculates the clipping distances based on a bounding box cuboid
     * specified in world coordinates. (This methods implementation uses
     * 'SetClip', 'Position', and 'Front'.)
     *
     * @param bbox The bounding box in world coordinates.
     * @param border Additional distance of the clipping distances to the
     *               bounding box.
     */
    virtual void CalcClipping(const vislib::math::Cuboid<float>& bbox, float border);
};
} // namespace view
} // namespace core
} // namespace megamol

#endif /* MEGAMOLCORE_CAMERA_2_H_INCLUDED */
