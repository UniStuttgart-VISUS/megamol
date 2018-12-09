/*
 * Camera_2.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/thecam/camera.h"
#include "mmcore/thecam/camera_maths.h"
#include "mmcore/thecam/arcball_manipulator.h"
#include "mmcore/thecam/translate_manipulator.h"

typedef megamol::core::thecam::glm_camera_maths<> cam_maths_type;
typedef megamol::core::thecam::camera<cam_maths_type> cam_type;
typedef megamol::core::thecam::arcball_manipulator<cam_type> arcball_type;
typedef megamol::core::thecam::translate_manipulator<cam_type> xlate_type;

namespace megamol {
namespace core {
namespace nextgen {
/*
 * Wrapper for the template-heavy camera class
 */
class MEGAMOLCORE_API Camera_2 {
public:

    /**
     * Constructor
     */
    Camera_2(void);

    /**
     * Destructor
     */
    virtual ~Camera_2(void);

private:
    
    
};
} // namespace nextgen
} // namespace core
} // namespace megamol
