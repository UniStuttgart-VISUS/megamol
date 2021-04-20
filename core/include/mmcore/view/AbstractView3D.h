/*
 * AbstractView3D.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <map>
#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/BaseView.h"
#include "mmcore/view/cam_typedefs.h"
#include "mmcore/view/Camera.h"
#include "mmcore/view/CallRender3D.h"

#include "vislib/sys/KeyCode.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "mmcore/CoreInstance.h"

#include "glm/gtx/rotate_vector.hpp" // glm::rotate(quaternion, vector)
#include "glm/gtx/quaternion.hpp" // glm::rotate(quat, vector)

#include "GlobalValueStore.h"

namespace megamol {
namespace core {
namespace view {

template<typename FBO_TYPE>
using RESIZEFUNC = void(std::shared_ptr<FBO_TYPE>&, int, int);

template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
    typename CAM_PARAMS_TYPE>
class MEGAMOLCORE_API AbstractView3D
        : public view::BaseView<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE> {

public:

    /**
     * Callback requesting a rendering of this view
     *
     * @param call The calling call
     *
     * @return The return value
     */
    virtual bool OnRenderView(Call& call);

protected:
    /** Ctor. */
    AbstractView3D(void);

    /** Dtor. */
    virtual ~AbstractView3D(void);

    /**
     * Resets the view. This normally sets the camera parameters to
     * default values.
     *
     * @param window_aspect Aspect ratio of the full window. Used to set the camera frustrum aspect ratio.
     */
    void ResetView(float window_aspect);

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    bool cameraOvrCallback(param::ParamSlot& p);
};

#include "AbstractView3D.inl"

} // namespace view
} /* end namespace core */
} /* end namespace megamol */

