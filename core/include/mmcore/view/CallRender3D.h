/*
 * CallRender3D.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <vector>
#include <glm/glm.hpp>

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/InputCall.h"
#include "mmcore/view/MouseFlags.h"
#include "mmcore/view/Camera_2.h"
#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/view/AbstractCallRender.h"

namespace megamol {
namespace core {
namespace view {

struct CPUFramebuffer {
    bool depthBufferActive = false;
    std::vector<uint32_t> colorBuffer;
    std::vector<float> depthBuffer;
    unsigned int width = 0;
    unsigned int height = 0;
    int x = 0;
    int y = 0;
};


/**
 * Class of CPU context rendering
 *
 * Function "Render" tells the callee to render in a CPU context
 *
 * Function "GetExtents" asks the callee to fill the extents member of the
 * call (bounding boxes, temporal extents).
 */
class MEGAMOLCORE_API CallRender3D : public AbstractCallRender {
public:

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "CallRender3D"; }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) { return "CPU Rendering call"; }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return AbstractCallRender::FunctionCount();
    }
    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        return AbstractCallRender::FunctionName(idx);
    }
    /** Ctor. */
    CallRender3D(void);

    /** Dtor. */
    virtual ~CallRender3D(void);

    /**
     * Accesses the bounding boxes of the output of the callee. This can
     * be called by the callee as answer to 'GetExtents'.
     *
     * @return The bounding boxes of the output of the callee.
     */
    inline BoundingBoxes_2& AccessBoundingBoxes(void) {
        return _bboxs;
    }

    /**
     * Gets the bounding boxes of the output of the callee. This can
     * be called by the callee as answer to 'GetExtents'.
     *
     * @return The bounding boxes of the output of the callee.
     */
    inline const BoundingBoxes_2& GetBoundingBoxes(void) const {
        return _bboxs;
    }

    /**
     * Gets the camera parameters .
     *
     * @return The camera parameters pointer.
     */
    inline const cam_type::minimal_state_type& GetCameraState(void) const {
        return _minCamState;
    }

    /**
     * Returns the camera containing the parameters transferred by this call.
     * Things like the view matrix are not calculated yet and have still to be retrieved from the object
     * by using the appropriate functions. THIS METHOD PERFORMS A COPY OF A WHOLE CAMERA OBJECT.
     * TO AVOID THIS, USE GetCameraState() or GetCamera(Camera_2&) INSTEAD.
     *
     * @return A camera object containing the minimal state transferred by this call.
     */
    inline const Camera_2 GetCamera(void) const {
        Camera_2 retval = _minCamState;
        return retval;
    }

    /**
     * Stores the transferred camera state in a given Camera_2 object to avoid the copy of whole camera objects.
     * This invalidates all present parameters in the given object. They have to be calculated again, using the
     * appropriate functions.
     *
     * @param cam The camera object the transferred state is stored in
     */
    inline void GetCamera(Camera_2& cam) const {
        cam = _minCamState;
    }

    /**
     * Sets the camera state. This has to be set by the
     * caller before calling 'Render'.
     *
     * @param camera The camera the state is adapted from.
     */
    inline void SetCameraState(Camera_2& camera) {
        _minCamState = camera.get_minimal_state(_minCamState);
    }

     /**
     * Sets the background color
     *
     * @param backCol The new background color
     */
    inline void SetBackgroundColor(glm::vec4 backCol) {
        _backgroundCol = backCol;
    }

    /**
     * Gets the background color
     *
     * @return The stored background color
     */
    inline glm::vec4 BackgroundColor(void) const {
        return _backgroundCol;
    }


    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    CallRender3D& operator=(const CallRender3D& rhs);

     /**
     * Sets the Framebuffer
     *
     * @param fb The framebuffer
     */
    inline void SetFramebuffer(std::shared_ptr<CPUFramebuffer> fb) {
        _framebuffer = fb;
    }

     /**
     * Gets the Framebuffer
     *
     */
    inline std::shared_ptr<CPUFramebuffer> GetFramebuffer() {
        return _framebuffer;
    }

private:

    std::shared_ptr<CPUFramebuffer> _framebuffer;

    cam_type::minimal_state_type _minCamState;
    BoundingBoxes_2 _bboxs;
    glm::vec4 _backgroundCol;
};

/** Description class typedef */
typedef factories::CallAutoDescription<CallRender3D> CallRender3DDescription;

} // namespace view
} /* end namespace core */
} /* end namespace megamol */
