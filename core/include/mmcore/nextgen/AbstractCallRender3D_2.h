/*
 * AbstractCallRender3D_2.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTCallRender3D_2_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTCallRender3D_2_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/nextgen/Camera_2.h"
#include "mmcore/view/AbstractCallRender.h"
#include "vislib/SmartPtr.h"
#include "vislib/assert.h"
#include "vislib/types.h"

namespace megamol {
namespace core {
namespace nextgen {

/**
 * New and improved base class of rendering graph calls
 *
 * Function "Render" tells the callee to render itself into the currently
 * active opengl context (TODO: Later on it could also be a FBO).
 *
 * Function "GetExtents" asks the callee to fill the extents member of the
 * call (bounding boxes, temporal extents).
 *
 * Function "GetCapabilities" asks the callee to set the capabilities
 * flags of the call.
 */
class MEGAMOLCORE_API AbstractCallRender3D_2 : public view::AbstractCallRender {
public:
    virtual ~AbstractCallRender3D_2(void);

    /**
     * Accesses the bounding boxes of the output of the callee. This can
     * be called by the callee as answer to 'GetExtents'.
     *
     * @return The bounding boxes of the output of the callee.
     */
    inline BoundingBoxes_2& AccessBoundingBoxes(void) { return this->bboxs; }

    /**
     * Gets the bounding boxes of the output of the callee. This can
     * be called by the callee as answer to 'GetExtents'.
     *
     * @return The bounding boxes of the output of the callee.
     */
    inline const BoundingBoxes_2& GetBoundingBoxes(void) const { return this->bboxs; }

    /**
     * Gets the camera parameters .
     *
     * @return The camera parameters pointer.
     */
    inline const cam_type::minimal_state_type& GetCameraState(void) const { return this->minCamState; }

    /**
     * Returns the camera containing the parameters transferred by this call.
     * Things like the view matrix are not calculated yet and have still to be retrieved from the object
     * by using the appropriate functions. THIS METHOD PERFORMS A COPY OF A WHOLE CAMERA OBJECT.
     * TO AVOID THIS, USE GetCameraState() or GetCamera(Camera_2&) INSTEAD.
     *
     * @return A camera object containing the minimal state transferred by this call.
     */
    inline const Camera_2 GetCamera(void) const {
        Camera_2 retval = this->minCamState;
        return retval;
    }

    /**
     * Stores the transferred camera state in a given Camera_2 object to avoid the copy of whole camera objects.
     * This invalidates all present parameters in the given object. They have to be calculated again, using the
     * appropriate functions.
     *
     * @param cam The camera object the transferred state is stored in
     */
    inline void GetCamera(Camera_2& cam) const { cam = this->minCamState; }

    /**
     * Sets the camera state. This has to be set by the
     * caller before calling 'Render'.
     *
     * @param camera The camera the state is adapted from.
     */
    inline void SetCameraState(Camera_2& camera) { this->minCamState = camera.get_minimal_state(this->minCamState); }

    /**
     * Gets the number of milliseconds required to render the last frame.
     *
     * @return The time required to render the last frame
     */
    inline double LastFrameTime(void) const { return this->lastFrameTime; }

    /**
     * Sets the number of milliseconds required to render the last frame.
     *
     * @param time The time required to render the last frame
     */
    inline void SetLastFrameTime(double time) { this->lastFrameTime = time; }

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    AbstractCallRender3D_2& operator=(const AbstractCallRender3D_2& rhs);

protected:
    /** Ctor. */
    AbstractCallRender3D_2(void);

private:
#ifdef _WIN32
#    pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** The transferred camera state */
    cam_type::minimal_state_type minCamState;
#ifdef _WIN32
#    pragma warning(default : 4251)
#endif /* _WIN32 */

    /** The bounding boxes */
    BoundingBoxes_2 bboxs;

    /** The number of milliseconds required to render the last frame */
    double lastFrameTime;
};

} // namespace nextgen
} /* end namespace core */
} /* end namespace megamol */


#endif /* MEGAMOLCORE_ABSTRACTCallRender3D_2_H_INCLUDED */
