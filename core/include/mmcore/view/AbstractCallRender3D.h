/*
 * AbstractCallRender3D.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTCALLRENDER3D_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTCALLRENDER3D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/BoundingBoxes.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/AbstractCallRender.h"
#include "vislib/assert.h"
#include "vislib/graphics/CameraParameters.h"
#include "vislib/SmartPtr.h"
#include "vislib/types.h"


namespace megamol {
namespace core {
namespace view {


#ifdef _WIN32
#pragma warning(disable: 4250)  // I know what I am doing ...
#endif /* _WIN32 */
    /**
     * Base class of rendering graph calls
     *
     * Function "Render" tells the callee to render itself into the currently
     * active opengl context (TODO: Late on it could also be a FBO).
     *
     * Function "GetExtents" asks the callee to fill the extents member of the
     * call (bounding boxes, temporal extents).
     */
    class MEGAMOLCORE_API AbstractCallRender3D : public AbstractCallRender {
    public:
        /** Dtor. */
        virtual ~AbstractCallRender3D(void);

        /**
         * Accesses the bounding boxes of the output of the callee. This can
         * be called by the callee as answer to 'GetExtents'.
         *
         * @return The bounding boxes of the output of the callee.
         */
        inline BoundingBoxes& AccessBoundingBoxes(void) {
            return this->bboxs;
        }

        /**
         * Gets the bounding boxes of the output of the callee. This can
         * be called by the callee as answer to 'GetExtents'.
         *
         * @return The bounding boxes of the output of the callee.
         */
        inline const BoundingBoxes& GetBoundingBoxes(void) const {
            return this->bboxs;
        }

        /**
         * Gets the camera parameters pointer.
         *
         * @return The camera parameters pointer.
         */
        inline const vislib::SmartPtr<vislib::graphics::CameraParameters>&
        GetCameraParameters(void) const {
            return this->camParams;
        }

        /**
         * Sets the camera parameters pointer. These are to be set by the
         * caller before calling 'Render'.
         *
         * @param camParams The new value for the camera parameters pointer.
         */
        inline void SetCameraParameters(const vislib::SmartPtr<
                vislib::graphics::CameraParameters>& camParams) {
            this->camParams = camParams;
        }

        /**
         * Gets the number of milliseconds required to render the last frame.
         *
         * @return The time required to render the last frame
         */
        inline double LastFrameTime(void) const {
            return this->lastFrameTime;
        }

        /**
         * Sets the number of milliseconds required to render the last frame.
         *
         * @param time The time required to render the last frame
         */
        inline void SetLastFrameTime(double time) {
            this->lastFrameTime = time;
        }

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        AbstractCallRender3D& operator=(const AbstractCallRender3D& rhs);

    protected:

        /** Ctor. */
        AbstractCallRender3D(void);

    private:

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /** The camera parameters */
        vislib::SmartPtr<vislib::graphics::CameraParameters> camParams;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

        /** The bounding boxes */
        BoundingBoxes bboxs;

        /** The number of milliseconds required to render the last frame */
        double lastFrameTime;

    };
#ifdef _WIN32
#pragma warning(default: 4250)
#endif /* _WIN32 */


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTCALLRENDER3D_H_INCLUDED */
