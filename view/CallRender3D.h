/*
 * CallRender3D.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLRENDER3D_H_INCLUDED
#define MEGAMOLCORE_CALLRENDER3D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include "BoundingBoxes.h"
#include "CallAutoDescription.h"
#include "view/AbstractCallRender.h"
#include "vislib/assert.h"
#include "vislib/CameraParameters.h"
#include "vislib/SmartPtr.h"
#include "vislib/types.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Base class of rendering graph calls
     *
     * Function "Render" tells the callee to render itself into the currently
     * active opengl context (TODO: Late on it could also be a FBO).
     *
     * Function "GetExtents" asks the callee to fill the extents member of the
     * call (bounding boxes, temporal extents).
     *
     * Function "GetCapabilities" asks the callee to set the capabilities
     * flags of the call.
     */
    class MEGAMOLCORE_API CallRender3D : public AbstractCallRender {
    public:

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "CallRender3D";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call for rendering a frame";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return 3;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            switch (idx) {
                case 0: return "Render";
                case 1: return "GetExtents";
                case 2: return "GetCapabilities";
                default: return NULL;
            }
        }

        /** Capability of rendering (must be present) */
        static const UINT64 CAP_RENDER;

        /** Capability to light the scene using the view-light */
        static const UINT64 CAP_LIGHTING;

        /** Capability to represent dynamic (time-varying) data */
        static const UINT64 CAP_ANIMATION;

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
        inline BoundingBoxes& AccessBoundingBoxes(void) {
            return this->bboxs;
        }

        /**
         * Sets the capability flags specified. This is to be set by the callee
         * as answer to 'GetCapabilities'.
         *
         * @param cap The capability flags to be set. Can by any '|' (or)
         *            combination of the 'CAP_*' members.
         */
        inline void AddCapability(UINT64 cap) {
            this->capabilities |= cap;
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
         * Gets the capabilitie flags.
         *
         * @return The capabilitie flags.
         */
        inline UINT64 GetCapabilities(void) const {
            return this->capabilities;
        }

        /**
         * Answers whether the specified capability flags are set.
         *
         * @param cap The requested capability flags.  Can by any '|' (or)
         *            combination of the 'CAP_*' members.
         *
         * @return 'true' if all requested flags are set, 'false' otherwise.
         */
        inline bool IsCapable(UINT64 cap) const {
            return (this->capabilities & cap) == cap;
        }

        /**
         * Unsets the capability flags specified. This is to be set by the
         * callee as answer to 'GetCapabilities'.
         *
         * @param cap The capability flags to be set. Can by any '|' (or)
         *            combination of the 'CAP_*' members.
         */
        inline void RemoveCapabilities(UINT64 cap) {
            this->capabilities &= ~cap;
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
         * Sets the capabilities of the callee. This is to be set by the callee
         * as answer to 'GetCapabilities'.
         *
         * @param cap The new capabilities value. Can by any '|' (or)
         *            combination of the 'CAP_*' members.
         */
        inline void SetCapabilities(UINT64 cap) {
            this->capabilities = cap;
        }

        /**
         * Sets the time code of the frame to render.
         *
         * @param time The time code of the frame to render.
         */
        inline void SetTime(float time) {
            this->time = time;
        }

        /**
         * Sets the number of time frames of the data the callee can render.
         * This is to be set by the callee as answer to 'GetExtents'.
         *
         * @param The number of time frames of the data the callee can render.
         *        Must not be zero.
         */
        inline void SetTimeFramesCount(unsigned int cnt) {
            ASSERT(cnt > 0);
            this->cntTimeFrames = cnt;
        }

        /**
         * Gets the time code of the frame requested to render.
         *
         * @return The time frame code of the frame to render.
         */
        inline float Time(void) {
            return time;
        }

        /**
         * Gets the number of time frames of the data the callee can render.
         *
         * @return The number of time frames of the data the callee can render.
         */
        inline unsigned int TimeFramesCount(void) const {
            return this->cntTimeFrames;
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
        CallRender3D& operator=(const CallRender3D& rhs);

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

        /** The number of time frames available to render */
        unsigned int cntTimeFrames;

        /** The capabilities flags */
        UINT64 capabilities;

        /** The time code requested to render */
        float time;

        /** The number of milliseconds required to render the last frame */
        double lastFrameTime;

    };


    /** Description class typedef */
    typedef CallAutoDescription<CallRender3D> CallRender3DDescription;


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLRENDER3D_H_INCLUDED */
