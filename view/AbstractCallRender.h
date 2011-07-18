/*
 * AbstractCallRender.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTCALLRENDER_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTCALLRENDER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Call.h"
#include "view/AbstractRenderOutput.h"
#include "vislib/Array.h"
#include "vislib/FramebufferObject.h"
#include "vislib/Rectangle.h"
#include "vislib/types.h"
#include <GL/gl.h>


namespace megamol {
namespace core {
namespace view {


    /**
     * Abstract base class of rendering graph calls
     *
     * Handles the output buffer control.
     */
    class MEGAMOLCORE_API AbstractCallRender : public Call, public virtual AbstractRenderOutput {
    public:

        /** Dtor. */
        virtual ~AbstractCallRender(void);

        /**
         * Gets the instance time code
         *
         * @return The instance time code
         */
        inline float InstanceTime(void) const {
            return this->instTime;
        }

        /**
         * Sets the instance time code
         *
         * @param time The time code of the frame to render
         */
        inline void SetInstanceTime(float time) {
            this->instTime = time;
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
        inline float Time(void) const {
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
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        AbstractCallRender& operator=(const AbstractCallRender& rhs);

    protected:

        /** Ctor. */
        AbstractCallRender(void);

    private:

        /** The number of time frames available to render */
        unsigned int cntTimeFrames;

        /** The time code requested to render */
        float time;

        /** The instance time code */
        float instTime;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTCALLRENDER_H_INCLUDED */
