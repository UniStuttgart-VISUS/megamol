/*
 * AbstractCallRender.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/view/AbstractRenderOutput.h"
#include "mmcore/view/InputCall.h"
#include "vislib/Array.h"

namespace megamol {
namespace core {
namespace view {


    /**
     * Abstract base class of rendering graph calls
     *
     * Handles the output buffer control.
     */
    class MEGAMOLCORE_API AbstractCallRender : public InputCall {
    public:
        static const unsigned int FnRender = 5;
        static const unsigned int FnGetExtents = 6;

		/**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            ASSERT(FnRender == InputCall::FunctionCount() && "Enum has bad magic number");
            ASSERT(FnGetExtents == InputCall::FunctionCount() + 1 && "Enum has bad magic number");
            return InputCall::FunctionCount() + 2;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
			#define CaseFunction(id) case Fn##id: return #id
            switch (idx) {
                CaseFunction(Render);
                CaseFunction(GetExtents);
                default: return InputCall::FunctionName(idx);
            }
			#undef CaseFunction
        }

        /** Dtor. */
        virtual ~AbstractCallRender(void) = default;

        /**
         * Sets the instance time code
         *
         * @param time The time code of the frame to render
         */
        inline void SetInstanceTime(double time) {
            this->instTime = time;
        }

         /**
         * Gets the instance time code
         *
         * @return The time frame code of instance time code
         */
        inline float InstanceTime(void) const {
            return instTime;
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
        AbstractCallRender& operator=(const AbstractCallRender& rhs);

    protected:

        /** Ctor. */
        AbstractCallRender(void);

    private:

        unsigned int cntTimeFrames;
        float time;
        double instTime;
        double lastFrameTime;

    };

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */
