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

#include "mmcore/Call.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/view/AbstractRenderOutput.h"
#include "mmcore/view/InputCall.h"
#include "vislib/Array.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/math/Rectangle.h"
#include "vislib/types.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Abstract base class of rendering graph calls
     *
     * Handles the output buffer control.
     */
    class MEGAMOLCORE_API AbstractCallRender : public InputCall, public virtual AbstractRenderOutput {
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

        /** Defines the type of a GPU handle specifying the GPU affinity. */
        typedef void *GpuHandleType;

        /** Constant value for specifying no GPU affinity is requested. */
        static const GpuHandleType NO_GPU_AFFINITY;

        /** Dtor. */
        virtual ~AbstractCallRender(void) = default;

        /**
         * Get the GPU affinity handle and convert it to its native type in
         * one step.
         *
         * This value is only meaningful, if IsGpuAffinity() is true.
         *
         * You must ensure that the handle type you request matches the GPU in
         * the system.
         *
         * @return The GPU affinity handle.
         */
        template<class T> T GpuAffinity(void) const {
            static_assert(sizeof(T) == sizeof(this->gpuAffinity), "The size of "
                "the GPU handle is unexpected. You are probably doing "
                "something very nasty.");
            return reinterpret_cast<T>(this->gpuAffinity);
        }

        /**
         * Gets the instance time code
         *
         * @return The instance time code
         */
        inline double InstanceTime(void) const {
            return this->instTime;
        }

        /**
         * Answer whether GPU affinity was requested for the rendering this view.
         *
         * @return true in case GPU affinity was requested, false otherwise.
         */
        inline bool IsGpuAffinity(void) const {
            return (this->gpuAffinity != NO_GPU_AFFINITY);
        }

        /**
         * Answer the flag for in situ timing.
         * If true 'TimeFramesCount' returns the number of the data frame
         * currently available from the in situ source.
         *
         * @return The flag for in situ timing
         */
        inline bool IsInSituTime(void) const {
            return this->isInSituTime;
        }

        /**
         * Sets the GPU that the renderer should use for the following frame.
         *
         * This parameter is set by the core and derived from the
         * mmcRenderViewContext. DO NOT USE THIS UNLESS YOU KNOW WHAT YOU ARE
         * DOING!
         *
         * @param gpuAffinity The handle for the GPU the renderer should use;
         *                    NO_GPU_AFFINITY in case affinity does not matter.
         */
        inline void SetGpuAffinity(const GpuHandleType gpuAffinity) {
            this->gpuAffinity = gpuAffinity;
        }

        /**
         * Sets the instance time code
         *
         * @param time The time code of the frame to render
         */
        inline void SetInstanceTime(double time) {
            this->instTime = time;
        }

        /**
         * Sets the flag for in situ timing.
         * If set to true 'TimeFramesCount' returns the number of the data
         * frame currently available from the in situ source.
         *
         * @param v The new value for the flag for in situ timing
         */
        inline void SetIsInSituTime(bool v) {
            this->isInSituTime = v;
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

        /** The number of time frames available to render */
        unsigned int cntTimeFrames;

        /** Some kind of GPU handle if GPU affinity is requested. */
        GpuHandleType gpuAffinity;

        /** The time code requested to render */
        float time;

        /** The instance time code */
        double instTime;

        /**
         * Flag marking that 'cntTimeFrames' store the number of the currently
         * available time frame when doing in situ visualization
         */
        bool isInSituTime;

        /** The number of milliseconds required to render the last frame */
        double lastFrameTime;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTCALLRENDER_H_INCLUDED */
