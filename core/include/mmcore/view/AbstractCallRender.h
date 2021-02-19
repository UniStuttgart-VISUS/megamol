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

#include <glm/glm.hpp>
#include "mmcore/view/InputCall.h"
#include "vislib/Array.h"
#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/view/Camera_2.h"



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
         * Gets the instance time code
         *
         * @return The instance time code
         */
        inline double InstanceTime(void) const {
            return this->_instTime;
        }

        /**
         * Answer the flag for in situ timing.
         * If true 'TimeFramesCount' returns the number of the data frame
         * currently available from the in situ source.
         *
         * @return The flag for in situ timing
         */
        inline bool IsInSituTime(void) const {
            return this->_isInSituTime;
        }

        /**
         * Sets the instance time code
         *
         * @param time The time code of the frame to render
         */
        inline void SetInstanceTime(double time) {
            this->_instTime = time;
        }

        /**
         * Sets the flag for in situ timing.
         * If set to true 'TimeFramesCount' returns the number of the data
         * frame currently available from the in situ source.
         *
         * @param v The new value for the flag for in situ timing
         */
        inline void SetIsInSituTime(bool v) {
            this->_isInSituTime = v;
        }

        /**
         * Sets the time code of the frame to render.
         *
         * @param time The time code of the frame to render.
         */
        inline void SetTime(float time) {
            this->_time = time;
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
            this->_cntTimeFrames = cnt;
        }

        /**
         * Gets the time code of the frame requested to render.
         *
         * @return The time frame code of the frame to render.
         */
        inline float Time(void) const {
            return _time;
        }

        /**
         * Gets the number of time frames of the data the callee can render.
         *
         * @return The number of time frames of the data the callee can render.
         */
        inline unsigned int TimeFramesCount(void) const {
            return this->_cntTimeFrames;
        }

        /**
         * Gets the number of milliseconds required to render the last frame.
         *
         * @return The time required to render the last frame
         */
        inline double LastFrameTime(void) const {
            return this->_lastFrameTime;
        }

        /**
         * Sets the number of milliseconds required to render the last frame.
         *
         * @param time The time required to render the last frame
         */
        inline void SetLastFrameTime(double time) {
            this->_lastFrameTime = time;
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
         * Accesses the bounding boxes of the output of the callee. This can
         * be called by the callee as answer to 'GetExtents'.
         *
         * @return The bounding boxes of the output of the callee.
         */
        inline BoundingBoxes_2& AccessBoundingBoxes(void) {
            return this->_bboxs;
        }

        /**
         * Gets the bounding boxes of the output of the callee. This can
         * be called by the callee as answer to 'GetExtents'.
         *
         * @return The bounding boxes of the output of the callee.
         */
        inline const BoundingBoxes_2& GetBoundingBoxes(void) const {
            return this->_bboxs;
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
            return this->_camera;
        }

        /**
         * Stores the transferred camera state in a given Camera_2 object to avoid the copy of whole camera objects.
         * This invalidates all present parameters in the given object. They have to be calculated again, using the
         * appropriate functions.
         *
         * @param cam The camera object the transferred state is stored in
         */
        inline void GetCamera(Camera_2& cam) const {
            cam = this->_camera;
        }

        /**
         * Sets the camera state. This has to be set by the
         * caller before calling 'Render'.
         *
         * @param camera The camera the state is adapted from.
         */
        inline void SetCamera(Camera_2& camera) {
            this->_camera = camera;
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
        unsigned int _cntTimeFrames;

        /** The time code requested to render */
        float _time;

        /** The instance time code */
        double _instTime;

        /**
         * Flag marking that 'cntTimeFrames' store the number of the currently
         * available time frame when doing in situ visualization
         */
        bool _isInSituTime;

        /** The number of milliseconds required to render the last frame */
        double _lastFrameTime;

        glm::vec4 _backgroundCol;

        /** The transferred camera state */
        Camera_2 _camera;


        /** The bounding boxes */
        BoundingBoxes_2 _bboxs;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTCALLRENDER_H_INCLUDED */
