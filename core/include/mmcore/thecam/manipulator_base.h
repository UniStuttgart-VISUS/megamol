/*
 * thecam/manipulator_base.h
 *
 * Copyright (C) 2016 TheLib Team (http://www.thelib.org/license)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of TheLib, TheLib Team, nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THELIB TEAM AS IS AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THELIB TEAM BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef THE_GRAPHICS_CAMERA_MANIPULATOR_BASE_H_INCLUDED
#define THE_GRAPHICS_CAMERA_MANIPULATOR_BASE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#include "mmcore/thecam/camera.h"

#include "mmcore/thecam/math/quaternion.h"
#include "mmcore/thecam/math/vector.h"


namespace megamol {
namespace core {
namespace thecam {

/**
 * Base class for camera manipulators.
 *
 * The purpose of the base class is the management of the camera to be
 * manipulated as well as the activation state of the derived manipulators.
 */
template <class T> class manipulator_base {

public:
    /** The type of the camera to be manipulated by the manipulator. */
    typedef T camera_type;

    /** The mathematical traits of the camera. */
    typedef typename T::maths_type maths_type;

    /**
     * Answer the camera to manipulated.
     *
     * @return The camera the manipulator is working on. It is valid for
     *         the method to return nullptr.
     */
    inline camera_type* camera(void) { return this->cam; }

    /**
     * Answer the camera to manipulated.
     *
     * @return The camera the manipulator is working on. It is valid for
     *         the method to return nullptr.
     */
    inline const camera_type* camera(void) const { return this->cam; }

    /**
     * Enable or disable the manipulator.
     *
     * @param isEnabled If true, the manipulator will be enabled;
     *                  otherwise, it will be disabled.
     */
    inline void enable(const bool isEnabled = true) { this->isEnabled = isEnabled; }

    /**
     * Answer whether the manipulator is enabled.
     *
     * Note that the manipulator is implicitly disabled if no valid camera
     * is set, ie it is guaranteed that this method return false if the
     * camera is nullptr.
     *
     * Subclasses can use this property to track whether the mouse is
     * currently dragging.
     *
     * @return true if the manipulator is enabled; false otherwise.
     */
    inline bool enabled(void) const { return (this->isEnabled && (this->cam != nullptr)); }

    /**
     * Answer whether the manipulator is currently changing the camera.
     *
     * Most manipulators translate a continuous input like mouse motion into
     * a continuous change of the camera, but only while some conditions
     * like a mouse button being pressed are met. This property tracks
     * whether this condition is met.
     *
     * @return true if the manipulator is modifying the camera,
     *         false otherwie.
     */
    inline bool manipulating(void) const { return this->isManipulating; }

    /**
     * Resets the target of the manipulation, effectively disabling the
     * manipulator.
     */
    inline void reset_target(void) { this->cam = nullptr; }

    /**
     * Sets the target of the manipulation.
     *
     * Note that the caller is responsible for that 'cam' lives as long as
     * it is used by this manipulator.
     *
     * @param cam A reference to the camera to be manipulated.
     */
    inline void set_target(camera_type& cam) { this->cam = &cam; }

protected:
    /** Disallow instances except from derived classes. */
    inline manipulator_base(camera_type* cam = nullptr) : cam(cam), isEnabled(false), isManipulating(false) {}

    /**
     * Sets the manipulation flag to true.
     */
    inline void begin_manipulation(void) { this->isManipulating = true; }

    /**
     * Sets the manipulation flag to false.
     */
    inline void end_manipulation(void) { this->isManipulating = false; }

private:
    /** The camera to be manipulated. */
    camera_type* cam;

    /** Determines whether the manipulator is enabled or not. */
    bool isEnabled;

    /**
     * Remembers whether the manipulator is currently modifyingt the
     * camera.
     */
    bool isManipulating;
};

} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_GRAPHICS_CAMERA_MANIPULATOR_BASE_H_INCLUDED */
