#ifndef THE_GRAPHICS_CAMERA_ROTATE_MANIPULATOR_H_INCLUDED
#define THE_GRAPHICS_CAMERA_ROTATE_MANIPULATOR_H_INCLUDED
#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include <glm/gtx/rotate_vector.hpp>

#include "mmcore/thecam/manipulator_base.h"
#include "mmcore/view/Camera.h"

namespace megamol::core::thecam {

/**
 * A manipulator for rotating the camera around its own position
 *
 * @tparam T The type of the camera to be manipulated.
 */
template<class T>
class rotate_manipulator : public manipulator_base<T> {
public:
    /** The type of the camera to be manipulated by the manipulator. */
    typedef typename manipulator_base<T>::camera_type camera_type;

    // Typedef all mathematical types we need in the manipulator.
    typedef typename glm::vec4 point_type;
    typedef typename glm::quat quaternion_type;
    typedef int screen_type;
    typedef typename glm::vec4 vector_type;
    typedef float world_type;

    /**
     * Constructor using a specific angle
     *
     * @param angle The angle to rotate in degrees. The default value is 1.
     */
    rotate_manipulator(const world_type angle = 1);

    /**
     * Destructor
     */
    virtual ~rotate_manipulator();

    /**
     * Rotates the camera around the right vector
     *
     * @param angle The angle to rotate in degrees.
     */
    void pitch(const world_type angle);

    /**
     * Rotates the camera around the right vector using the internally stored rotation angle.
     */
    inline void pitch() {
        this->pitch(this->rotationAngle);
    }

    /**
     * Rotates the camera around the up vector.
     *
     * @param angle The angle to rotate in degrees.
     */
    void yaw(const world_type angle, bool fixToWorldUp);

    /**
     * Rotates the camera around the up vector using the internally stored rotation angle.
     */
    inline void yaw(bool fixToWorldUp) {
        this->yaw(this->rotationAngle, fixToWorldUp);
    }

    /**
     * Rotates the camera around the view vector.
     *
     * @param angle The angle to rotate in degrees.
     */
    void roll(const world_type angle);

    /**
     * Rotates the camera around the view vector using the internally stored rotation angle.
     */
    inline void roll() {
        this->roll(this->rotationAngle);
    }

    /**
     * Sets the default rotation angle.
     *
     * @param angle The new default angle in degrees.
     */
    inline void set_rotation_angle(const world_type angle) {
        this->rotationAngle = angle;
    }

    /**
     * Returns the current default rotation angle in degrees.
     *
     * @return The default rotation angle in degrees.
     */
    inline world_type rotation_angle() const {
        return this->rotationAngle;
    }

    void setActive() {
        if (!this->manipulating() && this->enabled()) {
            this->begin_manipulation();
        }
    }

    /**
     * Set manipulator to inactive (usually on mouse button release).
     */
    inline void setInactive() {
        this->end_manipulation();
    }

private:
    /** The angle that is used for rotation */
    world_type rotationAngle;
};

} // namespace megamol::core::thecam

#include "mmcore/thecam/rotate_manipulator.inl"

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif
