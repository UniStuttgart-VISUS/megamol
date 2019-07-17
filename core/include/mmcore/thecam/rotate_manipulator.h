#ifndef THE_GRAPHICS_CAMERA_ROTATE_MANIPULATOR_H_INCLUDED
#define THE_GRAPHICS_CAMERA_ROTATE_MANIPULATOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/manipulator_base.h"
#include "mmcore/thecam/math/functions.h"
#include "mmcore/thecam/utility/config.h"

namespace megamol {
namespace core {
namespace thecam {

/**
 * A manipulator for rotating the camera around its own position
 *
 * @tparam T The type of the camera to be manipulated.
 */
template <class T> class rotate_manipulator : public manipulator_base<T> {
public:
    /** The type of the camera to be manipulated by the manipulator. */
    typedef typename manipulator_base<T>::camera_type camera_type;

    /** The mathematical traits of the camera. */
    typedef typename manipulator_base<T>::maths_type maths_type;

    // Typedef all mathematical types we need in the manipulator.
    typedef typename maths_type::point_type point_type;
    typedef typename maths_type::quaternion_type quaternion_type;
    typedef typename maths_type::screen_type screen_type;
    typedef typename maths_type::vector_type vector_type;
    typedef typename maths_type::world_type world_type;

    /**
     * Constructor using a specific angle
     *
     * @param angle The angle to rotate in degrees. The default value is 1.
     */
    rotate_manipulator(const world_type angle = 1);

    /**
     * Destructor
     */
    virtual ~rotate_manipulator(void);

    /**
     * Rotates the camera around the right vector
     *
     * @param angle The angle to rotate in degrees.
     */
    void pitch(const world_type angle);

    /**
     * Rotates the camera around the right vector using the internally stored rotation angle.
     */
    inline void pitch(void) { this->pitch(this->rotationAngle); }

    /**
     * Rotates the camera around the up vector.
     *
     * @param angle The angle to rotate in degrees.
     */
    void yaw(const world_type angle);

    /**
     * Rotates the camera around the up vector using the internally stored rotation angle.
     */
    inline void yaw(void) { this->yaw(this->rotationAngle); }

    /**
     * Rotates the camera around the view vector.
     *
     * @param angle The angle to rotate in degrees.
     */
    void roll(const world_type angle);

    /**
     * Rotates the camera around the view vector using the internally stored rotation angle.
     */
    inline void roll(void) { this->roll(this->rotationAngle); }

    /**
     * Sets the default rotation angle.
     *
     * @param angle The new default angle in degrees.
     */
    inline void set_rotation_angle(const world_type angle) { this->rotationAngle = angle; }

    /**
     * Returns the current default rotation angle in degrees.
     *
     * @return The default rotation angle in degrees.
     */
    inline world_type rotation_angle(void) const { return this->rotationAngle; }

private:
    /** The angle that is used for rotation */
    world_type rotationAngle;
};

} // namespace thecam
} // namespace core
} // namespace megamol

#include "mmcore/thecam/rotate_manipulator.inl"

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif
