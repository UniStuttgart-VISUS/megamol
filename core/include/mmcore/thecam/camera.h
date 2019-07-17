/*
 * thecam/camera.h
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

#ifndef THE_GRAPHICS_CAMERA_CAMERA_H_INCLUDED
#define THE_GRAPHICS_CAMERA_CAMERA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#include "mmcore/thecam/camera_snapshot.h"
#include "mmcore/thecam/functions.h"
#include "mmcore/thecam/minimal_camera_state.h"
#include "mmcore/thecam/property.h"
#include "mmcore/thecam/synchronisable_property.h"
#include "mmcore/thecam/types.h"
#include "mmcore/thecam/view_frustum.h"

#include "mmcore/thecam/math/functions.h"
#include "mmcore/thecam/math/quaternion.h"
#include "mmcore/thecam/math/vector.h"


namespace megamol {
namespace core {
namespace thecam {

/**
 *
 * @tparam M The configuration of the mathematical types the camera uses.
 * @tparam P The property type the camera uses. This defaults to the
 *           non-synchronisable megamol::core::thecam::graphics::camera::parameter type.
 */
template <class M, template <class> class P = property> class camera {

public:
    /**
     * The type used for expressing ratios. This must never be an
     * integral type, but always a rational one.
     */
    typedef typename M::fractional_type fractional_type;

    /** The type of matrices used by the camera. */
    typedef typename M::matrix_type matrix_type;

    /** The type of the camera parameter template used by the camera. */
    template <class Pp> using property_type = P<Pp>;

    /** The type used for world-space positions. */
    typedef typename M::point_type point_type;

    /** The type of rectangles in screen space (pixels). */
    typedef typename M::screen_rectangle_type screen_rectangle_type;

    /** The type used for expressing screen-space dimensions. */
    typedef typename M::screen_size_type screen_size_type;

    /** The type used for expressing screen coordinate values. */
    typedef typename M::screen_type screen_type;

    /** The type of quaternions used by the camera. */
    typedef typename M::quaternion_type quaternion_type;

    /** The mathematical traits of the camera. */
    typedef M maths_type;

    /** The minimal camera state type used for this camera type. */
    typedef minimal_camera_state<M> minimal_state_type;

    /** The type of the camera snapshot that is produced by the camera. */
    typedef camera_snapshot<M> snapshot_type;

    /** The type of vectors used by the camera. */
    typedef typename M::vector_type vector_type;

    /** The type of a view frustum in world coordinates. */
    typedef view_frustum<typename M::world_type> view_frustum_type;

    /** The type used for expressing world-space dimensions. */
    typedef typename M::world_size_type world_size_type;

    /** The type for expressing world-space values. */
    typedef typename M::world_type world_type;

    /**
     * Computes a left-handed projection matrix from the data in the given
     * camera_snapshot. It is assumed that valid camera-space view frustum
     * is stored in 'snapshot'. No runtime checks are performed.
     *
     * @param outMat   The matrix object which receives the projection
     *                 matrix. The object will be completely reset.
     * @param snapshot The camera snapshot holding at least the camera-space
     *                 view frustum parameters.
     *
     * @return 'outMat'.
     */
    static matrix_type& projection_matrix_left_handed(matrix_type& outMat, const snapshot_type& snapshot);

    /**
     * Computes a right-handed projection matrix from the data in the given
     * camera_snapshot. It is assumed that valid camera-space view frustum
     * is stored in 'snapshot'. No runtime checks are performed.
     *
     * @param outMat   The matrix object which receives the projection
     *                 matrix. The object will be completely reset.
     * @param snapshot The camera snapshot holding at least the camera-space
     *                 view frustum parameters.
     *
     * @return 'outMat'.
     */
    static matrix_type& projection_matrix_right_handed(matrix_type& outMat, const snapshot_type& snapshot);

    /**
     * Computes a left-handed view matrix from the data in the given
     * camera_snapshot. It is assumed that a valid camera coordinate system
     * is stored in 'snapshot'. No runtime checks are performed.
     *
     * @param outMat   The matrix object which receives the view
     *                 matrix. The object will be completely reset.
     * @param snapshot The camera snapshot holding at least the camera
     *                 coordinate system.
     *
     * @return 'outMat'.
     */
    static matrix_type& view_matrix_left_handed(matrix_type& outMat, const snapshot_type& snapshot);

    /**
     * Computes a right-handed view matrix from the data in the given
     * camera_snapshot. It is assumed that a valid camera coordinate system
     * is sotred in 'snapshot'. No runtime checks are performed.
     *
     * @param outMat   The matrix object which receives the view
     *                 matrix. The object will be completely reset.
     * @param snapshot The camera snapshot holding at least the camera
     *                 coordinate system.
     *
     * @return 'outMat'.
     */
    static matrix_type& view_matrix_right_handed(matrix_type& outMat, const snapshot_type& snapshot);

    /**
     * Initialises a new instance.
     */
    inline camera(void) { this->reset(); }

    /**
     * Clone 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    inline camera(const camera& rhs) { *this = rhs; }

    /**
     * Gets the vertical aperture angle of the camera in degrees.
     *
     * @return The vertical aperture angle of the camera.
     */
    inline fractional_type aperture_angle(void) const { return math::angle_rad2deg(this->aperture_angle_radians()); }

    /**
     * Sets the vertical aperture angle of the camera in degrees.
     *
     * @param value The new vertical aperture angle.
     */
    inline void aperture_angle(const fractional_type value) {
        this->aperture_angle_radians(math::angle_deg2rad(value));
    }

    /**
     * Gets the vertical aperture angle of the camera in radians.
     *
     * @return The vertical aperture angle of the camera.
     */
    inline fractional_type aperture_angle_radians(void) const {
        return (static_cast<fractional_type>(2) * this->half_aperture_angle_radians());
    }

    /**
     * Sets the vertical aperture angle of the camera in radians.
     *
     * @param value The new vertical aperture angle.
     */
    inline void aperture_angle_radians(const fractional_type value) {
        this->half_aperture_angle_radians(static_cast<fractional_type>(0.5) * value);
    }

    /**
     * Convenience method for (i) taking a camera snapshot, (ii) computing
     * the view matrix from the snapshot and (iii) computing the projection
     * matrix from the snapshot.
     *
     * @param outView Receives the view matrix. The handedness of the matrix
     *                matches the handedness of the camera.
     * @param outProj Receives the projection matrix. The handedness of the
     *                matrix matches the handedness of the camera.
     */
    inline void calc_matrices(matrix_type& outView, matrix_type& outProj) const {
        snapshot_type snapshot;
        this->calc_matrices(snapshot, outView, outProj);
    }

    /**
     * Convenience method for (i) taking a camera snapshot, (ii) computing
     * the view matrix from the snapshot and (iii) computing the projection
     * matrix from the snapshot.
     *
     * This is the most efficient way of retrieving both matrices as it does
     * not need top re-compute stuff and returns the data directly into an
     * out-variable.
     *
     * @parma outSnapshot Receives the snapshot which was used to compute
     *                    'outView' and 'outProj'. The snapshot will contain
     *                    all possible content.
     * @param outView     Receives the view matrix. The handedness of the
     *                    matrix matches the handedness of the camera.
     * @param outProj     Receives the projection matrix. The handedness of
     *                    the matrix matches the handedness of the camera.
     * @param sc          The type of content contained in 'outSnapshot'. By
     *                    default, this only includes the dependencies for
     *                    the two matrics.
     */
    void calc_matrices(snapshot_type& outSnapshot, matrix_type& outView, matrix_type& outProj,
        snapshot_content sc = snapshot_content::camera_coordinate_system |
                              snapshot_content::camera_space_frustum) const;

    /**
     * Gets the disparity of the camera in world coordinates at the position
     * of the camera (film).
     */
    inline world_type disparity(void) const { return static_cast<fractional_type>(2) * this->half_disparity(); }

    /**
     * Sets the disparity of the camera in world coordinates at the position
     * of the camera (film).
     */
    inline void disparity(const world_type value) { this->half_disparity(static_cast<fractional_type>(0.5) * value); }

    /**
     * Computes the current position of the camera (honouring the current
     * stereo settings).
     *
     * This is a convenience method, which is rather expensive in computing
     * the result, because the method needs to compute the whole camera
     * coordinate system. Consider taking a camera snapshot and using the
     * results from the snapshot if you need the result more than once or
     * more than one derived camera property.
     *
     * @return The current position of the camera.
     */
    inline point_type eye_position(void) const {
        snapshot_type s;
        this->take_snapshot(s, snapshot_content::camera_coordinate_system);
        return s.position;
    }

    /**
     * Gets the aspect ratio of the film gate.
     *
     * @return (width of film gate) / (height of film gate).
     */
    inline fractional_type film_gate_aspect(void) const {
        const auto& g = this->film_gate();
        auto h = static_cast<fractional_type>(g.height());
        auto w = static_cast<fractional_type>(g.width());
        const auto z = static_cast<fractional_type>(0);
        return (h == z) ? z : (w / h);
    }

    // inline world_type film_height(void) const {
    //    return this->film_gate()[1];
    //}

    // inline void film_height(const world_type height) {
    //    this->film_gate()[1] = height;
    //}

    // inline world_type film_width(void) const {
    //    return this->film_gate()[0];
    //}

    // inline void film_width(const world_type width) {
    //    this->film_gate()[0] = width;
    //}

    /**
     * Gets the focal length of the camera in world coordinates.
     *
     * The focal length is computed from the width of the film gate and the
     * aperture angle. If no film gate is set, the resolution gate is used.
     * This will result in pixels being interpreted as being the units of the
     * world coordinate system and therefore result in unexpected values. It
     * recommended to use the focal length only if a film gate is set.
     *
     * @return The focal length of the camera.
     */
    world_type focal_length(void) const;

    /**
     * Fills a minimal_state_type structure with the data which are required
     * for restoring the full state of the camera.
     *
     * @param outState Receives the camera state.
     *
     * @return 'outState'.
     */
    minimal_state_type& get_minimal_state(minimal_state_type& outState);

    /**
     * Answer the handedness of the coordinate system the matrices are
     * computed for if not requested differently.
     *
     * @return The default coordinate system handedness.
     */
    inline megamol::core::thecam::Handedness handedness(void) const { return maths_type::handedness; }

    /**
     * Answer whether a non-empty film gate has been configured.
     *
     * @return true if the camera has a film gate, false otherwise.
     */
    inline bool has_film_gate(void) const { return !this->film_gate().empty(); }

    // inline screen_type image_height(void) const {
    //    return this->resolution_gate().height();
    //}

    // inline void image_height(const screen_type height) {
    //    this->resolution_gate().height() = height;
    //}

    // inline screen_type image_width(void) const {
    //    return this->resolution_gate().width();
    //}

    // inline void image_width(const screen_type width) {
    //    this->resolution_gate().width() = width;
    //}

    /**
     * Answer whether the camera shows a sub-region of the whole image,
     * which means that its image tile is not empty.
     *
     * @return true if the image tile is not empty, false otherwise.
     */
    inline bool is_tiled(void) const { return !this->image_tile().empty(); }


    /**
     * Moves the camera to the specified position and makes it look at the
     * specified point.
     *
     * @param position The new position of the camera.
     * @param lookAt   The point to look at.
     */
    inline void look_at(const point_type& position, const point_type& lookAt) {
        this->position(position);
        this->look_at(lookAt);
    }

    /**
     * Rotates the camera such that it points to the given point.
     *
     * @param lookAt The point to look at.
     */
    void look_at(const point_type& lookAt);

    /**
     * Convenience method for computing the projection matrix.
     *
     * It is recommended to take a camera snapshot with at least
     * snapshot_content::camera_space_frustum using the take_snapshot()
     * method and computing the matrices using the static methods of the
     * camera instead as this will reduce duplicate computations of more
     * than the matrix is required in a single frame.
     *
     * Alternatively, camera::calc_matrices() provides a way to take a
     * snapshot and compute all matrices at ones. Note that this method
     * will always produce a full snapshot of all derived camera properties.
     *
     * @return The current projection matrix as defined by the camera.
     */
    matrix_type projection_matrix(void) const;

    /**
     * Resets all camera parameters to their initial value.
     */
    void reset(void);

    /**
     * Computes the current right-vector of the camera coordinate system.
     *
     * This is a convenience method, which is rather expensive in computing
     * the result, because the method needs to compute the whole camera
     * coordinate system. Consider taking a camera snapshot and using the
     * results from the snapshot if you need the result more than once or
     * more than one derived camera property.
     *
     * @return The current right-vector of the camera.
     */
    inline vector_type right_vector(void) {
        snapshot_type s;
        this->take_snapshot(s, snapshot_content::camera_coordinate_system);
        return s.right_vector;
    }

    /**
     * Gets the aspect ratio of the resolution gate.
     *
     * @return (width of resolution gate) / (height of resolution gate).
     */
    inline fractional_type resolution_gate_aspect(void) const {
        const auto& g = this->resolution_gate();
        auto h = static_cast<fractional_type>(g.height());
        auto w = static_cast<fractional_type>(g.width());
        const auto z = static_cast<fractional_type>(0);
        return (h == z) ? z : (w / h);
    }

    /**
     * Fills a camera_snapshot with at least the requested information.
     *
     * @param snapshot The snapshot object that will be filled.
     * @param which    A bitmask of the fields to compute.
     *
     * @return 'snapshot'.
     */
    snapshot_type& take_snapshot(snapshot_type& snapshot, const snapshot_content which = snapshot_content::all) const;

    /**
     * Computes the current up-vector of the camera.
     *
     * This is a convenience method, which is rather expensive in computing
     * the result. Consider taking a camera snapshot and using the results
     * from the snapshot if you need the result more than once or more than
     * one derived camera property.
     *
     * @return The current up-vector of the camera.
     */
    inline vector_type up_vector(void) const {
        snapshot_type s;
        this->take_snapshot(s, snapshot_content::up_vector);
        return s.up_vector;
    }

    /**
     * Computes the current view-vector of the camera.
     *
     * This is a convenience method, which is rather expensive in computing
     * the result. Consider taking a camera snapshot and using the results
     * from the snapshot if you need the result more than once or more than
     * one derived camera property.
     *
     * @return The current view-vector of the camera.
     */
    inline vector_type view_vector(void) const {
        snapshot_type s;
        this->take_snapshot(s, snapshot_content::view_vector);
        return s.view_vector;
    }

    // TODO
    // view_frustum_type view_frustum(void) const;

    /**
     * Convenience method for computing the view matrix.
     *
     * It is recommended to take a camera snapshot with at least
     * snapshot_content::camera_coordinate_system using the take_snapshot()
     * method and computing the matrices using the static methods of the
     * camera instead as this will reduce duplicate computations of more
     * than the matrix is required in a single frame.
     *
     * Alternatively, camera::calc_matrices() provides a way to take a
     * snapshot and compute all matrices at ones. Note that this method
     * will always produce a full snapshot of all derived camera properties.
     *
     * @return The current view matrix as defined by the camera.
     */
    inline matrix_type view_matrix(void) const {
        matrix_type view, proj;
        // TODO only the calculation of the view matrix should be performed...
        this->calc_matrices(view, proj);
        return view;
    }

    // TODO: do we want to allow this?
    // camera& operator =(const camera& rhs);

    /**
     * Assign the camera's properties from a minimal state snapshot.
     *
     * @param rhs The minimal camera state to be applied.
     *
     * @return *this.
     */
    camera& operator=(const minimal_state_type& rhs);

    /**
     * Contains the relative offset of the camera position from the centre
     * of the image wrt the image size.
     *
     * Usually, this value is zero. It can be used for implementing user
     * tracking.
     */
    property_type<math::vector<fractional_type, 2>> centre_offset;
    // TODO: Add z-coordinate?

    /**
     * Gets or sets the distance of the plane of zero parallax from the near
     * clipping plane.
     *
     * This is the distance from the near clipping plane on the camera view
     * axis where zero parallax occurs, ie the point where objects appear on
     * the screen. For most cases you want your objects to always be behind
     * the zero parallax plane. That is they have a camera distance greater
     * than the near clipping plane value plus the zero parallax plane
     * value.
     *
     * The zero parallax value, the disparity and focal length are all used
     * to determine the shift that must be applied to film back on the
     * respective left and right camera. The zero parallax distance is only
     * applicable when the projection type is either
     * projection_type::off_axis or projection_type::toe_in/
     * projection_type::converged.
     */
    property_type<world_type> convergence_plane;

    /**
     * Gets or sets the stereo eye that the camera should compute the
     * transformation matrices for.
     */
    property_type<Eye> eye;

    /**
     * Gets or sets the distance of the far clipping plane in world
     * coordinates.
     */
    property_type<world_type> far_clipping_plane;

    /**
     * Gets or sets the size of the film in world coordinates.
     *
     * The film size will be used to apply gate_scaling and to derive the
     * aperture angle from the the focal length. If the film gate is empty,
     * it will be ignored.
     */
    property_type<world_size_type> film_gate;

    /**
     * Gets or sets how different aspect ratios of the film gate and the
     * resolution gate are handled.
     *
     * Note that no gate scaling is performed if no film gate is set for
     * the camera.
     */
    property_type<Gate_scaling> gate_scaling;

    /**
     * Gets or sets the half vertical aperture angle of the camera in
     * radians.
     */
    property_type<fractional_type> half_aperture_angle_radians;

    /**
     * Gets or sets half of the stereo disparity of the camera in world
     * coordinates at the position of the camera (film).
     */
    property_type<world_type> half_disparity;

    /**
     * Gets or sets a tile within the resolution gate which the camera
     * should render instead of the whole image.
     *
     * If the rectangle is empty, the tile will be ignored and the whole
     * resolution gate will be rendered.
     */
    property_type<screen_rectangle_type&> image_tile;

    /**
     * Gets or sets the distance of the near clipping plane in world
     * coordinates.
     */
    property_type<world_type> near_clipping_plane;

    /**
     * Gets or sets the orientation of the camera.
     */
    property_type<quaternion_type&> orientation;

    /**
     * Gets or sets the position of the (centre) camera in world
     * coordinates.
     *
     * Take a snapshot or use camera::eye_position() to evaluate the actual
     * position used to compute the matrices.
     */
    property_type<point_type&> position;

    /**
     * Gets or sets the type of projection that the camera computes matrices
     * for.
     */
    property_type<Projection_type> projection_type;

    /**
     * Gets or sets the size of the (total) image in pixels.
     */
    property_type<screen_size_type&> resolution_gate;
};

} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#include "mmcore/thecam/camera.inl"

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_GRAPHICS_CAMERA_CAMERA_H_INCLUDED */
