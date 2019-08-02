/*
 * thecam/camera.inl
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


/*
 * megamol::core::thecam::camera<M, P>::projection_matrix_left_handed
 */
template <class M, template <class> class P>
typename megamol::core::thecam::camera<M, P>::matrix_type&
megamol::core::thecam::camera<M, P>::projection_matrix_left_handed(matrix_type& outMat, const snapshot_type& snapshot) {
    THE_ASSERT(snapshot.contains(snapshot_content::camera_space_frustum));
    THE_ASSERT(snapshot.frustum_near != snapshot.frustum_far);

    const auto ZERO = static_cast<world_type>(0);
    const auto ONE = static_cast<world_type>(1);
    const auto TWO = static_cast<world_type>(2);
    auto l = snapshot.frustum_left;
    auto r = snapshot.frustum_right;
    auto t = snapshot.frustum_top;
    auto b = snapshot.frustum_bottom;
    auto n = snapshot.frustum_near;
    auto f = snapshot.frustum_far;

    // https://msdn.microsoft.com/en-us/library/windows/desktop/bb205353(v=vs.85).aspx
    outMat(0, 0) = (TWO * n) / (r - l);
    outMat(0, 1) = ZERO;
    outMat(0, 2) = ZERO;
    outMat(0, 3) = ZERO;

    outMat(1, 0) = ZERO;
    outMat(1, 1) = (TWO * n) / (t - b);
    outMat(1, 2) = ZERO;
    outMat(1, 3) = ZERO;

    outMat(2, 0) = (l + r) / (r - l); // TODO l - r????
    outMat(2, 1) = (t + b) / (b - t);
    outMat(2, 2) = f / (f - n);
    outMat(2, 3) = ONE;

    outMat(3, 0) = ZERO;
    outMat(3, 1) = ZERO;
    outMat(3, 2) = (n * f) / (n - f);
    outMat(3, 3) = ZERO;

    return outMat;
}


/*
 * megamol::core::thecam::camera<M, P>::projection_matrix_right_handed
 */
template <class M, template <class> class P>
typename megamol::core::thecam::camera<M, P>::matrix_type&
megamol::core::thecam::camera<M, P>::projection_matrix_right_handed(
    matrix_type& outMat, const snapshot_type& snapshot) {
    THE_ASSERT(snapshot.contains(snapshot_content::camera_space_frustum));
    THE_ASSERT(snapshot.frustum_near != snapshot.frustum_far);
    static const auto ZERO = static_cast<world_type>(0);
    static const auto ONE = static_cast<world_type>(1);
    static const auto TWO = static_cast<world_type>(2);

    auto l = snapshot.frustum_left;
    auto r = snapshot.frustum_right;
    auto t = snapshot.frustum_top;
    auto b = snapshot.frustum_bottom;
    auto n = snapshot.frustum_near;
    auto f = snapshot.frustum_far;

    // https://msdn.microsoft.com/en-us/library/windows/desktop/bb205354(v=vs.85).aspx
    // https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix
    // http://seanmiddleditch.com/matrices-handedness-pre-and-post-multiplication-row-vs-column-major-and-notations/
    outMat(0, 0) = (TWO * n) / (r - l);
    outMat(0, 1) = ZERO;
    outMat(0, 2) = ZERO;
    outMat(0, 3) = ZERO;

    outMat(1, 0) = ZERO;
    outMat(1, 1) = (TWO * n) / (t - b);
    outMat(1, 2) = ZERO;
    outMat(1, 3) = ZERO;

    outMat(2, 0) = (l + r) / (r - l);
    outMat(2, 1) = (t + b) / (t - b);
    outMat(2, 2) = f / (n - f); // -((f + n) / (f - n)) // (-f - n) / (-n - f)
    outMat(2, 3) = -ONE;

    outMat(3, 0) = ZERO;
    outMat(3, 1) = ZERO;
    outMat(3, 2) = (n * f) / (n - f); // -((2 * f * n) / (f - n)
    outMat(3, 3) = ZERO;

    return outMat;
}


/*
 * megamol::core::thecam::camera<M, P>::view_matrix_left_handed
 */
template <class M, template <class> class P>
typename megamol::core::thecam::camera<M, P>::matrix_type& megamol::core::thecam::camera<M, P>::view_matrix_left_handed(
    matrix_type& outMat, const snapshot_type& snapshot) {
    static const auto ZERO = static_cast<world_type>(0);
    static const auto ONE = static_cast<world_type>(1);

    auto& xAxis = snapshot.right_vector;
    auto& yAxis = snapshot.up_vector;
    auto zAxis = snapshot.view_vector;

    // TODO: Does it make sense not raising an error directly?
    if (maths_type::handedness != Handedness::left_handed) {
        zAxis *= -ONE;
    }

    // https://msdn.microsoft.com/de-de/library/windows/desktop/bb205342(v=vs.85).aspx
    outMat(0, 0) = xAxis.x();
    outMat(1, 0) = xAxis.y();
    outMat(2, 0) = xAxis.z();
    outMat(3, 0) = -math::dot(xAxis, snapshot.position);

    outMat(0, 1) = yAxis.x();
    outMat(1, 1) = yAxis.y();
    outMat(2, 1) = yAxis.z();
    outMat(3, 1) = -math::dot(yAxis, snapshot.position);

    outMat(0, 2) = zAxis.x();
    outMat(1, 2) = zAxis.y();
    outMat(2, 2) = zAxis.z();
    outMat(3, 2) = -math::dot(zAxis, snapshot.position);

    outMat(0, 3) = ZERO;
    outMat(1, 3) = ZERO;
    outMat(2, 3) = ZERO;
    outMat(3, 3) = ONE;

    return outMat;
}

/*
 * megamol::core::thecam::camera<M, P>::view_matrix_right_handed
 */
template <class M, template <class> class P>
typename megamol::core::thecam::camera<M, P>::matrix_type&
megamol::core::thecam::camera<M, P>::view_matrix_right_handed(matrix_type& outMat, const snapshot_type& snapshot) {
    static const auto ZERO = static_cast<world_type>(0);
    static const auto ONE = static_cast<world_type>(1);

    auto& xAxis = snapshot.right_vector;
    auto& yAxis = snapshot.up_vector;
    auto zAxis = snapshot.view_vector;

    // TODO: Does it make sense not raising an error directly?
    if (maths_type::handedness != Handedness::right_handed) {
        zAxis *= -ONE;
    }

    // https://msdn.microsoft.com/de-de/library/windows/desktop/bb205342(v=vs.85).aspx
    // EDIT by schatzkn: the signs have been adapted to comply with the opengl coordinate system with inversed z-axis
    // this is hacky because it does not work for direct3D anymore.
    outMat(0, 0) = xAxis.x();
    outMat(1, 0) = xAxis.y();
    outMat(2, 0) = xAxis.z();
    outMat(3, 0) = -math::dot(xAxis, snapshot.position);

    outMat(0, 1) = yAxis.x();
    outMat(1, 1) = yAxis.y();
    outMat(2, 1) = yAxis.z();
    outMat(3, 1) = -math::dot(yAxis, snapshot.position);

    outMat(0, 2) = -zAxis.x();
    outMat(1, 2) = -zAxis.y();
    outMat(2, 2) = -zAxis.z();
    outMat(3, 2) = math::dot(zAxis, snapshot.position);

    outMat(0, 3) = ZERO;
    outMat(1, 3) = ZERO;
    outMat(2, 3) = ZERO;
    outMat(3, 3) = ONE;

    return outMat;
}


/*
 * megamol::core::thecam::camera<M, P>::calc_matrices
 */
template <class M, template <class> class P>
void megamol::core::thecam::camera<M, P>::calc_matrices(
    snapshot_type& outSnapshot, matrix_type& outView, matrix_type& outProj, snapshot_content sc) const {
    sc |= snapshot_content::camera_coordinate_system; // Just paranoia ...
    sc |= snapshot_content::camera_space_frustum;     // Just paranoia ...
    this->take_snapshot(outSnapshot, sc);

    switch (this->handedness()) {
    case Handedness::left_handed:
        camera::view_matrix_left_handed(outView, outSnapshot);
        camera::projection_matrix_left_handed(outProj, outSnapshot);
        break;

    case Handedness::right_handed:
        camera::view_matrix_right_handed(outView, outSnapshot);
        camera::projection_matrix_right_handed(outProj, outSnapshot);
        break;

    default:
        throw std::runtime_error("Unknown handedness in camera::calc_matrices(). This should never happen.");
    }
}


/*
 * megamol::core::thecam::camera<M, P>::focal_length
 */
template <class M, template <class> class P>
typename megamol::core::thecam::camera<M, P>::world_type megamol::core::thecam::camera<M, P>::focal_length(void) const {
    // See
    // http://www.scratchapixel.com/lessons/3d-basic-rendering/3d-viewing-pinhole-camera/how-pinhole-camera-works-part-2
    // for connection between film gate, resolution gate and focal length.
    auto w = this->has_film_gate() ? static_cast<world_type>(this->film_gate().width())
                                   : static_cast<world_type>(this->resolution_gate().width());
    auto t = std::tan(this->aperture_angle_radians());
    return (w / t);
}


/*
 * megamol::core::thecam::camera<M, P>::get_minimal_state
 */
template <class M, template <class> class P>
typename megamol::core::thecam::camera<M, P>::minimal_state_type&
megamol::core::thecam::camera<M, P>::get_minimal_state(minimal_state_type& outState) {
    outState.centre_offset[0] = this->centre_offset().x();
    outState.centre_offset[1] = this->centre_offset().y();
    outState.convergence_plane = this->convergence_plane();
    outState.eye = this->eye();
    outState.far_clipping_plane = this->far_clipping_plane();
    outState.film_gate[0] = this->film_gate().width();
    outState.film_gate[1] = this->film_gate().height();
    outState.gate_scaling = this->gate_scaling();
    outState.half_aperture_angle_radians = this->half_aperture_angle_radians();
    outState.half_disparity = this->half_disparity();
    outState.image_tile[0] = this->image_tile().left();
    outState.image_tile[1] = this->image_tile().top();
    outState.image_tile[2] = this->image_tile().right();
    outState.image_tile[3] = this->image_tile().bottom();
    outState.near_clipping_plane = this->near_clipping_plane();
    outState.orientation[0] = this->orientation()[math::quaternion_component::x];
    outState.orientation[1] = this->orientation()[math::quaternion_component::y];
    outState.orientation[2] = this->orientation()[math::quaternion_component::z];
    outState.orientation[3] = this->orientation()[math::quaternion_component::w];
    outState.position[0] = this->position()[0];
    outState.position[1] = this->position()[1];
    outState.position[2] = this->position()[2];
    outState.projection_type = this->projection_type();
    outState.resolution_gate[0] = this->resolution_gate().width();
    outState.resolution_gate[1] = this->resolution_gate().height();
    return outState;
}


/*
 * megamol::core::thecam::camera<M, P>::look_at
 */
template <class M, template <class> class P>
void megamol::core::thecam::camera<M, P>::look_at(const point_type& lookAt) {
    // http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
    const auto& sv = maths_type::view_vector;
    const auto tv = math::perspective_divide(lookAt) - this->position();
    this->orientation(quaternion_type::from_vectors(sv, tv));
}


/*
 * megamol::core::thecam::camera<M, P>::projection_matrix
 */
template <class M, template <class> class P>
typename megamol::core::thecam::camera<M, P>::matrix_type megamol::core::thecam::camera<M, P>::projection_matrix(
    void) const {
    matrix_type retval;
    snapshot_type snapshot;

    this->take_snapshot(snapshot, snapshot_content::camera_space_frustum);

    switch (this->handedness()) {
    case Handedness::left_handed:
        camera::projection_matrix_right_handed(retval, snapshot);
        break;

    case Handedness::right_handed:
        camera::projection_matrix_right_handed(retval, snapshot);
        break;

    default:
        throw std::runtime_error("Unknown handedness in camera::calc_matrices(). This should never happen.");
    }

    return std::move(retval);
}


/*
 * megamol::core::thecam::camera<M, P>::reset
 */
template <class M, template <class> class P> void megamol::core::thecam::camera<M, P>::reset(void) {
    const auto WZ = static_cast<world_type>(0);
    const auto WO = static_cast<world_type>(1);

    this->centre_offset(thecam::math::vector<fractional_type, 2>());
    this->convergence_plane(WZ);
    this->half_disparity(WZ);
    this->eye(megamol::core::thecam::Eye::centre);
    this->far_clipping_plane(WZ);
    this->film_gate(world_size_type());
    this->gate_scaling(megamol::core::thecam::Gate_scaling::none);
    this->half_aperture_angle_radians(static_cast<fractional_type>(0));
    this->image_tile(screen_rectangle_type());
    this->near_clipping_plane(WZ);
    this->orientation(quaternion_type::create_identity());
    this->position(point_type(WZ, WZ, WZ, WO));
    this->projection_type(megamol::core::thecam::Projection_type::perspective);
    this->resolution_gate(screen_size_type());
}


/*
 * megamol::core::thecam::camera<M, P>::take_snapshot
 */
template <class M, template <class> class P>
typename megamol::core::thecam::camera<M, P>::snapshot_type& megamol::core::thecam::camera<M, P>::take_snapshot(
    snapshot_type& snapshot, const snapshot_content which) const {
    const auto EYE_DIR = static_cast<world_type>(this->eye());
    const auto HALF = static_cast<world_type>(0.5);

    /* We guarantee that we return at least the requested information. */
    snapshot.content = which;

    /* Resolve some dependencies. */
    if (snapshot.content != snapshot_content::all) {
        if (snapshot.contains(snapshot_content::camera_space_frustum)) {
            // If we want the frustum, we need to account for mismatches in the
            // aspect ratios and we need to know the aspect ratios.
            snapshot.content |= snapshot_content::gate_scaling;
            snapshot.content |= snapshot_content::resolution_aspect;
            snapshot.content |= snapshot_content::film_aspect;
        }

        if (snapshot.contains(snapshot_content::gate_scaling)) {
            // We need both aspect ratios to compute the gate scaling factor.
            snapshot.content |= snapshot_content::resolution_aspect;
            snapshot.content |= snapshot_content::film_aspect;
        }

        if (snapshot.contains(snapshot_content::view_vector) && (this->eye() != Eye::centre) &&
            (this->projection_type() == Projection_type::toe_in) &&
            (this->half_disparity() != static_cast<world_type>(0))) {
            // If we have non-trivial toe-in stereo, we need the up-vector to
            // compute the final view-vector.
            snapshot.content |= snapshot_content::up_vector;
        }

        if (snapshot.contains(snapshot_content::camera_coordinate_system)) {
            // Up and view vector are parts of the camera system.
            snapshot.content |= snapshot_content::up_vector;
            snapshot.content |= snapshot_content::view_vector;
        }
    } /* end if (snapshot.content != snapshot_content::all) */

    /* Compute device/image/resolution aspect ratio. */
    if (snapshot.contains(snapshot_content::resolution_aspect)) {
        snapshot.resolution_aspect = this->resolution_gate_aspect();
    }

    /* Compute film aspect ratio. */
    if (snapshot.contains(snapshot_content::film_aspect)) {
        snapshot.film_aspect = this->film_gate_aspect();
        THE_ASSERT(this->has_film_gate() || (snapshot.film_aspect == 0));
    }

    /* Compute scaling between film and resolution gate. */
    if (snapshot.contains(snapshot_content::gate_scaling)) {
        snapshot.gate_scaling[0] = static_cast<fractional_type>(1);
        snapshot.gate_scaling[1] = static_cast<fractional_type>(1);

        switch (this->gate_scaling()) {
        case Gate_scaling::uniform_to_fill:
            if (snapshot.film_aspect > snapshot.resolution_aspect) {
                THE_ASSERT(snapshot.film_aspect != 0);
                snapshot.gate_scaling[0] = snapshot.resolution_aspect / snapshot.film_aspect;
            } else {
                THE_ASSERT(snapshot.resolution_aspect != 0);
                snapshot.gate_scaling[1] = snapshot.film_aspect / snapshot.resolution_aspect;
            }
            break;

        case Gate_scaling::uniform_to_fit:
            if (snapshot.film_aspect > snapshot.resolution_aspect) {
                THE_ASSERT(snapshot.resolution_aspect != 0);
                snapshot.gate_scaling[1] = snapshot.film_aspect / snapshot.resolution_aspect;
            } else {
                THE_ASSERT(snapshot.film_aspect != 0);
                snapshot.gate_scaling[0] = snapshot.resolution_aspect / snapshot.film_aspect;
            }
            break;
        }
    }

    /* Compute bounds of view frustum in camera space. */
    if (snapshot.contains(snapshot_content::camera_space_frustum)) {
        auto iw = static_cast<world_type>(this->resolution_gate().width());
        auto ih = static_cast<world_type>(this->resolution_gate().height());
        // TODO: account for empty tile
        auto tl = static_cast<world_type>(this->image_tile().left());
        auto tt = static_cast<world_type>(this->image_tile().top());
        auto tr = static_cast<world_type>(this->image_tile().right());
        auto tb = static_cast<world_type>(this->image_tile().bottom());
        if (this->image_tile().width() == 0 || this->image_tile().height() == 0) {
            tl = tb = static_cast<world_type>(0);
            tr = iw;
            tt = ih;
        }

        snapshot.frustum_near = this->near_clipping_plane();
        snapshot.frustum_far = this->far_clipping_plane();

        switch (this->projection_type()) {
        case Projection_type::perspective:
        case Projection_type::parallel:
        case Projection_type::toe_in:
        case Projection_type::off_axis: {
            // See http://paulbourke.net/stereographics/stereorender/
            auto h = std::tan(this->half_aperture_angle_radians()) * snapshot.frustum_near;
            auto w = h * snapshot.resolution_aspect; // TODO: film gate

            /*
            l = this->Parameters()->TileRect().GetLeft()
                * w / (this->Parameters()->VirtualViewSize().Width() * 0.5f);
            r = this->Parameters()->TileRect().GetRight()
                * w / (this->Parameters()->VirtualViewSize().Width() * 0.5f);
            b = this->Parameters()->TileRect().GetBottom()
                * h / (this->Parameters()->VirtualViewSize().Height() * 0.5f);
            t = this->Parameters()->TileRect().GetTop()
                * h / (this->Parameters()->VirtualViewSize().Height() * 0.5f);
            */
            // TODO: could use SSE2 here
            snapshot.frustum_left = (tl * w) / (iw * HALF);
            snapshot.frustum_top = (tt * h) / (ih * HALF);
            snapshot.frustum_right = (tr * w) / (iw * HALF);
            snapshot.frustum_bottom = (tb * h) / (ih * HALF);

            // TODO: tracking here?
            // TODO: toe-in/parallel in projection or in view matrix?

            if (this->projection_type() == Projection_type::off_axis) {
                auto c = snapshot.frustum_near + this->convergence_plane();
                THE_ASSERT(c != static_cast<world_type>(0));
                // w += static_cast<world_type>(this->eye()) * (snapshot.frustum_near * this->half_disparity()) /
                // this->focal_length();  // TODO: convergence plane
                w += EYE_DIR * this->half_disparity() * snapshot.frustum_near / c;
            }

            // TODO: could use SSE2 here
            // cut out local frustum for tile rect
            snapshot.frustum_left -= w;
            snapshot.frustum_right -= w;
            snapshot.frustum_bottom -= h;
            snapshot.frustum_top -= h;
        } break;

        case Projection_type::orthographic:
            // TODO: could use SSE2 here
            snapshot.frustum_left = tl - (iw * HALF);
            snapshot.frustum_top = tt - (ih * HALF);
            snapshot.frustum_right = tr - (iw * HALF);
            snapshot.frustum_bottom = tb - (ih * HALF);
            break;

        default:
            throw std::runtime_error("invalid operation");
        }

        // TODO: gate scaling here?
    }

    // TODO: normalise only once?
    // const auto q = math::normalise(this->orientation());

    /* Compute final up-vector. */
    if (snapshot.contains(snapshot_content::up_vector)) {
        auto v = maths_type::up_vector;
        auto q = math::normalise(this->orientation());

        snapshot.up_vector = math::rotate(v, q);
        THE_ASSERT(
            math::is_equal(snapshot.up_vector.w(), static_cast<world_type>(0), static_cast<world_type>(0.00001)));
        THE_ASSERT(math::is_equal(
            math::length(snapshot.up_vector), static_cast<world_type>(1), static_cast<world_type>(0.00001)));
    }

    /* Compute final view-vector. */
    if (snapshot.contains(snapshot_content::view_vector)) {
        auto v = maths_type::view_vector;
        auto q = math::normalise(this->orientation());

        snapshot.view_vector = math::rotate(v, q);
        THE_ASSERT(
            math::is_equal(snapshot.view_vector.w(), static_cast<world_type>(0), static_cast<world_type>(0.00001)));
        THE_ASSERT(math::is_equal(
            math::length(snapshot.view_vector), static_cast<world_type>(1), static_cast<world_type>(0.00001)));
    }

    /* Compute final position and the right-vector of the camera. */
    if (snapshot.contains(snapshot_content::camera_coordinate_system)) {
        THE_ASSERT(snapshot.contains(snapshot_content::up_vector));
        THE_ASSERT(snapshot.contains(snapshot_content::view_vector));

        if (this->handedness() == Handedness::right_handed) {
            snapshot.right_vector = math::cross(snapshot.view_vector, snapshot.up_vector);
        } else {
            snapshot.right_vector = math::cross(snapshot.up_vector, snapshot.view_vector);
        }
        THE_ASSERT(math::is_equal(
            math::length(snapshot.right_vector), static_cast<world_type>(1), static_cast<world_type>(0.00001)));

        if (this->half_disparity() != static_cast<world_type>(0)) {
            auto p = vector_type(this->position());
            auto e = static_cast<world_type>(this->eye());
            auto d = e * this->half_disparity() * snapshot.right_vector;
            p += d;
            snapshot.position = point_type(p);
        } else {
            snapshot.position = this->position();
        }
    }

    /*
     * Toe-in/converged stereo requires rotating the camera itself.
     * Note that the right-vector must have been computed before rotating the
     * view-vector now.
     */
    if (snapshot.contains(snapshot_content::view_vector) && (this->projection_type() == Projection_type::toe_in) &&
        (EYE_DIR != static_cast<world_type>(0)) && (this->half_disparity() != static_cast<world_type>(0))) {
        /*
         *                   look at
         *                      ^
         *                      | (initial view); c = |near + convergence|
         *                      | opposite
         *                      |
         *  eye<-------------centre
         *    e = |half disparity|
         *    adjacent
         */
        auto e = EYE_DIR * this->half_disparity();
        auto c = this->near_clipping_plane() + this->convergence_plane();
        THE_ASSERT(snapshot.contains(snapshot_content::up_vector));
        THE_ASSERT(e != static_cast<world_type>(0));
        // TODO: is this correct?
        auto a = std::atan2(c, e);
        a = static_cast<world_type>(0.5 * math::pi<double>::value) - a;
        if (this->handedness() == Handedness::left_handed) {
            // Rotation direction in left-handed systems is clockwise, ie
            // mathematically negative.
            a = -a;
        }
        quaternion_type q(thecam::utility::do_not_initialise);
        math::set_from_angle_axis(q, a, snapshot.up_vector);
        snapshot.view_vector = math::rotate(snapshot.view_vector, q);
        // The following result will be invalid if the camera system was not
        // requested (but only the view vector), but the user will hopefully not
        // use the right-vector if not requested anyway. If the camera system
        // was requested (which implies the right-vector9, the view-vector will
        // be requested, too, so we can use this code path for computing both.
        snapshot.right_vector = math::rotate(snapshot.right_vector, q);
    }

    return snapshot;
}


/*
 * megamol::core::thecam::camera<M, P>::operator =
 */
template <class M, template <class> class P>
megamol::core::thecam::camera<M, P>& megamol::core::thecam::camera<M, P>::operator=(const minimal_state_type& rhs) {
    this->centre_offset(
        {rhs.centre_offset[0], rhs.centre_offset[1]}); // TODO is this correct for all template possibilities?
    this->convergence_plane(rhs.convergence_plane);
    this->eye(rhs.eye);
    this->far_clipping_plane(rhs.far_clipping_plane);
    this->film_gate({rhs.film_gate[0], rhs.film_gate[1]}); // TODO is this correct for all template possibilities?
    this->gate_scaling(rhs.gate_scaling);
    this->half_aperture_angle_radians(rhs.half_aperture_angle_radians);
    this->half_disparity(rhs.half_disparity);
    this->image_tile({rhs.image_tile[0], rhs.image_tile[1], rhs.image_tile[2],
        rhs.image_tile[3]}); // TODO is this correct for all template possibilities?
    this->near_clipping_plane(rhs.near_clipping_plane);
    this->orientation(quaternion_type(rhs.orientation[0], rhs.orientation[1], rhs.orientation[2], rhs.orientation[3]));
    this->position(vector_type(rhs.position[0], rhs.position[1], rhs.position[2], 1.0f));
    this->projection_type(rhs.projection_type);
    this->resolution_gate(
        {rhs.resolution_gate[0], rhs.resolution_gate[1]}); // TODO is this correct for all template possibilities?
    return *this;
}
