/*
 * RenderInput.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <glm/glm.hpp>
#include <optional>

namespace megamol {
namespace frontend_resources {

struct RenderInput {
    glm::uvec2 global_framebuffer_resolution{0, 0}; //< e.g. overall powerwall resolution when tiling, not intended to be used by view
    glm::uvec2 local_view_framebuffer_resolution{0, 0}; //< resolution with which the View should render into own framebuffer

    glm::dvec2 local_tile_relative_begin{0.0, 0.0}, local_tile_relative_end{1.0, 1.0}; //< relative start and end of image tile in global framebuffer this view is rendering with local resolution. local resolution and effective tile resolution can differ. the view framebuffer should have local resolution.

    struct CameraMatrices {
        glm::mat4 view;
        glm::mat4 projection;
    };
    std::optional<CameraMatrices> camera_matrices_override = std::nullopt; //< if camera matrices are overridden, this view still needs to render in local resolution

    double instanceTime_sec = 0.0; //< monotone high resolution time in seconds since first frame rendering of some (any) view
    double time_sec = 0.0; //< time computed by view::TimeControl(instanceTime)
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
