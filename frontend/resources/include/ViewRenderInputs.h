/*
 * ViewRenderInputs.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "EntryPoint.h"
#include "RenderInput.h"

namespace megamol {
namespace frontend_resources {

    using UintPair = std::pair<unsigned int, unsigned int>;
    using DoublePair = std::pair<double, double>;

    struct ViewportTile {
        UintPair global_resolution;
        DoublePair tile_start_normalized;
        DoublePair tile_end_normalized;
    };

    struct ViewRenderInputs : public frontend_resources::RenderInputsUpdate {
        static constexpr const char* Name = "ViewRenderInputs";

        // individual inputs used by view for rendering of next frame
        megamol::frontend_resources::RenderInput render_input;

        // sets (local) fbo resolution of render_input from various sources
        std::function<UintPair()> render_input_framebuffer_size_handler;
        std::function<ViewportTile()> render_input_tile_handler;
        std::function<std::optional<frontend_resources::RenderInput::CameraMatrices>()> render_input_camera_handler = []() { return std::nullopt; };

        void update() override {
            auto fbo_size = render_input_framebuffer_size_handler();
            render_input.local_view_framebuffer_resolution = {fbo_size.first, fbo_size.second};

            auto tile = render_input_tile_handler();
            render_input.global_framebuffer_resolution = {tile.global_resolution.first, tile.global_resolution.second};
            render_input.local_tile_relative_begin = {tile.tile_start_normalized.first, tile.tile_start_normalized.second};
            render_input.local_tile_relative_end = {tile.tile_end_normalized.first, tile.tile_end_normalized.second};

            render_input.camera_matrices_override = render_input_camera_handler();
        }

        frontend::FrontendResource get_resource() override{
            return {Name, render_input};
        };
    };
#define accessViewRenderInput(unique_ptr) (*static_cast<frontend_resources::ViewRenderInputs*>(unique_ptr.get()))

} /* end namespace frontend_resources */
} /* end namespace megamol */
