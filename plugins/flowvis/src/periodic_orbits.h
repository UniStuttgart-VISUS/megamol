/*
 * periodic_orbits.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "tpf/data/tpf_grid.h"

#include "Eigen/Dense"

#include <utility>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Module for computing periodic orbits of a vector field.
        *
        * @author Alexander Straub
        */
        class periodic_orbits : public core::Module
        {
        public:
            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static inline const char* ClassName() { return "periodic_orbits"; }

            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static inline const char* Description() { return "Compute periodic orbits of a 2D vector field"; }

            /**
             * Answers whether this module is available on the current system.
             *
             * @return 'true' if the module is available, 'false' otherwise.
             */
            static inline bool IsAvailable() { return true; }

            /**
             * Initialises a new instance.
             */
            periodic_orbits();

            /**
             * Finalises an instance.
             */
            virtual ~periodic_orbits();

        protected:
            /**
             * Implementation of 'Create'.
             *
             * @return 'true' on success, 'false' otherwise.
             */
            virtual bool create() override;

            /**
             * Implementation of 'Release'.
             */
            virtual void release() override;

        private:

            std::vector<Eigen::Vector2f> extract_periodic_orbit(const tpf::data::grid<float, float, 2, 2>& grid,
                const std::vector<Eigen::Vector2f>& critical_points, Eigen::Vector2f seed, float sign) const;


            std::pair<Eigen::Vector2f, float> advect_RK45(const tpf::data::grid<float, float, 2, 2>& grid,
                const Eigen::Vector2f& position, float delta, float sign, float max_error) const;


            std::list<typename tpf::data::grid<float, float, 2, 2>::coords_t> find_turn(const tpf::data::grid<float, float, 2, 2>& grid,
                const std::vector<Eigen::Vector2f>& critical_points, Eigen::Vector2f& position, float& delta, float sign, float max_error,
                std::size_t max_new_cells = -1) const;


            /** Callbacks for the triangle mesh */
            bool get_glyph_data_callback(core::Call& call);
            bool get_glyph_extent_callback(core::Call& call);

            /** Callback for the mouse event */
            bool get_mouse_coordinates_callback(core::Call& call);

            /** Output slot for the glyphs */
            core::CalleeSlot glyph_slot;
            SIZE_T glyph_hash;
            std::vector<std::vector<Eigen::Vector2f>> glyph_output;

            /** Output slot for receiving mouse clicks */
            core::CalleeSlot mouse_slot;

            /** Input slot for getting an input vector field */
            core::CallerSlot vector_field_slot;
            SIZE_T vector_field_hash;

            /** Input slot for getting critical points, needed as seed */
            core::CallerSlot critical_points_slot;
            SIZE_T critical_points_hash;

            /** Stored vector field */
            tpf::data::grid<float, float, 2, 2> grid;

            /** Stored critical points */
            std::vector<Eigen::Vector2f> critical_points;
        };
    }
}