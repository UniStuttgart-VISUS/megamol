/*
 * implicit_topology.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "implicit_topology_computation.h"
#include "implicit_topology_results.h"
#include "triangulation.h"

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "glad/glad.h"

#include <array>
#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Module for computing and visualizing the implicit topology of a vector field.
        *
        * @author Alexander Straub
        * @author Grzegorz K. Karch
        */
        class implicit_topology : public core::Module
        {
            static_assert(std::is_same<GLfloat, float>::value, "'GLfloat' and 'float' must be the same type!");
            static_assert(std::is_same<GLuint, unsigned int>::value, "'GLuint' and 'unsigned int' must be the same type!");

        public:
            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static inline const char* ClassName() { return "implicit_topology"; }

            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static inline const char* Description() { return "Compute and visualize implicit topology of a 2D vector field"; }

            /**
             * Answers whether this module is available on the current system.
             *
             * @return 'true' if the module is available, 'false' otherwise.
             */
            static inline bool IsAvailable() { return true; }

            /**
             * Initialises a new instance.
             */
            implicit_topology();

            /**
             * Finalises an instance.
             */
            virtual ~implicit_topology();

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
            /** Callbacks for the triangle mesh */
            bool get_triangle_data_callback(core::Call& call);
            bool get_triangle_extent_callback(core::Call& call);

            /** Callbacks for the mesh data */
            bool get_data_data_callback(core::Call& call);
            bool get_data_extent_callback(core::Call& call);

            /** Callback for the result writer */
            bool get_result_writer_cb_callback(core::Call& call);
            std::function<bool(const implicit_topology_results&)> get_result_writer_callback;

            /** Callbacks for the log stream */
            bool get_log_cb_callback(core::Call& call);
            std::function<std::ostream&()> get_log_callback;

            /** Callbacks for the performance log stream */
            bool get_performance_cb_callback(core::Call& call);
            std::function<std::ostream&()> get_performance_callback;

            /** Callbacks for starting/stopping/resetting the computation */
            bool start_computation_callback(core::param::ParamSlot& parameter = core::param::ParamSlot("", ""));
            bool stop_computation_callback(core::param::ParamSlot& parameter = core::param::ParamSlot("", ""));
            bool reset_computation_callback(core::param::ParamSlot& parameter = core::param::ParamSlot("", ""));
            bool load_computation_callback(core::param::ParamSlot& parameter = core::param::ParamSlot("", ""));
            bool save_computation_callback(core::param::ParamSlot& parameter = core::param::ParamSlot("", ""));

            /**
            * Initialize computation.
            *
            * @return Success
            */
            bool initialize_computation();

            /**
            * Load input from file.
            *
            * @param resolution   Domain resolution (number of vectors per direction)
            * @param domain       Domain size (minimum and maximum coordinates)
            * @param positions    Positions of the vectors, also used as initial seed
            * @param vectors      Vectors of the vector field
            * @param points       Convergence structure points (e.g., critical points, periodic orbits, ...)
            * @param point_ids    Unique IDs (or labels) of the given points
            * @param lines        Convergence structure lines (e.g., domain boundaries, obstacles, ...)
            * @param line_ids     (Unique) IDs (or labels) of the given lines
            *
            * @return Success
            */
            bool load_input(std::array<int, 2>& resolution, std::array<float, 4>& domain, std::vector<float>& positions, std::vector<float>& vectors,
                std::vector<float>& points, std::vector<int>& point_ids, std::vector<float>& lines, std::vector<int>& line_ids);

            /**
            * Update results obtained from computation.
            */
            void update_results();

            /**
            * Manipulate accessibility of fixed parameters
            *
            * @param read_only True: read-only, false: writable
            */
            void set_readonly_fixed_parameters(bool read_only);

            /**
            * Manipulate accessibility of variable parameters
            *
            * @param read_only True: read-only, false: writable
            */
            void set_readonly_variable_parameters(bool read_only);

            /** Output slot for the triangle mesh */
            core::CalleeSlot triangle_mesh_slot;

            /** Output slot for data attached to the triangles or their nodes */
            core::CalleeSlot mesh_data_slot;

            /** Output slot for writing results to file */
            core::CalleeSlot result_writer_slot;

            /** Output slots for logging */
            core::CalleeSlot log_slot;
            core::CalleeSlot performance_slot;

            /** Output slot for reading results from file */
            core::CallerSlot result_reader_slot;

            /** Start or reset the computation */
            core::param::ParamSlot start_computation;
            core::param::ParamSlot stop_computation;
            core::param::ParamSlot reset_computation;
            core::param::ParamSlot load_computation;
            core::param::ParamSlot save_computation;

            /** Path to input vector field, and input convergence structures */
            core::param::ParamSlot vector_field_path;
            core::param::ParamSlot convergence_structures_path;
            
            /** Transfer function for labels, distances, reasons of termination, and gradients */
            core::param::ParamSlot label_transfer_function;
            core::param::ParamSlot distance_transfer_function;
            core::param::ParamSlot termination_transfer_function;
            core::param::ParamSlot gradient_transfer_function;

            /** Checkboxes for fixing the data range */
            core::param::ParamSlot label_fixed_range;
            core::param::ParamSlot distance_fixed_range;
            core::param::ParamSlot termination_fixed_range;
            core::param::ParamSlot gradient_fixed_range;

            /** Values of the fixed range */
            core::param::ParamSlot label_range_min, label_range_max;
            core::param::ParamSlot distance_range_min, distance_range_max;
            core::param::ParamSlot termination_range_min, termination_range_max;
            core::param::ParamSlot gradient_range_min, gradient_range_max;

            /** Parameters for stream line computation */
            core::param::ParamSlot num_integration_steps;
            core::param::ParamSlot integration_timestep;
            core::param::ParamSlot max_integration_error;
            core::param::ParamSlot num_particles_per_batch;
            core::param::ParamSlot num_integration_steps_per_batch;

            /** Parameters for grid refinement */
            core::param::ParamSlot refinement_threshold;
            core::param::ParamSlot refine_at_labels;
            core::param::ParamSlot distance_difference_threshold;

            /** Input information */
            std::array<int, 2> resolution;

            /** Indicator for changed output */
            bool computation_running;

            bool mesh_output_changed;
            bool data_output_changed;

            /** Output vertices and indices of the triangle mesh */
            std::shared_ptr<std::vector<GLfloat>> vertices;
            std::shared_ptr<std::vector<GLuint>> indices;

            /** Output labels */
            std::shared_ptr<std::vector<GLfloat>> labels_forward;
            std::shared_ptr<std::vector<GLfloat>> labels_backward;

            /** Output distances */
            std::shared_ptr<std::vector<GLfloat>> distances_forward;
            std::shared_ptr<std::vector<GLfloat>> distances_backward;

            /** Output reasons for termination */
            std::shared_ptr<std::vector<GLfloat>> terminations_forward;
            std::shared_ptr<std::vector<GLfloat>> terminations_backward;

            /** Output gradients */
            std::shared_ptr<std::vector<GLfloat>> gradients_forward;
            std::shared_ptr<std::vector<GLfloat>> gradients_backward;

            /** Computation class */
            std::unique_ptr<implicit_topology_computation> computation;

            /** Store last promised result */
            std::shared_future<implicit_topology_results> last_result;

            /** Store previous result */
            std::unique_ptr<implicit_topology_results> previous_result;
        };
    }
}