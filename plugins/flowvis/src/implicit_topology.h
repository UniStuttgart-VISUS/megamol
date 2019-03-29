/*
 * implicit_topology.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "implicit_topology_computation.h"
#include "triangulation.h"

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "glad/glad.h"

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
            bool get_results_cb_callback(core::Call& call);
            // TODO

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

            /**
            * Initialize computation.
            *
            * @return Success
            */
            bool initialize_computation();

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
            core::CalleeSlot results_slot;

            /** Output slots for logging */
            core::CalleeSlot log_slot;
            core::CalleeSlot performance_slot;

            /** Start or reset the computation */
            core::param::ParamSlot start_computation;
            core::param::ParamSlot stop_computation;
            core::param::ParamSlot reset_computation;

            /** Path to input vector field, and input convergence structures */
            core::param::ParamSlot vector_field_path;
            core::param::ParamSlot convergence_structures_path;
            
            /** Transfer function for labels, distances and reasons of termination */
            core::param::ParamSlot label_transfer_function;
            core::param::ParamSlot distance_transfer_function;
            core::param::ParamSlot termination_transfer_function;

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

            /** Computation class */
            std::unique_ptr<implicit_topology_computation> computation;

            /** Store last promised result */
            std::shared_future<implicit_topology_computation::result> last_result;
        };
    }
}