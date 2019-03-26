#pragma once

#include "triangulation.h"

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "glad/glad.h"

#include <memory>

namespace megamol
{
    namespace flowvis
    {
        class implicit_topology : public core::Module
        {
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

            /** Output slot for the triangle mesh */
            core::CalleeSlot triangle_mesh_slot;

            /** Output slot for data attached to the triangles or their nodes */
            core::CalleeSlot mesh_data_slot;

            /** Path to input vector field */
            core::param::ParamSlot vector_field_path;

            /** Path to input convergence structures */
            core::param::ParamSlot convergence_structures_path;
            
            /** Transfer function for labels */
            core::param::ParamSlot label_transfer_function;
            
            /** Transfer function for distances */
            core::param::ParamSlot distance_transfer_function;

            /** Indicator for changed output */
            bool output_changed;

            /** Output labels */
            std::shared_ptr<std::vector<GLfloat>> labels;

            /** Output distances */
            std::shared_ptr<std::vector<GLfloat>> distances;
        };
    }
}