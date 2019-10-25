/*
 * triangle_mesh_renderer_3d.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mesh_data_call.h"

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/math/Cuboid.h"

#include "glad/glad.h"

#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace megamol
{
    namespace flowvis
    {
        /**
        * Module for uploading a 3D triangle mesh to the GPU.
        *
        * @author Alexander Straub
        */
        class triangle_mesh_renderer_3d : public core::Module
        {
            static_assert(std::is_same<GLfloat, float>::value, "'GLfloat' and 'float' must be the same type!");
            static_assert(std::is_same<GLuint, unsigned int>::value, "'GLuint' and 'unsigned int' must be the same type!");

        public:
            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static inline const char* ClassName() { return "triangle_mesh_renderer_3d"; }

            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static inline const char* Description() { return "Upload 3D data to the GPU for use with the mesh plugin"; }

            /**
             * Answers whether this module is available on the current system.
             *
             * @return 'true' if the module is available, 'false' otherwise.
             */
            static inline bool IsAvailable() { return true; }

            /**
             * Initialises a new instance.
             */
            triangle_mesh_renderer_3d();

            /**
             * Finalises an instance.
             */
            virtual ~triangle_mesh_renderer_3d();

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
            /** Callbacks for setting up the render tasks */
            bool get_task_callback(core::Call& caller);
            bool get_task_extent_callback(core::Call& caller);

            /** Callbacks for uploading the mesh to the GPU */
            bool get_mesh_callback(core::Call& caller);
            bool get_mesh_extent_callback(core::Call& caller);

            /** Input slot for the triangle mesh */
            core::CallerSlot triangle_mesh_slot;
            SIZE_T triangle_mesh_hash;

            /** Input slot for data attached to the triangles or their nodes */
            core::CallerSlot mesh_data_slot;
            SIZE_T mesh_data_hash;

            /** Output slots for rendering */
            core::CalleeSlot render_task;
            core::CalleeSlot gpu_mesh;

            /** Parameter slot for choosing data sets to visualize */
            core::param::ParamSlot data_set;

            /** Parameter slot for choosing validity masks */
            core::param::ParamSlot mask;
            core::param::ParamSlot mask_color;

            /** Parameter slot for choosing between filled and wireframe mode */
            core::param::ParamSlot wireframe;

            /** Bounding box */
            vislib::math::Cuboid<float> bounds;

            /** Struct for storing data needed for rendering */
            struct render_data_t
            {
                std::shared_ptr<std::vector<GLfloat>> vertices;
                std::shared_ptr<std::vector<GLuint>> indices;

                std::shared_ptr<mesh_data_call::data_set> values;

                std::shared_ptr<std::vector<GLfloat>> mask;

            } render_data;
        };
    }
}