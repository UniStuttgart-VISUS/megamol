/*
 * PlaneRenderer.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CallerSlot.h"

#include "mmcore_gl/view/CallRender3DGL.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"

#include "glowl/GLSLProgram.hpp"

#include "vislib/math/Plane.h"

#include <array>
#include <memory>

namespace megamol {
namespace core_gl {
namespace view {

        /**
         * Module for rendering (clip) plane.
         *
         * @author Alexander Straub
         */
        class PlaneRenderer : public Renderer3DModuleGL {
        public:
            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static inline const char* ClassName() {
                return "PlaneRenderer";
            }

            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static inline const char* Description() {
                return "Render a (clip) plane";
            }

            /**
             * Answers whether this module is available on the current system.
             *
             * @return 'true' if the module is available, 'false' otherwise.
             */
            static inline bool IsAvailable() {
                return true;
            }

            /**
             * Initialises a new instance.
             */
            PlaneRenderer();

            /**
             * Finalises an instance.
             */
            virtual ~PlaneRenderer();

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

            /** Callbacks for the computed streamlines */
            virtual bool GetExtents(CallRender3DGL& call) override;
            virtual bool Render(CallRender3DGL& call) override;

        private:
            /** Call for getting the input plane */
            core::CallerSlot input_plane_slot;

            /** The plane color */
            std::array<float, 4> color;

            /** The (clip) plane */
            vislib::math::Plane<float> plane;

            /** Data needed for rendering */
            std::unique_ptr<glowl::GLSLProgram> render_data;

            /** Initialization status */
            bool initialized;
        };

    } // namespace view
} // namespace core
} // namespace megamol
