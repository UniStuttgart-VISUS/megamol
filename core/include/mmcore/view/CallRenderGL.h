/*
 * CallRenderGL.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/view/AbstractCallRender.h"

#define GLOWL_OPENGL_INCLUDE_GLAD
#include <glowl/FramebufferObject.hpp>

namespace megamol {
namespace core {
namespace view {


    class CallRenderGL : public AbstractCallRender {
    public:
        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        CallRenderGL& operator=(const CallRenderGL& rhs);

        inline void SetFramebufferObject(std::shared_ptr<glowl::FramebufferObject> fbo) {
            _framebuffer = fbo;
        }

        inline std::shared_ptr<glowl::FramebufferObject> GetFramebufferObject() {
            return _framebuffer;
        }

    protected:
        /** Ctor. */
        CallRenderGL(void);

        /** Dtor. */
        virtual ~CallRenderGL(void);

    private:
        std::shared_ptr<glowl::FramebufferObject> _framebuffer;
    };

} // namespace view
} /* end namespace core */
} /* end namespace megamol */
