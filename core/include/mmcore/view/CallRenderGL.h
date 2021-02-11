/*
 * CallRenderGL.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/view/AbstractCallRender.h"
#include "vislib/graphics/gl/FramebufferObject.h"

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

    inline void SetFramebufferObject(std::shared_ptr<vislib::graphics::gl::FramebufferObject> fbo) {
        _framebuffer = fbo;
    }

    inline std::shared_ptr<vislib::graphics::gl::FramebufferObject> GetFramebufferObject() {
        return _framebuffer;
    }

protected:
    /** Ctor. */
    CallRenderGL(void);

    /** Dtor. */
    virtual ~CallRenderGL(void);

private:

    std::shared_ptr<vislib::graphics::gl::FramebufferObject> _framebuffer;

};

} // namespace view
} /* end namespace core */
} /* end namespace megamol */
