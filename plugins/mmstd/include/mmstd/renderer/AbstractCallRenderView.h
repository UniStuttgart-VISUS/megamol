/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/Input.h"
#include "mmstd/renderer/CallRender3D.h"

namespace megamol::core::view {

/**
 * Call for rendering visual elements (from separate sources) into a single target, i.e.,
 * FBO-based compositing and cluster display.
 */
template<typename FBO, const char* NAME, const char* DESC>
class AbstractCallRenderView : public AbstractCallRender {
public:
    using FBO_TYPE = FBO;

    /** Function index of 'render' */
    static const unsigned int CALL_RENDER = AbstractCallRender::FnRender;

    /** Function index of 'getExtents' */
    static const unsigned int CALL_EXTENTS = AbstractCallRender::FnGetExtents;

    /** Function index of 'ResetView' */
    static const unsigned int CALL_RESETVIEW = 7;

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName() {
        return NAME;
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return DESC;
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
        ASSERT(CALL_RESETVIEW == AbstractCallRender::FunctionCount() && "Enum has bad magic number");
        return AbstractCallRender::FunctionCount() + 1;
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        if (idx == CALL_RESETVIEW) {
            return "ResetView";
        }
        return AbstractCallRender::FunctionName(idx);
    }

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to 'this'
     */
    AbstractCallRenderView& operator=(const AbstractCallRenderView& rhs) {
        view::AbstractCallRender::operator=(rhs);
        _framebuffer = rhs._framebuffer;
        return *this;
    }

    /**
     * Sets the Framebuffer
     *
     * @param fb The framebuffer
     */
    inline void SetFramebuffer(std::shared_ptr<FBO_TYPE> fb) {
        _framebuffer = fb;
    }

    /**
     * Gets the Framebuffer
     *
     */
    inline std::shared_ptr<FBO_TYPE> GetFramebuffer() {
        return _framebuffer;
    }

    /**
     * Ctor.
     */
    AbstractCallRenderView() = default;

    /**
     * Dtor.
     */
    ~AbstractCallRenderView() override = default;

private:
    std::shared_ptr<FBO_TYPE> _framebuffer;
};

} // namespace megamol::core::view
