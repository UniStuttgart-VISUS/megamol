/*
 * CallRenderViewGL.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/AbstractCallRenderView.h"
#include "mmcore/view/Input.h"

#define GLOWL_OPENGL_INCLUDE_GLAD
#include "glowl/FramebufferObject.hpp"


namespace megamol {
namespace core {
namespace view {

    /**
     * Call for rendering visual elements (from separate sources) into a single target, i.e.,
	 * FBO-based compositing and cluster display.
     */
    class CallRenderViewGL : public AbstractCallRenderView {
    public:

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "CallRenderViewGL";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call for rendering visual elements into a single target";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            ASSERT(CALL_RESETVIEW == AbstractCallRender::FunctionCount()
				&& "Enum has bad magic number");
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
         * Ctor.
         */
        CallRenderViewGL(void);

        /**
         * Copy ctor.
         *
         * @param src Object to clone
         */
        CallRenderViewGL(const CallRenderViewGL& src);

        /**
         * ~Dtor.
         */
        virtual ~CallRenderViewGL(void);

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to 'this'
         */
        CallRenderViewGL& operator=(const CallRenderViewGL& rhs);

        inline void SetFramebufferObject(std::shared_ptr<glowl::FramebufferObject> fbo) {
            _framebuffer = fbo;
        }

        inline std::shared_ptr<glowl::FramebufferObject> GetFramebufferObject() {
            return _framebuffer;
        }

    private:

        std::shared_ptr<glowl::FramebufferObject> _framebuffer;

    };

    /** Description class typedef */
    typedef factories::CallAutoDescription<CallRenderViewGL>
        CallRenderViewGLDescription;


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

