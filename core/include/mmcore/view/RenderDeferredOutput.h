/*
 * RenderDeferredOutput.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_RENDERDEFERREDOUTPUT_H_INCLUDED
#define MEGAMOLCORE_RENDERDEFERREDOUTPUT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/AbstractRenderOutput.h"
#include "vislib/graphics/gl/FramebufferObject.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Abstract base class of rendering graph calls for deferred rendering
     *
     * The OutputBuffer must be a FramebufferObject with two colour attachments.
     *  Attachment 0 will be used to store the normal vector information
     *  Attachment 1 will be used to store the base colour information
     *  In addition the depth attachment will be written
     */
    class MEGAMOLCORE_API RenderDeferredOutput : public virtual AbstractRenderOutput {
    public:

        /**
         * Deactivates the output buffer
         */
        virtual void DisableOutputBuffer(void);

        /**
         * Activates the output buffer
         */
        virtual void EnableOutputBuffer(void);

        /**
         * Answer the framebuffer object to be used.
         *
         * @return The framebuffer object to be used
         */
        inline vislib::graphics::gl::FramebufferObject *FrameBufferObject(void) const {
            return this->outputFBO;
        }

        /**
         * Copies the output buffer settings of 'output'
         *
         * @param call The source object to copy from
         */
        inline void SetOutputBuffer(RenderDeferredOutput& output) {
            this->SetOutputBuffer(output.outputFBO);
        }

        /**
         * Sets the output buffer to an framebuffer object.
         *
         * @param fbo Pointer to the framebuffer object to be used; Must not
         *            be NULL. The caller must ensure that the object remains
         *            valid as long as the pointer is set.
         */
        void SetOutputBuffer(vislib::graphics::gl::FramebufferObject *fbo);

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        inline RenderDeferredOutput& operator=(const RenderDeferredOutput& rhs) {
            this->SetOutputBuffer(rhs.outputFBO);
            return *this;
        }

    protected:

        /** Ctor. */
        RenderDeferredOutput(void);

        /** Dtor. */
        virtual ~RenderDeferredOutput(void);

    private:

        /**
         * The framebuffer object the callee should render to
         */
        vislib::graphics::gl::FramebufferObject *outputFBO;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_RENDEROUTPUT_H_INCLUDED */
