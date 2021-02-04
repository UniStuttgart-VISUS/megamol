/*
 * RenderOutput.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */


#pragma once

#include <vector>
#include <memory>
#include "mmcore/view/AbstractRenderOutput.h"



namespace megamol {
namespace core {
namespace view {

    struct CPUFramebuffer {
            bool depthBufferActive = false;
            std::vector<uint32_t> colorBuffer;
            std::vector<float> depthBuffer;
            unsigned int width = 0;
            unsigned int height = 0;
            int x = 0;
            int y = 0;
    };

    /**
     * Abstract base class of rendering graph calls
     *
     * Handles the output buffer control.
     */
    class RenderOutput : public virtual AbstractRenderOutput {
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
        std::shared_ptr<CPUFramebuffer> getGenericFramebuffer(void) const;


        /**
         * Set the FBO.
         *
         */
        void setGenericFramebuffer(std::shared_ptr<CPUFramebuffer> fbo);


        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        RenderOutput& operator=(const RenderOutput& rhs);

    protected:

        /** Ctor. */
        RenderOutput(void);

        /** Dtor. */
        virtual ~RenderOutput(void);

    private:

        std::shared_ptr<CPUFramebuffer> _framebuffer;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

