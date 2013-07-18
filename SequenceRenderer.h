/*
 * SequenceRenderer.h
 *
 * Author: Michael Krone
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MEGAMOLCORE_SEQUENCERENDERER_H_INCLUDED
#define MEGAMOLCORE_SEQUENCERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "param/ParamSlot.h"
#include "CallerSlot.h"
#include "view/Renderer2DModule.h"
#include "vislib/GLSLShader.h"
#include <vislib/OutlineFont.h>
#include <vislib/OpenGLTexture2D.h>

namespace megamol {
namespace protein {

    class SequenceRenderer : public megamol::core::view::Renderer2DModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "SequenceRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Offers sequence renderings.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** ctor */
        SequenceRenderer(void);

        /** dtor */
        ~SequenceRenderer(void);

    protected:
        
        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);
        
        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

        /**
         * Callback for mouse events (move, press, and release)
         *
         * @param x The x coordinate of the mouse in world space
         * @param y The y coordinate of the mouse in world space
         * @param flags The mouse flags
         */
        virtual bool MouseEvent(float x, float y, megamol::core::view::MouseFlags flags);

    private:

        /**********************************************************************
         * 'render'-functions
         **********************************************************************/

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents(megamol::core::view::CallRender2D& call);

        /**
        * The Open GL Render callback.
        *
        * @param call The calling call.
        * @return The return value of the function.
        */
        virtual bool Render(megamol::core::view::CallRender2D& call);

        /**********************************************************************
         * variables
         **********************************************************************/

        /** caller slot */
        core::CallerSlot dataCallerSlot;
        

        // the number of residues in one row
        megamol::core::param::ParamSlot resCountPerRowParam;

    };

} /* end namespace protein */
} /* end namespace megamol */

#endif // MEGAMOLCORE_SEQUENCERENDERER_H_INCLUDED
