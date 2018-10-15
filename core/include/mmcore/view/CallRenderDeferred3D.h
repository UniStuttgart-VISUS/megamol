/*
 * CallRenderDeferred3D.h
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLRENDERDEFERRED3D_H_INCLUDED
#define MEGAMOLCORE_CALLRENDERDEFERRED3D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/AbstractCallRender3D.h"
#include "mmcore/view/RenderDeferredOutput.h"


namespace megamol {
namespace core {
namespace view {


#ifdef _WIN32
#pragma warning(disable: 4250)  // I know what I am doing ...
#endif /* _WIN32 */
    /**
     * Base class of rendering graph calls
     *
     * Function "Render" tells the callee to render itself into the currently
     * active opengl context (TODO: Late on it could also be a FBO).
     *
     * Function "GetExtents" asks the callee to fill the extents member of the
     * call (bounding boxes, temporal extents).
     */
    class MEGAMOLCORE_API CallRenderDeferred3D : public AbstractCallRender3D, public RenderDeferredOutput {
    public:

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "CallRenderDeferred3D";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call for rendering a frame with deferred shading";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return 2;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            switch (idx) {
                case 0: return "Render";
                case 1: return "GetExtents";
                default: return NULL;
            }
        }

        /** Ctor. */
        CallRenderDeferred3D(void);

        /** Dtor. */
        virtual ~CallRenderDeferred3D(void);

        /**
         * Assignment operator
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        CallRenderDeferred3D& operator=(const CallRenderDeferred3D& rhs);

    };
#ifdef _WIN32
#pragma warning(default: 4250)
#endif /* _WIN32 */


    /** Description class typedef */
    typedef factories::CallAutoDescription<CallRenderDeferred3D> CallRenderDeferred3DDescription;


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLRENDERDEFERRED3D_H_INCLUDED */
