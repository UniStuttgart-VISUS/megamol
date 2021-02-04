/*
 * CallRenderViewGL.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLRENDERVIEW_H_INCLUDED
#define MEGAMOLCORE_CALLRENDERVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/api/MegaMolCore.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/CallRenderView.h"
#include "mmcore/view/RenderOutputOpenGL.h" 
#include "mmcore/view/Input.h"
#include "mmcore/thecam/camera.h"
#include "vislib/graphics/graphicstypes.h"
#include "mmcore/view/GPUAffinity.h"


namespace megamol {
namespace core {
namespace view {

#ifdef _WIN32
#pragma warning(disable: 4250)  // I know what I am doing ...
#endif /* _WIN32 */
    /**
     * Call for rendering visual elements (from separate sources) into a single target, i.e.,
	 * FBO-based compositing and cluster display.
     */
    class MEGAMOLCORE_API
    CallRenderViewGL : public CallRenderView, public RenderOutputOpenGL, public GPUAffinity {
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

	/** Function index of 'render' */
        static const unsigned int CALL_RENDER = AbstractCallRender::FnRender;

        /** Function index of 'getExtents' */
        static const unsigned int CALL_EXTENTS = AbstractCallRender::FnGetExtents;

        /** Function index of 'freeze' */
        static const unsigned int CALL_FREEZE = 7;

        /** Function index of 'unfreeze' */
        static const unsigned int CALL_UNFREEZE = 8;

        /** Function index of 'ResetView' */
        static const unsigned int CALL_RESETVIEW = 9;

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
			ASSERT(CALL_FREEZE == AbstractCallRender::FunctionCount()
				&& "Enum has bad magic number");
			ASSERT(CALL_UNFREEZE == AbstractCallRender::FunctionCount() + 1
				&& "Enum has bad magic number");
			ASSERT(CALL_RESETVIEW  == AbstractCallRender::FunctionCount() + 2
				&& "Enum has bad magic number");
            return AbstractCallRender::FunctionCount() + 3;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char* FunctionName(unsigned int idx) {
            if (idx == CALL_FREEZE) {
                return "freeze";
            } else if (idx == CALL_UNFREEZE) {
                return "unfreeze";
            } else if (idx == CALL_RESETVIEW) {
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

    };
#ifdef _WIN32
#pragma warning(default: 4250)
#endif /* _WIN32 */


    /** Description class typedef */
    typedef factories::CallAutoDescription<CallRenderViewGL>
        CallRenderViewGLDescription;


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLRENDERVIEW_H_INCLUDED */
