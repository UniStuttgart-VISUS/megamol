/*
 * CallRender2DGL.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLRENDER2D_H_INCLUDED
#define MEGAMOLCORE_CALLRENDER2D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/CallRenderGL.h"


namespace megamol {
namespace core {
namespace view {


/**
 * Call for rendering 2d images
 *
 * Function "Render" tells the callee to render itself into the currently
 * active opengl context (TODO: Later on it could also be a FBO).
 * The bounding box member will be set to the world space rectangle
 * containing the visible part.
 *
 * Function "GetExtents" asks the callee to fill the extents member of the
 * call (bounding boxes).
 * The renderer should not draw anything outside the bounding box
 */
class CallRender2DGL : public CallRenderGL {
public:

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char *ClassName(void) {
        return "CallRender2DGL";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char *Description(void) {
        return "Call for rendering a frame";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return AbstractCallRender::FunctionCount();
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char * FunctionName(unsigned int idx) {
        return AbstractCallRender::FunctionName(idx);
    }

    /** Ctor. */
    CallRender2DGL(void) = default;

    /** Dtor. */
    virtual ~CallRender2DGL(void) = default;


    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    CallRender2DGL& operator=(const CallRender2DGL& rhs) {
        AbstractCallRender::operator=(rhs);
        return *this;
    }

};


/** Description class typedef */
typedef factories::CallAutoDescription<CallRender2DGL> CallRender2DGLDescription;


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLRENDER2D_H_INCLUDED */
