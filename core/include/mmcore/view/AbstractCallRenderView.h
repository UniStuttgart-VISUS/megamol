/*
 * AbstractCallRenderView.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/api/MegaMolCore.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/Input.h"
#include "mmcore/thecam/camera.h"



namespace megamol {
namespace core {
namespace view {

/**
 * Call for rendering visual elements (from separate sources) into a single target, i.e.,
     * FBO-based compositing and cluster display.
 */
class MEGAMOLCORE_API AbstractCallRenderView : public AbstractCallRender {
public:

    /** Function index of 'render' */
    static const unsigned int CALL_RENDER = AbstractCallRender::FnRender;

    /** Function index of 'getExtents' */
    static const unsigned int CALL_EXTENTS = AbstractCallRender::FnGetExtents;

    /** Function index of 'ResetView' */
    static const unsigned int CALL_RESETVIEW = 7;

    void setRequestedHeight(int height) {
        _height = height;
    }

    void setRequestedWidth(int width) {
        _width = width;
    }

    /**
     * Answer the height of the virtual viewport
     *
     * @return The height of the virtual viewport
     */
    inline int getRequestedHeight(void) const {
        return _height;
    }

    /**
     * Answer the width of the virtual viewport
     *
     * @return The width of the virtual viewport
     */
    inline int getRequestedWidth(void) const {
        return _width;
    }

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to 'this'
     */
    AbstractCallRenderView& operator=(const AbstractCallRenderView& rhs);

protected:
    /**
     * Ctor.
     */
    AbstractCallRenderView(void);

private:

    /** The height of the requested virtual viewport */
    int _height;

    /** The width of the requested virtual viewport */
    int _width;
};



} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */
