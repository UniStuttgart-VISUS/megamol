/*
 * CallRender3D.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <glm/glm.hpp>

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/Camera_2.h"
#include "mmcore/view/AbstractCallRender3D.h"
#include "mmcore/view/RenderOutput.h"

namespace megamol {
namespace core {
namespace view {


/**
 * Class of CPU context rendering
 *
 * Function "Render" tells the callee to render in a CPU context
 *
 * Function "GetExtents" asks the callee to fill the extents member of the
 * call (bounding boxes, temporal extents).
 */
class MEGAMOLCORE_API CallRender3D : public AbstractCallRender3D, public RenderOutput {
public:

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "CallRender3D"; }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) { return "CPU Rendering call"; }

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
    static const char* FunctionName(unsigned int idx) {
        return AbstractCallRender::FunctionName(idx);
    }

    /** Ctor. */
    CallRender3D(void);

    /** Dtor. */
    virtual ~CallRender3D(void);

     /**
     * Sets the background color
     *
     * @param backCol The new background color
     */
    inline void SetBackgroundColor(glm::vec4 backCol) {
        _backgroundCol = backCol;
    }

    /**
     * Gets the background color
     *
     * @return The stored background color
     */
    inline glm::vec4 BackgroundColor(void) const {
        return _backgroundCol;
    }

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    CallRender3D& operator=(const CallRender3D& rhs);

private:

    glm::vec4 _backgroundCol;
};

/** Description class typedef */
typedef factories::CallAutoDescription<CallRender3D> CallRender3DDescription;

} // namespace view
} /* end namespace core */
} /* end namespace megamol */
