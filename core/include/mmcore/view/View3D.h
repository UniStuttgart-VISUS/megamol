/*
 * View3D.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/view/AbstractView3D.h"

namespace megamol {
namespace core {
namespace view {

class MEGAMOLCORE_API View3D : public view::AbstractView3D {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "View3D"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "View 3D module"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    View3D(void);

    /** Dtor. */
    virtual ~View3D(void);

      /**
     * Renders this AbstractView3D in the currently active OpenGL context.
     *
     * @param context
     */
    virtual void Render(const mmcRenderViewContext& context, Call* call);

 protected:
 
    std::shared_ptr<CPUFramebuffer> _framebuffer;
};

} // namespace view
} /* end namespace core */
} /* end namespace megamol */

