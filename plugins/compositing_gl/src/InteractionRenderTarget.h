/*
 * InteractionRenderTarget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef INTERACTION_RENDER_TARGET_H_INCLUDED
#define INTERACTION_RENDER_TARGET_H_INCLUDED


#include "SimpleRenderTarget.h"

namespace megamol {
namespace compositing {

class InteractionRenderTarget : public SimpleRenderTarget {
public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "InteractionRenderTarget"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Binds a FBO with color, normal, depth and objectID render targets."; }

    InteractionRenderTarget();
    ~InteractionRenderTarget() = default;

protected:

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create();

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(core::view::CallRender3D_2& call);

     /**
     *
     */
    bool getObjectIdRenderTarget(core::Call& caller);

    /**
     *
     */
    bool getMetaDataCallback(core::Call& caller);

private:
    core::CalleeSlot m_objId_render_target;
};

}
}


#endif // !INTERACTION_RENDER_TARGET_H_INCLUDED
