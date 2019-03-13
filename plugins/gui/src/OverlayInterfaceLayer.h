/*
 * OverlayInterfaceLayer.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#ifndef MEGAMOL_GUI_OVERLAYINTERFACELAYER_H_INCLUDED
#define MEGAMOL_GUI_OVERLAYINTERFACELAYER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/view/Input.h"
#include "mmcore/view/Renderer2DModule.h"
#include "mmcore/view/Renderer3DModule.h"

#include "vislib/sys/Log.h"

#include "mmcore/view/CallSplitViewOverlay.h"


namespace megamol {
namespace gui {


/**
 * Layer providing interface to overlay slot for GUIRenderer.
 */
template <class M> class OverlayInterfaceLayer: public M {

public:

    /**
     * Callback forwarding OnRender request.
     *
     * (Implemented by GUIRenderer)
     */
    virtual bool OnGUIRenderCallback(megamol::core::Call& call) { return false; }

    /**
     * Callback forwarding OnKey request.
     */
    bool OnGUIKeyCallback(megamol::core::Call& call);

    /**
     * Callback forwarding OnChar request.
     */
    bool OnGUICharCallback(megamol::core::Call& call);

    /**
     * Callback forwarding OnMouse request.
     */
    bool OnGUIMouseButtonCallback(megamol::core::Call& call);

    /**
     * Callback forwarding OnMouseMove request.
     */
    bool OnGUIMouseMoveCallback(megamol::core::Call& call);

    /**
     * Callback forwarding OnMouseScroll request.
     */
    bool OnGUIMouseScrollCallback(megamol::core::Call& call);


protected:

    /**
     * Ctor
     */
    OverlayInterfaceLayer(void);

    /**
     * Dtor
     */
    ~OverlayInterfaceLayer(void);

    /** The overlay callee slot */
    megamol::core::CalleeSlot overlay_slot;

private:

};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_OVERLAYINTERFACELAYER_H_INCLUDED
