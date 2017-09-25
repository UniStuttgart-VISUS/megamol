//
// View3DMouse.h
//
// Copyright (C) 2012 by University of Stuttgart (VISUS).
// All rights reserved.
//

#ifndef MMPROTEINPLUGIN_VIEW3DMOUSE_H_INCLUDED
#define MMPROTEINPLUGIN_VIEW3DMOUSE_H_INCLUDED

#include "mmcore/view/View3D.h"
#include "mmcore/view/MouseFlags.h"
#include "mmcore/CallerSlot.h"

namespace megamol {
namespace protein {

/**
 * TODO
 */
class View3DMouse : public core::view::View3D {

public:

    /** Ctor. */
    View3DMouse();

    /** Dtor. */
    ~View3DMouse();

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "View3DMouse";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "3D View Module sending mouse events to the renderer";
    }

    /**
     * Sets the button state of a button of the 2d cursor. See
     * 'vislib::graphics::Cursor2D' for additional information.
     *
     * @param[in] button The button.
     * @param[in] down Flag whether the button is pressed, or not.
     */
    virtual void SetCursor2DButtonState(unsigned int btn, bool down);

    /**
     * Sets the position of the 2d cursor. See 'vislib::graphics::Cursor2D'
     * for additional information.
     *
     * @param[in] x The x coordinate
     * @param[in] y The y coordinate
     */
    virtual void SetCursor2DPosition(float x, float y);

    /**
     * TODO
     */
    bool OnButtonEvent(core::param::ParamSlot& p);


    /// Slot to send a mouse event to the renderer
    core::CallerSlot mouseSlot;

    /// Enable selecting mode of mouse (disables camera movement)
    core::param::ParamSlot enableSelectingSlot;

    /// The mouse x coordinate
    float mouseX;

    /// The mouse y coordinate
    float mouseY;

    /// The mouse flags
    core::view::MouseFlags mouseFlags;

    /// Flag whether mouse control is to be handed over to the renderer
    bool toggleSelect;

};

} // end namespace protein
} // end namespace megamol
#endif // MMPROTEINPLUGIN_VIEW3DMOUSE_H_INCLUDED
