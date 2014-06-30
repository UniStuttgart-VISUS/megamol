//
// LinkedView3D.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//

#ifndef MEGAMOLCORE_LINKEDVIEW3D_H_INCLUDED
#define MEGAMOLCORE_LINKEDVIEW3D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "view/View3D.h"
#include "Call.h"
#include "CallerSlot.h"

namespace megamol {
namespace core {
namespace view {

class LinkedView3D : public core::view::View3D {

public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "LinkedView3D";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "3D View Module enabling linked views.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor. */
    LinkedView3D(void);

    /** Dtor. */
    virtual ~LinkedView3D(void);

protected:

    /**
     * Renders this AbstractView3D in the currently active OpenGL context.
     *
     * @param context
     */
    virtual void Render(const mmcRenderViewContext& context);

    /**
     * Sets the button state of a button of the 2d cursor. See
     * 'vislib::graphics::Cursor2D' for additional information.
     * This also sets the 'drag' flag
     *
     * @param button The button.
     * @param down Flag whether the button is pressed, or not.
     */
    virtual void SetCursor2DButtonState(unsigned int btn, bool down);

    /**
     * Sets the position of the 2d cursor. See 'vislib::graphics::Cursor2D'
     * for additional information. Test also whether the camera has changed,
     * if yes, set the cnew camera parameters in the shared camera module.
     *
     * @param x The x coordinate
     * @param y The y coordinate
     */
    virtual void SetCursor2DPosition(float x, float y);

private:

    /// Caller slot to access shared camera parameters
    core::CallerSlot sharedCamParamsSlot;

    /// Flags whether dragging is activated (i.e. whether the left mouse button
    /// is pressed
    bool drag;

    /// Flags whether the camera has changed and, therefore, the shared cam
    /// params have to be updated
    bool camChanged;

    /// Stores the cursor position at the previous cursor event
    float oldPosX, oldPosY;

    /// The scene camera (camera of parent class is private)
    vislib::graphics::gl::CameraOpenGL cam;

};

} // namespace view
} // namespace core
} // namespace megamol

#endif // MEGAMOLCORE_LINKEDVIEW3D_H_INCLUDED
