//
// Renderer3DModuleMouse.h
//
// Copyright (C) 2012 by University of Stuttgart (VISUS).
// All rights reserved.
//

#ifndef MMPROTEINPLUGIN_RENDERER3DMODULEMOUSE_H_INCLUDED
#define MMPROTEINPLUGIN_RENDERER3DMODULEMOUSE_H_INCLUDED

#include "mmcore/view/Renderer3DModuleDS.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/view/MouseFlags.h"
#include "protein_calls/CallMouseInput.h"

namespace megamol {
namespace protein {

/**
 * TODO
 */
class Renderer3DModuleMouse : public core::view::Renderer3DModuleDS {

public:

    /** Ctor. */
    Renderer3DModuleMouse(void);

    /** Dtor. */
    virtual ~Renderer3DModuleMouse(void);

protected:

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param[in] call The calling call.
     * @return The return value of the function.
     */
    virtual bool GetExtents(core::Call& call)= 0;

    /**
     * The render callback.
     *
     * @param[in] call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(core::Call& call) = 0;

	/**
	 * Callback for mouse events (move, press, and release)
	 *
	 * @param[in] x The x coordinate of the mouse in screen space
	 * @param[in] y The y coordinate of the mouse in screen space
	 * @param[in] flags The mouse flags
	 * @return 'true' on success
	 */
	virtual bool MouseEvent(int x, int y, core::view::MouseFlags flags) = 0;

private:

    /**
     * The mouse event callback.
     *
     * @param[in] call The calling call.
     * @return The return value of the function.
     */
    bool MouseEventCallback(core::Call& call) {
        try {
			protein_calls::CallMouseInput &cm = dynamic_cast<protein_calls::CallMouseInput&>(call);
            return this->MouseEvent(cm.GetMouseX(), cm.GetMouseY(), cm.GetMouseFlags());
        } catch(...) {
            ASSERT("MouseEventCallback call cast failed\n");
        }
        return false;
    }

    /// The mouse nput callee slot
    core::CalleeSlot mouseSlot;
};

} // end namespace protein
} // end namespace megamol
#endif // MMPROTEINPLUGIN_RENDERER3DMODULEMOUSE_H_INCLUDED
