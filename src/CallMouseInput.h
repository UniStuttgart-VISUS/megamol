//
// CallMouseInput.h
//
// Copyright (C) 2012 by University of Stuttgart (VISUS).
// All rights reserved.
//

#ifndef MMPROTEINPLUGIN_CALLMOUSEINPUT_H_INCLUDED
#define MMPROTEINPLUGIN_CALLMOUSEINPUT_H_INCLUDED

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/MouseFlags.h"

namespace megamol {
namespace protein {

/**
 * TODO
 */
class CallMouseInput : public core::Call {

public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "CallMouseInput";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Call for rendering a frame and sending mouse information";
    }

    /**
      * Answer the number of functions used for this call.
      *
      * @return The number of functions used for this call.
      */
     static unsigned int FunctionCount(void) {
         return 1;
     }

     /**
      * Answer the name of the function used for this call.
      *
      * @param idx The index of the function to return it's name.
      *
      * @return The name of the requested function.
      */
     static const char * FunctionName(unsigned int idx) {
         switch (idx) {
             case 0: return "MouseEvent";
             default: return NULL;
         }
     }

    /**
     * Answer the mouse flags
     *
     * @return The mouse flags
     */
    inline core::view::MouseFlags GetMouseFlags(void) const {
        return this->mouseFlags;
    }

    /**
     * Answer the mouse x coordinate in world space
     *
     * @return The mouse x coordinate in world space
     */
    inline int GetMouseX(void) const {
        return this->mouseX;
    }

    /**
     * Answer the mouse y coordinate in world space
     *
     * @return The mouse y coordinate in world space
     */
    inline int GetMouseY(void) const {
        return this->mouseY;
    }

    /**
     * Sets the mouse informations.
     *
     * @param x The mouse x coordinate in world space
     * @param y The mouse y coordinate in world space
     * @param flags The mouse flags
     */
    inline void SetMouseInfo(int x, int y, core::view::MouseFlags flags) {
        this->mouseX = x;
        this->mouseY = y;
        this->mouseFlags = flags;
    }

private:

    /// The mouse coordinates for the mouse event
    int mouseX, mouseY;

    /// The mouse flags for the mouse event
    core::view::MouseFlags mouseFlags;
};

} // end namespace protein
} // end namespace megamol

#endif // MMPROTEINPLUGIN_CALLMOUSEINPUT_H_INCLUDED
