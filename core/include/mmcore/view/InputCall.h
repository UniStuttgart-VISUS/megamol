/*
 * InputCall.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_INPUTCALL_H_INCLUDED
#define MEGAMOLCORE_INPUTCALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/view/MouseFlags.h"

namespace megamol {
namespace core {
namespace view {


/**
 * Base class of input calls
 */
class MEGAMOLCORE_API InputCall : public Call {
public:
    static const unsigned int FnOnKey = 0;
    static const unsigned int FnOnChar = 1;
    static const unsigned int FnOnMouseButton = 2;
    static const unsigned int FnOnMouseMove = 3;
    static const unsigned int FnOnMouseScroll = 4;

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) { return 5; }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        // clang-format off
#define CaseFunction(id) case Fn##id: return #id
        // clang-format on
        switch (idx) {
            CaseFunction(OnKey);
            CaseFunction(OnChar);
            CaseFunction(OnMouseButton);
            CaseFunction(OnMouseMove);
            CaseFunction(OnMouseScroll);
        default:
            return NULL;
        }
#undef CaseFunction
    }

	/** Ctor. */
    InputCall() : mouseX(0.0f), mouseY(0.0f), mouseFlags(0) {}

    /** Dtor. */
    virtual ~InputCall(void) = default;


    /**
     * Answer the mouse flags
     *
     * @return The mouse flags
     */
    inline MouseFlags GetMouseFlags(void) const { return this->mouseFlags; }

    /**
     * Answer the mouse x coordinate in world space
     *
     * @return The mouse x coordinate in world space
     */
    inline float GetMouseX(void) const { return this->mouseX; }

    /**
     * Answer the mouse y coordinate in world space
     *
     * @return The mouse y coordinate in world space
     */
    inline float GetMouseY(void) const { return this->mouseY; }

    /**
     * Sets the mouse informations.
     *
     * @param x The mouse x coordinate in world space
     * @param y The mouse y coordinate in world space
     * @param flags The mouse flags
     */
    inline void SetMouseInfo(float x, float y, MouseFlags flags) {
        this->mouseX = x;
        this->mouseY = y;
        this->mouseFlags = flags;
    }

    /**
     * Gets the state of the mouse selection.
     *
     * @return The current state of the mouse selection
     */
    inline bool MouseSelection(void) { return this->mouseSelection; }

    /**
     * Sets the state of the mouse selection.
     *
     * @param selection The current state of the mouse selection
     */
    inline void SetMouseSelection(bool selection) { this->mouseSelection = selection; }

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    InputCall& operator=(const InputCall& rhs) = default;

protected:
    /** The mouse coordinates for the mouse event */
    float mouseX, mouseY;

    /** The mouse flags for the mouse event */
    MouseFlags mouseFlags;


    /** The current state of the mouse toggle selection */
    bool mouseSelection;
};


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_INPUTCALL_H_INCLUDED */
