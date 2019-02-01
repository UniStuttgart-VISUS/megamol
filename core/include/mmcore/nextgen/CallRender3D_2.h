/*
 * CallRender3D_2.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CallRender3D_2_H_INCLUDED
#define MEGAMOLCORE_CallRender3D_2_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <glm/glm.hpp>
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/nextgen/AbstractCallRender3D_2.h"
#include "mmcore/view/MouseFlags.h"
#include "mmcore/view/RenderOutputOpenGL.h"

namespace megamol {
namespace core {
namespace nextgen {
#ifdef _WIN32
#    pragma warning(disable : 4250) // I know what I am doing ...
#endif                              /* _WIN32 */
/**
 * New and improved base class of rendering graph calls
 *
 * Function "Render" tells the callee to render itself into the currently
 * active opengl context (TODO: Late on it could also be a FBO).
 *
 * Function "GetExtents" asks the callee to fill the extents member of the
 * call (bounding boxes, temporal extents).
 */
class MEGAMOLCORE_API CallRender3D_2 : public AbstractCallRender3D_2, public view::RenderOutputOpenGL {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "CallRender3D_2"; }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) { return "New and improved call for rendering a frame"; }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) { return AbstractCallRender3D_2::FunctionCount(); }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) { return AbstractCallRender3D_2::FunctionName(idx); }

    /** Ctor. */
    CallRender3D_2(void);

    /** Dtor. */
    virtual ~CallRender3D_2(void);

    /**
     * Answer the mouse flags
     *
     * @return The mouse flags
     */
    inline view::MouseFlags GetMouseFlags(void) const { return this->mouseFlags; }

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
    inline void SetMouseInfo(float x, float y, view::MouseFlags flags) {
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
     * Sets the background color
     *
     * @param backCol The new background color
     */
    inline void SetBackgroundColor(glm::vec4 backCol) { this->backgroundCol = backCol; }

    /**
     * Gets the background color
     *
     * @return The stored background color
     */
    inline glm::vec4 BackgroundColor(void) const { return this->backgroundCol; }

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    CallRender3D_2& operator=(const CallRender3D_2& rhs);

private:
    /** x-coordinate of the mouse pointer */
    float mouseX;

    /** y-coordinate of the mouse pointer */
    float mouseY;

    /** The mouse flags for the mouse event */
    view::MouseFlags mouseFlags;

    /** The background color */
    glm::vec4 backgroundCol;

    /** The current state of the mouse toggle selection */
    bool mouseSelection;
};
#ifdef _WIN32
#    pragma warning(default : 4250)
#endif /* _WIN32 */

/** Description class typedef */
typedef factories::CallAutoDescription<CallRender3D_2> CallRender3D_2Description;

} // namespace nextgen
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CallRender3D_2_H_INCLUDED */
