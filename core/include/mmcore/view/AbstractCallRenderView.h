/*
 * AbstractCallRenderView.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/api/MegaMolCore.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/Input.h"
#include "mmcore/thecam/camera.h"



namespace megamol {
namespace core {
namespace view {

/**
 * Call for rendering visual elements (from separate sources) into a single target, i.e.,
     * FBO-based compositing and cluster display.
 */
class MEGAMOLCORE_API AbstractCallRenderView : public AbstractCallRender {
public:

    /** Function index of 'render' */
    static const unsigned int CALL_RENDER = AbstractCallRender::FnRender;

    /** Function index of 'getExtents' */
    static const unsigned int CALL_EXTENTS = AbstractCallRender::FnGetExtents;

    /** Function index of 'freeze' */
    static const unsigned int CALL_FREEZE = 7;

    /** Function index of 'unfreeze' */
    static const unsigned int CALL_UNFREEZE = 8;

    /** Function index of 'ResetView' */
    static const unsigned int CALL_RESETVIEW = 9;

    /**
     * Gets the input modifier
     *
     * @return The input modifier
     */
    inline Modifier InputModifier(void) const {
        return this->mod;
    }

    /**
     * Answers the flag indicating that the background information has been set
     *
     * @return 'true' if the background information has been set
     */
    inline bool IsBackgroundSet(void) const {
        return this->flagBkgnd;
    }

    /**
     * Answers the flag indicating that the viewport information has been set
     *
     * @return 'true' if the viewport information has been set
     */
    inline bool IsViewportSet(void) const {
        return true;
    }

    /**
     * Gets the button.
     *
     * @return The button
     */
    inline unsigned int MouseButton(void) const {
        return this->btn;
    }

    /**
     * Gets the 'down' flag.
     *
     * @return The 'down' flag
     */
    inline bool MouseButtonDown(void) const{
        return this->down;
    }

    /**
     * Gets the x coordinate.
     *
     * @return The x coordinate
     */
    inline float MouseX(void) const {
        return this->x;
    }

    /**
     * Gets the y coordinate.
     *
     * @return The y coordinate
     */
    inline float MouseY(void) const {
        return this->y;
    }

    /**
     * Resets all flags
     */
    inline void ResetAll(void) {
        this->flagBkgnd = false;
    }

    /**
     * Resets the flag indicating that the background had been set.
     */
    inline void ResetBackground(void) {
        this->flagBkgnd = false;
    }

    /**
     * Gets the height of the viewport in pixel.
     *
     * @return The height of the viewport in pixel
     */
    inline unsigned int ViewportHeight(void) const {
        return _framebuffer->height;
    }

    /**
     * Gets the width of the viewport in pixel.
     *
     * @return The width of the viewport in pixel
     */
    inline unsigned int ViewportWidth(void) const {
        return _framebuffer->width;
    }

    /**
     * Answer the height of the virtual viewport
     *
     * @return The height of the virtual viewport
     */
    inline float VirtualHeight(void) const {
        return this->height;
    }

    /**
     * Answer the width of the virtual viewport
     *
     * @return The width of the virtual viewport
     */
    inline float VirtualWidth(void) const {
        return this->width;
    }

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to 'this'
     */
    AbstractCallRenderView& operator=(const AbstractCallRenderView& rhs);

protected:
    /**
     * Ctor.
     */
    AbstractCallRenderView(void);

private:

    /** Flag indicating that the background colour information has been set */
    bool flagBkgnd : 1;

    /** The height of the virtual viewport */
    float height;

    /** The width of the virtual viewport */
    float width;

    /** The button */
    unsigned int btn;

    /**
     * Flag whether the button is pressed, or not, or the new input
     * modifier state
     */
    bool down;

    /** The x coordinate */
    float x;

    /** The y coordinate */
    float y;

    /** The input modifier to be set */
    Modifier mod;

    std::shared_ptr<CPUFramebuffer> _framebuffer;
};



} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */
