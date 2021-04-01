/*
 * View2DGL.h
 *
 * Copyright (C) 2008 - 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VIEW2DGL_H_INCLUDED
#define MEGAMOLCORE_VIEW2DGL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore/view/TimeControl.h"

#define GLOWL_OPENGL_INCLUDE_GLAD
#include <glowl/FramebufferObject.hpp>

namespace megamol {
namespace core {
namespace view {

/*
 * Forward declaration of incoming render calls
 */
class CallRenderViewGL;


/**
 * Base class of rendering graph calls
 */
class View2DGL: public AbstractView {
public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "View2DGL";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "2D View Module";
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
    View2DGL(void);

    /** Dtor. */
    virtual ~View2DGL(void);

    /**
     * Answer the camera synchronization number.
     *
     * @return The camera synchronization number
     */
    virtual unsigned int GetCameraSyncNumber(void) const;

    /**
     * ...
     */
    virtual void Render(double time, double instanceTime, bool present_fbo) override;

    /**
     * Resets the view. This normally sets the camera parameters to
     * default values.
     */
    virtual void ResetView(void);

    /**
     * Resizes the View2DGl framebuffer object.
     *
     * @param width The new width.
     * @param height The new height.
     */
    virtual void Resize(unsigned int width, unsigned int height) override;

    /**
     * Callback requesting a rendering of this view
     *
     * @param call The calling call
     *
     * @return The return value
     */
    virtual bool OnRenderView(Call& call);

    virtual bool GetExtents(Call& call) override;

    virtual bool OnKey(Key key, KeyAction action, Modifiers mods) override;

    virtual bool OnChar(unsigned int codePoint) override;

    virtual bool OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) override;

    virtual bool OnMouseMove(double x, double y) override;

    virtual bool OnMouseScroll(double dx, double dy) override;

    enum MouseMode : uint8_t { Propagate, Pan, Zoom };

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

private:

    /** Track state of ctrl key for camera controls */
    bool _ctrlDown;

    /** The mouse drag mode */
    MouseMode _mouseMode;

    /** The mouse x coordinate */
    float _mouseX;

    /** The mouse y coordinate */
    float _mouseY;

    /** the update counter for the view settings */
    unsigned int _viewUpdateCnt;

    std::shared_ptr<glowl::FramebufferObject> _fbo;
};
} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VIEW2DGL_H_INCLUDED */
