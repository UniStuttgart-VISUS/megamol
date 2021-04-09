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
#include "vislib/graphics/gl/FramebufferObject.h"

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
     * Renders this AbstractView3D in the currently active OpenGL context.
     *
     * @param context
     */
    virtual void Render(const mmcRenderViewContext& context, Call* call = nullptr);

    /**
     * Resets the view. This normally sets the camera parameters to
     * default values.
     */
    virtual void ResetView(void);

    /**
     * Resizes the AbstractView3D.
     *
     * @param width The new width.
     * @param height The new height.
     */
    virtual void Resize(unsigned int width, unsigned int height);

    /**
     * Callback requesting a rendering of this view
     *
     * @param call The calling call
     *
     * @return The return value
     */
    virtual bool OnRenderView(Call& call);

    virtual bool GetExtents(Call& call) override;

    /**
     * Freezes, updates, or unfreezes the view onto the scene (not the
     * rendering, but camera settings, timing, etc).
     *
     * @param freeze true means freeze or update freezed settings,
     *               false means unfreeze
     */
    virtual void UpdateFreeze(bool freeze);

    virtual bool OnKey(Key key, KeyAction action, Modifiers mods) override;

    virtual bool OnChar(unsigned int codePoint) override;

    virtual bool OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) override;

    virtual bool OnMouseMove(double x, double y) override;

    virtual bool OnMouseScroll(double dx, double dy) override;

    /**
     * Unpacks the mouse coordinates, which are relative to the virtual
     * viewport size.
     *
     * @param x The x coordinate of the mouse position
     * @param y The y coordinate of the mouse position
     */
    virtual void unpackMouseCoordinates(float &x, float &y);

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

    /** The viewport height */
    float _height;

    /** The mouse drag mode */
    MouseMode _mouseMode;

    /** The mouse x coordinate */
    float _mouseX;

    /** The mouse y coordinate */
    float _mouseY;

    /** The view focus x coordinate */
    float _viewX;

    /** The view focus y coordinate */
    float _viewY;

    /** The view zoom factor */
    float _viewZoom;

    /** the update counter for the view settings */
    unsigned int _viewUpdateCnt;

    /** the viewport width */
    float _width;

    std::shared_ptr<vislib::graphics::gl::FramebufferObject> _fbo;

    ImageWrapper GetRenderingResult() const override {
        ImageWrapper::DataChannels channels = ImageWrapper::DataChannels::RGBA8; // vislib::graphics::gl::FramebufferObject seems to use RGBA8
        unsigned int fbo_color_buffer_gl_handle = _fbo->GetColourTextureID(0); // IS THIS SAFE?? IS THIS THE COLOR BUFFER??
        size_t fbo_width = _fbo->GetWidth();
        size_t fbo_height = _fbo->GetHeight();

        return frontend_resources::wrap_image({fbo_width, fbo_height}, fbo_color_buffer_gl_handle, channels);
    }
};
} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VIEW2DGL_H_INCLUDED */
