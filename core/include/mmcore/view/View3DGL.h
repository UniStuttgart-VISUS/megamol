/*
 * View3DGL.h
 *
 * Copyright (C) 2018, 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once
#include "mmcore/view/AbstractView3D.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/graphics/Cursor2D.h"

namespace megamol {
namespace core {
namespace view {

class MEGAMOLCORE_API View3DGL : public view::AbstractView3D {

public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "View3DGL"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "New and improved 3D View Module"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    View3DGL(void);

    /** Dtor. */
    virtual ~View3DGL(void);

    /**
     * Renders this AbstractView3D in the currently active OpenGL context.
     *
     * @param context
     */
    virtual void Render(const mmcRenderViewContext& context, Call* call) override;

    virtual bool OnKey(view::Key key, view::KeyAction action, view::Modifiers mods) override;

    virtual bool OnChar(unsigned int codePoint) override;

    virtual bool OnMouseButton(view::MouseButton button, view::MouseButtonAction action, view::Modifiers mods) override;

    virtual bool OnMouseMove(double x, double y) override;

    virtual bool OnMouseScroll(double dx, double dy) override;

protected:

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

    std::shared_ptr<vislib::graphics::gl::FramebufferObject> _fbo;

private:

    /** The mouse x coordinate */
    float _mouseX;

    /** The mouse y coordinate */
    float _mouseY;

    /** the 2d cursor of this view */
    vislib::graphics::Cursor2D _cursor2d;

};

} // namespace view
} /* end namespace core */
} /* end namespace megamol */

