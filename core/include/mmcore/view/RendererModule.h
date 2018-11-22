/*
 * RendererModule.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_RENDERERMODULE_H_INCLUDED
#define MEGAMOLCORE_RENDERERMODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/view/AbstractCallRender.h"
#include "mmcore/view/AbstractInputScope.h"
#include "mmcore/view/MouseFlags.h"
#include "mmcore/view/InputCall.h"
#include "mmcore/view/AbstractCallRender.h"

namespace megamol {
namespace core {
namespace view {

/**
 * Base class of rendering modules.
 */
template <class C> class MEGAMOLCORE_API RendererModule : public Module, public AbstractInputScope {
public:
    /** Ctor. */
    RendererModule() : Module(), renderSlot("rendering", "Connects the Renderer to a calling renderer or view") {
        // InputCall
        this->renderSlot.SetCallback(
            C::ClassName(), InputCall::FunctionName(InputCall::FnOnKey), &RendererModule::OnKeyCallback);
        this->renderSlot.SetCallback(
            C::ClassName(), InputCall::FunctionName(InputCall::FnOnChar), &RendererModule::OnCharCallback);
        this->renderSlot.SetCallback(C::ClassName(), InputCall::FunctionName(InputCall::FnOnMouseButton),
            &RendererModule::OnMouseButtonCallback);
        this->renderSlot.SetCallback(
            C::ClassName(), InputCall::FunctionName(InputCall::FnOnMouseMove), &RendererModule::OnMouseMoveCallback);
        this->renderSlot.SetCallback(C::ClassName(), InputCall::FunctionName(InputCall::FnOnMouseScroll),
            &RendererModule::OnMouseScrollCallback);
        // AbstractCallRender
        this->renderSlot.SetCallback(C::ClassName(), AbstractCallRender::FunctionName(AbstractCallRender::FnRender),
            &RendererModule::RenderCallback);
        this->renderSlot.SetCallback(C::ClassName(), AbstractCallRender::FunctionName(AbstractCallRender::FnGetExtents),
            &RendererModule::GetExtentsCallback);
        // Do not make it avaiable yet (extensibility).
    }

    /** Dtor. */
    virtual ~RendererModule(void) = default;

protected:
    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(C& call) = 0;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(C& call) = 0;

    [[deprecated("Implement AbstractInputScope methods instead")]] virtual bool MouseEvent(
        float x, float y, MouseFlags flags) {
        return false;
    }

    virtual bool OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) {
        MouseFlags mouseFlags;
        // Ugly mapping to deprecated functions (can be removed some day).
        auto down = action == core::view::MouseButtonAction::PRESS;
        if (mods.test(core::view::Modifier::SHIFT)) {
            view::MouseFlagsSetFlag(mouseFlags, view::MOUSEFLAG_MODKEY_SHIFT_DOWN, true);
        } else if (mods.test(core::view::Modifier::CTRL)) {
            view::MouseFlagsSetFlag(mouseFlags, view::MOUSEFLAG_MODKEY_CTRL_DOWN, true);
        } else if (mods.test(core::view::Modifier::ALT)) {
            view::MouseFlagsSetFlag(mouseFlags, view::MOUSEFLAG_MODKEY_ALT_DOWN, true);
        }
        if (button == MouseButton::BUTTON_LEFT) {
            view::MouseFlagsSetFlag(mouseFlags, view::MOUSEFLAG_BUTTON_LEFT_DOWN, down);
        } else if (button == MouseButton::BUTTON_RIGHT) {
            view::MouseFlagsSetFlag(mouseFlags, view::MOUSEFLAG_BUTTON_RIGHT_DOWN, down);
        } else if (button == MouseButton::BUTTON_MIDDLE) {
            view::MouseFlagsSetFlag(mouseFlags, view::MOUSEFLAG_BUTTON_MIDDLE_DOWN, down);
        }
		//TODO: Verify semantics of (X,Y) coordinates...
		// - Could be "world space" (see View2D/View3D) instead of window space!
		// - If so, then provide a freaking method in the Call to the the transformation instead of passing around black magic!
        this->MouseEvent(lastX, lastY, mouseFlags);
        return down;
    }

    virtual bool OnMouseMove(double x, double y) {
        // Ugly mapping to deprecated functions (can be removed some day).
        this->lastX = x;
        this->lastY = y;
        this->MouseEvent(x, y, 0);
        return false;
    }

    bool RenderCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
            return this->Render(cr);
        } catch (...) {
            ASSERT("onRenderCallback call cast failed\n");
        }
        return false;
    }

    bool GetExtentsCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
            return this->GetExtents(cr);
        } catch (...) {
            ASSERT("onGetExtentsCallback call cast failed\n");
        }
        return false;
    }

    bool OnKeyCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
            auto& evt = cr.GetInputEvent();
            ASSERT(evt.tag == InputEvent::Tag::Key && "Callback invocation mismatched input event");
            return this->OnKey(evt.keyData.key, evt.keyData.action, evt.keyData.mods);
        } catch (...) {
            ASSERT("onRenderCallback call cast failed\n");
        }
        return false;
    }

    bool OnCharCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
            auto& evt = cr.GetInputEvent();
            ASSERT(evt.tag == InputEvent::Tag::Char && "Callback invocation mismatched input event");
            return this->OnChar(evt.charData.codePoint);
        } catch (...) {
            ASSERT("onRenderCallback call cast failed\n");
        }
        return false;
    }

    bool OnMouseButtonCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
            auto& evt = cr.GetInputEvent();
            ASSERT(evt.tag == InputEvent::Tag::MouseButton && "Callback invocation mismatched input event");
            return this->OnMouseButton(
                evt.mouseButtonData.button, evt.mouseButtonData.action, evt.mouseButtonData.mods);
        } catch (...) {
            ASSERT("onRenderCallback call cast failed\n");
        }
        return false;
    }

    bool OnMouseMoveCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
            auto& evt = cr.GetInputEvent();
            ASSERT(evt.tag == InputEvent::Tag::MouseMove && "Callback invocation mismatched input event");
            return this->OnMouseMove(evt.mouseMoveData.x, evt.mouseMoveData.y);
        } catch (...) {
            ASSERT("onRenderCallback call cast failed\n");
        }
        return false;
    }

    bool OnMouseScrollCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
            auto& evt = cr.GetInputEvent();
            ASSERT(evt.tag == InputEvent::Tag::MouseScroll && "Callback invocation mismatched input event");
            return this->OnMouseScroll(evt.mouseScrollData.dx, evt.mouseScrollData.dy);
        } catch (...) {
            ASSERT("onRenderCallback call cast failed\n");
        }
        return false;
    }

    /** The render callee slot */
    CalleeSlot renderSlot;

private:
    /** Last mouse position (for deprecation mapping) */
    float lastX, lastY;
};

} /* end namespace view */
} // namespace core
} // namespace megamol

#endif /* MEGAMOLCORE_RENDERERMODULE_H_INCLUDED */
