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
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/view/AbstractCallRender.h"
#include "mmcore/view/AbstractInputScope.h"
#include "mmcore/view/InputCall.h"
#include "mmcore/view/MouseFlags.h"

namespace megamol {
namespace core {
namespace view {

/**
 * Base class of rendering modules.
 */
template <class C> class MEGAMOLCORE_API RendererModule : public Module, public AbstractInputScope {
public:
    /** Ctor. */
    RendererModule()
        : Module()
        , chainRenderSlot("chainRendering", "Connects the renderer to and additional renderer")
        , renderSlot("rendering", "Connects the Renderer to a calling renderer or view") {

        // OutputCall
        this->chainRenderSlot.template SetCompatibleCall<factories::CallAutoDescription<C>>();
        // Do not make it available yet (features has to be turned off for legacy code).

        // InputCall
        this->renderSlot.SetCallback(
            C::ClassName(), InputCall::FunctionName(InputCall::FnOnKey), &RendererModule::OnKeyChainCallback);
        this->renderSlot.SetCallback(
            C::ClassName(), InputCall::FunctionName(InputCall::FnOnChar), &RendererModule::OnCharChainCallback);
        this->renderSlot.SetCallback(C::ClassName(), InputCall::FunctionName(InputCall::FnOnMouseButton),
            &RendererModule::OnMouseButtonChainCallback);
        this->renderSlot.SetCallback(C::ClassName(), InputCall::FunctionName(InputCall::FnOnMouseMove),
            &RendererModule::OnMouseMoveChainCallback);
        this->renderSlot.SetCallback(C::ClassName(), InputCall::FunctionName(InputCall::FnOnMouseScroll),
            &RendererModule::OnMouseScrollChainCallback);
        // AbstractCallRender
        this->renderSlot.SetCallback(C::ClassName(), AbstractCallRender::FunctionName(AbstractCallRender::FnRender),
            &RendererModule::RenderChainCallback);
        this->renderSlot.SetCallback(C::ClassName(), AbstractCallRender::FunctionName(AbstractCallRender::FnGetExtents),
            &RendererModule::GetExtentsChainCallback);
        // Do not make it available yet (extensibility).
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
        if (mods.test(core::view::Modifier::SHIFT)) {
            view::MouseFlagsSetFlag(mouseFlags, view::MOUSEFLAG_MODKEY_SHIFT_DOWN, true);
        } else if (mods.test(core::view::Modifier::CTRL)) {
            view::MouseFlagsSetFlag(mouseFlags, view::MOUSEFLAG_MODKEY_CTRL_DOWN, true);
        } else if (mods.test(core::view::Modifier::ALT)) {
            view::MouseFlagsSetFlag(mouseFlags, view::MOUSEFLAG_MODKEY_ALT_DOWN, true);
        }
        auto down = action == core::view::MouseButtonAction::PRESS;
        if (button == MouseButton::BUTTON_LEFT) {
            view::MouseFlagsSetFlag(mouseFlags, view::MOUSEFLAG_BUTTON_LEFT_DOWN, down);
        } else if (button == MouseButton::BUTTON_RIGHT) {
            view::MouseFlagsSetFlag(mouseFlags, view::MOUSEFLAG_BUTTON_RIGHT_DOWN, down);
        } else if (button == MouseButton::BUTTON_MIDDLE) {
            view::MouseFlagsSetFlag(mouseFlags, view::MOUSEFLAG_BUTTON_MIDDLE_DOWN, down);
        }
        // TODO: Verify semantics of (X,Y) coordinates...
        // - Could be "world space" (see View2D/View3D) instead of window space!
        // - If so, then provide a freaking method in the Call to the the transformation instead of passing around black
        // magic!
        this->MouseEvent(lastX, lastY, mouseFlags);
        // Ignore deprecated "event was processed" flag because too many renderers fail to use it properly
        return false;
    }

    virtual bool OnMouseMove(double x, double y) {
        // Ugly mapping to deprecated functions (can be removed some day).
        this->lastX = x;
        this->lastY = y;
        this->MouseEvent(x, y, 0);
        // Ignore deprecated "event was processed" flag because too many renderers fail to use it properly
        return false;
    }

    virtual bool RenderChain(C& call) { return this->Render(call); }

    virtual bool GetExtentsChain(C& call) { return this->GetExtents(call); }

    virtual bool OnKeyChain(Key key, KeyAction action, Modifiers mods) { 
        auto* cr = this->chainRenderSlot.template CallAs<C>();
        if (cr != nullptr) {
            view::InputEvent evt;
            evt.tag = view::InputEvent::Tag::Key;
            evt.keyData.key = key;
            evt.keyData.action = action;
            evt.keyData.mods = mods;
            cr->SetInputEvent(evt);
            if ((*cr)(C::FnOnKey)) return true;
        }
        return this->OnKey(key, action, mods); 
    }

    virtual bool OnCharChain(unsigned int codePoint) { 
        auto* cr = this->chainRenderSlot.template CallAs<C>();
        if (cr != nullptr) {
            view::InputEvent evt;
            evt.tag = view::InputEvent::Tag::Char;
            evt.charData.codePoint = codePoint;
            cr->SetInputEvent(evt);
            if ((*cr)(C::FnOnChar)) return true;
        }
        return this->OnChar(codePoint); 
    }

    virtual bool OnMouseButtonChain(MouseButton button, MouseButtonAction action, Modifiers mods) {
        auto* cr = this->chainRenderSlot.template CallAs<C>();
        if (cr != nullptr) {
            view::InputEvent evt;
            evt.tag = view::InputEvent::Tag::MouseButton;
            evt.mouseButtonData.button = button;
            evt.mouseButtonData.action = action;
            evt.mouseButtonData.mods = mods;
            cr->SetInputEvent(evt);
            if ((*cr)(C::FnOnMouseButton)) return true;
        }
        return this->OnMouseButton(button, action, mods);
    }

    virtual bool OnMouseMoveChain(double x, double y) { 
        auto* cr = this->chainRenderSlot.template CallAs<C>();
        if (cr != nullptr) {
            view::InputEvent evt;
            evt.tag = view::InputEvent::Tag::MouseMove;
            evt.mouseMoveData.x = x;
            evt.mouseMoveData.y = y;
            cr->SetInputEvent(evt);
            if ((*cr)(C::FnOnMouseMove)) return true;
        }
        return this->OnMouseMove(x, y); 
    }

    virtual bool OnMouseScrollChain(double dx, double dy) { 
        auto* cr = this->chainRenderSlot.template CallAs<C>();
        if (cr != nullptr) {
            view::InputEvent evt;
            evt.tag = view::InputEvent::Tag::MouseScroll;
            evt.mouseScrollData.dx = dx;
            evt.mouseScrollData.dy = dy;
            cr->SetInputEvent(evt);
            if ((*cr)(C::FnOnMouseScroll)) return true;
        }
        return this->OnMouseScroll(dx, dy); 
    }

    bool GetExtentsChainCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
            return this->GetExtentsChain(cr);
        } catch (...) {
            ASSERT("onGetExtentsCallback call cast failed\n");
        }
        return false;
    }

    bool RenderChainCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
            return this->RenderChain(cr);
        } catch (...) {
            ASSERT("onRenderCallback call cast failed\n");
        }
        return false;
    }

    bool OnMouseButtonChainCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
            auto& evt = cr.GetInputEvent();
            ASSERT(evt.tag == InputEvent::Tag::MouseButton && "Callback invocation mismatched input event");
            return this->OnMouseButtonChain(
                evt.mouseButtonData.button, evt.mouseButtonData.action, evt.mouseButtonData.mods);
        } catch (...) {
            ASSERT("onRenderCallback call cast failed\n");
        }
        return false;
    }

    bool OnMouseMoveChainCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
            auto& evt = cr.GetInputEvent();
            ASSERT(evt.tag == InputEvent::Tag::MouseMove && "Callback invocation mismatched input event");
            return this->OnMouseMoveChain(evt.mouseMoveData.x, evt.mouseMoveData.y);
        } catch (...) {
            ASSERT("onRenderCallback call cast failed\n");
        }
        return false;
    }

    bool OnMouseScrollChainCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
            auto& evt = cr.GetInputEvent();
            ASSERT(evt.tag == InputEvent::Tag::MouseScroll && "Callback invocation mismatched input event");
            return this->OnMouseScrollChain(evt.mouseScrollData.dx, evt.mouseScrollData.dy);
        } catch (...) {
            ASSERT("onRenderCallback call cast failed\n");
        }
        return false;
    }

    bool OnCharChainCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
            auto& evt = cr.GetInputEvent();
            ASSERT(evt.tag == InputEvent::Tag::Char && "Callback invocation mismatched input event");
            return this->OnCharChain(evt.charData.codePoint);
        } catch (...) {
            ASSERT("onRenderCallback call cast failed\n");
        }
        return false;
    }

    bool OnKeyChainCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
            auto& evt = cr.GetInputEvent();
            ASSERT(evt.tag == InputEvent::Tag::Key && "Callback invocation mismatched input event");
            return this->OnKeyChain(evt.keyData.key, evt.keyData.action, evt.keyData.mods);
        } catch (...) {
            ASSERT("onRenderCallback call cast failed\n");
        }
        return false;
    }

    /** Slot for the daisy-chained renderer */
    CallerSlot chainRenderSlot;

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
