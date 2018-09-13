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
#include "mmcore/view/AbstractInputScope.h"
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
    RendererModule() : Module(), renderSlot("rendering", "Connects the Renderer to a calling renderer or view") {
		// InputCall
		this->renderSlot.SetCallback(C::ClassName(),
            InputCall::FunctionName(InputCall::FnOnKey), 
			&RendererModule::OnKeyCallback);
        this->renderSlot.SetCallback(C::ClassName(),
            InputCall::FunctionName(InputCall::FnOnChar),
			&RendererModule::OnCharCallback);
        this->renderSlot.SetCallback(C::ClassName(),
            InputCall::FunctionName(InputCall::FnOnMouseButton),
            &RendererModule::OnMouseButtonCallback);
        this->renderSlot.SetCallback(C::ClassName(),
            InputCall::FunctionName(InputCall::FnOnMouseMove),
            &RendererModule::OnMouseMoveCallback);
        this->renderSlot.SetCallback(C::ClassName(),
            InputCall::FunctionName(InputCall::FnOnMouseScroll),
            &RendererModule::OnMouseScrollCallback);
		// AbstractCallRender
		this->renderSlot.SetCallback(C::ClassName(),
            AbstractCallRender::FunctionName(AbstractCallRender::FnRender), 
			&RendererModule::RenderCallback);
        this->renderSlot.SetCallback(C::ClassName(),
            AbstractCallRender::FunctionName(AbstractCallRender::FnGetExtents),
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

    [[deprecated("Implement AbstractInputScope methods instead")]] 
	virtual bool MouseEvent(float x, float y, MouseFlags flags) { return false; }

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
            return false; // this->OnKey(cr.GetMouseX(), cr.GetMouseY(), cr.GetMouseFlags());
        } catch (...) {
            ASSERT("onRenderCallback call cast failed\n");
        }
        return false;
    }

    bool OnCharCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
            return false; // this->OnKey(cr.GetMouseX(), cr.GetMouseY(), cr.GetMouseFlags());
        } catch (...) {
            ASSERT("onRenderCallback call cast failed\n");
        }
        return false;
    }

    bool OnMouseButtonCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
			//return MouseEvent(cr.GetMouseX(), cr.GetMouseY(), cr.GetMouseFlags());
            return false; // this->OnKey(cr.GetMouseX(), cr.GetMouseY(), cr.GetMouseFlags());
                          
        } catch (...) {
            ASSERT("onRenderCallback call cast failed\n");
        }
        return false;
    }

    bool OnMouseMoveCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
			//return MouseEvent(cr.GetMouseX(), cr.GetMouseY(), cr.GetMouseFlags());
            return false; // this->OnKey(cr.GetMouseX(), cr.GetMouseY(), cr.GetMouseFlags());
        } catch (...) {
            ASSERT("onRenderCallback call cast failed\n");
        }
        return false;
    }

    bool OnMouseScrollCallback(Call& call) {
        try {
            C& cr = dynamic_cast<C&>(call);
            return false; // this->OnKey(cr.GetMouseX(), cr.GetMouseY(), cr.GetMouseFlags());
        } catch (...) {
            ASSERT("onRenderCallback call cast failed\n");
        }
        return false;
    }

	    /** The render callee slot */
    CalleeSlot renderSlot;
};

} /* end namespace view */
} // namespace core
} // namespace megamol

#endif /* MEGAMOLCORE_RENDERERMODULE_H_INCLUDED */
