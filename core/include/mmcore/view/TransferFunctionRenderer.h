
/*
 * TransferFunctionRenderer.h
 *
 * Copyright (C) 2014 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/api/MegaMolCore.std.h"
#include "Renderer2DModule.h"
#include "mmcore/CallerSlot.h"
#include "vislib/graphics/gl/SimpleFont.h"

namespace megamol {
namespace core {
namespace view {

class TransferFunctionRenderer :  public Renderer2DModule
{
public:
	/**
	 * Answer the name of this module.
	 *
	 * @return The name of this module.
	 */
	static const char *ClassName(void) {
		return "TransferFunctionRenderer";
	}

	/**
	 * Answer a human readable description of this module.
	 *
	 * @return A human readable description of this module.
	 */
	static const char *Description(void) {
		return "Displays a small legend for the transfer function";
	}

	
    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
		return true;
    }

	TransferFunctionRenderer(void);
	~TransferFunctionRenderer(void);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(CallRender2D& call);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(CallRender2D& call);

	/**
	 * Implementation of 'Create'.
	 *
	 * @return 'true' on success, 'false' otherwise.
	 */
	bool create(void);

	/**
	 * Implementation of 'Release'.
	 */
	void release(void);

	
    /** The render callee slot */
	CallerSlot getTFSlot;
	vislib::graphics::gl::SimpleFont *ctFont;

};

}
}
}