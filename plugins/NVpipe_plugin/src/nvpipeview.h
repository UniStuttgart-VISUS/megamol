/*
 * nvpipeview.h
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/view/View3D.h"
#include "mmcore/param/ParamSlot.h"


 // NVpipe header
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "nvpipe.h"
#include "socket.h"

using namespace megamol::core;

namespace megamol {
namespace nvpipe {


/**
* Base class of rendering graph calls
*/
class  NVpipeView : public view::View3D {

public:

	/**
	* Answer the name of this module.
	*
	* @return The name of this module.
	*/
	static const char *ClassName(void) {
		return "NVpipeView";
	}

	/**
	* Answer a human readable description of this module.
	*
	* @return A human readable description of this module.
	*/
	static const char *Description(void) {
		return "3D View Module with NVpipe encoding";
	}


	/** Ctor. */
	NVpipeView(void);

	/** Dtor. */
	virtual ~NVpipeView(void);



	/**
	* Renders this AbstractView3D in the currently active OpenGL context.
	*
	* @param context
	*/
	virtual void Render(const mmcRenderViewContext& context);



protected:

	/**
	* NvPipe encoding parameters
	*/
	cudaGraphicsResource_t graphicsResource;
	cudaArray_t serverArray;
	::nvpipe* encoder;
	size_t sendBufferSize;
	void* deviceBuffer;
	uint8_t* sendBuffer;
	size_t deviceBufferSize;
	size_t numBytes;
	param::ParamSlot serverNameSlot;
	param::ParamSlot portSlot;
	nvpipe::socket socket;


	// Camera manipulation
	vislib::SmartPtr<vislib::graphics::CameraParameters> offscreenOverride;
	vislib::graphics::ImageSpaceRectangle offscreenTile;
	vislib::graphics::ImageSpaceDimension offscreenSize;

	// encoded FBO
	vislib::graphics::gl::FramebufferObject fbo;


};


} /* end namespace view */
} /* end namespace nvpipe*/

