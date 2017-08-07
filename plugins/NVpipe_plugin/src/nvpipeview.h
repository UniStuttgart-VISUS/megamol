/*
 * nvpipeview.h
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/view/View3D.h"
#include "mmcore/param/ParamSlot.h"
#include <thread>
#include <atomic>


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
	* Implementation of 'Release'.
	*/
	virtual void release(void);

	/**
	* NvPipe encoding parameters
	*/
	core::param::ParamSlot paramClipMachine;
	bool isClipMachine;
	cudaGraphicsResource_t graphicsResource;
	cudaArray_t serverArray;
	::nvpipe* encoder;
	void* deviceBuffer;
	size_t deviceBufferSize;
	size_t numBytes;
	param::ParamSlot serverNameSlot;
	param::ParamSlot portSlot;
	nvpipe::socket socket;

	// thread stuff
	std::thread sender;
	param::ParamSlot queueLength;

	// ringbuffer stuff
	std::vector<vislib::RawStorage> sendQueue;

	/** The index of the next element to be read. */
	std::atomic<size_t> curRead;

	/** The index of the next element to be written. */
	std::atomic<size_t> curWrite;

	// Camera manipulation
	vislib::SmartPtr<vislib::graphics::CameraParameters> offscreenOverride;
	vislib::graphics::ImageSpaceRectangle offscreenTile;
	vislib::graphics::ImageSpaceDimension offscreenSize;

	// encoded FBO
	vislib::graphics::gl::FramebufferObject fbo;

	/**
	* Increments the given index, honouring the size of the ring buffer.
	*
	* @param idx The current index.
	*
	* @return The next index.
	*/
	inline size_t advanceIndex(const size_t idx) const {
		return ((idx + 1) % this->sendQueue.size());
	}

	void doSend(void);
};


} /* end namespace view */
} /* end namespace nvpipe*/

