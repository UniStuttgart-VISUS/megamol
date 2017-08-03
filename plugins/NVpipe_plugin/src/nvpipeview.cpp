/*
* nvpipeview.cpp
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/



#include "stdafx.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/CameraParamsTileRectOverride.h"
#include "vislib/graphics/CameraParamsVirtualViewOverride.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "vislib/sys/Log.h"

#include "nvpipeview.h"

#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <array>


using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::view;
using vislib::sys::Log;


/*
* nvpipe::NVpipeView::NVpipeView
*/
nvpipe::NVpipeView::NVpipeView(void) : View3D(), socket(),
serverNameSlot("serverName", "Server to push encoded stream to"),
portSlot("port", "Communication port"),
offscreenOverride(new vislib::graphics::CameraParamsStore) {

	this->serverNameSlot << new param::StringParam("");
	this->MakeSlotAvailable(&this->serverNameSlot);

	this->portSlot << new param::IntParam(0);
	this->MakeSlotAvailable(&this->portSlot);
}


nvpipe::NVpipeView::~NVpipeView(void) {
	// empty
}



/*
* nvpipe::NVpipeView::Render
*/
void nvpipe::NVpipeView::Render(const mmcRenderViewContext& context) {

	/*
	* init Network stuff and camera
	*/

	if (!this->socket.isInitialized()) {
		using namespace vislib::graphics;

		/*
		* Init Connection
		*/
		std::string sn = this->serverNameSlot.Param<param::StringParam>()->Value();
		std::string pt_str = std::to_string(this->portSlot.Param<param::IntParam>()->Value());
		PCSTR pt = (PCSTR)pt_str.c_str();

		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Connecting to %hs:%hs.",
			sn.c_str(), pt_str.c_str());
		this->socket.init(sn, pt);
		this->socket.connect();
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Connecting to %hs:%hs - success.",
			sn.c_str(), pt_str.c_str());
		// receive initial buffer - ltrbwh
		std::array<int32_t, 6> bounds;
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Receiving bounds data.");
		this->socket.receive(bounds.data(), bounds.size() * sizeof(bounds[0]));
		std::transform(bounds.begin(), bounds.end(), bounds.begin(), ::ntohl);
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Receiving bounds data - complete.");
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "left: %i top: %i right: %i bottom: %i width: %i height: %i",
			bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]);


		// Init the HAMMER
		this->offscreenTile = ImageSpaceRectangle(bounds[0], bounds[5] - bounds[3],
			bounds[2], bounds[5] - bounds[1]);
		this->offscreenSize = ImageSpaceDimension(bounds[4], bounds[5]);

		this->deviceBufferSize = this->offscreenTile.Width() * this->offscreenTile.Height() * 4;

		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Creating FrameBufferObject for NVPipe with [%f, %f] px.",
			this->offscreenTile.Width(), this->offscreenTile.Height());
		this->fbo.Create(this->offscreenTile.Width(), this->offscreenTile.Height(), GL_RGBA, GL_RGBA);

		/*
		* Init Encoder
		*/
		const uint64_t bitrate = this->deviceBufferSize * 30 * 0.07;
		
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Creating NVPipe encoder with %i bps.",
			bitrate);
		encoder = nvpipe_create_encoder(NVPIPE_H264_NV, bitrate);
		if (encoder == NULL) {
			throw vislib::Exception("Creating NVPipe encoder - failed", __FILE__, __LINE__);
		}
		
		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Allocating device buffer with %u bytes.",
			deviceBufferSize);
		if (cudaMalloc(&deviceBuffer, deviceBufferSize) != cudaSuccess) {
			throw vislib::Exception("Failed to allocate device memory", __FILE__, __LINE__);
		}

		sendBufferSize = deviceBufferSize; // Reserve enough space for encoded output
		sendBuffer = new uint8_t[sendBufferSize];

		Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Registering FBO with CUDA .");
		auto returnValue = cudaGraphicsGLRegisterImage(&graphicsResource, this->fbo.GetColourTextureID(),
			GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
		if (returnValue != cudaSuccess) {
			throw vislib::Exception("Registering FBO with CUDA - failed", __FILE__, __LINE__);
		}
	}

	/*
	* Override camera and render image into FBO 
	*/
	{
		// Use the HAMMER on camera
		auto oldcam = this->cam.Parameters();
		this->offscreenOverride->CopyFrom(oldcam);
		this->offscreenOverride->SetVirtualViewSize(this->offscreenSize);
		this->offscreenOverride->SetTileRect(this->offscreenTile);
		this->cam.SetParameters(this->offscreenOverride);


		if (this->fbo.IsValid()) {
			// Redirect output to CUDA-mapped FBO
			this->fbo.Enable();
		}

		View3D::Render(context);

		// Undo the HAMMER
		if (this->fbo.IsValid()) {
			this->fbo.Disable();
			::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			// Not the HAMMER
			//this->fbo.BindColourTexture();
			//glTexEnvi(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPLACE);
			this->fbo.DrawColourTexture();
		}
		this->cam.SetParameters(oldcam);
	}

	/*
	* Grab frame and encode
	*/
	{
		cudaGraphicsMapResources(1, &graphicsResource);
		cudaGraphicsSubResourceGetMappedArray(&serverArray, graphicsResource, 0, 0);
		cudaMemcpy2DFromArray(this->deviceBuffer, this->offscreenTile.Width() * 4, serverArray, 0, 0,
			this->offscreenTile.Width() * 4, this->offscreenTile.Height(), cudaMemcpyDeviceToDevice);
		cudaGraphicsUnmapResources(1, &graphicsResource);


		numBytes = this->sendBufferSize;
		nvp_err_t encodeStatus = nvpipe_encode(this->encoder, this->deviceBuffer, this->deviceBufferSize,
			sendBuffer, &numBytes, this->offscreenTile.Width(), this->offscreenTile.Height(), NVPIPE_RGBA);
		if (encodeStatus != NVPIPE_SUCCESS) {
			Log::DefaultLog.WriteError(Log::LEVEL_ERROR, "NVPipe encode failed with error code %i.", encodeStatus);
			throw std::exception("Encode failed");
		}

		// Send frame
		this->socket.sendFrame(numBytes, sendBuffer);
	}
}