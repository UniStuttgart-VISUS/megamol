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
#include "vislib/sys/SystemInformation.h"
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
#include <cstdint>

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
paramClipMachine("clipMachine", "Clips everything on the specified machine"),
queueLength("queueLength", "Defines the size of the sending ringbuffer"),
offscreenOverride(new vislib::graphics::CameraParamsStore),
isClipMachine(false),
sleep(0), fullLoop(0),
isInitialized(false),
curRead(0), curWrite(0){

	this->serverNameSlot << new param::StringParam("");
	this->MakeSlotAvailable(&this->serverNameSlot);

	this->portSlot << new param::IntParam(0);
	this->MakeSlotAvailable(&this->portSlot);

	this->paramClipMachine << new core::param::StringParam("");
	this->MakeSlotAvailable(&this->paramClipMachine);

	this->queueLength << new core::param::IntParam(5);
	this->MakeSlotAvailable(&this->queueLength);
}


nvpipe::NVpipeView::~NVpipeView(void) {
	// empty
}



/*
* nvpipe::NVpipeView::Render
*/
void nvpipe::NVpipeView::Render(const mmcRenderViewContext& context) {

	// Determine whether the machine is completely excluded from rendering and 
	// remember the result for skipping the rendering later on.
	if (this->paramClipMachine.IsDirty()) {
		auto machines = vislib::StringTokeniserA::Split(
			this->paramClipMachine.Param<core::param::StringParam>()->Value(), ",");
		vislib::TString machine;
		vislib::sys::SystemInformation::ComputerName(machine);

		for (SIZE_T i = 0; i < machines.Count(); ++i) {
			machines[i].TrimSpaces();
			if (machine.Equals(machines[i], false)) {
				this->isClipMachine = true;
				break;
			}
		}
		this->paramClipMachine.ResetDirty();
	}

	if (!this->isClipMachine) {

	// #########################################################################################

		/*
		* init Network, encoder and the HAMMER
		*/
		if (!this->isInitialized) {
			using namespace vislib::graphics;
			/*
			* Init socket Connection
			*/
			std::string sn = (const char*)T2A(this->serverNameSlot.Param<param::StringParam>()->Value());
      
			uint32_t pt = this->portSlot.Param<param::IntParam>()->Value();
			std::string pt_str = std::to_string(pt);

			Log::DefaultLog.WriteInfo("Connecting to %hs:%hs ...",
				sn.c_str(), pt_str.c_str());

			// try to start up socket
			vislib::net::Socket::Startup();
			// create socket
			this->socket.Create(vislib::net::Socket::FAMILY_INET, vislib::net::Socket::TYPE_STREAM, vislib::net::Socket::PROTOCOL_TCP);
			// connect socket
			this->socket.Connect(vislib::net::IPEndPoint::CreateIPv4(sn.c_str(), pt));
			Log::DefaultLog.WriteInfo("Connecting to %hs:%hs - success.",
				sn.c_str(), pt_str.c_str());
			// receive initial buffer - ltrbwh
			std::array<int32_t, 6> bounds;
			Log::DefaultLog.WriteInfo("Receiving bounds data ...");
			this->socket.Receive(bounds.data(), bounds.size() * sizeof(bounds[0]));
			std::transform(bounds.begin(), bounds.end(), bounds.begin(), ::ntohl);
			Log::DefaultLog.WriteInfo("Receiving bounds data - complete.");
			Log::DefaultLog.WriteInfo("left: %i top: %i right: %i bottom: %i width: %i height: %i",
				bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]);


			// Init the HAMMER
			this->offscreenTile = ImageSpaceRectangle(bounds[0], bounds[5] - bounds[3],
				bounds[2], bounds[5] - bounds[1]);
			this->offscreenSize = ImageSpaceDimension(bounds[4], bounds[5]);

			this->deviceBufferSize = this->offscreenTile.Width() * this->offscreenTile.Height() * 4;

			Log::DefaultLog.WriteInfo("Loading all GL extensions ...");
			vislib::graphics::gl::LoadAllGL();
			Log::DefaultLog.WriteInfo("Creating FrameBufferObject for NVPipe with [%f, %f] px ...",
				this->offscreenTile.Width(), this->offscreenTile.Height());
			this->fbo.Create(this->offscreenTile.Width(), this->offscreenTile.Height(), GL_RGBA, GL_RGBA);

			/*
			* Init Encoder
			*/
			const uint64_t bitrate = this->deviceBufferSize * 30 * 0.07;

			Log::DefaultLog.WriteInfo("Creating NVPipe encoder with %i bps ...",
				bitrate);
			encoder = nvpipe_create_encoder(NVPIPE_H264_NV, bitrate);
			if (encoder == NULL) {
				throw vislib::Exception("Creating NVPipe encoder - failed", __FILE__, __LINE__);
			}

			Log::DefaultLog.WriteInfo("Allocating device buffer with %u bytes ...",
				deviceBufferSize);
			if (cudaMalloc(&deviceBuffer, deviceBufferSize) != cudaSuccess) {
				throw vislib::Exception("Failed to allocate device memory", __FILE__, __LINE__);
			}

			auto ql = this->queueLength.Param<param::IntParam>()->Value();
			if (ql < 2) ql = 2;

			Log::DefaultLog.WriteInfo("Ringbuffer has size %i.", ql);
			this->sendQueue.resize(ql);
			
			for (auto& sq : this->sendQueue) {
				sq.AssertSize(deviceBufferSize + sizeof(uint32_t));
			}

			Log::DefaultLog.WriteInfo("Starting sender thread ...");
			this->sender = std::thread(&NVpipeView::doSend, this);


			Log::DefaultLog.WriteInfo("Registering FBO with CUDA ...");
			auto returnValue = cudaGraphicsGLRegisterImage(&graphicsResource, this->fbo.GetColourTextureID(),
				GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
			if (returnValue != cudaSuccess) {
			  Log::DefaultLog.WriteError("Registering FBO with CUDA failed with error code %i", returnValue);
			  throw vislib::Exception("Registering FBO with CUDA - failed", __FILE__, __LINE__);
			}
			this->isInitialized = true;
		} // end init Network, encoder and the HAMMER


		/*
		* Check if ringbuffer is full and sleep untill it there is free space again
		*/

		if (this->fullLoop < -1) {
			this->sleep *= 0.9;
		}

		auto cur = this->curWrite.load();
		auto nxt = this->advanceIndex(cur);

		if (nxt == this->curRead.load()) {
			Log::DefaultLog.WriteWarn("SendQueue full ...");
			auto zero_mus = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::duration<uint32_t>(0));
			auto t0 = std::chrono::high_resolution_clock::now();
			this->fullLoop = 0;
			while (nxt == this->curRead.load()) {
				if (this->sleep != zero_mus) {
					Log::DefaultLog.WriteWarn("Sleeping for %i microseconds ...", this->sleep);
					std::this_thread::sleep_for(this->sleep);
					this->fullLoop++;
				}
			}
			if (this->sleep == zero_mus) {
				auto t1 = std::chrono::high_resolution_clock::now();
				auto dur = t1 - t0;
				this->sleep = std::chrono::duration_cast<std::chrono::microseconds>(dur);
			} else {
				if (this->fullLoop > 1) {
					this->sleep *= 1.1;
				}
			}
			this->fullLoop = 0;
		} else {
			this->fullLoop--;
		}// end if queue full

			

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
		} // end  Override camera and render image into FBO

		/*
		* Grab frame and encode
		*/
		{
			cudaGraphicsMapResources(1, &graphicsResource);
			cudaGraphicsSubResourceGetMappedArray(&serverArray, graphicsResource, 0, 0);
			cudaMemcpy2DFromArray(this->deviceBuffer, this->offscreenTile.Width() * 4, serverArray, 0, 0,
				this->offscreenTile.Width() * 4, this->offscreenTile.Height(), cudaMemcpyDeviceToDevice);
			cudaGraphicsUnmapResources(1, &graphicsResource);

			Log::DefaultLog.WriteInfo("Starting NVPipe encode ...");
			auto cntEncoded = static_cast<size_t>(this->sendQueue[cur].GetSize()) - sizeof(uint32_t);
			nvp_err_t encodeStatus = nvpipe_encode(this->encoder,
				this->deviceBuffer, this->deviceBufferSize,
				this->sendQueue[cur].At(sizeof(uint32_t)), &cntEncoded,
				this->offscreenTile.Width(), this->offscreenTile.Height(), NVPIPE_RGBA);
			if (encodeStatus != NVPIPE_SUCCESS) {
				Log::DefaultLog.WriteError("NVPipe encode failed with error code %i.", encodeStatus);
				throw vislib::Exception("Encode failed", __FILE__, __LINE__);
			}
			*this->sendQueue[cur].As<uint32_t>() = cntEncoded;

			this->curWrite = nxt;

		} // end Grab frame and encode


    // ##################################################################################################################

	} else {
		View3D::Render(context);
	} // end if isClipMachine
}


void nvpipe::NVpipeView::release(void) {
	try {
		this->socket.Close();
	}
	catch (...) {
		Log::DefaultLog.WriteWarn("Close connection failed.");
	}
	if (this->sender.joinable()) {
		this->sender.join();
	}
}

void nvpipe::NVpipeView::doSend(void) {
	Log::DefaultLog.WriteInfo("NVPipe sender thread is running.");

	try {
		while (true) {

			auto cur = this->curRead.load();

			if (cur != this->curWrite.load()) {
				auto len = *this->sendQueue[cur].As<uint32_t>() + sizeof(uint32_t);
				// Send frame
				this->sendFrame(len, this->sendQueue[cur].As<uint8_t>());
				this->curRead = this->advanceIndex(cur);
			} else {
				std::this_thread::yield();
			}
		}
	}
	catch (...) {
		Log::DefaultLog.WriteInfo("NVPipe thread is exiting.");
	}

}

void nvpipe::NVpipeView::sendFrame(size_t numBytes, uint8_t* sendBuffer) {
	uint32_t nl_numBytes = ::htonl(numBytes);
	
	// send encoded size
	this->socket.Send(&nl_numBytes, sizeof(nl_numBytes));
	// send encoded frame
	this->socket.Send(sendBuffer, numBytes);
}
