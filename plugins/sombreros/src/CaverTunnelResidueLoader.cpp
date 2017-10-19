/*
 * CaverTunnelResidueLoader.cpp
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "CaverTunnelResidueLoader.h"

#include "mmcore/param/FloatParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::sombreros;

/********************************************************************************************************************************************/

/*
 * CaverTunnelResidueLoader::Frame::Frame
 */
CaverTunnelResidueLoader::Frame::Frame(view::AnimDataModule& owner) : 
	view::AnimDataModule::Frame(owner), dat() {
	// intentionally empty
}

/*
 * CaverTunnelResidueLoader::Frame::~Frame
 */
CaverTunnelResidueLoader::Frame::~Frame(void) {
	this->Clear();
}

/*
 * CaverTunnelResidueLoader::Frame::LoadFrame
 */
bool CaverTunnelResidueLoader::Frame::LoadFrame(vislib::sys::File * file, unsigned int idx, UINT64 size, unsigned int version) {
	this->frame = idx;
	this->fileVersion = version;
	this->dat.EnforceSize(static_cast<SIZE_T>(size));
	return (file->Read(this->dat, size) == size);
}

/*
 * CaverTunnelResidueLoader::Frame::SetData
 */
void CaverTunnelResidueLoader::Frame::SetData(TunnelResidueDataCall& call) {
	// TODO
}

/********************************************************************************************************************************************/

/*
 * CaverTunnelResidueLoader::CaverTunnelResidueLoader
 */
CaverTunnelResidueLoader::CaverTunnelResidueLoader(void) : view::AnimDataModule(),
		getData("getData", "The slot providing the data loaded by this module."),
		filenameSlot("filename", "The path to the input file.") {

	this->filenameSlot.SetParameter(new param::FilePathParam(""));
	this->filenameSlot.SetUpdateCallback(&CaverTunnelResidueLoader::filenameChanged);
	this->MakeSlotAvailable(&this->filenameSlot);

	this->getData.SetCallback(TunnelResidueDataCall::ClassName(), TunnelResidueDataCall::FunctionName(0), &CaverTunnelResidueLoader::getDataCallback);
	this->getData.SetCallback(TunnelResidueDataCall::ClassName(), TunnelResidueDataCall::FunctionName(1), &CaverTunnelResidueLoader::getExtentCallback);
	this->MakeSlotAvailable(&this->getData);
}

/*
 * CaverTunnelResidueLoader::~CaverTunnelResidueLoader
 */
CaverTunnelResidueLoader::~CaverTunnelResidueLoader(void) {
	this->Release();
}

/*
 * CaverTunnelResidueLoader::constructFrame
 */
core::view::AnimDataModule::Frame* CaverTunnelResidueLoader::constructFrame(void) const {
	Frame * f = new Frame(*const_cast<CaverTunnelResidueLoader*>(this));
	return f;
}

/*
 * CaverTunnelResidueLoader::create
 */
bool CaverTunnelResidueLoader::create(void) {
	// TODO
	return true;
}

/*
 * CaverTunnelResidueLoader::loadFrame
 */
void CaverTunnelResidueLoader::loadFrame(core::view::AnimDataModule::Frame * frame, unsigned int idx) {
	using vislib::sys::Log;
	Frame * f = dynamic_cast<Frame*>(frame);
	if (f == nullptr) return;
	if (this->file == nullptr) {
		f->Clear();
		return;
	}
	ASSERT(idx < this->FrameCount());
	// TODO
	//this->file->Seek(this->frameIdx[idx]);
	//if (!f->LoadFrame(this->file, idx, this->frameIdx[idx + 1] - this->frameIdx[idx], this->fileVersion)) {
	//	// failed
	//	Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to read frame %d from MMPLD file\n", idx);
	//}
}

/*
 * CaverTunnelResidueLoader::release
 */
void CaverTunnelResidueLoader::release(void) {
	// TODO
}

/*
 * CaverTunnelResidueLoader::filenameChanged
 */
bool CaverTunnelResidueLoader::filenameChanged(core::param::ParamSlot& slot) {
	// TODO
	return true;
}

/*
 * CaverTunnelResidueLoader::getDataCallback
 */
bool CaverTunnelResidueLoader::getDataCallback(core::Call& caller) {
	// TODO
	return true;
}

/*
 * CaverTunnelResidueLoader::getExtentCallback
 */
bool CaverTunnelResidueLoader::getExtentCallback(core::Call& caller) {
	// TODO
	return true;
}