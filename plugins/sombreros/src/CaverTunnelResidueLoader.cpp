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

#include "vislib/sys/FastFile.h"
#include "vislib/sys/TextFileReader.h"
#include "vislib/sys/Log.h"

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
		filenameSlot("filename", "The path to the input file."),
		data_hash(0) {

	this->filenameSlot.SetParameter(new param::FilePathParam(""));
	this->filenameSlot.SetUpdateCallback(&CaverTunnelResidueLoader::filenameChanged);
	this->MakeSlotAvailable(&this->filenameSlot);

	this->getData.SetCallback(TunnelResidueDataCall::ClassName(), TunnelResidueDataCall::FunctionName(0), &CaverTunnelResidueLoader::getDataCallback);
	this->getData.SetCallback(TunnelResidueDataCall::ClassName(), TunnelResidueDataCall::FunctionName(1), &CaverTunnelResidueLoader::getExtentCallback);
	this->MakeSlotAvailable(&this->getData);

	this->file = nullptr;
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
	this->resetFrameCache();
	if (this->file != nullptr) {
		vislib::sys::File * f = this->file;
		this->file = nullptr;
		f->Close();
		delete f;
	}
}

/*
 * CaverTunnelResidueLoader::filenameChanged
 */
bool CaverTunnelResidueLoader::filenameChanged(core::param::ParamSlot& slot) {
	using vislib::sys::Log;
	using vislib::sys::File;

	this->resetFrameCache();
	this->data_hash++;

	if (this->file == nullptr) {
		this->file = new File();
	} else {
		this->file->Close();
	}
	ASSERT(this->filenameSlot.Param<param::FilePathParam>() != nullptr);

	if (!this->file->Open(this->filenameSlot.Param<param::FilePathParam>()->Value(), File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY)) {
		Log::DefaultLog.WriteError("Unable to open tunnel-File \"%s\".", vislib::StringA(
			this->filenameSlot.Param<param::FilePathParam>()->Value()).PeekBuffer());
		SAFE_DELETE(this->file);
		this->setFrameCount(1);
		this->initFrameCache(1);
		return true;
	}

	vislib::sys::TextFileReader fileReader(this->file);
	this->tunnelVector.clear();

	vislib::StringA line;
	while (fileReader.ReadLine(line)) {
		if (line.StartsWith('#')) { // comment line
			continue;
		}
		if (line.StartsWith('=')) { // new tunnel
			this->tunnelVector.push_back(TunnelResidueDataCall::Tunnel());
			continue;
		}

		// parse the line
		line.TrimSpaces();
		if (line.IsEmpty()) continue;

		// split the line into the different parts
		std::vector<vislib::StringA> parts = splitLine(line);
	}

	return true;
}

/*
 * CaverTunnelResidueLoader::splitLine
 */
std::vector<vislib::StringA> CaverTunnelResidueLoader::splitLine(vislib::StringA line, char splitChar) {
	std::vector<vislib::StringA> result;
	line.TrimSpaces();
	if (line.IsEmpty()) return result;

	// special case when there is only one word in the line
	if (line.Find(vislib::StringA(std::string(1, splitChar).c_str())) == line.INVALID_POS) {
		result.push_back(line);
		return result;
	}

	int pos = 0;
	while (pos != line.INVALID_POS) {
		int newpos = line.Find(std::string(1, splitChar).c_str(), pos);
		if (newpos != pos || newpos == line.INVALID_POS) {
			vislib::StringA s;
			if (newpos == line.INVALID_POS) {
				s = line.Substring(pos);
			} else {
				s = line.Substring(pos, newpos - pos);
			}
			s.TrimSpaces();
			if (!s.IsEmpty()) {
				result.push_back(s);
			}
		} else {
			newpos++;
		}
		pos = newpos;
	}
	return result;
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