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

#include <string>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::sombreros;

/*
 * CaverTunnelResidueLoader::CaverTunnelResidueLoader
 */
CaverTunnelResidueLoader::CaverTunnelResidueLoader(void) : Module(),
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
 * CaverTunnelResidueLoader::create
 */
bool CaverTunnelResidueLoader::create(void) {
	// TODO
	return true;
}


/*
 * CaverTunnelResidueLoader::release
 */
void CaverTunnelResidueLoader::release(void) {
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

		if (this->tunnelVector.empty()) {
			// should not happen
			break;
		}

		// parse the line
		line.TrimSpaces();
		if (line.IsEmpty()) continue;

		// split the line into the different parts
		std::vector<vislib::StringA> parts = splitLine(line);
		TunnelResidueDataCall::Tunnel * tp = &this->tunnelVector[this->tunnelVector.size() - 1];

		if (parts.size() > 5) { // only in this case we have atom indices
			tp->atomNumbers.push_back(static_cast<int>(parts.size() - 5));
			tp->firstAtomIndices.push_back(static_cast<int>(tp->atomIdentifiers.size()));
			// parse the small entries of the form <snapshots>:<element_symbol>_<serial_number>
			for (int i = 5; i < static_cast<int>(parts.size()); i++) {
				// TODO this may be dangerous...
				auto numberString = splitLine(parts[i], ':')[0];
				auto idxString = splitLine(parts[i], '_')[1];
				tp->atomIdentifiers.push_back(std::pair<int, int>(std::stoi(idxString.PeekBuffer()), std::stoi(numberString.PeekBuffer())));
			}
		}
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
	TunnelResidueDataCall * trdc = dynamic_cast<TunnelResidueDataCall*>(&caller);
	if (trdc == nullptr) return false;

	trdc->setTunnelNumber(static_cast<int>(this->tunnelVector.size()));
	trdc->setTunnelDescriptions(this->tunnelVector.data());

	return true;
}

/*
 * CaverTunnelResidueLoader::getExtentCallback
 */
bool CaverTunnelResidueLoader::getExtentCallback(core::Call& caller) {
	TunnelResidueDataCall * trdc = dynamic_cast<TunnelResidueDataCall*>(&caller);

	if (trdc != nullptr) {
		trdc->SetFrameCount(1); // TODO
		trdc->setTunnelNumber(static_cast<int>(this->tunnelVector.size()));
		trdc->AccessBoundingBoxes().Clear();
		trdc->SetDataHash(this->data_hash);
		return true;
	}

	return false;
}