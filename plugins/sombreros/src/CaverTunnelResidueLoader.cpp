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
		filenameSlot("filename", "The path to the residues.txt input file produced by caver."),
		tunnelFilenameSlot("tunnelFilename", "The path to the tunnel_profiles.csv input file produced by caver."),
		data_hash(0) {

	this->filenameSlot.SetParameter(new param::FilePathParam(""));
	this->filenameSlot.SetUpdateCallback(&CaverTunnelResidueLoader::filenameChanged);
	this->MakeSlotAvailable(&this->filenameSlot);

	this->tunnelFilenameSlot.SetParameter(new param::FilePathParam(""));
	this->tunnelFilenameSlot.SetUpdateCallback(&CaverTunnelResidueLoader::filenameChanged);
	this->MakeSlotAvailable(&this->tunnelFilenameSlot);

	this->getData.SetCallback(TunnelResidueDataCall::ClassName(), TunnelResidueDataCall::FunctionName(0), &CaverTunnelResidueLoader::getDataCallback);
	this->getData.SetCallback(TunnelResidueDataCall::ClassName(), TunnelResidueDataCall::FunctionName(1), &CaverTunnelResidueLoader::getExtentCallback);
	this->MakeSlotAvailable(&this->getData);

	this->file = nullptr;
	this->tunnelFile = nullptr;
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
	if (this->tunnelFile != nullptr) {
		vislib::sys::File * f = this->tunnelFile;
		this->tunnelFile = nullptr;
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

	/*********************************************** Read Residues file ******************************************************/
	if (slot.Name().Equals(vislib::StringA("filename"))) {

		if (this->file == nullptr) {
			this->file = new File();
		}
		else {
			this->file->Close();
		}
		ASSERT(this->filenameSlot.Param<param::FilePathParam>() != nullptr);

		if (!this->file->Open(this->filenameSlot.Param<param::FilePathParam>()->Value(), File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY)) {
			Log::DefaultLog.WriteError("Unable to open residues-File \"%s\".", vislib::StringA(
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
	}

	/*********************************************** Read Tunnel file ******************************************************/

	if (slot.Name().Equals(vislib::StringA("tunnelFilename"))) {

		if (this->tunnelFile == nullptr) {
			this->tunnelFile = new File();
		}
		else {
			this->tunnelFile->Close();
		}
		ASSERT(this->tunnelFilenameSlot.Param<param::FilePathParam>() != nullptr);

		if (!this->tunnelFile->Open(this->tunnelFilenameSlot.Param<param::FilePathParam>()->Value(), File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY)) {
			Log::DefaultLog.WriteError("Unable to open tunnel-File \"%s\".", vislib::StringA(
				this->tunnelFilenameSlot.Param<param::FilePathParam>()->Value()).PeekBuffer());
			SAFE_DELETE(this->file);
			return true;
		}

		vislib::sys::TextFileReader tunnelFileReader(this->tunnelFile);

		vislib::StringA line;
		bool firstline = true;
		while (tunnelFileReader.ReadLine(line)) {
			if (firstline) { // ignore the first line
				firstline = false;
				continue;
			}

			// TODO multiple snapshots

			auto values = splitLine(line, ',');
			 // 12 eintraege + werte
			// read number of clusters
			int clusterNum = std::stoi(values[1].PeekBuffer());

			// read tunnel number
			int tunnelNum = std::stoi(values[2].PeekBuffer());

			if (tunnelNum > static_cast<int>(this->tunnelVector.size())) {
				// no tunnel exists
				Log::DefaultLog.WriteWarn("The tunnel with index %i does not exist, all values for it will be ignored", tunnelNum);
				continue;
			}

			std::vector<float> * tp = &this->tunnelVector[tunnelNum - 1].coordinates;
			if (tp->size() != (values.size() - 12) * 4) { // 4 entries for each vertex
				tp->resize((values.size() - 12) * 4, 0.0f);
			}

			if (values[11].Equals(vislib::StringA("X"))) {
				for (int i = 12; i < static_cast<int>(values.size()); i++) {
					tp->operator[]((i - 12) * 4) = static_cast<float>(std::stof(values[i].PeekBuffer()));
				}
			} else if (values[11].Equals(vislib::StringA("Y"))) {
				for (int i = 12; i < static_cast<int>(values.size()); i++) {
					tp->operator[]((i - 12) * 4 + 1) = static_cast<float>(std::stof(values[i].PeekBuffer()));
				}
			} else if (values[11].Equals(vislib::StringA("Z"))) {
				for (int i = 12; i < static_cast<int>(values.size()); i++) {
					tp->operator[]((i - 12) * 4 + 2) = static_cast<float>(std::stof(values[i].PeekBuffer()));
				}
			} else if (values[11].Equals(vislib::StringA("R"))) {
				for (int i = 12; i < static_cast<int>(values.size()); i++) {
					tp->operator[]((i - 12) * 4 + 3) = static_cast<float>(std::stof(values[i].PeekBuffer()));
				}
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