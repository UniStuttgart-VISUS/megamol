/*
 * ParticleBoxGeneratorDataSource.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ParticleBoxGeneratorDataSource.h"
#include <climits>
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"


using namespace megamol;
using namespace megamol::stdplugin::datatools;

namespace {
	enum MyColorType : int {
		COLOR_NONE = 0,
		COLOR_RGBu8,
		COLOR_RGBAu8,
		COLOR_If,
		COLOR_RGBf,
		COLOR_RGBAf
	};
}


/*
 * ParticleBoxGeneratorDataSource::ParticleBoxGeneratorDataSource
 */
ParticleBoxGeneratorDataSource::ParticleBoxGeneratorDataSource(void) : core::Module(),
		particleCountSlot("count", "Number of particles to be generated"),
		radiusPerParticleSlot("store::explicitRadius", "Flag to explicitly store radii at each particle"),
		colorDataSlot("store::color", "Type of color information to be generated"),
		interleavePosAndColorSlot("store::interleaved", "Flag to interleave position and color information"),
		radiusScaleSlot("radiusScale", "Scale factor for particle radii"),
		positionNoiseSlot("positionNoise", "Amount of noise for the position values"),
		dataHash(0) {
	particleCountSlot.SetParameter(new core::param::IntParam(1000, 0));
	MakeSlotAvailable(&particleCountSlot);

	radiusPerParticleSlot.SetParameter(new core::param::BoolParam(false));
	MakeSlotAvailable(&radiusPerParticleSlot);

	core::param::EnumParam *colType = new core::param::EnumParam(0);
	colType->SetTypePair(COLOR_NONE, "none");
	colType->SetTypePair(COLOR_RGBu8, "RGB (bytes)");
	colType->SetTypePair(COLOR_RGBAu8, "RGBA (bytes)");
	colType->SetTypePair(COLOR_If, "Lum (float)");
	colType->SetTypePair(COLOR_RGBf, "RGB (floats)");
	colType->SetTypePair(COLOR_RGBAf, "RGBA (floats)");
	colorDataSlot.SetParameter(colType);
	MakeSlotAvailable(&colorDataSlot);

	interleavePosAndColorSlot.SetParameter(new core::param::BoolParam(true));
	MakeSlotAvailable(&interleavePosAndColorSlot);

	radiusScaleSlot.SetParameter(new core::param::FloatParam(1.0f, 0.0f));
	MakeSlotAvailable(&radiusScaleSlot);

	positionNoiseSlot.SetParameter(new core::param::FloatParam(0.0f, 0.0f));
	MakeSlotAvailable(&positionNoiseSlot);
}


/*
 * ParticleBoxGeneratorDataSource::~ParticleBoxGeneratorDataSource
 */
 ParticleBoxGeneratorDataSource::~ParticleBoxGeneratorDataSource(void) {
    this->Release();
}


/*
 * ParticleBoxGeneratorDataSource::create
 */
 bool ParticleBoxGeneratorDataSource::create(void) {
    // intentionally empty
    return true;
}


/*
 * ParticleBoxGeneratorDataSource::release
 */
 void ParticleBoxGeneratorDataSource::release(void) {
	 this->clear();
 }


/*
 * ParticleBoxGeneratorDataSource::getDataCallback
 */
 bool ParticleBoxGeneratorDataSource::getDataCallback(core::Call& caller) {
    core::moldyn::MultiParticleDataCall *mpdc = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&caller);
    if (mpdc == nullptr) return false;
	if (particleCountSlot.IsDirty() || radiusPerParticleSlot.IsDirty() || colorDataSlot.IsDirty()
			|| interleavePosAndColorSlot.IsDirty() || radiusScaleSlot.IsDirty() || positionNoiseSlot.IsDirty()) {
		this->assertData();
	}

	mpdc->SetParticleListCount(1);
	mpdc->SetDataHash(dataHash);
	mpdc->SetExtent(1, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
	core::moldyn::MultiParticleDataCall::Particles& parties = mpdc->AccessParticles(0);

	// TODO: Implement

    //if (this->colourSlot.IsDirty()) {
    //    this->colourSlot.ResetDirty();
    //    float r, g, b;
    //    if (core::utility::ColourParser::FromString(this->colourSlot.Param<core::param::StringParam>()->Value(), r, g, b)) {
    //        this->defCol[0] = static_cast<unsigned char>(vislib::math::Clamp<int>(static_cast<int>(r * 255.0f), 0, 255));
    //        this->defCol[1] = static_cast<unsigned char>(vislib::math::Clamp<int>(static_cast<int>(g * 255.0f), 0, 255));
    //        this->defCol[2] = static_cast<unsigned char>(vislib::math::Clamp<int>(static_cast<int>(b * 255.0f), 0, 255));
    //    } else {
    //        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
    //            "Unable to parse default colour \"%s\"\n",
    //            vislib::StringA(this->colourSlot.Param<core::param::StringParam>()->Value()).PeekBuffer());
    //    }
    //}
    //vislib::Array<int> colMode(this->posData.Count(), this->colourModeSlot.Param<core::param::EnumParam>()->Value(),
    //    vislib::Array<int>::DEFAULT_CAPACITY_INCREMENT);
    //for (int i = 0; i < static_cast<int>(this->posData.Count()); i++) {
    //    if ((colMode[i] == 1) && (this->colData[i]->GetSize() == 0)) {
    //        colMode[i] = 0;
    //    }
    //}
    //if (this->dircolourSlot.IsDirty()) {
    //    this->dircolourSlot.ResetDirty();
    //    float r, g, b;
    //    if (core::utility::ColourParser::FromString(this->dircolourSlot.Param<core::param::StringParam>()->Value(), r, g, b)) {
    //        this->dirdefCol[0] = static_cast<unsigned char>(vislib::math::Clamp<int>(static_cast<int>(r * 255.0f), 0, 255));
    //        this->dirdefCol[1] = static_cast<unsigned char>(vislib::math::Clamp<int>(static_cast<int>(g * 255.0f), 0, 255));
    //        this->dirdefCol[2] = static_cast<unsigned char>(vislib::math::Clamp<int>(static_cast<int>(b * 255.0f), 0, 255));
    //    } else {
    //        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
    //            "Unable to parse default dir::colour \"%s\"\n",
    //            vislib::StringA(this->dircolourSlot.Param<core::param::StringParam>()->Value()).PeekBuffer());
    //    }
    //}
    //int dircolMode = this->dircolourModeSlot.Param<core::param::EnumParam>()->Value();

    //if (mpdc != NULL) {
    //    mpdc->SetFrameID(0);
    //    mpdc->SetDataHash(this->datahash);
    //    mpdc->SetParticleListCount(static_cast<unsigned int>(this->posData.Count()));
    //    // TODO hier ne for-schleife um die listen...
    //    for (int idx = 0; idx < static_cast<int>(this->posData.Count()); idx++) {
    //        mpdc->AccessParticles(idx).SetGlobalColour(this->defCol[0], this->defCol[1], this->defCol[2]);
    //        mpdc->AccessParticles(idx).SetGlobalRadius(this->radiusSlot.Param<core::param::FloatParam>()->Value());
    //        mpdc->AccessParticles(idx).SetCount(this->posData[idx]->GetSize() / (3 * sizeof(float)));
    //        mpdc->AccessParticles(idx).SetGlobalType(this->typeData[idx]);
    //        switch (colMode[idx]) {
    //            case 0:
    //                mpdc->AccessParticles(idx).SetColourData(core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE, NULL);
    //                break;
    //            case 1:
    //                if (this->autoColumnRangeSlot.Param<core::param::BoolParam>()->Value()) {
    //                    mpdc->AccessParticles(idx).SetColourMapIndexValues(this->minC[idx], this->maxC[idx]);
    //                } else {
    //                    mpdc->AccessParticles(idx).SetColourMapIndexValues(
    //                        this->minColumnValSlot.Param<core::param::FloatParam>()->Value(),
    //                        this->maxColumnValSlot.Param<core::param::FloatParam>()->Value());
    //                }
    //                mpdc->AccessParticles(idx).SetColourData(core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I, this->colData[idx]->As<void>());
    //                break;
    //            default:
    //                mpdc->AccessParticles(idx).SetColourData( // some internal error
    //                    core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE, NULL);
    //                break;
    //        }

    //        if (!this->posData[idx]->IsEmpty()) {
    //            mpdc->AccessParticles(idx).SetVertexData(core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ, this->posData[idx]->As<void>());
    //        } else {
    //            mpdc->AccessParticles(idx).SetVertexData(core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE, NULL);
    //        }
    //    }
    //    mpdc->SetUnlocker(NULL);

    //} else if (dpdc != NULL) {
    //    dpdc->SetFrameID(0);
    //    dpdc->SetDataHash(this->datahash);
    //    dpdc->SetParticleListCount(static_cast<unsigned int>(this->posData.Count())); // For the moment
    //    for (int idx = 0; idx < static_cast<int>(this->posData.Count()); idx++) {
    //        dpdc->AccessParticles(idx).SetGlobalColour(this->dirdefCol[0], this->dirdefCol[1], this->dirdefCol[2]);
    //        dpdc->AccessParticles(idx).SetGlobalRadius(this->dirradiusSlot.Param<core::param::FloatParam>()->Value());
    //        dpdc->AccessParticles(idx).SetGlobalType(this->typeData[idx]);
    //        if (this->allDirData[idx]->GetSize() > 0) {
    //            unsigned int fpp = (dircolMode == 1) ? 7 : ((dircolMode == 2) ? 9 : 6); // floats per particle
    //            dpdc->AccessParticles(idx).SetCount(this->allDirData[idx]->GetSize() / (fpp * sizeof(float)));
    //            if (dpdc->AccessParticles(idx).GetCount() == 0) {
    //                dpdc->AccessParticles(idx).SetVertexData(core::moldyn::DirectionalParticleDataCall::Particles::VERTDATA_NONE, NULL);
    //                dpdc->AccessParticles(idx).SetColourData(core::moldyn::DirectionalParticleDataCall::Particles::COLDATA_NONE, NULL);
    //                dpdc->AccessParticles(idx).SetDirData(core::moldyn::DirectionalParticleDataCall::Particles::DIRDATA_NONE, NULL);
    //            } else {
    //                dpdc->AccessParticles(idx).SetVertexData(core::moldyn::DirectionalParticleDataCall::Particles::VERTDATA_FLOAT_XYZ,
    //                    this->allDirData[idx]->As<void>(), fpp * sizeof(float));
    //                if (dircolMode == 1) {
    //                    if (this->dirautoColumnRangeSlot.Param<core::param::BoolParam>()->Value()) {
    //                        dpdc->AccessParticles(idx).SetColourMapIndexValues(this->minC[idx], this->maxC[idx]);
    //                    } else {
    //                        dpdc->AccessParticles(idx).SetColourMapIndexValues(
    //                            this->dirminColumnValSlot.Param<core::param::FloatParam>()->Value(),
    //                            this->dirmaxColumnValSlot.Param<core::param::FloatParam>()->Value());
    //                    }
    //                    dpdc->AccessParticles(idx).SetColourData(core::moldyn::DirectionalParticleDataCall::Particles::COLDATA_FLOAT_I,
    //                        this->allDirData[idx]->At(3 * sizeof(float)), fpp * sizeof(float));
    //                } else if (dircolMode == 2) {
    //                    dpdc->AccessParticles(idx).SetColourData(core::moldyn::DirectionalParticleDataCall::Particles::COLDATA_FLOAT_RGB,
    //                        this->allDirData[idx]->At(3 * sizeof(float)), fpp * sizeof(float));
    //                } else {
    //                    dpdc->AccessParticles(idx).SetColourData(core::moldyn::DirectionalParticleDataCall::Particles::COLDATA_NONE, NULL);
    //                }
    //                dpdc->AccessParticles(idx).SetDirData(core::moldyn::DirectionalParticleDataCall::Particles::DIRDATA_FLOAT_XYZ,
    //                    this->allDirData[idx]->At(((dircolMode == 1) ? 4 : ((dircolMode == 2) ? 6 : 3)) * sizeof(float)), fpp * sizeof(float));
    //            }
    //        } else {
    //            dpdc->AccessParticles(idx).SetCount(this->posData[idx]->GetSize() / (3 * sizeof(float)));
    //            switch (dircolMode) {
    //                case 0:
    //                    dpdc->AccessParticles(idx).SetColourData(core::moldyn::DirectionalParticleDataCall::Particles::COLDATA_NONE, NULL);
    //                    break;
    //                case 1:
    //                    if (this->dirautoColumnRangeSlot.Param<core::param::BoolParam>()->Value()) {
    //                        dpdc->AccessParticles(idx).SetColourMapIndexValues(this->minC[idx], this->maxC[idx]);
    //                    } else {
    //                        dpdc->AccessParticles(idx).SetColourMapIndexValues(
    //                            this->dirminColumnValSlot.Param<core::param::FloatParam>()->Value(),
    //                            this->dirmaxColumnValSlot.Param<core::param::FloatParam>()->Value());
    //                    }
    //                    dpdc->AccessParticles(idx).SetColourData(core::moldyn::DirectionalParticleDataCall::Particles::COLDATA_FLOAT_I, this->colData[idx]->As<void>());
    //                    break;
    //                default:
    //                    dpdc->AccessParticles(idx).SetColourData( // some internal error
    //                        core::moldyn::DirectionalParticleDataCall::Particles::COLDATA_NONE, NULL);
    //                    break;
    //            }

    //            if (!this->posData.IsEmpty()) {
    //                dpdc->AccessParticles(idx).SetVertexData(core::moldyn::DirectionalParticleDataCall::Particles::VERTDATA_FLOAT_XYZ, this->posData[idx]->As<void>());
    //            } else {
    //                dpdc->AccessParticles(idx).SetVertexData(core::moldyn::DirectionalParticleDataCall::Particles::VERTDATA_NONE, NULL);
    //            }
    //            dpdc->AccessParticles(idx).SetDirData(core::moldyn::DirectionalParticleDataCall::Particles::DIRDATA_NONE, NULL);
    //        }
    //    }
    //    dpdc->SetUnlocker(NULL);
    //}

    return true;
}


/*
 * ParticleBoxGeneratorDataSource::getExtentCallback
 */
bool ParticleBoxGeneratorDataSource::getExtentCallback(core::Call& caller) {
    core::moldyn::MultiParticleDataCall *mpdc = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&caller);
    if (mpdc == nullptr) return false;
	if (particleCountSlot.IsDirty() || radiusPerParticleSlot.IsDirty() || colorDataSlot.IsDirty()
			|| interleavePosAndColorSlot.IsDirty() || radiusScaleSlot.IsDirty() || positionNoiseSlot.IsDirty()) {
		this->assertData();
	}

	mpdc->SetParticleListCount(1);
	mpdc->SetDataHash(dataHash);
	mpdc->SetExtent(1, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);

    return true;
}


/*
 * ParticleBoxGeneratorDataSource::clear
 */
void ParticleBoxGeneratorDataSource::clear(void) {

	// TODO: Implement

    //for (int i = 0; i < static_cast<int>(this->posData.Count()); i++) {
    //    this->posData[i]->EnforceSize(0);
    //    this->colData[i]->EnforceSize(0);
    //    this->allDirData[i]->EnforceSize(0);
    //}
    //this->headerMinX = this->headerMinY = this->headerMinZ = 0.0f;
    //this->headerMaxX = this->headerMaxY = this->headerMaxZ = 1.0f;
    //this->minX = this->minY = this->minZ = 0.0f;
    //this->maxX = this->maxY = this->maxZ = 1.0f;
    //this->datahash++;
}


/*
 * ParticleBoxGeneratorDataSource::assertData
 */
void ParticleBoxGeneratorDataSource::assertData(void) {
	dataHash++;

	// TODO: Implement


//    using vislib::sys::Log;
//    if (!this->filenameSlot.IsDirty()
//            && !this->colourModeSlot.IsDirty()
//            && !this->colourColumnSlot.IsDirty()
//            && !this->splitLoadDiredDataSlot.IsDirty()
//            && !this->dirXColNameSlot.IsDirty()
//            && !this->dirYColNameSlot.IsDirty()
//            && !this->dirZColNameSlot.IsDirty()
//            && !this->dircolourModeSlot.IsDirty()
//            && !this->dircolourColumnSlot.IsDirty()
//            && !this->typeColumnSlot.IsDirty()
//			&& !this->bboxEnabledSlot.IsDirty()
//			&& !this->bboxMaxSlot.IsDirty()
//			&& !this->bboxMinSlot.IsDirty()
//        ) return;
//    this->filenameSlot.ResetDirty();
//    this->colourModeSlot.ResetDirty();
//    this->colourColumnSlot.ResetDirty();
//    this->splitLoadDiredDataSlot.ResetDirty();
//    this->dirXColNameSlot.ResetDirty();
//    this->dirYColNameSlot.ResetDirty();
//    this->dirZColNameSlot.ResetDirty();
//    this->dircolourModeSlot.ResetDirty();
//    this->dircolourColumnSlot.ResetDirty();
//    this->typeColumnSlot.ResetDirty();
//	this->bboxEnabledSlot.ResetDirty();
//	this->bboxMaxSlot.ResetDirty();
//	this->bboxMinSlot.ResetDirty();
//
//    this->clear();
//
//    vislib::sys::FastFile file;
//    vislib::TString filename = this->filenameSlot.Param<core::param::FilePathParam>()->Value();
////    Log::DefaultLog.WriteInfo(50, _T("Loading \"%s\""), filename.PeekBuffer());
//    //this->datahash = static_cast<SIZE_T>(filename.HashCode());
//    if (!file.Open(filename, vislib::sys::File::READ_ONLY,
//            vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
//        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
//            "Unable to open imd file %s\n", vislib::StringA(
//            this->filenameSlot.Param<core::param::FilePathParam>()->Value()).PeekBuffer());
//        return;
//    }
//
//    HeaderData header;
//    if (!this->readHeader(file, header)) {
//        // error already logged
//        file.Close();
//        return;
//    }
//
////    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100,
////        "IMDAtom with %d data colums:\n", static_cast<int>(header.captions.Count()));
////    for (SIZE_T i = 0; i < header.captions.Count(); i++) {
////        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100,
////            "\t%s\n", header.captions[i].PeekBuffer());
////    }
//
//    UINT32 endianTestInt = 0x12345678;
//    UINT8 endianTestBytes[4];
//    ::memcpy(endianTestBytes, &endianTestInt, 4);
//    bool machineLittleEndian = ((endianTestBytes[0] == 0x78)
//        && (endianTestBytes[1] == 0x56)
//        && (endianTestBytes[2] == 0x34)
//        && (endianTestBytes[3] == 0x12));
//
//    vislib::StringA dirXColName = this->dirXColNameSlot.Param<core::param::StringParam>()->Value();
//    vislib::StringA dirYColName = this->dirYColNameSlot.Param<core::param::StringParam>()->Value();
//    vislib::StringA dirZColName = this->dirZColNameSlot.Param<core::param::StringParam>()->Value();
//    INT_PTR dirXCol = dirXColName.IsEmpty() ? -1 : header.captions.IndexOf(dirXColName);
//    INT_PTR dirYCol = dirYColName.IsEmpty() ? -1 : header.captions.IndexOf(dirYColName);
//    INT_PTR dirZCol = dirZColName.IsEmpty() ? -1 : header.captions.IndexOf(dirZColName);
//    // TODO hier die type column? vermutlich nicht.
//    bool loadDir = (dirXCol >= 0) && (dirYCol >= 0) && (dirZCol >= 0);
//    bool splitLoadDir = this->splitLoadDiredDataSlot.Param<core::param::BoolParam>()->Value();
//
//    bool retval = false;
//    switch (header.format) {
//        case 'A': // ASCII
//            retval = this->readData<AtomReaderASCII>(file, header, loadDir, splitLoadDir);
//            break;
//        case 'B': // binary, big endian, double
//            retval = (machineLittleEndian)
//                ? this->readData<AtomReaderDoubleSwitched>(file, header, loadDir, splitLoadDir)
//                : this->readData<AtomReaderDouble>(file, header, loadDir, splitLoadDir);
//            break;
//        case 'b': // binary, big endian, float
//            retval = (machineLittleEndian)
//                ? this->readData<AtomReaderFloatSwitched>(file, header, loadDir, splitLoadDir)
//                : this->readData<AtomReaderFloat>(file, header, loadDir, splitLoadDir);
//            break;
//        case 'L': // binary, little endian, double
//            retval = (machineLittleEndian)
//                ? this->readData<AtomReaderDouble>(file, header, loadDir, splitLoadDir)
//                : this->readData<AtomReaderDoubleSwitched>(file, header, loadDir, splitLoadDir);
//            break;
//        case 'l': // binary, little endian float
//            retval = (machineLittleEndian)
//                ? this->readData<AtomReaderFloat>(file, header, loadDir, splitLoadDir)
//                : this->readData<AtomReaderFloatSwitched>(file, header, loadDir, splitLoadDir);
//            break;
//        default:
//            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
//                "Unable to read imd file: Illegal format\n");
//            break;
//    }
//
//    if (retval) {
//        // TODO inside readData!
//        //this->posData.EnforceSize(posWriter.End(), true);
//        //this->colData.EnforceSize(colWriter.End(), true);
//        //this->allDirData.EnforceSize(dirWriter.End(), true);
//        // TODO allTypeData
//
//        SIZE_T cnt = 0;
//        for (int i = 0; i < static_cast<int>(this->posData.Count()); i++) {
//            if (this->allDirData[i]->GetSize() > 0) {
//                unsigned int dircolMode = this->dircolourModeSlot.Param<core::param::EnumParam>()->Value();
//                unsigned int fpp = (dircolMode == 1) ? 7 : ((dircolMode == 2) ? 9 : 6); // floats per particle
//                cnt += this->allDirData[i]->GetSize() / (fpp * sizeof(float));
//            } else {
//                cnt += this->posData[i]->GetSize() / (sizeof(float) * 3);
//            }
//        }
////        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%d Atoms loaded\n", cnt);
////        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100, "Data bounding box = (%f, %f, %f) ... (%f, %f, %f)\n",
////            this->minX, this->minY, this->minZ, this->maxX, this->maxY, this->maxZ);
//
//        //this->datahash = (this->datahash << (sizeof(SIZE_T) / 2))
//        //    || (this->datahash >> (sizeof(SIZE_T) / 2));
//        //this->datahash ^= this->posData.GetSize();
//        this->datahash++;
//
//        // All parameters must influence the data hash
//
//    } else {
//        // error already logged
//        //this->posData.EnforceSize(0, true);
//        //this->colData.EnforceSize(0, true);
//        //this->allDirData.EnforceSize(0, true);
//        //this->datahash = 0;
//    }
//
//    file.Close();
//
//    // apply filter (if activated)
//    this->posXFilterUpdate(this->posXFilterNow);

}


#if 0

#include "DataGenerator.h"
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>

DataGenerator::DataGenerator(uint32_t c, DataScenario s, DataLayout l, uint32_t r_seed)
	: cnt(c), scenario(s), layout(l), rnd_seed(r_seed), rad_param(1.0f), pos_param(0.0f) {
}

DataGenerator::~DataGenerator() {
}

void DataGenerator::Generate(DataStore & store) {

	// clear store first
	store.Clear();

	// initialize debugger and distribution
	std::mt19937 rnd_engine(this->rnd_seed);
	std::uniform_real_distribution<float> rnd_uni; // [0, 1]
	std::normal_distribution<float> rnd_norm; // normal at 0
	std::uniform_int_distribution<int> rnd_uni_int(0, 255);

	// allocate memory
	float *mem = new float[cnt * 5];

	// prepare memory write positions and position increments depending on the layout
	float *x, *y, *z, *r, *i;
	int x_step, y_step, z_step, r_step, i_step;
	switch (layout) {
	case DataLayout::X_Y_Z:
	case DataLayout::X_Y_Z_R_I:
		x = mem; x_step = 1;
		y = mem + cnt; y_step = 1;
		z = mem + 2 * cnt; z_step = 1;
		r = mem + 3 * cnt; r_step = 1;
		i = mem + 4 * cnt; i_step = 1;
		break;
	case DataLayout::XYZ:
	case DataLayout::XYZ_R_I:
		x = mem; x_step = 3;
		y = mem + 1; y_step = 3;
		z = mem + 2; z_step = 3;
		r = mem + 3 * cnt; r_step = 1;
		i = mem + 4 * cnt; i_step = 1;
		break;
	case DataLayout::XYZR_I:
		x = mem; x_step = 4;
		y = mem + 1; y_step = 4;
		z = mem + 2; z_step = 4;
		r = mem + 3; r_step = 4;
		i = mem + 4 * cnt; i_step = 1;
		break;
	case DataLayout::XYZRI:
		x = mem; x_step = 5;
		y = mem + 1; y_step = 5;
		z = mem + 2; z_step = 5;
		r = mem + 3; r_step = 5;
		i = mem + 4; i_step = 5;
		break;
	default:
		delete[] mem;
		std::cerr << "Internal particle generation error @" << __FILE__ << ":" << __LINE__ << std::endl;
		return;
	}

	// generate data
	switch (scenario) {
	case DataScenario::Box: {
		uint32_t xcnt, ycnt, zcnt;
		float yzcnt;
		xcnt = static_cast<uint32_t>(
			std::ceil(
			static_cast<float>(std::pow(static_cast<double>(cnt), 1.0 / 3.0))
			)
			);
		yzcnt = std::ceil(static_cast<float>(cnt) / static_cast<float>(xcnt));
		ycnt = static_cast<uint32_t>(
			std::ceil(std::sqrt(yzcnt))
			);
		zcnt = static_cast<uint32_t>(
			std::ceil(yzcnt / static_cast<float>(ycnt))
			);
		std::cout << "Boxing: " << xcnt << " x " << ycnt << " x " << zcnt << std::endl;

		float x_a = 2.0f / static_cast<float>(xcnt);
		float x_b = x_a / 2.0f;
		float y_a = 2.0f / static_cast<float>(ycnt);
		float y_b = y_a / 2.0f;
		float z_a = 2.0f / static_cast<float>(zcnt);
		float z_b = z_a / 2.0f;

		float rad = std::min(std::min(x_b, y_b), z_b) * rad_param;
		float pn = rad * pos_param;

		uint32_t i_all = 0;
		for (uint32_t iz = 0; (iz < zcnt) && (i_all < cnt); ++iz) {
			float pz = -1.0f + z_b + z_a * iz;
			for (uint32_t iy = 0; (iy < ycnt) && (i_all < cnt); ++iy) {
				float py = -1.0f + y_b + y_a * iy;
				for (uint32_t ix = 0; (ix < xcnt) && (i_all < cnt); ++ix, ++i_all) {
					float px = -1.0f + x_b + x_a * ix;

					*x = px;
					*y = py;
					*z = -pz;
					*r = rad;
					*i = makePackedColorRGBA(rnd_uni_int, rnd_engine);
					// rnd_uni(rnd_engine);

					this->addNoise(*x, *y, *z, pn, rnd_uni, rnd_engine);

					x += x_step;
					y += y_step;
					z += z_step;
					r += r_step;
					i += i_step;

				}
			}
		}

	} break;
	case DataScenario::Line: {
		float a = 2.0f / static_cast<float>(cnt);
		float b = a / 2.0f;
		float rad = b * rad_param;
		float pn = rad * pos_param;

		for (uint32_t p = 0; p < cnt; ++p) {
			*x = -1.0f + b + a * p;
			*y = 0.0f;
			*z = 0.0f;
			*r = rad;
			*i = makePackedColorRGBA(rnd_uni_int, rnd_engine);
			// rnd_uni(rnd_engine);

			this->addNoise(*x, *y, *z, pn, rnd_uni, rnd_engine);

			x += x_step;
			y += y_step;
			z += z_step;
			r += r_step;
			i += i_step;
		}

	} break;
	}

	// store generated data in DataStore object
	switch (layout) {
	case DataLayout::XYZ:
		store.SetXYZOnly(cnt, mem);
		break;
	case DataLayout::X_Y_Z:
		store.SetXYZOnly(cnt, mem, mem + cnt, mem + 2 * cnt);
		break;
	case DataLayout::X_Y_Z_R_I:
		store.Set(cnt, mem, mem + cnt, mem + 2 * cnt, mem + 3 * cnt, mem + 4 * cnt);
		break;
	case DataLayout::XYZ_R_I:
		store.Set(cnt, mem, mem + 3 * cnt, mem + 4 * cnt);
		break;
	case DataLayout::XYZR_I:
		store.Set(cnt, mem, mem + 4 * cnt);
		break;
	case DataLayout::XYZRI:
		store.Set(cnt, mem);
		break;
	}

	// fin

}

#endif
