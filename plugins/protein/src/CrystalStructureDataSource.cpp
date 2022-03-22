/*
 * CrystalStructureDataSource.cpp
 *
 * Copyright (C) 2012 by University of Stuttgart (VISUS).
 * All rights reserved.
 *
 * $Id$
 */

#include "stdafx.h"
#include <sstream>

#include "CrystalStructureDataSource.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"
#include "protein_calls/CrystalStructureDataCall.h"
#include "vislib/math/Vector.h"

using namespace megamol;

#define VECFIELD_PROC_SIZE 2
#define VECFIELD_PROC_SCL 1.0f

#define HACK

using namespace megamol::core::utility::log;

/*
 * protein::CrystalStructureDataSource::CrystalStructureDataSource
 */
protein::CrystalStructureDataSource::CrystalStructureDataSource(void)
        : AnimDataModule()
        , dataOutSlot("dataout", "The slot providing the loaded data")
        , dataChkptCallerSlot("chkptData", "The caller slot to connect a chkpt-source.")
        , fileFramesSlot("fileFrames", "The path to the frame file, e.g.: /PathToFile/bto_625000at_500fr.bin")
        , fileAtomsSlot("fileAtoms", "The path to the atom file, e.g.: /PathToFile/bto_625000at.bin")
        , fileCellsSlot("fileCells", "The path to the file containing cells, e.g.: /PathToFile/bto_625000at_cells.bin")
        , frameCacheSizeParam("frameCacheSize", "The size of the frame cache")
        , displOffsParam("displOffs", "The frame offset for displacement vectors")
        , dSourceParam("dipoleScr", "The dipole source")
        , cells(NULL)
        , bbox(-105.0f, -105.0f, -105.0f, 105.0f, 105.0f, 105.0f)
        , frameCnt(0) {

    // Filename slots
    this->fileFramesSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->fileFramesSlot);
    this->fileCellsSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->fileCellsSlot);
    this->fileAtomsSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->fileAtomsSlot);

    // Data caller slot for chkpt source
    this->dataChkptCallerSlot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->dataChkptCallerSlot);

    // Data out slot
    this->dataOutSlot.SetCallback(protein_calls::CrystalStructureDataCall::ClassName(),
        protein_calls::CrystalStructureDataCall::FunctionName(protein_calls::CrystalStructureDataCall::CallForGetData),
        &CrystalStructureDataSource::getData);
    this->dataOutSlot.SetCallback(protein_calls::CrystalStructureDataCall::ClassName(),
        protein_calls::CrystalStructureDataCall::FunctionName(
            protein_calls::CrystalStructureDataCall::CallForGetExtent),
        &CrystalStructureDataSource::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    // Param displacement vector frame offset
    this->frameCacheSize = 10;
    this->frameCacheSizeParam.SetParameter(new core::param::IntParam(this->frameCacheSize, 1, 75));
    this->MakeSlotAvailable(&this->frameCacheSizeParam);

    // Param displacement vector frame offset
    this->displOffs = 10;
    this->displOffsParam.SetParameter(new core::param::IntParam(static_cast<int>(this->displOffs), 0, 499));
    this->MakeSlotAvailable(&this->displOffsParam);

    geocalls::MultiParticleDataCall* dirc = this->dataChkptCallerSlot.CallAs<geocalls::MultiParticleDataCall>();

    // Param for dipole calculation
    this->dSource = DIPOLE_DISPL;
    core::param::EnumParam* gb = new core::param::EnumParam(this->dSource);
    gb->SetTypePair(DIPOLE_DISPL, "Displ");
    gb->SetTypePair(DIPOLE_DISPLTI, "Displ (Ti only)");
    gb->SetTypePair(DIPOLE_CELL, "Dipole (cell)");
    gb->SetTypePair(DIPOLE_NOBA, "Dipole (No Ba)");
    gb->SetTypePair(DIPOLE_BATI, "Ti/Ba-Center");
    gb->SetTypePair(CHKPT_SOURCE, "*.chkpt source");
    gb->SetTypePair(VECFIELD_PROC, "Procedural vecfield");
    this->dSourceParam << gb;
    this->MakeSlotAvailable(&this->dSourceParam);

    // Init dipole and atom count according to dipole source
    switch (this->dSource) {
    case DIPOLE_DISPL:
        this->atomCnt = 625000;
        this->dipoleCnt = 625000;
        this->cellCnt = 125000;
        break;
    case DIPOLE_DISPLTI:
        this->atomCnt = 625000;
        this->dipoleCnt = 125000;
        this->cellCnt = 125000;
        break;
    case DIPOLE_CELL:
        this->atomCnt = 625000;
        this->dipoleCnt = 125000;
        this->cellCnt = 125000;
        break;
    case DIPOLE_NOBA:
        this->atomCnt = 625000;
        this->dipoleCnt = 125000;
        this->cellCnt = 125000;
        break;
    case DIPOLE_BATI:
        this->atomCnt = 625000;
        this->dipoleCnt = 125000;
        this->cellCnt = 125000;
        break;
    case CHKPT_SOURCE:
        (*dirc)(1); // Call for get extend
        (*dirc)(0); // Call for get data
        this->atomCnt = (unsigned int)dirc->AccessParticles(0).GetCount();
        this->dipoleCnt = (unsigned int)dirc->AccessParticles(0).GetCount();
        this->cellCnt = 0;
        this->frameCnt = 1;
        break;
    case VECFIELD_PROC:
        this->atomCnt = 0;                                                              // TODO
        this->dipoleCnt = VECFIELD_PROC_SIZE * VECFIELD_PROC_SIZE * VECFIELD_PROC_SIZE; // TODO
        this->cellCnt = 0;
        this->frameCnt = 1;
        break;
    default:
        this->atomCnt = 0;
        this->dipoleCnt = 0;
        this->cellCnt = 0;
        break;
    }
}


/*
 * protein::CrystalStructureDataSource::~CrystalStructureDataSource
 */
protein::CrystalStructureDataSource::~CrystalStructureDataSource(void) {
    this->Release();
}


/*
 * protein::CrystalStructureDataSource::create
 */
bool protein::CrystalStructureDataSource::create(void) {
    // intentionally empty
    return true;
}


/*
 * protein::CrystalStructureDataSource::getData
 */
bool protein::CrystalStructureDataSource::getData(core::Call& call) {
    using megamol::core::utility::log::Log;

    protein_calls::CrystalStructureDataCall* dc = dynamic_cast<protein_calls::CrystalStructureDataCall*>(&call);

    // TODO Invalid filepointer if frameIdx == 0?

    if (dc == NULL)
        return false;

    if ((this->dSource == CHKPT_SOURCE) || (this->dSource == VECFIELD_PROC)) {
        if (dc->FrameID() > 0) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: unable to load frame %u (no of frames is %u)",
                this->ClassName(), dc->FrameID(), 1);
            return false;
        }
    }

    if (dc->FrameID() >= this->frameCnt) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: unable to load frame %u (no of frames is %u)",
            this->ClassName(), dc->FrameID(), dc->FrameCount());
        return false;
    }

    //printf("Requesting frame # %u ... \n", dc->FrameID()); // DEBUG

    // Search frame chache until a matching frame is found
    Frame *fr, *frTmp = NULL;
    bool ready = false;
    if (dc->IsFrameForced()) {
        do {
            ready = true;
            fr = dynamic_cast<CrystalStructureDataSource::Frame*>(this->requestLockedFrame(dc->FrameID()));
            //printf("  requested locked frame (fr) # %u ... \n", fr->FrameNumber()); // DEBUG

            if (dc->FrameID() != fr->GetFrameIdx()) {
                ready = false;
                //printf("FORCE (got %u)\n", fr->GetFrameIdx()); // DEBUG
                fr->Unlock();
                // Request idx-1 to get loading thread to load idx
                // TODO This is ugly
                frTmp = dynamic_cast<CrystalStructureDataSource::Frame*>(this->requestLockedFrame(dc->FrameID() - 1));
                frTmp->Unlock();
                //printf("  requested locked frame (frTmp) # %u ... \n", frTmp->FrameNumber()); // DEBUG
            }
        } while (!ready);


    } else {
        fr = dynamic_cast<CrystalStructureDataSource::Frame*>(this->requestLockedFrame(dc->FrameID()));
    }
    if (fr == NULL)
        return false;

    //printf("--> got %u\n", fr->GetFrameIdx()); // DEBUG

    dc->SetUnlocker(new Unlocker(*fr));
    dc->SetAtoms(fr->GetAtomPos(), this->atomType.PeekElements(), this->atomCnt);
    dc->SetDipoles(fr->GetDipolePos(), fr->GetDipole(), this->dipoleCnt);
    dc->SetCells(this->cells, this->cellCnt);
    dc->SetAtomCon(this->atomCon.PeekElements(), static_cast<unsigned int>(this->atomCon.Count()) / 2);

    return true;
}


/*
 * protein::CrystalStructureDataSource::getExtend
 */
bool protein::CrystalStructureDataSource::getExtent(core::Call& call) {
    using megamol::core::utility::log::Log;

    // Update parameters
    updateParams();


    protein_calls::CrystalStructureDataCall* dc = dynamic_cast<protein_calls::CrystalStructureDataCall*>(&call);
    if (dc == NULL)
        return false;


    if (this->dSource == CHKPT_SOURCE) { // Use *.chkpt source
        geocalls::MultiParticleDataCall* dirc = this->dataChkptCallerSlot.CallAs<geocalls::MultiParticleDataCall>();
        if (!(*dirc)(1)) { // Try call for extent
            return false;
        }
#ifdef HACK // Grow bounding boxes to prevent clipping of the isosurfaces
        this->bbox = dirc->AccessBoundingBoxes().ObjectSpaceBBox();
        this->bbox.Grow(10.0f);
        dc->AccessBoundingBoxes().Clear();
        dc->AccessBoundingBoxes().SetWorldSpaceBBox(this->bbox);
        dc->AccessBoundingBoxes().SetWorldSpaceClipBox(this->bbox);
        dc->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
        dc->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);
#else
        dc->AccessBoundingBoxes() = dirc->AccessBoundingBoxes();
#endif
        float scaling = dirc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (scaling > 0.0000001) {
            scaling = 10.0f / scaling;
        } else {
            scaling = 1.0f;
        }
        dc->AccessBoundingBoxes().MakeScaledWorld(scaling);
        dc->SetFrameCount(1);
    } else if (this->dSource == VECFIELD_PROC) {
        dc->AccessBoundingBoxes().Clear();
        dc->AccessBoundingBoxes().SetWorldSpaceBBox(this->bbox);
        dc->AccessBoundingBoxes().SetWorldSpaceClipBox(this->bbox);
        dc->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
        dc->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);
        dc->SetFrameCount(1);
    } else { // Use binary source

        dc->AccessBoundingBoxes().Clear();
        dc->AccessBoundingBoxes().SetWorldSpaceBBox(this->bbox);
        dc->AccessBoundingBoxes().SetWorldSpaceClipBox(this->bbox);
        dc->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
        dc->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);
        dc->SetFrameCount(vislib::math::Max(1U, this->frameCnt));
    }

    return true;
}


/*
 * protein::CrystalStructureDataSource::release
 */
void protein::CrystalStructureDataSource::release(void) {
    // Stop frame-loading thread
    this->resetFrameCache();
}


/*
 * protein::CrystalStructureDataSource::loadFiles
 */
bool protein::CrystalStructureDataSource::loadFiles() {

    //printf("Start loading files...\n"); // DEBUG


    std::fstream fileFrames, fileCells, fileAtoms;
    std::string line;
    int* bufferAtoms;

    time_t t = clock();

    // Allocate memory if necessary
    this->atomCon.Clear();
    this->atomCon.SetCount(this->atomCnt * 6);
    bufferAtoms = new int[this->atomCnt * 7];

    fileAtoms.open(this->fileAtomsSlot.Param<core::param::FilePathParam>()->Value(), std::ios::in | std::ios::binary);

    if (!fileAtoms.is_open()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Could not open file %s", this->ClassName(),
            this->fileAtomsSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
        return false;
    }

    fileAtoms.seekg(0, std::ios::beg);
    fileAtoms.read((char*)bufferAtoms, this->atomCnt * 7 * sizeof(int));
    fileAtoms.close();

    // Set atom types and colors
    this->atomType.SetCount(this->atomCnt);
    for (unsigned int i = 0; i < this->atomCnt; i++) {
        if (bufferAtoms[i * 7] == 1) {
            this->atomType[i] = protein_calls::CrystalStructureDataCall::O;
        } else if (bufferAtoms[i * 7] == 0) {
            this->atomType[i] = protein_calls::CrystalStructureDataCall::BA;
        } else if (bufferAtoms[i * 7] == 2) {
            this->atomType[i] = protein_calls::CrystalStructureDataCall::TI;
        } else {
            this->atomType[i] = protein_calls::CrystalStructureDataCall::GENERIC;
            return false;
        }
        // Get connectivity information of this atom
        for (int cnt = 0; cnt < 6; cnt++) {
            this->atomCon[6 * i + cnt] = bufferAtoms[i * 7 + cnt + 1];
        }
    }

    //printf("  atom types done ...\n"); // DEBUG

    // Get number of frames
    fileFrames.open(this->fileFramesSlot.Param<core::param::FilePathParam>()->Value(), std::ios::in | std::ios::binary);

    if (!fileFrames) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Could not open file %s", this->ClassName(),
            this->fileFramesSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
        return false;
    }

    fileFrames.seekg(0, std::fstream::beg);
    this->frameCnt = 0;
    do {
        this->frameCnt++;
        fileFrames.seekg(this->atomCnt * 3 * sizeof(float) - 1, std::fstream::cur);
        fileFrames.get();
    } while (fileFrames.good());
    fileFrames.close();
    this->frameCnt--;

    //printf("  frames done ...\n"); // DEBUG

    // Read cell file and load indices
    this->cells = new int[this->cellCnt * 15];
    fileCells.open(this->fileCellsSlot.Param<core::param::FilePathParam>()->Value(), std::ios::in | std::ios::binary);
    if (!fileCells) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Could not open file %s", this->ClassName(),
            this->fileCellsSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
        return false;
    }

    fileCells.seekg(0, std::ios::beg);
    fileCells.read((char*)this->cells, this->cellCnt * 15 * sizeof(int));
    fileCells.close();

    //printf("  cells done ...\n"); // DEBUG

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: Number of atoms       %u", this->ClassName(), this->atomCnt);

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: Number of frames      %u", this->ClassName(), this->frameCnt);

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: Number of cells       %u", this->ClassName(), this->cellCnt);


    // Init frame cache and start loading thread
    this->setFrameCount(this->frameCnt);        // Number of frames in the dataset
    this->initFrameCache(this->frameCacheSize); // Number of frames in the cache

    Log::DefaultLog.WriteMsg(
        Log::LEVEL_INFO, "%s: Loading thread started (cache size %u)", this->ClassName(), this->frameCacheSize);

    delete[] bufferAtoms;

    /*for(int i = 0; i < 100; i++) {
        printf("CELL %i (%i %i %i %i %i %i %i %i %i %i %i %i %i %i %i)\n", i,
            this->cells[i*15+0], this->cells[i*15+1], this->cells[i*15+2],
            this->cells[i*15+3], this->cells[i*15+4], this->cells[i*15+5],
            this->cells[i*15+6], this->cells[i*15+7], this->cells[i*15+8],
            this->cells[i*15+9], this->cells[i*15+10], this->cells[i*15+11],
            this->cells[i*15+12], this->cells[i*15+13], this->cells[i*15+14]);
    }*/ // DEBUG

    // TODO chkpt source

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: time for loading files %f", this->ClassName(),
        (double(clock() - t) / double(CLOCKS_PER_SEC))); // DEBUG

    return true;
}


/*
 * protein::CrystalStructureDataSource::convertStrTo
 */
template<class T>
T protein::CrystalStructureDataSource::convertStrTo(std::string str) {
    std::istringstream stream(str);
    T var;
    stream >> var;
    return var;
}


/*
 * protein::CrystalStructureDataSource::constructFrame
 */
core::view::AnimDataModule::Frame* protein::CrystalStructureDataSource::constructFrame(void) const {
    Frame* f = new Frame(*const_cast<CrystalStructureDataSource*>(this));
    f->AllocBufs(this->atomCnt, this->dipoleCnt);
    return f;
}


/*
 * protein::CrystalStructureDataSource::loadFrame
 */
void protein::CrystalStructureDataSource::loadFrame(core::view::AnimDataModule::Frame* frame, unsigned int idx) {
    CrystalStructureDataSource::Frame* fr = dynamic_cast<CrystalStructureDataSource::Frame*>(frame);
    fr->SetFrameIdx(idx);
    if (!this->WriteFrameData(fr)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Could not write frame data.", this->ClassName());
    }
    //printf("Frame %u loaded into cache\n", idx); // DEBUG
}


/*
 * protein::CrystalStructureDataSource::updateParams
 */
void protein::CrystalStructureDataSource::updateParams() {

    //printf("Updating params ...\n"); // DEBUG

    // Filename params
    if (this->fileAtomsSlot.IsDirty() || this->fileFramesSlot.IsDirty() || this->fileCellsSlot.IsDirty()) {
        this->fileAtomsSlot.ResetDirty();
        this->fileFramesSlot.ResetDirty();
        this->fileCellsSlot.ResetDirty();
        this->loadFiles(); // Note: this also restarts the loading thread
    }
    // Framecache size param
    if (this->frameCacheSizeParam.IsDirty()) {
        this->frameCacheSizeParam.ResetDirty();
        this->frameCacheSize = this->frameCacheSizeParam.Param<core::param::IntParam>()->Value();
        this->resetFrameCache();                    // Restart loading thread
        this->setFrameCount(this->frameCnt);        // Number of frames in the dataset
        this->initFrameCache(this->frameCacheSize); // Number of frames in the cache
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_INFO, "%s: Loading thread (re)started (cache size %u)", this->ClassName(), this->frameCacheSize);
    }
    // Displacement offset
    if (this->displOffsParam.IsDirty()) {
        this->displOffsParam.ResetDirty();
        this->displOffs = this->displOffsParam.Param<core::param::IntParam>()->Value();
        this->resetFrameCache();                    // Restart loading thread
        this->setFrameCount(this->frameCnt);        // Number of frames in the dataset
        this->initFrameCache(this->frameCacheSize); // Number of frames in the cache
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_INFO, "%s: Loading thread (re)started (frame offset %i)", this->ClassName(), this->displOffs);
    }

    // Dipole source
    if (this->dSourceParam.IsDirty()) {
        this->dSourceParam.ResetDirty();
        this->dSource = static_cast<DipoleSrc>(this->dSourceParam.Param<core::param::EnumParam>()->Value());

        geocalls::MultiParticleDataCall* dirc = this->dataChkptCallerSlot.CallAs<geocalls::MultiParticleDataCall>();


        // Set dipole and atom count according to source
        switch (this->dSource) {
        case DIPOLE_DISPL:
            this->atomCnt = 625000;
            this->dipoleCnt = 625000;
            this->cellCnt = 125000;
            break;
        case DIPOLE_DISPLTI:
            this->atomCnt = 625000;
            this->dipoleCnt = 125000;
            this->cellCnt = 125000;
            break;
        case DIPOLE_CELL:
            this->atomCnt = 625000;
            this->dipoleCnt = 125000;
            this->cellCnt = 125000;
            break;
        case DIPOLE_NOBA:
            this->atomCnt = 625000;
            this->dipoleCnt = 125000;
            this->cellCnt = 125000;
            break;
        case DIPOLE_BATI:
            this->atomCnt = 625000;
            this->dipoleCnt = 125000;
            this->cellCnt = 125000;
            break;
        case CHKPT_SOURCE:
            (*dirc)(1); // Call for get extend
            (*dirc)(0); // Call for get data
            this->atomCnt = (unsigned int)dirc->AccessParticles(0).GetCount();
            this->dipoleCnt = (unsigned int)dirc->AccessParticles(0).GetCount();
            this->cellCnt = 0;
            this->frameCnt = 1;
            break;
        case VECFIELD_PROC:
            this->atomCnt = 0;
            this->dipoleCnt = VECFIELD_PROC_SIZE * VECFIELD_PROC_SIZE * VECFIELD_PROC_SIZE;
            this->cellCnt = 0;
            this->frameCnt = 1;
            break;
        default:
            this->atomCnt = 0;
            this->dipoleCnt = 0;
            this->cellCnt = 0;
            break;
        }


        // Reload files with time independent information
        //this->loadFiles();


        this->resetFrameCache(); // Restart loading thread
        if ((this->dSource == CHKPT_SOURCE) || (this->dSource == VECFIELD_PROC)) {
            this->setFrameCount(1); // Number of frames in the dataset
        } else {
            this->setFrameCount(this->frameCnt);
        }
        this->initFrameCache(this->frameCacheSize); // Number of frames in the cache


        Log::DefaultLog.WriteMsg(
            Log::LEVEL_INFO, "%s: Loading thread (re)started (dipole source changed)", this->ClassName());
    }

    //printf("  done.\n"); // DEBUG
}


/*
 * protein::CrystalStructureDataSource::Frame::WriteFrameData
 */
bool protein::CrystalStructureDataSource::WriteFrameData(CrystalStructureDataSource::Frame* fr) {

    std::fstream file;
    int displFrameIdx;

    time_t t = clock();

    // Use displacement of all atoms
    if (this->dSource == DIPOLE_DISPL) {

        file.open(this->fileFramesSlot.Param<core::param::FilePathParam>()->Value(), std::ios::in | std::ios::binary);
        if (!file.good()) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "CrystalStructureDataSource::Frame: Could not open file %s",
                this->fileFramesSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
            return false;
        }
        file.seekg(0, std::ios::beg);

        if (static_cast<int>(fr->GetFrameIdx()) - static_cast<int>(this->displOffs) < 0) {
            displFrameIdx = fr->GetFrameIdx(); // displacement = 0 in this case
        } else {
            displFrameIdx = static_cast<int>(fr->FrameNumber()) - static_cast<int>(this->displOffs);
        }

        // Skip preceeding frames
        file.seekg(this->atomCnt * 3 * sizeof(float) * displFrameIdx, std::ios::beg);

        if (!file.good()) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "CrystalStructureDataSource::Frame: Could not parse file %s (invalid filepointer %i)",
                this->fileFramesSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str(),
                static_cast<int>(file.tellg()));
            return false;
        }

        // Read frame data of the displacement frame from file
        file.read((char*)(fr->dipole), this->dipoleCnt * 3 * sizeof(float));

        // Skip frames
        // Note: -1 because we already skipped one frame by reading it
        //file.seekg(this->atomCnt*3*sizeof(float)*(displOffs-1), std::ios::cur);
        file.seekg(0, std::ios::beg);
        file.seekg(this->atomCnt * 3 * sizeof(float) * fr->GetFrameIdx(), std::ios::beg);

        // Read atom pos from file
        file.read((char*)(fr->atomPos), this->atomCnt * 3 * sizeof(float));

        // Close file
        file.close();

        // Calc displacement
#pragma omp parallel for
        for (int at = 0; at < static_cast<int>(dipoleCnt); at++) {
            fr->dipole[at * 3 + 0] *= -1.0f;
            fr->dipole[at * 3 + 0] += fr->atomPos[at * 3 + 0];
            fr->dipole[at * 3 + 1] *= -1.0f;
            fr->dipole[at * 3 + 1] += fr->atomPos[at * 3 + 1];
            fr->dipole[at * 3 + 2] *= -1.0f;
            fr->dipole[at * 3 + 2] += fr->atomPos[at * 3 + 2];
            fr->dipolePos[at * 3 + 0] = fr->atomPos[at * 3 + 0]; // TODO Use memcpy
            fr->dipolePos[at * 3 + 1] = fr->atomPos[at * 3 + 1];
            fr->dipolePos[at * 3 + 2] = fr->atomPos[at * 3 + 2];
        }
    }
    // Use ONLY displacement of Ti atoms, however use positions of all atoms
    else if (this->dSource == DIPOLE_DISPLTI) {

        file.open(this->fileFramesSlot.Param<core::param::FilePathParam>()->Value(), std::ios::in | std::ios::binary);
        if (!file.good()) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "CrystalStructureDataSource::Frame: Could not open file %s",
                this->fileFramesSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
            return false;
        }
        file.seekg(0, std::ios::beg);

        if (static_cast<int>(fr->GetFrameIdx()) - static_cast<int>(this->displOffs) < 0) {
            displFrameIdx = fr->GetFrameIdx(); // displacement = 0 in this case
        } else {
            displFrameIdx = static_cast<int>(fr->FrameNumber()) - static_cast<int>(this->displOffs);
        }

        // Skip preceeding frames
        file.seekg(this->atomCnt * 3 * sizeof(float) * displFrameIdx, std::ios::beg);

        if (!file.good()) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "CrystalStructureDataSource::Frame: Could not parse file %s (invalid filepointer %i)",
                this->fileFramesSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str(),
                static_cast<int>(file.tellg()));
            return false;
        }

        // Read frame data of the displacement frame from file
        file.read((char*)(fr->atomPos), this->atomCnt * 3 * sizeof(float));
        // Calc displacement
#pragma omp parallel for
        for (int di = 0; di < static_cast<int>(dipoleCnt); di++) {
            int tiIdx = this->cells[di * 15 + 14];
            fr->dipole[di * 3 + 0] = -fr->atomPos[tiIdx * 3 + 0];
            fr->dipole[di * 3 + 1] = -fr->atomPos[tiIdx * 3 + 1];
            fr->dipole[di * 3 + 2] = -fr->atomPos[tiIdx * 3 + 2];
        }

        // Skip frames
        // Note: -1 because we already skipped one frame by reading it
        //file.seekg(this->atomCnt*3*sizeof(float)*(displOffs-1), std::ios::cur);
        file.seekg(0, std::ios::beg);
        file.seekg(this->atomCnt * 3 * sizeof(float) * fr->GetFrameIdx(), std::ios::beg);

        // Read atom pos from file
        file.read((char*)(fr->atomPos), this->atomCnt * 3 * sizeof(float));

        // Close file
        file.close();

        // Calc displacement
#pragma omp parallel for
        for (int di = 0; di < static_cast<int>(dipoleCnt); di++) {
            unsigned int tiIdx = this->cells[di * 15 + 14];
            fr->dipole[di * 3 + 0] += fr->atomPos[tiIdx * 3 + 0];
            fr->dipole[di * 3 + 1] += fr->atomPos[tiIdx * 3 + 1];
            fr->dipole[di * 3 + 2] += fr->atomPos[tiIdx * 3 + 2];
            fr->dipolePos[di * 3 + 0] = fr->atomPos[tiIdx * 3 + 0];
            fr->dipolePos[di * 3 + 1] = fr->atomPos[tiIdx * 3 + 1];
            fr->dipolePos[di * 3 + 2] = fr->atomPos[tiIdx * 3 + 2];
        }
    } else if (this->dSource == DIPOLE_CELL) {

        // Read atom positions
        file.open(this->fileFramesSlot.Param<core::param::FilePathParam>()->Value(), std::ios::in | std::ios::binary);
        if (!file.good()) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "CrystalStructureDataSource::Frame: Could not open file %s",
                this->fileFramesSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
            return false;
        }

        // Skip preceeding frames
        file.seekg(0, std::ios::beg);
        file.seekg(this->atomCnt * 3 * sizeof(float) * fr->GetFrameIdx(), std::ios::beg);

        // Read atom pos from file
        file.read((char*)(fr->atomPos), this->atomCnt * 3 * sizeof(float));

        // Close file
        file.close();

        // Loop through all cells
#pragma omp parallel for
        for (int di = 0; di < static_cast<int>(dipoleCnt); di++) {

            int idxTiAtom = this->cells[15 * di + 14];

            // Check whether the cell is valid, if not go to next cell
            bool isValid = true;
            for (int cnt = 0; cnt < 15; cnt++) {
                if (this->cells[di * 15 + cnt] == -1)
                    isValid = false;
            }
            if (!isValid) {
                fr->dipole[3 * di + 0] = 0.0f;
                fr->dipole[3 * di + 1] = 0.0f;
                fr->dipole[3 * di + 2] = 0.0f;
                continue;
            }

            vislib::math::Vector<float, 3> tmpVec1, tmpVec2, diffVec;

            // Calculate spacial center of anions (= oxygen atoms)
            float anionCenter[] = {0.0, 0.0, 0.0};
            for (int oxy = 8; oxy < 14; oxy++) {
                anionCenter[0] += fr->atomPos[3 * this->cells[15 * di + oxy] + 0];
                anionCenter[1] += fr->atomPos[3 * this->cells[15 * di + oxy] + 1];
                anionCenter[2] += fr->atomPos[3 * this->cells[15 * di + oxy] + 2];
            }
            /*anionCenter[0] /= 6.0f;
            anionCenter[1] /= 6.0f;
            anionCenter[2] /= 6.0f;*/

            tmpVec1.Set(anionCenter[0], anionCenter[1], anionCenter[2]);

            // Calculate spacial center of cations (= titanium and barium atoms)
            float cationCenter[] = {0.0f, 0.0f, 0.0f};
            for (int at = 0; at < 8; at++) {
                cationCenter[0] += 2.0f * fr->atomPos[3 * this->cells[15 * di + at] + 0];
                cationCenter[1] += 2.0f * fr->atomPos[3 * this->cells[15 * di + at] + 1];
                cationCenter[2] += 2.0f * fr->atomPos[3 * this->cells[15 * di + at] + 2];
            }
            cationCenter[0] /= 8.0f;
            cationCenter[1] /= 8.0f;
            cationCenter[2] /= 8.0f;

            cationCenter[0] += 4.0f * fr->atomPos[3 * idxTiAtom + 0];
            cationCenter[1] += 4.0f * fr->atomPos[3 * idxTiAtom + 1];
            cationCenter[2] += 4.0f * fr->atomPos[3 * idxTiAtom + 2];

            tmpVec2.Set(cationCenter[0], cationCenter[1], cationCenter[2]);

            diffVec = tmpVec2 - tmpVec1;

            fr->dipole[3 * di + 0] = diffVec.X();
            fr->dipole[3 * di + 1] = diffVec.Y();
            fr->dipole[3 * di + 2] = diffVec.Z();

            fr->dipolePos[3 * di + 0] = fr->atomPos[3 * idxTiAtom + 0];
            fr->dipolePos[3 * di + 1] = fr->atomPos[3 * idxTiAtom + 1];
            fr->dipolePos[3 * di + 2] = fr->atomPos[3 * idxTiAtom + 2];
        }
    } else if (this->dSource == CHKPT_SOURCE) {


        geocalls::MultiParticleDataCall* dirc = this->dataChkptCallerSlot.CallAs<geocalls::MultiParticleDataCall>();

        dirc->SetFrameID(0);
        if (!(*dirc)(0)) { // Try call for data
            return false;
        }
        if (!(*dirc)(1))
            return false;

        this->dipoleCnt = (unsigned int)dirc->AccessParticles(0).GetCount();

        // Note: we only have one particle list in this case
        geocalls::MultiParticleDataCall::Particles& parts = dirc->AccessParticles(0);
        const float* pos = static_cast<const float*>(dirc->AccessParticles(0).GetVertexData());
        const float* dir = static_cast<const float*>(dirc->AccessParticles(0).GetDirData());

        //printf("*.chkpt dipole count %u\n", this->dipoleCnt); // DEBUG

        // Get dipoles
        if (parts.GetDirDataType() == geocalls::MultiParticleDataCall::Particles::DIRDATA_FLOAT_XYZ) {
#pragma omp parallel for
            for (int c = 0; c < static_cast<int>(this->dipoleCnt); c++) {
                fr->dipole[3 * c + 0] = dir[7 * c + 0];
                fr->dipole[3 * c + 1] = dir[7 * c + 1];
                fr->dipole[3 * c + 2] = dir[7 * c + 2];
            }
        } else {
#pragma omp parallel for
            for (int c = 0; c < static_cast<int>(this->dipoleCnt); c++) {
                fr->dipole[3 * c + 0] = 0.0f;
                fr->dipole[3 * c + 1] = 0.0f;
                fr->dipole[3 * c + 2] = 0.0f;
            }
        }

        // Get dipole positions
        if (parts.GetVertexDataType() == geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) {
#pragma omp parallel for
            for (int c = 0; c < static_cast<int>(this->dipoleCnt); c++) {
                fr->dipolePos[3 * c + 0] = pos[7 * c + 0];
                fr->dipolePos[3 * c + 1] = pos[7 * c + 1];
                fr->dipolePos[3 * c + 2] = pos[7 * c + 2];

                fr->atomPos[3 * c + 0] = pos[7 * c + 0];
                fr->atomPos[3 * c + 1] = pos[7 * c + 1];
                fr->atomPos[3 * c + 2] = pos[7 * c + 2];
            }
        } else {
#pragma omp parallel for
            for (int c = 0; c < static_cast<int>(this->dipoleCnt); c++) {
                fr->dipolePos[3 * c + 0] = 0.0f;
                fr->dipolePos[3 * c + 1] = 0.0f;
                fr->dipolePos[3 * c + 2] = 0.0f;

                fr->atomPos[3 * c + 0] = 0.0f;
                fr->atomPos[3 * c + 1] = 0.0f;
                fr->atomPos[3 * c + 2] = 0.0f;
            }
        }
    } else if (this->dSource == VECFIELD_PROC) {
        unsigned int idx = 0;
        for (int x = 0; x < VECFIELD_PROC_SIZE; x++) {
            for (int y = 0; y < VECFIELD_PROC_SIZE; y++) {
                for (int z = 0; z < VECFIELD_PROC_SIZE; z++) {
                    //unsigned int idx = VECFIELD_PROC_SIZE*(VECFIELD_PROC_SIZE*z+y)+x;
                    fr->dipolePos[3 * idx + 0] = static_cast<float>(x) * VECFIELD_PROC_SCL;
                    fr->dipolePos[3 * idx + 1] = static_cast<float>(y) * VECFIELD_PROC_SCL;
                    fr->dipolePos[3 * idx + 2] = static_cast<float>(z) * VECFIELD_PROC_SCL;
                    fr->dipole[3 * idx + 0] = static_cast<float>(x);
                    fr->dipole[3 * idx + 1] = static_cast<float>(y);
                    fr->dipole[3 * idx + 2] = static_cast<float>(z);
                    idx++;
                }
            }
        }
        /*printf("Number of Dipoles %u\n", idx);
        for(unsigned int dp = 0;  dp < this->dipoleCnt; dp++) {
            printf("%i (%f %f %f): %f %f %f\n", dp,
                    fr->dipolePos[dp*3+0], fr->dipolePos[dp*3+1], fr->dipolePos[dp*3+2],
                    fr->dipole[dp*3+0], fr->dipole[dp*3+1], fr->dipole[dp*3+2]);
        } // DEBUG*/
    } else if (this->dSource == DIPOLE_BATI) {

        // Read atom positions
        file.open(this->fileFramesSlot.Param<core::param::FilePathParam>()->Value(), std::ios::in | std::ios::binary);
        if (!file.good()) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "CrystalStructureDataSource::Frame: Could not open file %s",
                this->fileFramesSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
            return false;
        }

        // Skip preceeding frames
        file.seekg(0, std::ios::beg);
        file.seekg(this->atomCnt * 3 * sizeof(float) * fr->GetFrameIdx(), std::ios::beg);

        // Read atom pos from file
        file.read((char*)(fr->atomPos), this->atomCnt * 3 * sizeof(float));

        // Close file
        file.close();

        // Loop through all cells
#pragma omp parallel for
        for (int di = 0; di < static_cast<int>(dipoleCnt); di++) {

            int idxTiAtom = this->cells[15 * di + 14];

            // Check whether the cell is valid, if not go to next cell
            bool isValid = true;
            for (int cnt = 0; cnt < 15; cnt++) {
                if (this->cells[di * 15 + cnt] == -1)
                    isValid = false;
            }
            if (!isValid) {
                fr->dipole[3 * di + 0] = 0.0f;
                fr->dipole[3 * di + 1] = 0.0f;
                fr->dipole[3 * di + 2] = 0.0f;
                continue;
            }

            vislib::math::Vector<float, 3> tmpVec1, tmpVec2, diffVec;

            tmpVec1.Set(fr->atomPos[3 * idxTiAtom + 0], fr->atomPos[3 * idxTiAtom + 1], fr->atomPos[3 * idxTiAtom + 2]);

            // Calculate spacial center of cations (= titanium and barium atoms)
            float baCenter[] = {0.0f, 0.0f, 0.0f};
            for (int at = 0; at < 8; at++) {
                baCenter[0] += fr->atomPos[3 * this->cells[15 * di + at] + 0];
                baCenter[1] += fr->atomPos[3 * this->cells[15 * di + at] + 1];
                baCenter[2] += fr->atomPos[3 * this->cells[15 * di + at] + 2];
            }
            baCenter[0] /= 8.0f;
            baCenter[1] /= 8.0f;
            baCenter[2] /= 8.0f;

            tmpVec2.Set(baCenter[0], baCenter[1], baCenter[2]);

            diffVec = tmpVec1 - tmpVec2;

            fr->dipole[3 * di + 0] = diffVec.X();
            fr->dipole[3 * di + 1] = diffVec.Y();
            fr->dipole[3 * di + 2] = diffVec.Z();

            fr->dipolePos[3 * di + 0] = fr->atomPos[3 * idxTiAtom + 0];
            fr->dipolePos[3 * di + 1] = fr->atomPos[3 * idxTiAtom + 1];
            fr->dipolePos[3 * di + 2] = fr->atomPos[3 * idxTiAtom + 2];
        }
    } else if (this->dSource == DIPOLE_NOBA) {

        // Read atom positions
        file.open(this->fileFramesSlot.Param<core::param::FilePathParam>()->Value(), std::ios::in | std::ios::binary);
        if (!file.good()) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "CrystalStructureDataSource::Frame: Could not open file %s",
                this->fileFramesSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
            return false;
        }

        // Skip preceeding frames
        file.seekg(0, std::ios::beg);
        file.seekg(this->atomCnt * 3 * sizeof(float) * fr->GetFrameIdx(), std::ios::beg);

        // Read atom pos from file
        file.read((char*)(fr->atomPos), this->atomCnt * 3 * sizeof(float));

        // Close file
        file.close();

        // Loop through all cells
#pragma omp parallel for
        for (int di = 0; di < static_cast<int>(dipoleCnt); di++) {

            int idxTiAtom = this->cells[15 * di + 14];

            // Check whether the cell is valid, if not go to next cell
            bool isValid = true;
            for (int cnt = 0; cnt < 15; cnt++) {
                if (this->cells[di * 15 + cnt] == -1)
                    isValid = false;
            }
            if (!isValid) {
                fr->dipole[3 * di + 0] = 0.0f;
                fr->dipole[3 * di + 1] = 0.0f;
                fr->dipole[3 * di + 2] = 0.0f;
                continue;
            }

            vislib::math::Vector<float, 3> tmpVec1, tmpVec2, diffVec;

            // Calculate spacial center of anions (= oxygen atoms)
            float anionCenter[] = {0.0, 0.0, 0.0};
            for (int oxy = 8; oxy < 14; oxy++) {
                anionCenter[0] += fr->atomPos[3 * this->cells[15 * di + oxy] + 0];
                anionCenter[1] += fr->atomPos[3 * this->cells[15 * di + oxy] + 1];
                anionCenter[2] += fr->atomPos[3 * this->cells[15 * di + oxy] + 2];
            }
            anionCenter[0] /= 6.0f;
            anionCenter[1] /= 6.0f;
            anionCenter[2] /= 6.0f;

            tmpVec1.Set(anionCenter[0], anionCenter[1], anionCenter[2]);

            // Calculate spacial center of cations (= titanium and barium atoms)
            float cationCenter[] = {0.0f, 0.0f, 0.0f};
            cationCenter[0] += fr->atomPos[3 * idxTiAtom + 0];
            cationCenter[1] += fr->atomPos[3 * idxTiAtom + 1];
            cationCenter[2] += fr->atomPos[3 * idxTiAtom + 2];

            tmpVec2.Set(cationCenter[0], cationCenter[1], cationCenter[2]);

            diffVec = tmpVec2 - tmpVec1;

            fr->dipole[3 * di + 0] = diffVec.X();
            fr->dipole[3 * di + 1] = diffVec.Y();
            fr->dipole[3 * di + 2] = diffVec.Z();

            fr->dipolePos[3 * di + 0] = fr->atomPos[3 * idxTiAtom + 0];
            fr->dipolePos[3 * di + 1] = fr->atomPos[3 * idxTiAtom + 1];
            fr->dipolePos[3 * di + 2] = fr->atomPos[3 * idxTiAtom + 2];
        }
    }
    //for(unsigned int at = 1000; at < 2000; at++) {
    //  printf("%i (DIPOLE %u, offset %u): %f %f %f\n", at, this->frame,
    //          displOffs, this->dipole[at*3+0],
    //          this->dipole[at*3+1], this->dipole[at*3+2]);
    //} // DEBUG

    /*Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
            "%s: time for loading frame %f",
            this->ClassName(),
            (double(clock()-t)/double(CLOCKS_PER_SEC)));*/ // DEBUG

    return true;
}


/*
 * protein::CrystalStructureDataSource::Frame::Frame
 */
protein::CrystalStructureDataSource::Frame::Frame(core::view::AnimDataModule& owner)
        : core::view::AnimDataModule::Frame(owner)
        , atomPos(NULL)
        , dipolePos(NULL)
        , dipole(NULL)
        , atomCnt(0)
        , dipoleCnt(0) {}


/*
 * protein::CrystalStructureDataSource::Frame::~Frame
 */
protein::CrystalStructureDataSource::Frame::~Frame(void) {
    if (this->atomPos != NULL)
        delete[] this->atomPos;
    if (this->dipole != NULL)
        delete[] this->dipole;
    if (this->dipolePos != NULL)
        delete[] this->dipolePos;
}
