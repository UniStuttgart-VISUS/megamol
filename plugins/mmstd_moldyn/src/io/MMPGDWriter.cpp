/*
 * MMPGDWriter.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MMPGDWriter.h"
#include "mmcore/BoundingBoxes.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/DataWriterCtrlCall.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/FastFile.h"
#include "vislib/String.h"
#include "vislib/sys/Thread.h"

using namespace megamol::stdplugin::moldyn::io;
using namespace megamol::stdplugin::moldyn::rendering;
using namespace megamol::core;

/*
 * MMPGDWriter::MMPGDWriter
 */
MMPGDWriter::MMPGDWriter(void) : AbstractDataWriter(),
        filenameSlot("filename", "The path to the MMPGD file to be written"),
        dataSlot("data", "The slot requesting the data to be written") {

    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->dataSlot.SetCompatibleCall<ParticleGridDataCallDescription>();
    this->MakeSlotAvailable(&this->dataSlot);
}


/*
 * MMPGDWriter::~MMPGDWriter
 */
MMPGDWriter::~MMPGDWriter(void) {
    this->Release();
}


/*
 * MMPGDWriter::create
 */
bool MMPGDWriter::create(void) {
    return true;
}


/*
 * MMPGDWriter::release
 */
void MMPGDWriter::release(void) {
}


/*
 * MMPGDWriter::run
 */
bool MMPGDWriter::run(void) {
    using vislib::sys::Log;
    vislib::TString filename(this->filenameSlot.Param<core::param::FilePathParam>()->Value());
    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "No file name specified. Abort.");
        return false;
    }

    ParticleGridDataCall *pgdc = this->dataSlot.CallAs<ParticleGridDataCall>();
    if (pgdc == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "No data source connected. Abort.");
        return false;
    }

    if (vislib::sys::File::Exists(filename)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
            "File %s already exists and will be overwritten.",
            vislib::StringA(filename).PeekBuffer());
    }

    vislib::math::Cuboid<float> bbox;
    vislib::math::Cuboid<float> cbox;
    UINT32 frameCnt = 1;
    if (!(*pgdc)(1)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Unable to query data extents.");
        bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        cbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    } else {
        if (pgdc->AccessBoundingBoxes().IsObjectSpaceBBoxValid()
                || pgdc->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
            if (pgdc->AccessBoundingBoxes().IsObjectSpaceBBoxValid()) {
                bbox = pgdc->AccessBoundingBoxes().ObjectSpaceBBox();
            } else {
                bbox = pgdc->AccessBoundingBoxes().ObjectSpaceClipBox();
            }
            if (pgdc->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
                cbox = pgdc->AccessBoundingBoxes().ObjectSpaceClipBox();
            } else {
                cbox = pgdc->AccessBoundingBoxes().ObjectSpaceBBox();
            }
        } else {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Object space bounding boxes not valid. Using defaults");
            bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
            cbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        }
        frameCnt = pgdc->FrameCount();
        if (frameCnt == 0) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Data source counts zero frames. Abort.");
            return false;
        }
    }

    vislib::sys::FastFile file;
    if (!file.Open(filename, vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE, vislib::sys::File::CREATE_OVERWRITE)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to create output file \"%s\". Abort.",
            vislib::StringA(filename).PeekBuffer());
        return false;
    }

#define ASSERT_WRITEOUT(A, S) if (file.Write((A), (S)) != (S)) { \
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Write error %d", __LINE__); \
        file.Close(); \
        return false; \
    }

    vislib::StringA magicID("MMPGD");
    ASSERT_WRITEOUT(magicID.PeekBuffer(), 6);
    UINT16 version = 0;
    ASSERT_WRITEOUT(&version, 2);
    ASSERT_WRITEOUT(&frameCnt, 4);
    ASSERT_WRITEOUT(bbox.PeekBounds(), 6 * 4);
    ASSERT_WRITEOUT(cbox.PeekBounds(), 6 * 4);

    UINT64 seekTable = static_cast<UINT64>(file.Tell());
    UINT64 frameOffset = 0;
    for (UINT32 i = 0; i <= frameCnt; i++) {
        ASSERT_WRITEOUT(&frameOffset, 8);
    }

    for (UINT32 i = 0; i < frameCnt; i++) {
        frameOffset = static_cast<UINT64>(file.Tell());
        file.Seek(seekTable + i * 8);
        ASSERT_WRITEOUT(&frameOffset, 8);
        file.Seek(frameOffset);

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Started writing data frame %u\n", i);

        unsigned int missCnt = 0;
        do {
            pgdc->SetFrameID(i, true);
            if (!(*pgdc)(0)) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Cannot get data frame %u. Abort.\n", i);
                file.Close();
                return false;
            }
            if (pgdc->FrameID() != i) {
                if ((missCnt % 10) == 0) {
                    Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
                        "Frame %u returned on request for frame %u\n", pgdc->FrameID(), i);
                }
                missCnt++;
                vislib::sys::Thread::Sleep(missCnt * 100);
            }
        } while(pgdc->FrameID() != i);

        if (!this->writeFrame(file, *pgdc)) {
            pgdc->Unlock();
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Cannot write data frame %u. Abort.\n", i);
            file.Close();
            return false;
        }
        pgdc->Unlock();
    }

    frameOffset = static_cast<UINT64>(file.Tell());
    file.Seek(seekTable + frameCnt * 8);
    ASSERT_WRITEOUT(&frameOffset, 8);

    file.Seek(6); // set correct version to show that file is complete
    version = 100;
    ASSERT_WRITEOUT(&version, 2);

    file.Seek(frameOffset);

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Completed writing data\n");
    file.Close();

    return true;
}


/*
 * MMPGDWriter::getCapabilities
 */
bool MMPGDWriter::getCapabilities(DataWriterCtrlCall& call) {
    call.SetAbortable(false);
    return true;
}


/*
 * MMPGDWriter::writeFrame
 */
bool MMPGDWriter::writeFrame(vislib::sys::File& file, ParticleGridDataCall& data) {
    using vislib::sys::Log;

    UINT32 typeCnt = static_cast<UINT32>(data.TypesCount());
    ASSERT_WRITEOUT(&typeCnt, 4);

    UINT32 gridX = static_cast<UINT32>(data.CellsXCount());
    UINT32 gridY = static_cast<UINT32>(data.CellsYCount());
    UINT32 gridZ = static_cast<UINT32>(data.CellsZCount());
    ASSERT_WRITEOUT(&gridX, 4);
    ASSERT_WRITEOUT(&gridY, 4);
    ASSERT_WRITEOUT(&gridZ, 4);

    for (UINT32 i = 0; i < typeCnt; i++) {
        const ParticleGridDataCall::ParticleType &type = data.Types()[i];
        UINT8 vt = 0, ct = 0;
        switch(type.GetVertexDataType()) {
            case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE: vt = 0; break;
            case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ: vt = 1; break;
            case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR: vt = 2; break;
            case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ: vt = 3; break;
            default: vt = 0; break;
        }
        if (vt != 0) {
            switch(type.GetColourDataType()) {
                case core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE: ct = 0; break;
                case core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB: ct = 1; break;
                case core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA: ct = 2; break;
                case core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I: ct = 3; break;
                case core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB: ct = 4; break;
                case core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA: ct = 5; break;
                default: ct = 0; break;
            }
        } else {
            ct = 0;
        }
        ASSERT_WRITEOUT(&vt, 1);
        ASSERT_WRITEOUT(&ct, 1);

        if ((vt == 1) || (vt == 3)) {
            float f = type.GetGlobalRadius();
            ASSERT_WRITEOUT(&f, 4);
        }
        if (ct == 0) {
            const unsigned char *col = type.GetGlobalColour();
            ASSERT_WRITEOUT(col, 4);
        } else if (ct == 3) {
            float f = type.GetMinColourIndexValue();
            ASSERT_WRITEOUT(&f, 4);
            f = type.GetMaxColourIndexValue();
            ASSERT_WRITEOUT(&f, 4);
        }
    }

    for (UINT32 i = 0; i < gridX * gridY * gridZ; i++) {
        const ParticleGridDataCall::GridCell &cell = data.Cells()[i];
        ASSERT_WRITEOUT(cell.GetBoundingBox().PeekBounds(), 4 * 6);
        for (UINT32 t = 0; t < typeCnt; t++) {
            const ParticleGridDataCall::ParticleType &type = data.Types()[t];
            const ParticleGridDataCall::Particles &points = cell.AccessParticleLists()[t];

            UINT8 vt = 0, ct = 0;
            unsigned int vs = 0, vo = 0, cs = 0, co = 0;
            switch(type.GetVertexDataType()) {
                case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE: vt = 0; vs = 0; break;
                case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ: vt = 1; vs = 12; break;
                case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR: vt = 2; vs = 16; break;
                case core::moldyn::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ: vt = 3; vs = 6; break;
                default: vt = 0; vs = 0; break;
            }
            if (vt != 0) {
                switch(type.GetColourDataType()) {
                    case core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE: ct = 0; cs = 0; break;
                    case core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB: ct = 1; cs = 3; break;
                    case core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA: ct = 2; cs = 4; break;
                    case core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I: ct = 3; cs = 4; break;
                    case core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB: ct = 4; cs = 12; break;
                    case core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA: ct = 5; cs = 16; break;
                    default: ct = 0; cs = 0; break;
                }
            } else {
                ct = 0;
            }

            if (points.GetVertexDataStride() > vs) {
                vo = points.GetVertexDataStride();
            } else {
                vo = vs;
            }
            if (points.GetColourDataStride() > cs) {
                co = points.GetColourDataStride();
            } else {
                co = cs;
            }

            UINT64 cnt = points.GetCount();
            if (vt == 0) cnt = 0;
            ASSERT_WRITEOUT(&cnt, 8);
            float maxRad = points.GetMaxRadius();
            ASSERT_WRITEOUT(&maxRad, 4);
            if (vt == 0) continue;
            const unsigned char *vp = static_cast<const unsigned char *>(points.GetVertexData());
            const unsigned char *cp = static_cast<const unsigned char *>(points.GetColourData());
            for (UINT64 i = 0; i < cnt; i++) {
                ASSERT_WRITEOUT(vp, vs);
                vp += vo;
                if (ct != 0) {
                    ASSERT_WRITEOUT(cp, cs);
                    cp += co;
                }
            }
        }
    }

    return true;
}

#undef ASSERT_WRITEOUT
