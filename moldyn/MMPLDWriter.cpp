/*
 * MMPLDWriter.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MMPLDWriter.h"
#include "BoundingBoxes.h"
#include "param/FilePathParam.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/MemmappedFile.h"
#include "vislib/String.h"
#include "vislib/sys/Thread.h"
#include <algorithm>

using namespace megamol::core;

//#define WITH_CLUSTERINFO

/*
 * moldyn::MMPLDWriter::MMPLDWriter
 */
moldyn::MMPLDWriter::MMPLDWriter(void) : AbstractDataWriter(),
        filenameSlot("filename", "The path to the MMPLD file to be written"),
        dataSlot("data", "The slot requesting the data to be written") {

    this->filenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->dataSlot.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->dataSlot);
}


/*
 * moldyn::MMPLDWriter::~MMPLDWriter
 */
moldyn::MMPLDWriter::~MMPLDWriter(void) {
    this->Release();
}


/*
 * moldyn::MMPLDWriter::create
 */
bool moldyn::MMPLDWriter::create(void) {
    return true;
}


/*
 * moldyn::MMPLDWriter::release
 */
void moldyn::MMPLDWriter::release(void) {
}


/*
 * moldyn::MMPLDWriter::run
 */
bool moldyn::MMPLDWriter::run(void) {
    using vislib::sys::Log;
    vislib::TString filename(this->filenameSlot.Param<param::FilePathParam>()->Value());
    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "No file name specified. Abort.");
        return false;
    }

    MultiParticleDataCall *mpdc = this->dataSlot.CallAs<MultiParticleDataCall>();
    if (mpdc == NULL) {
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
    if (!(*mpdc)(1)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Unable to query data extents.");
        bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        cbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    } else {
        if (mpdc->AccessBoundingBoxes().IsObjectSpaceBBoxValid()
                || mpdc->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
            if (mpdc->AccessBoundingBoxes().IsObjectSpaceBBoxValid()) {
                bbox = mpdc->AccessBoundingBoxes().ObjectSpaceBBox();
            } else {
                bbox = mpdc->AccessBoundingBoxes().ObjectSpaceClipBox();
            }
            if (mpdc->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
                cbox = mpdc->AccessBoundingBoxes().ObjectSpaceClipBox();
            } else {
                cbox = mpdc->AccessBoundingBoxes().ObjectSpaceBBox();
            }
        } else {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Object space bounding boxes not valid. Using defaults");
            bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
            cbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        }
        frameCnt = mpdc->FrameCount();
        if (frameCnt == 0) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "Data source counts zero frames. Abort.");
            mpdc->Unlock();
            return false;
        }
    }

    // DEBUG
//    frameCnt = 10;
    // END DEBUG

    vislib::sys::MemmappedFile file;
    if (!file.Open(filename, vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE, vislib::sys::File::CREATE_OVERWRITE)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to create output file \"%s\". Abort.",
            vislib::StringA(filename).PeekBuffer());
        mpdc->Unlock();
        return false;
    }

#define ASSERT_WRITEOUT(A, S) if (file.Write((A), (S)) != (S)) { \
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Write error %d", __LINE__); \
        file.Close(); \
        mpdc->Unlock(); \
        return false; \
    }

    vislib::StringA magicID("MMPLD");
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

    mpdc->Unlock();
    for (UINT32 i = 0; i < frameCnt; i++) {
        frameOffset = static_cast<UINT64>(file.Tell());
        file.Seek(seekTable + i * 8);
        ASSERT_WRITEOUT(&frameOffset, 8);
        file.Seek(frameOffset);

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Started writing data frame %u\n", i);

        int missCnt = -9;
        do {
            mpdc->Unlock();
            mpdc->SetFrameID(i, true);
            if (!(*mpdc)(0)) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Cannot get data frame %u. Abort.\n", i);
                file.Close();
                return false;
            }
            if (mpdc->FrameID() != i) {
                if ((missCnt % 10) == 0) {
                    Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
                        "Frame %u returned on request for frame %u\n", mpdc->FrameID(), i);
                }
                ++missCnt;
                vislib::sys::Thread::Sleep(static_cast<DWORD>(1 + std::max(missCnt, 0) * 100));
            }
        } while(mpdc->FrameID() != i);

        if (!this->writeFrame(file, *mpdc)) {
            mpdc->Unlock();
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Cannot write data frame %u. Abort.\n", i);
            file.Close();
            return false;
        }
        mpdc->Unlock();
    }

    frameOffset = static_cast<UINT64>(file.Tell());
    file.Seek(seekTable + frameCnt * 8);
    ASSERT_WRITEOUT(&frameOffset, 8);

    file.Seek(6); // set correct version to show that file is complete
    version = 100;
#ifdef WITH_CLUSTERINFO
    version++;
#endif
    ASSERT_WRITEOUT(&version, 2);

    file.Seek(frameOffset);

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Completed writing data\n");
    file.Close();

#undef ASSERT_WRITEOUT
    return true;
}


/*
 * moldyn::MMPLDWriter::getCapabilities
 */
bool moldyn::MMPLDWriter::getCapabilities(DataWriterCtrlCall& call) {
    call.SetAbortable(false);
    return true;
}


/*
 * moldyn::MMPLDWriter::writeFrame
 */
bool moldyn::MMPLDWriter::writeFrame(vislib::sys::File& file, moldyn::MultiParticleDataCall& data) {
#define ASSERT_WRITEOUT(A, S) if (file.Write((A), (S)) != (S)) { \
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Write error %d", __LINE__); \
        file.Close(); \
        return false; \
    }
    using vislib::sys::Log;
    UINT32 listCnt = data.GetParticleListCount();
    ASSERT_WRITEOUT(&listCnt, 4);

    for (UINT32 li = 0; li < listCnt; li++) {
        MultiParticleDataCall::Particles &points = data.AccessParticles(li);
        UINT8 vt = 0, ct = 0;
        unsigned int vs = 0, vo = 0, cs = 0, co = 0;
        switch(points.GetVertexDataType()) {
            case MultiParticleDataCall::Particles::VERTDATA_NONE: vt = 0; vs = 0; break;
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ: vt = 1; vs = 12; break;
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR: vt = 2; vs = 16; break;
            case MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ: vt = 3; vs = 6; break;
            default: vt = 0; vs = 0; break;
        }
        if (vt != 0) {
            switch(points.GetColourDataType()) {
                case MultiParticleDataCall::Particles::COLDATA_NONE: ct = 0; cs = 0; break;
                case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB: ct = 1; cs = 3; break;
                case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA: ct = 2; cs = 4; break;
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: ct = 3; cs = 4; break;
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB: ct = 4; cs = 12; break;
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA: ct = 5; cs = 16; break;
                default: ct = 0; cs = 0; break;
            }
        } else {
            ct = 0;
        }
        ASSERT_WRITEOUT(&vt, 1);
        ASSERT_WRITEOUT(&ct, 1);

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

        if ((vt == 1) || (vt == 3)) {
            float f = points.GetGlobalRadius();
            ASSERT_WRITEOUT(&f, 4);
        }
        if (ct == 0) {
            const unsigned char *col = points.GetGlobalColour();
            ASSERT_WRITEOUT(col, 4);
        } else if (ct == 3) {
            float f = points.GetMinColourIndexValue();
            ASSERT_WRITEOUT(&f, 4);
            f = points.GetMaxColourIndexValue();
            ASSERT_WRITEOUT(&f, 4);
        }

        UINT64 cnt = points.GetCount();
        if (vt == 0) cnt = 0;
        ASSERT_WRITEOUT(&cnt, 8);
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
#ifdef WITH_CLUSTERINFO
        if (points.GetClusterInfos() != NULL) {
            ASSERT_WRITEOUT(&points.GetClusterInfos()->numClusters, sizeof(unsigned int));
            ASSERT_WRITEOUT(&points.GetClusterInfos()->sizeofPlainData, sizeof(size_t));
            ASSERT_WRITEOUT(points.GetClusterInfos()->plainData, points.GetClusterInfos()->sizeofPlainData);
        } else {
            unsigned int zero1 = 0u;
            size_t zero2 = 0;
            ASSERT_WRITEOUT(&zero1, sizeof(unsigned int));
            ASSERT_WRITEOUT(&zero2, sizeof(size_t));
        }
#endif
    }

    return true;
#undef ASSERT_WRITEOUT
}
