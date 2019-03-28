/*
 * MMPLDWriter.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include <algorithm>
#include "mmcore/BoundingBoxes.h"
#include "mmcore/moldyn/MMPLDWriter.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/String.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/Thread.h"

using namespace megamol::core;

//#define WITH_CLUSTERINFO

/*
 * moldyn::MMPLDWriter::MMPLDWriter
 */
moldyn::MMPLDWriter::MMPLDWriter(void)
    : AbstractDataWriter()
    , filenameSlot("filename", "The path to the MMPLD file to be written")
    , versionSlot("version", "The file format version to be written")
    , dataSlot("data", "The slot requesting the data to be written") {

    this->filenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    param::EnumParam* verPar = new param::EnumParam(100);
    verPar->SetTypePair(100, "1.0");
#ifdef WITH_CLUSTERINFO
    verPar->SetTypePair(101, "1.1");
#endif
    verPar->SetTypePair(102, "1.2");
    verPar->SetTypePair(103, "1.3");
    this->versionSlot.SetParameter(verPar);
    this->MakeSlotAvailable(&this->versionSlot);

    this->dataSlot.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->dataSlot);
}


/*
 * moldyn::MMPLDWriter::~MMPLDWriter
 */
moldyn::MMPLDWriter::~MMPLDWriter(void) { this->Release(); }


/*
 * moldyn::MMPLDWriter::create
 */
bool moldyn::MMPLDWriter::create(void) { return true; }


/*
 * moldyn::MMPLDWriter::release
 */
void moldyn::MMPLDWriter::release(void) {}


/*
 * moldyn::MMPLDWriter::`
 */
bool moldyn::MMPLDWriter::run(void) {
    using vislib::sys::Log;
    vislib::TString filename(this->filenameSlot.Param<param::FilePathParam>()->Value());
    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "No file name specified. Abort.");
        return false;
    }

    MultiParticleDataCall* mpdc = this->dataSlot.CallAs<MultiParticleDataCall>();
    if (mpdc == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "No data source connected. Abort.");
        return false;
    }

    if (vislib::sys::File::Exists(filename)) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_WARN, "File %s already exists and will be overwritten.", vislib::StringA(filename).PeekBuffer());
    }

    vislib::math::Cuboid<float> bbox;
    vislib::math::Cuboid<float> cbox;
    UINT32 frameCnt = 1;
    if (!(*mpdc)(1)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Unable to query data extents.");
        bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        cbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    } else {
        if (mpdc->AccessBoundingBoxes().IsObjectSpaceBBoxValid() ||
            mpdc->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
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
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Data source counts zero frames. Abort.");
            mpdc->Unlock();
            return false;
        }
    }

    // DEBUG
    //    frameCnt = 10;
    // END DEBUG

    vislib::sys::FastFile file;
    if (!file.Open(filename, vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE,
            vislib::sys::File::CREATE_OVERWRITE)) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Unable to create output file \"%s\". Abort.", vislib::StringA(filename).PeekBuffer());
        mpdc->Unlock();
        return false;
    }

#define ASSERT_WRITEOUT(A, S)                                                                                          \
    if (file.Write((A), (S)) != (S)) {                                                                                 \
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Write error %d", __LINE__);                                        \
        file.Close();                                                                                                  \
        mpdc->Unlock();                                                                                                \
        return false;                                                                                                  \
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
            if (!(*mpdc)(1)) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Cannot request frame %u. Abort.\n", i);
                file.Close();
                return false;
            }
            if (!(*mpdc)(0)) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Cannot get data frame %u. Abort.\n", i);
                file.Close();
                return false;
            }
            if (mpdc->FrameID() != i) {
                if ((missCnt % 10) == 0) {
                    Log::DefaultLog.WriteMsg(
                        Log::LEVEL_WARN, "Frame %u returned on request for frame %u\n", mpdc->FrameID(), i);
                }
                ++missCnt;
                vislib::sys::Thread::Sleep(static_cast<DWORD>(1 + std::max<int>(missCnt, 0) * 100));
            }
        } while (mpdc->FrameID() != i);

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
    version = this->versionSlot.Param<param::EnumParam>()->Value();
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
#define ASSERT_WRITEOUT(A, S)                                                                                          \
    if (file.Write((A), (S)) != (S)) {                                                                                 \
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Write error %d", __LINE__);                                        \
        file.Close();                                                                                                  \
        return false;                                                                                                  \
    }
    using vislib::sys::Log;
    uint8_t const alpha = 255;
    int ver = this->versionSlot.Param<param::EnumParam>()->Value();

    if (ver == 102) {
        float ts = data.GetTimeStamp();
        ASSERT_WRITEOUT(&ts, 4);
    }

    UINT32 listCnt = data.GetParticleListCount();
    ASSERT_WRITEOUT(&listCnt, 4);

    for (UINT32 li = 0; li < listCnt; li++) {
        MultiParticleDataCall::Particles& points = data.AccessParticles(li);
        UINT8 vt = 0, ct = 0;
        unsigned int vs = 0, vo = 0, cs = 0, co = 0;
        switch (points.GetVertexDataType()) {
        case MultiParticleDataCall::Particles::VERTDATA_NONE:
            vt = 0;
            vs = 0;
            break;
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
            vt = 1;
            vs = 12;
            break;
        case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            vt = 2;
            vs = 16;
            break;
        case MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
            vt = 3;
            vs = 6;
            break;
        case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
            vt = 4;
            vs = 24;
            break;
        default:
            vt = 0;
            vs = 0;
            break;
        }
        if (vt != 0) {
            switch (points.GetColourDataType()) {
            case MultiParticleDataCall::Particles::COLDATA_NONE:
                ct = 0;
                cs = 0;
                break;
            case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                ct = 1;
                cs = 3;
                break;
            case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                ct = 2;
                cs = 4;
                break;
            case MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
                ct = 3;
                cs = 4;
                break;
            case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                ct = 4;
                cs = 12;
                break;
            case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                ct = 5;
                cs = 16;
                break;
            case MultiParticleDataCall::Particles::COLDATA_USHORT_RGBA:
                ct = 6;
                cs = 8;
                break;
            case MultiParticleDataCall::Particles::COLDATA_DOUBLE_I:
                ct = 7;
                cs = 8;
                break;
            default:
                ct = 0;
                cs = 0;
                break;
            }
        } else {
            ct = 0;
        }
        ASSERT_WRITEOUT(&vt, 1);
        if (ct == 1) ct = 2; // UINT8_RGB is unaligned and will never be written again.
        if (vt == 4 && ct < 5) { // TODO: fragile if we add another color type beyond DOUBLE_I!
            if (ct == 3) { // VERTDATA_DOUBLE_XYZ needs COLDATA_DOUBLE_I instead of COLDATA_FLOAT_I to be aligned for modern renderers (NG and OPSRay)
                UINT8 x = 7;
                ASSERT_WRITEOUT(&x, 1);
            } else { // VERTDATA_DOUBLE_XYZ needs COLDATA_USHORT_RGBA to be aligned for modern renderers (NG and OPSRay)
                UINT8 x = 6;
                ASSERT_WRITEOUT(&x, 1);
            }
        } else {
            ASSERT_WRITEOUT(&ct, 1);
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

        if ((vt == 1) || (vt == 3) || (vt == 4)) {
            float f = points.GetGlobalRadius();
            ASSERT_WRITEOUT(&f, 4);
        }
        if (ct == 0) {
            const unsigned char* col = points.GetGlobalColour();
            ASSERT_WRITEOUT(col, 4);
        } else if (ct == 3 || ct == 7) {
            float f = points.GetMinColourIndexValue();
            ASSERT_WRITEOUT(&f, 4);
            f = points.GetMaxColourIndexValue();
            ASSERT_WRITEOUT(&f, 4);
        }

        UINT64 cnt = points.GetCount();
        if (vt == 0) cnt = 0;
        ASSERT_WRITEOUT(&cnt, 8);

        if (ver == 103) {
            ASSERT_WRITEOUT(points.GetBBox().PeekBounds(), 24);
        }

        if (vt == 0) continue;
        const unsigned char* vp = static_cast<const unsigned char*>(points.GetVertexData());
        const unsigned char* cp = static_cast<const unsigned char*>(points.GetColourData());
        if (vt == 4 && ct < 5) {
            switch (points.GetColourDataType()) {
            case MultiParticleDataCall::Particles::COLDATA_NONE:
                {
                    auto col = points.GetGlobalColour();
                    uint16_t colNew[4] = {col[0] * 257, col[1] * 257, col[2] * 257, col[3] * 257};
                    for (UINT64 i = 0; i < cnt; ++i) {
                        ASSERT_WRITEOUT(vp, vs);
                        vp += vo;
                        ASSERT_WRITEOUT(colNew, 8);
                    }
                }
                break;
            case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                {
                    uint16_t colNew[4];
                    for (UINT64 i = 0; i < cnt; ++i) {
                        ASSERT_WRITEOUT(vp, vs);
                        vp += vo;
                        colNew[0] = cp[0] * 257;
                        colNew[1] = cp[1] * 257;
                        colNew[2] = cp[2] * 257;
                        colNew[3] = 65535;
                        ASSERT_WRITEOUT(colNew, 8);
                        cp += co;
                    }
                }
                break;
            case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                {
                    uint16_t colNew[4];
                    for (UINT64 i = 0; i < cnt; ++i) {
                        ASSERT_WRITEOUT(vp, vs);
                        vp += vo;
                        colNew[0] = cp[0] * 257;
                        colNew[1] = cp[1] * 257;
                        colNew[2] = cp[2] * 257;
                        colNew[3] = cp[3] * 257;
                        ASSERT_WRITEOUT(colNew, 8);
                        cp += co;
                    }
                }
                break;
            case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
                double iNew;
                for (UINT64 i = 0; i < cnt; ++i) {
                    ASSERT_WRITEOUT(vp, vs);
                    vp += vo;
                    iNew = *(reinterpret_cast<const float *>(cp));
                    ASSERT_WRITEOUT(&iNew, 8);
                    cp += co;
                }
            } break;
            case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB: {
                uint16_t colNew[4];
                for (UINT64 i = 0; i < cnt; ++i) {
                    ASSERT_WRITEOUT(vp, vs);
                    vp += vo;
                    const auto * col = reinterpret_cast<const float*>(cp);
                    colNew[0] = col[0] * 65535.0f;
                    colNew[1] = col[1] * 65535.0f;
                    colNew[2] = col[2] * 65535.0f;
                    colNew[3] = 65535.0f;
                    ASSERT_WRITEOUT(colNew, 8);
                    cp += co;
                }
            } break;
            default:
                vislib::sys::Log::DefaultLog.WriteError(
                    "MMPLDWriter: incoming unknown color type %u", points.GetColourDataType());
                break;
            }
        } else {
            for (UINT64 i = 0; i < cnt; i++) {
                ASSERT_WRITEOUT(vp, vs);
                vp += vo;
                if (ct != 0) {
                    ASSERT_WRITEOUT(cp, cs);
                    // warning: this only works since only one format is 3 bytes long, the illegal ct = 1
                    if (cs == 3) { // the unaligned ct == 1, UINT8_RGB, will be silently upgraded to ct 2 / cs 4
                        ASSERT_WRITEOUT(&alpha, 1);
                    }
                    cp += co;
                }
            }
        }
#ifdef WITH_CLUSTERINFO
        if (ver == 101) {
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
        }
#endif
    }

    return true;
#undef ASSERT_WRITEOUT
}
