/*
 * MMPLDWriter.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "MMPLDWriter.h"
#include "mmcore/BoundingBoxes.h"
#include <algorithm>

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/String.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/Thread.h"

namespace megamol::moldyn::io {

/*
 * :MMPLDWriter::MMPLDWriter
 */
MMPLDWriter::MMPLDWriter()
        : AbstractDataWriter()
        , filenameSlot("filename", "The path to the MMPLD file to be written")
        , versionSlot("version", "The file format version to be written")
        , dataSlot("data", "The slot requesting the data to be written")
        , startFrameSlot("startFrame", "the first frame to write")
        , endFrameSlot("endFrame", "the last frame to write")
        , subsetSlot("writeSubset", "use the specified start and end") {

    this->filenameSlot << new core::param::FilePathParam(
        "", megamol::core::param::FilePathParam::Flag_File_ToBeCreatedWithRestrExts, {"mmpld"});
    this->MakeSlotAvailable(&this->filenameSlot);

    core::param::EnumParam* verPar = new core::param::EnumParam(100);
    verPar->SetTypePair(100, "1.0");
#ifdef WITH_CLUSTERINFO
    verPar->SetTypePair(101, "1.1");
#endif
    verPar->SetTypePair(102, "1.2");
    verPar->SetTypePair(103, "1.3");
    this->versionSlot.SetParameter(verPar);
    this->MakeSlotAvailable(&this->versionSlot);

    this->startFrameSlot << new core::param::IntParam(0);
    this->MakeSlotAvailable(&startFrameSlot);
    this->endFrameSlot << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->endFrameSlot);
    this->subsetSlot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->subsetSlot);

    this->dataSlot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->dataSlot);
}


/*
 * MMPLDWriter::~MMPLDWriter
 */
MMPLDWriter::~MMPLDWriter() {
    this->Release();
}


/*
 * MMPLDWriter::create
 */
bool MMPLDWriter::create() {
    return true;
}


/*
 * MMPLDWriter::release
 */
void MMPLDWriter::release() {}


/*
 * MMPLDWriter::`
 */
bool MMPLDWriter::run() {
    using megamol::core::utility::log::Log;
    vislib::TString filename(this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_string().c_str());
    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteError("No file name specified. Abort.");
        return false;
    }

    geocalls::MultiParticleDataCall* mpdc = this->dataSlot.CallAs<geocalls::MultiParticleDataCall>();
    if (mpdc == NULL) {
        Log::DefaultLog.WriteError("No data source connected. Abort.");
        return false;
    }

    if (vislib::sys::File::Exists(filename)) {
        Log::DefaultLog.WriteWarn(
            "File %s already exists and will be overwritten.", vislib::StringA(filename).PeekBuffer());
    }

    vislib::math::Cuboid<float> bbox;
    vislib::math::Cuboid<float> cbox;
    UINT32 frameCnt = 1;
    if (!(*mpdc)(1)) {
        Log::DefaultLog.WriteWarn("Unable to query data extents.");
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
            Log::DefaultLog.WriteWarn("Object space bounding boxes not valid. Using defaults");
            bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
            cbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        }
        frameCnt = mpdc->FrameCount();
        if (frameCnt == 0) {
            Log::DefaultLog.WriteError("Data source counts zero frames. Abort.");
            mpdc->Unlock();
            return false;
        }
    }

    const bool overrideSubset = this->subsetSlot.Param<core::param::BoolParam>()->Value();
    const UINT32 theStart = overrideSubset ? this->startFrameSlot.Param<core::param::IntParam>()->Value() : 0;
    const UINT32 theEnd = overrideSubset ? this->endFrameSlot.Param<core::param::IntParam>()->Value() + 1 : frameCnt;

    // DEBUG
    //    frameCnt = 10;
    // END DEBUG
    if (overrideSubset)
        frameCnt = theEnd - theStart;

    vislib::sys::FastFile file;
    if (!file.Open(filename, vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE,
            vislib::sys::File::CREATE_OVERWRITE)) {
        Log::DefaultLog.WriteError(
            "Unable to create output file \"%s\". Abort.", vislib::StringA(filename).PeekBuffer());
        mpdc->Unlock();
        return false;
    }

#define ASSERT_WRITEOUT(A, S)                                   \
    if (file.Write((A), (S)) != (S)) {                          \
        Log::DefaultLog.WriteError("Write error %d", __LINE__); \
        file.Close();                                           \
        mpdc->Unlock();                                         \
        return false;                                           \
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
    for (UINT32 i = theStart; i < theEnd; i++) {
        frameOffset = static_cast<UINT64>(file.Tell());
        file.Seek(seekTable + (i - theStart) * 8);
        ASSERT_WRITEOUT(&frameOffset, 8);
        file.Seek(frameOffset);

        Log::DefaultLog.WriteInfo("Started writing data frame %u\n", i);

        int missCnt = -9;
        do {
            mpdc->Unlock();
            mpdc->SetFrameID(i, true);
            if (!(*mpdc)(1)) {
                Log::DefaultLog.WriteError("Cannot request frame %u. Abort.\n", i);
                file.Close();
                return false;
            }
            if (!(*mpdc)(0)) {
                Log::DefaultLog.WriteError("Cannot get data frame %u. Abort.\n", i);
                file.Close();
                return false;
            }
            if (mpdc->FrameID() != i) {
                if ((missCnt % 10) == 0) {
                    Log::DefaultLog.WriteWarn("Frame %u returned on request for frame %u\n", mpdc->FrameID(), i);
                }
                ++missCnt;
                vislib::sys::Thread::Sleep(static_cast<DWORD>(1 + std::max<int>(missCnt, 0) * 100));
            }
        } while (mpdc->FrameID() != i);

        if (!this->writeFrame(file, *mpdc)) {
            mpdc->Unlock();
            Log::DefaultLog.WriteError("Cannot write data frame %u. Abort.\n", i);
            file.Close();
            return false;
        }
        mpdc->Unlock();
    }

    frameOffset = static_cast<UINT64>(file.Tell());
    file.Seek(seekTable + frameCnt * 8);
    ASSERT_WRITEOUT(&frameOffset, 8);

    file.Seek(6); // set correct version to show that file is complete
    version = this->versionSlot.Param<core::param::EnumParam>()->Value();
    ASSERT_WRITEOUT(&version, 2);

    file.Seek(frameOffset);

    Log::DefaultLog.WriteInfo("Completed writing data\n");
    file.Close();

#undef ASSERT_WRITEOUT
    return true;
}


/*
 * MMPLDWriter::getCapabilities
 */
bool MMPLDWriter::getCapabilities(core::DataWriterCtrlCall& call) {
    call.SetAbortable(false);
    return true;
}


/*
 * MMPLDWriter::writeFrame
 */
bool MMPLDWriter::writeFrame(vislib::sys::File& file, geocalls::MultiParticleDataCall& data) {
#define ASSERT_WRITEOUT(A, S)                                   \
    if (file.Write((A), (S)) != (S)) {                          \
        Log::DefaultLog.WriteError("Write error %d", __LINE__); \
        file.Close();                                           \
        return false;                                           \
    }
    using megamol::core::utility::log::Log;
    uint8_t const alpha = 255;
    int ver = this->versionSlot.Param<core::param::EnumParam>()->Value();

    // HAZARD for megamol up to fc4e784dae531953ad4cd3180f424605474dd18b this reads == 102
    // which means that many MMPLDs out there with version 103 are written wrongly (no timestamp)!
    if (ver >= 102) {
        float ts = data.GetTimeStamp();
        ASSERT_WRITEOUT(&ts, 4);
    }

    UINT32 listCnt = data.GetParticleListCount();
    ASSERT_WRITEOUT(&listCnt, 4);

    for (UINT32 li = 0; li < listCnt; li++) {
        geocalls::MultiParticleDataCall::Particles& points = data.AccessParticles(li);
        UINT8 vt = 0, ct = 0;
        unsigned int vs = 0, vo = 0, cs = 0, co = 0;
        switch (points.GetVertexDataType()) {
        case geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE:
            vt = 0;
            vs = 0;
            break;
        case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
            vt = 1;
            vs = 12;
            break;
        case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            vt = 2;
            vs = 16;
            break;
        case geocalls::MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
            vt = 3;
            vs = 6;
            break;
        case geocalls::MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
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
            case geocalls::MultiParticleDataCall::Particles::COLDATA_NONE:
                ct = 0;
                cs = 0;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                ct = 1;
                cs = 3;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                ct = 2;
                cs = 4;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
                ct = 3;
                cs = 4;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                ct = 4;
                cs = 12;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                ct = 5;
                cs = 16;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_USHORT_RGBA:
                ct = 6;
                cs = 8;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_DOUBLE_I:
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
        if (ct == 1)
            ct = 2;              // UINT8_RGB is unaligned and will never be written again.
        if (vt == 4 && ct < 5) { // TODO: fragile if we add another color type beyond DOUBLE_I!
            if (ct ==
                3) { // VERTDATA_DOUBLE_XYZ needs COLDATA_DOUBLE_I instead of COLDATA_FLOAT_I to be aligned for modern renderers (NG and OPSRay)
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
        if (vt == 0)
            cnt = 0;
        ASSERT_WRITEOUT(&cnt, 8);

        if (ver >= 103) {
            ASSERT_WRITEOUT(points.GetBBox().PeekBounds(), 24);
        }

        if (vt == 0)
            continue;
        const unsigned char* vp = static_cast<const unsigned char*>(points.GetVertexData());
        const unsigned char* cp = static_cast<const unsigned char*>(points.GetColourData());
        if (vt == 4 && ct < 5) {
            switch (points.GetColourDataType()) {
            case geocalls::MultiParticleDataCall::Particles::COLDATA_NONE: {
                auto col = points.GetGlobalColour();
                uint16_t colNew[4] = {
                    static_cast<uint16_t>(col[0] * 257),
                    static_cast<uint16_t>(col[1] * 257),
                    static_cast<uint16_t>(col[2] * 257),
                    static_cast<uint16_t>(col[3] * 257),
                };
                for (UINT64 i = 0; i < cnt; ++i) {
                    ASSERT_WRITEOUT(vp, vs);
                    vp += vo;
                    ASSERT_WRITEOUT(colNew, 8);
                }
            } break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB: {
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
            } break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA: {
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
            } break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
                double iNew;
                for (UINT64 i = 0; i < cnt; ++i) {
                    ASSERT_WRITEOUT(vp, vs);
                    vp += vo;
                    iNew = *(reinterpret_cast<const float*>(cp));
                    ASSERT_WRITEOUT(&iNew, 8);
                    cp += co;
                }
            } break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB: {
                uint16_t colNew[4];
                for (UINT64 i = 0; i < cnt; ++i) {
                    ASSERT_WRITEOUT(vp, vs);
                    vp += vo;
                    const auto* col = reinterpret_cast<const float*>(cp);
                    colNew[0] = col[0] * 65535.0f;
                    colNew[1] = col[1] * 65535.0f;
                    colNew[2] = col[2] * 65535.0f;
                    colNew[3] = 65535.0f;
                    ASSERT_WRITEOUT(colNew, 8);
                    cp += co;
                }
            } break;
            default:
                megamol::core::utility::log::Log::DefaultLog.WriteError(
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
} // namespace megamol::moldyn::io
