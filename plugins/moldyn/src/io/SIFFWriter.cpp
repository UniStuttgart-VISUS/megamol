/*
 * SIFFWriter.cpp
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */

#include "io/SIFFWriter.h"
#include "mmcore/BoundingBoxes.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/utility/sys/Thread.h"
#include "vislib/String.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/sysfunctions.h"

using namespace megamol;
using namespace megamol::moldyn::io;


/*
 * SIFFWriter::SIFFWriter
 */
SIFFWriter::SIFFWriter(void)
        : AbstractDataWriter()
        , filenameSlot("filename", "The path to the MMPLD file to be written")
        , asciiSlot("ascii", "Set to true to write ASCII-versions of SIFF")
        , versionSlot("version", "The file format version to write")
        , dataSlot("data", "The slot requesting the data to be written") {

    this->filenameSlot << new core::param::FilePathParam(
        "", megamol::core::param::FilePathParam::Flag_File_ToBeCreatedWithRestrExts, {"siff"});
    this->MakeSlotAvailable(&this->filenameSlot);

    this->asciiSlot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->asciiSlot);

    core::param::EnumParam* version = new core::param::EnumParam(100);
    version->SetTypePair(100, "1.0");
    version->SetTypePair(101, "1.1");
    this->versionSlot << version;
    this->MakeSlotAvailable(&this->versionSlot);

    this->dataSlot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->dataSlot);
}


/*
 * SIFFWriter::~SIFFWriter
 */
SIFFWriter::~SIFFWriter(void) {
    this->Release();
}


/*
 * SIFFWriter::create
 */
bool SIFFWriter::create(void) {
    // intentionally empty
    return true;
}


/*
 * SIFFWriter::release
 */
void SIFFWriter::release(void) {
    // intentionally empty
}


/*
 * SIFFWriter::run
 */
bool SIFFWriter::run(void) {
    using megamol::core::utility::log::Log;
    using vislib::Exception;

    auto filename = this->filenameSlot.Param<core::param::FilePathParam>()->Value();
    if (filename.empty()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "No file name specified. Abort.");
        return false;
    }

    geocalls::MultiParticleDataCall* mpdc = this->dataSlot.CallAs<geocalls::MultiParticleDataCall>();
    if (mpdc == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "No data source connected. Abort.");
        return false;
    }

    bool useAscii = this->asciiSlot.Param<core::param::BoolParam>()->Value();

    int version = this->versionSlot.Param<core::param::EnumParam>()->Value();
    if ((version != 100) && (version != 101)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unsupported version requested. Abort.");
        return false;
    }

    if (vislib::sys::File::Exists(filename.native().c_str())) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_WARN, "File %s already exists and will be overwritten.", filename.generic_u8string().c_str());
    }

    mpdc->SetFrameID(0, true);
    if (!(*mpdc)(0)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Could not get data (Frame 0). Abort.");
        return false;
    }

    vislib::sys::FastFile file;
    if (!file.Open(filename.native().c_str(), vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE,
            vislib::sys::File::CREATE_OVERWRITE)) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "Unable to create output file \"%s\". Abort.", filename.generic_u8string().c_str());
        return false;
    }

    try {

        if (file.Write("SIFF", 4) != 4)
            throw Exception("Write failed", __FILE__, __LINE__);
        if (file.Write((useAscii ? "a" : "b"), 1) != 1)
            throw Exception("Write failed", __FILE__, __LINE__);
        if (useAscii) {
            vislib::sys::WriteFormattedLineToFile(file, "%u.%u\n", version / 100, version % 100);
        } else {
            unsigned short v = static_cast<unsigned short>(version);
            if (file.Write(&v, 2) != 2)
                throw Exception("Write failed", __FILE__, __LINE__);
        }

        for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
            geocalls::MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);
            const char* colPtr = static_cast<const char*>(parts.GetColourData());
            const char* vertPtr = static_cast<const char*>(parts.GetVertexData());
            size_t colStep = 0;
            size_t vertStep = 0;
            unsigned char col[3];
            bool colFloats = false;
            bool singleCol = false;
            bool hasRad = false;
            float minC = 0.0f, maxC = 0.0f;

            // colour
            switch (parts.GetColourDataType()) {
            case geocalls::MultiParticleDataCall::Particles::COLDATA_NONE:
                ::memcpy(col, parts.GetGlobalColour(), 3);
                colPtr = NULL;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                colStep = 3;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                colStep = 4;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                colStep = 12;
                colFloats = true;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                colStep = 16;
                colFloats = true;
                break;
            case geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
                colStep = 4;
                colFloats = true;
                singleCol = true;
                minC = parts.GetMinColourIndexValue();
                maxC = parts.GetMaxColourIndexValue();
                break;
            default:
                col[0] = col[1] = col[2] = 127;
                colPtr = NULL;
                break;
            }
            colStep = vislib::math::Max<size_t>(colStep, parts.GetColourDataStride());

            // radius and position
            switch (parts.GetVertexDataType()) {
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_NONE:
                continue;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                vertStep = 12;
                ASSERT(vertPtr != NULL);
                break;
            case geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                vertStep = 16;
                hasRad = true;
                ASSERT(vertPtr != NULL);
                break;
            default:
                continue;
            }
            vertStep = vislib::math::Max<size_t>(vertStep, parts.GetVertexDataStride());

            for (UINT64 i = 0, cnt = parts.GetCount(); i < cnt; ++i) {

                if (colPtr != NULL) {
                    if (colFloats) {
                        float c[3];
                        if (singleCol) {
                            c[0] = c[1] = c[2] = (*reinterpret_cast<const float*>(colPtr) - minC) / (maxC - minC);
                        } else {
                            ::memcpy(c, colPtr, 3 * sizeof(float));
                        }
                        for (int j = 0; j < 3; j++) {
                            col[j] = static_cast<unsigned char>(
                                vislib::math::Clamp<int>(static_cast<int>(c[j] * 255.0f), 0, 255));
                        }
                    } else {
                        ::memcpy(col, colPtr, 3);
                    }
                    colPtr += colStep;
                }

                float v[4];
                v[3] = parts.GetGlobalRadius();
                ::memcpy(v, vertPtr, (hasRad ? 4 : 3) * sizeof(float));
                vertPtr += vertStep;

                if (useAscii) {
                    vislib::sys::WriteFormattedLineToFile(file,
                        (version == 101) ? "%f %f %f\n" : "%f %f %f %f %u %u %u\n", v[0], v[1], v[2], v[3], col[0],
                        col[1], col[2]);
                } else {
                    if (version == 101) {
                        if (file.Write(&v, 12) != 12)
                            throw Exception("Write failed", __FILE__, __LINE__);
                    } else if (version == 100) {
                        if (file.Write(&v, 16) != 16)
                            throw Exception("Write failed", __FILE__, __LINE__);
                        if (file.Write(&col, 3) != 3)
                            throw Exception("Write failed", __FILE__, __LINE__);
                    }
                }
            }
        }

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Completed writing data\n");
    } catch (Exception ex) {
        Log::DefaultLog.WriteError("Failed to write: %s (%s, %d)\n", ex.GetMsgA(), ex.GetFile(), ex.GetLine());
    } catch (...) { Log::DefaultLog.WriteError("Failed to write: unexpected exception\n"); }
    file.Close();

    return true;
}


/*
 * SIFFWriter::getCapabilities
 */
bool SIFFWriter::getCapabilities(core::DataWriterCtrlCall& call) {
    call.SetAbortable(false);
    return true;
}
