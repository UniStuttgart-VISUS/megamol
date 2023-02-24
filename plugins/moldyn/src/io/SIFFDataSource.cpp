/*
 * SIFFDataSource.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "io/SIFFDataSource.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/String.h"
#include "vislib/VersionNumber.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/sysfunctions.h"

using namespace megamol;
using namespace megamol::moldyn::io;


/*
 * SIFFDataSource::SIFFDataSource
 */
SIFFDataSource::SIFFDataSource()
        : core::Module()
        , filenameSlot("filename", "The path to the trisoup file to load.")
        , radSlot("radius", "The radius used when loading a version 1.1 file")
        , getDataSlot("getdata", "Slot to request data from this data source.")
        , bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
        , data()
        , datahash(0)
        , verNum(100)
        , hasAlpha(false) {

    this->filenameSlot.SetParameter(new core::param::FilePathParam(""));
    this->filenameSlot.SetUpdateCallback(&SIFFDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filenameSlot);

    this->radSlot << new core::param::FloatParam(0.1f, 0.0f);
    this->MakeSlotAvailable(&this->radSlot);

    this->getDataSlot.SetCallback("MultiParticleDataCall", "GetData", &SIFFDataSource::getDataCallback);
    this->getDataSlot.SetCallback("MultiParticleDataCall", "GetExtent", &SIFFDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);
}


/*
 * SIFFDataSource::~SIFFDataSource
 */
SIFFDataSource::~SIFFDataSource() {
    this->Release(); // implicitly calls 'release'
}


/*
 * SIFFDataSource::create
 */
bool SIFFDataSource::create() {
    if (!this->filenameSlot.Param<core::param::FilePathParam>()->Value().empty()) {
        this->filenameChanged(this->filenameSlot);
    }
    return true;
}


/*
 * SIFFDataSource::release
 */
void SIFFDataSource::release() {
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    this->data.EnforceSize(0);
}


#ifdef SIFFREAD
#error WTF? Why is SIFFREAD already defined?
#endif
#define SIFFREAD(BUF, SIZE, LINE)                             \
    if (file.Read(BUF, SIZE) != SIZE) {                       \
        Log::DefaultLog.WriteError("SIFF-IO-Error@%d", LINE); \
        file.Close();                                         \
        return true;                                          \
    }

/*
 * SIFFDataSource::filenameChanged
 */
bool SIFFDataSource::filenameChanged(core::param::ParamSlot& slot) {
    using megamol::core::utility::log::Log;
    using vislib::sys::File;
    vislib::sys::FastFile file;

    if (!file.Open(this->filenameSlot.Param<core::param::FilePathParam>()->Value().native().c_str(), File::READ_ONLY,
            File::SHARE_READ, File::OPEN_ONLY)) {
        Log::DefaultLog.WriteError("Unable to open file \"%s\"",
            this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str());
        return true; // reset dirty flag!
    }

    // SIFF-Header
    char header[5];

    SIFFREAD(header, 5, __LINE__);
    if (!vislib::StringA(header, 4).Equals("SIFF")) {
        Log::DefaultLog.WriteError("SIFF-Header-Error");
        file.Close();
        return true;
    }
    unsigned int bpp = 0;
    if (header[4] == 'b') {
        // binary siff
        UINT32 version;
        this->hasAlpha = false;

        SIFFREAD(&version, 4, __LINE__);
        if (version == 100) {
            this->verNum = 100;
        } else if (version == 101) {
            this->verNum = 101;
        } else {
            Log::DefaultLog.WriteError("SIFF-Version-Error");
            file.Close();
            return true;
        }
        bpp = (this->verNum == 100) ? 19 : 12;

        // Header ok, so now we load data
        //  version 1.0 body:  4*floats (xyzr) + 3*bytes (rgb-colour) = 19 Bytes pro Sphere
        //  version 1.1 body:  3*floats (xyz) = 12 Bytes pro Sphere
        File::FileSize size = file.GetSize() - 9; // remaining bytes
        if ((size % bpp) != 0) {
            Log::DefaultLog.WriteWarn("SIFF-Size not aligned, ignoring last %d bytes", (size % bpp));
            size -= (size % bpp);
        }
        this->data.EnforceSize(static_cast<SIZE_T>(size));
        SIFFREAD(this->data.As<void>(), size, __LINE__);

    } else if (header[4] == 'a') {
        // ascii siff
        this->hasAlpha = false;
        vislib::StringA verstr = vislib::sys::ReadLineFromFileA(file);
        verstr.TrimSpaces();
        vislib::VersionNumber version(verstr);
        if (version == vislib::VersionNumber(1, 0)) {
            this->verNum = 100;
        } else if (version == vislib::VersionNumber(1, 1)) {
            this->verNum = 101;
        } else {
            Log::DefaultLog.WriteError("SIFF-Version-Error");
            file.Close();
            return true;
        }
        bpp = (this->verNum == 100) ? 19 : 12;

        SIZE_T cnt;
        const SIZE_T blockGrow = 1000000;

        vislib::sys::File::FileSize s1 = file.Tell();
        verstr = vislib::sys::ReadLineFromFileA(file);
        vislib::sys::File::FileSize s2 = file.Tell();
        file.Seek(static_cast<vislib::sys::File::FileOffset>(s1));
        s2 -= s1; // s2 is now size of line
        vislib::sys::File::FileSize fileSize = file.GetSize();
        s1 = fileSize - s1;

        cnt = 1 + static_cast<SIZE_T>(s1 / s2);
        SIZE_T blocks = 1 + cnt / blockGrow;
        this->data.AssertSize(blocks * bpp * blockGrow, false);
        cnt = 0;

        unsigned int t1 = vislib::sys::GetTicksOfDay();

        const SIZE_T bufferSize = 1024 * 1024 * 8;
        char* buffer = new char[bufferSize];
        vislib::StringA lineOnBreak;
        float x, y, z, rad;
        int r, g, b, colA;
        while (!file.IsEOF()) {
            SIZE_T read = static_cast<SIZE_T>(file.Read(buffer, bufferSize));
            SIZE_T sspos = 0;
            SIZE_T sepos = 0;
            while (sepos < read) {
                while ((sepos < read) && (buffer[sepos] != '\n') && (buffer[sepos] != '\r')) {
                    sepos++;
                }
                if ((sepos < read) || file.IsEOF()) {
                    // handle string input
                    const char* line;
                    if ((sepos == read) || !lineOnBreak.IsEmpty()) {
                        lineOnBreak += vislib::StringA(buffer + sspos, static_cast<unsigned int>(sepos - sspos));
                        line = lineOnBreak.PeekBuffer();
                    } else {
                        buffer[sepos] = 0;
                        line = buffer + sspos;
                    }
                    bool valid = true;

                    if (this->verNum == 100) {
                        int srv =
#ifdef _WIN32
                            sscanf_s
#else  /* _WIN32 */
                            sscanf
#endif /* _WIN32 */
                            (line, "%f %f %f %f %d %d %d %d\n", &x, &y, &z, &rad, &r, &g, &b, &colA);
                        if (srv == 8) {
                            if (cnt == 0) {
                                this->hasAlpha = true;
                                bpp = 20; // because we now store alpha too
                            }
                            if (colA < 0)
                                colA = 0;
                            else if (colA > 255)
                                colA = 255;
                            srv = 7;
                        } else
                            colA = 255;
                        if (srv == 7) {
                            if (r < 0)
                                r = 0;
                            else if (r > 255)
                                r = 255;
                            if (g < 0)
                                g = 0;
                            else if (g > 255)
                                g = 255;
                            if (b < 0)
                                b = 0;
                            else if (b > 255)
                                b = 255;
                        } else
                            valid = false;
                    } else if (this->verNum == 101) {
                        if (
#ifdef _WIN32
                            sscanf_s
#else  /* _WIN32 */
                            sscanf
#endif /* _WIN32 */
                            (line, "%f %f %f\n", &x, &y, &z) == 3) {
                            // everything fine
                        } else
                            valid = false;
                    } else
                        valid = false;

                    if (valid) {
                        blocks = this->data.GetSize() / (bpp * blockGrow);
                        if ((1 + cnt / blockGrow) > blocks) {
                            blocks = 1 + cnt / blockGrow;
                        }
                        this->data.AssertSize(blocks * bpp * blockGrow, true);

                        *this->data.AsAt<float>(cnt * bpp + 0) = x;
                        *this->data.AsAt<float>(cnt * bpp + 4) = y;
                        *this->data.AsAt<float>(cnt * bpp + 8) = z;
                        if (this->verNum == 100) {
                            *this->data.AsAt<float>(cnt * bpp + 12) = rad;
                            *this->data.AsAt<unsigned char>(cnt * bpp + 16) = static_cast<unsigned char>(r);
                            *this->data.AsAt<unsigned char>(cnt * bpp + 17) = static_cast<unsigned char>(g);
                            *this->data.AsAt<unsigned char>(cnt * bpp + 18) = static_cast<unsigned char>(b);
                            if (this->hasAlpha) {
                                *this->data.AsAt<unsigned char>(cnt * bpp + 19) = static_cast<unsigned char>(colA);
                            }
                        }
                        cnt++;
                    }

                    lineOnBreak.Clear();
                    sepos++;
                    sspos = sepos;

                } else {
                    // store line part for the next loop cycle
                    lineOnBreak += vislib::StringA(buffer + sspos, static_cast<unsigned int>(sepos - sspos));
                }
            }
        }

        delete[] buffer;
        unsigned int t2 = vislib::sys::GetTicksOfDay();

        // pure FileIO (no parsing):
        //  File.Read + Loop:   Loaded 6265625 spheres in 6355 milliseconds
        //  ReadLineFromFileA:  Loaded 6265625 spheres in 40615 milliseconds
        // with parsing:
        //  scanf:              Loaded 6265625 spheres in 78893 milliseconds
        //                      Loaded 0 spheres in      115283 milliseconds
        Log::DefaultLog.WriteInfo("Loaded %u spheres in %u milliseconds", cnt, t2 - t1);

        this->data.EnforceSize(cnt * bpp, true);

    } else {
        // unknown siff
        Log::DefaultLog.WriteError("SIFF-Header-Error: Unknown subformat");
    }

    // calc bounding box
    if (this->data.GetSize() >= bpp) {
        float* ptr = this->data.As<float>();
        float rad = 0.0f;
        if (this->verNum == 100)
            rad = ptr[3];
        this->bbox.Set(ptr[0] - rad, ptr[1] - rad, ptr[2] - rad, ptr[0] + rad, ptr[1] + rad, ptr[2] + rad);

        for (unsigned int i = bpp; i < this->data.GetSize(); i += bpp) {
            ptr = this->data.AsAt<float>(i);
            if (this->verNum == 100)
                rad = ptr[3];
            this->bbox.GrowToPoint(ptr[0] - rad, ptr[1] - rad, ptr[2] - rad);
            this->bbox.GrowToPoint(ptr[0] + rad, ptr[1] + rad, ptr[2] + rad);
        }

    } else {
        this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    }

    this->datahash++; // so simple, it might work

    file.Close();
    return true; // to reset the dirty flag of the param slot
}

#undef SIFFREAD


/*
 * SIFFDataSource::getDataCallback
 */
bool SIFFDataSource::getDataCallback(core::Call& caller) {
    geocalls::MultiParticleDataCall* c2 = dynamic_cast<geocalls::MultiParticleDataCall*>(&caller);
    if (c2 == NULL)
        return false;

    c2->SetUnlocker(NULL);
    c2->SetParticleListCount(1);

    c2->AccessParticles(0).SetCount(0);

    unsigned int bpp = (this->verNum == 100) ? (this->hasAlpha ? 20 : 19) : 12;

    c2->AccessParticles(0).SetCount(this->data.GetSize() / bpp);
    if (this->data.GetSize() >= bpp) {
        if (this->verNum == 100) {
            c2->AccessParticles(0).SetColourData(this->hasAlpha
                                                     ? geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA
                                                     : geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB,
                this->data.At(16), bpp);
            c2->AccessParticles(0).SetVertexData(
                geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR, this->data, bpp);
        } else if (this->verNum == 101) {
            c2->AccessParticles(0).SetGlobalColour(192, 192, 192);
            c2->AccessParticles(0).SetColourData(geocalls::MultiParticleDataCall::Particles::COLDATA_NONE, NULL);
            c2->AccessParticles(0).SetGlobalRadius(this->radSlot.Param<core::param::FloatParam>()->Value());
            c2->AccessParticles(0).SetVertexData(
                geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ, this->data);
        } else {
            c2->AccessParticles(0).SetCount(0);
        }
    }
    c2->SetDataHash(this->datahash);

    return true;
}


/*
 * SIFFDataSource::getExtentCallback
 */
bool SIFFDataSource::getExtentCallback(core::Call& caller) {
    geocalls::MultiParticleDataCall* c2 = dynamic_cast<geocalls::MultiParticleDataCall*>(&caller);
    if (c2 == NULL)
        return false;

    c2->SetExtent(1, this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back(), this->bbox.Right(), this->bbox.Top(),
        this->bbox.Front());
    c2->SetDataHash(this->datahash);

    return true;
}
