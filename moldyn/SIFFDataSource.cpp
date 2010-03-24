/*
 * SIFFDataSource.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "SIFFDataSource.h"
#include "param/FilePathParam.h"
#include "param/StringParam.h"
#include "MultiParticleDataCall.h"
#include "vislib/Log.h"
#include "vislib/MemmappedFile.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/String.h"
#include "vislib/sysfunctions.h"
#include "vislib/VersionNumber.h"

using namespace megamol::core;


/*
 * moldyn::SIFFDataSource::SIFFDataSource
 */
moldyn::SIFFDataSource::SIFFDataSource(void) : Module(),
        filenameSlot("filename", "The path to the trisoup file to load."),
        getDataSlot("getdata", "Slot to request data from this data source."),
        bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f), data(), datahash(0) {

    this->filenameSlot.SetParameter(new param::FilePathParam(""));
    this->filenameSlot.SetUpdateCallback(&SIFFDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filenameSlot);

    this->getDataSlot.SetCallback("MultiParticleDataCall", "GetData",
        &SIFFDataSource::getDataCallback);
    this->getDataSlot.SetCallback("MultiParticleDataCall", "GetExtent",
        &SIFFDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

}


/*
 * moldyn::SIFFDataSource::~SIFFDataSource
 */
moldyn::SIFFDataSource::~SIFFDataSource(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * moldyn::SIFFDataSource::create
 */
bool moldyn::SIFFDataSource::create(void) {
    if (!this->filenameSlot.Param<param::FilePathParam>()->Value().IsEmpty()) {
        this->filenameChanged(this->filenameSlot);
    }
    return true;
}


/*
 * moldyn::SIFFDataSource::release
 */
void moldyn::SIFFDataSource::release(void) {
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    this->data.EnforceSize(0);
}


#ifdef SIFFREAD
#error WTF? Why is SIFFREAD already defined?
#endif
#define SIFFREAD(BUF, SIZE, LINE) if (file.Read(BUF, SIZE) != SIZE) {\
    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "SIFF-IO-Error@%d", LINE);\
    file.Close();\
    return true;\
}

/*
 * moldyn::SIFFDataSource::filenameChanged
 */
bool moldyn::SIFFDataSource::filenameChanged(param::ParamSlot& slot) {
    using vislib::sys::File;
    using vislib::sys::Log;
    vislib::sys::MemmappedFile file;

    if (!file.Open(this->filenameSlot.Param<param::FilePathParam>()->Value(),
            File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to open file \"%s\"", vislib::StringA(
            this->filenameSlot.Param<param::FilePathParam>()->Value()).PeekBuffer());
        return true; // reset dirty flag!
    }

    // SIFF-Header
    char header[5];

    SIFFREAD(header, 5, __LINE__);
    if (!vislib::StringA(header, 4).Equals("SIFF")) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "SIFF-Header-Error");
        file.Close();
        return true;
    }
    if (header[4] == 'b') {
        // binary siff
        UINT32 version;

        SIFFREAD(&version, 4, __LINE__);
        if (version != 100) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "SIFF-Version-Error");
            file.Close();
            return true;
        }

        // Header ok, so now we load data
        // 4*floats (xyzr) + 3*bytes (rgb-colour) = 19 Byte pro Sphere
        File::FileSize size = file.GetSize() - 8; // remaining bytes
        if ((size % 19) != 0) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "SIFF-Size not aligned, ignoring last %d bytes",
                (size % 19));
            size -= (size % 19);
        }
        this->data.EnforceSize(static_cast<SIZE_T>(size));
        SIFFREAD(this->data.As<void>(), size, __LINE__);

    } else if (header[4] == 'a') {
        // ascii siff
        vislib::StringA verstr = vislib::sys::ReadLineFromFileA(file);
        verstr.TrimSpaces();
        vislib::VersionNumber version(verstr);
        if (version != vislib::VersionNumber(1, 0)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "SIFF-Version-Error");
            file.Close();
            return true;
        }

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
        this->data.AssertSize(blocks * 19 * blockGrow, false);
        cnt = 0;

        unsigned int t1 = vislib::sys::GetTicksOfDay();

        const SIZE_T bufferSize = 1024 * 1024 * 8;
        char *buffer = new char[bufferSize];
        vislib::StringA lineOnBreak;
        float x, y, z, rad;
        int r, g, b;
        while (!file.IsEOF()) {
            SIZE_T read = file.Read(buffer, bufferSize);
            SIZE_T sspos = 0;
            SIZE_T sepos = 0;
            while (sepos < read) {
                while ((sepos < read) && (buffer[sepos] != '\n') && (buffer[sepos] != '\r')) {
                    sepos++;
                }
                if ((sepos < read) || file.IsEOF()) {
                    // handle string input
                    const char *line;
                    if ((sepos == read) || !lineOnBreak.IsEmpty()) {
                        lineOnBreak += vislib::StringA(buffer + sspos, static_cast<unsigned int>(sepos - sspos));
                        line = lineOnBreak.PeekBuffer();
                    } else {
                        buffer[sepos] = 0;
                        line = buffer + sspos;
                    }

                    if (
#ifdef _WIN32
                        sscanf_s
#else /* _WIN32 */
                        sscanf
#endif /* _WIN32 */
                            (line, "%f %f %f %f %d %d %d\n", &x, &y, &z, &rad, &r, &g, &b) == 7) {

                        if (r < 0) r = 0; else if (r > 255) r = 255;
                        if (g < 0) g = 0; else if (g > 255) g = 255;
                        if (b < 0) b = 0; else if (b > 255) b = 255;

                        blocks = this->data.GetSize() / (19 * blockGrow);
                        if ((1 + cnt / blockGrow) > blocks) {
                            blocks = 1 + cnt / blockGrow;
                        }
                        this->data.AssertSize(blocks * 19 * blockGrow, true);

                        *this->data.AsAt<float>(cnt * 19 + 0) = x;
                        *this->data.AsAt<float>(cnt * 19 + 4) = y;
                        *this->data.AsAt<float>(cnt * 19 + 8) = z;
                        *this->data.AsAt<float>(cnt * 19 + 12) = rad;
                        *this->data.AsAt<unsigned char>(cnt * 19 + 16) = static_cast<unsigned char>(r);
                        *this->data.AsAt<unsigned char>(cnt * 19 + 17) = static_cast<unsigned char>(g);
                        *this->data.AsAt<unsigned char>(cnt * 19 + 18) = static_cast<unsigned char>(b);
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
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100, "Loaded %u spheres in %u milliseconds", cnt, t2 - t1);

        this->data.EnforceSize(cnt * 19, true);

    } else {
        // unknown siff
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "SIFF-Header-Error: Unknown subformat");

    }

    // calc bounding box
    if (this->data.GetSize() >= 19) {
        float *ptr = this->data.As<float>();
        this->bbox.Set(
            ptr[0] - ptr[3], ptr[1] - ptr[3], ptr[2] - ptr[3],
            ptr[0] + ptr[3], ptr[1] + ptr[3], ptr[2] + ptr[3]);

        for (unsigned int i = 19; i < this->data.GetSize(); i += 19) {
            ptr = this->data.AsAt<float>(i);
            this->bbox.GrowToPoint(ptr[0] - ptr[3], ptr[1] - ptr[3], ptr[2] - ptr[3]);
            this->bbox.GrowToPoint(ptr[0] + ptr[3], ptr[1] + ptr[3], ptr[2] + ptr[3]);
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
 * moldyn::SIFFDataSource::getDataCallback
 */
bool moldyn::SIFFDataSource::getDataCallback(Call& caller) {
    MultiParticleDataCall *c2 = dynamic_cast<MultiParticleDataCall*>(&caller);
    if (c2 == NULL) return false;

    c2->SetUnlocker(NULL);
    c2->SetParticleListCount(1);
    c2->AccessParticles(0).SetCount(this->data.GetSize() / 19);
    if ((this->data.GetSize() / 19) > 0) {
        c2->AccessParticles(0).SetColourData(
            moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB,
            this->data.At(16), 19);
        c2->AccessParticles(0).SetVertexData(
            moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR,
            this->data, 19);
    }
    c2->SetDataHash(this->datahash);

    return true;
}


/*
 * moldyn::SIFFDataSource::getExtentCallback
 */
bool moldyn::SIFFDataSource::getExtentCallback(Call& caller) {
    MultiParticleDataCall *c2 = dynamic_cast<MultiParticleDataCall*>(&caller);
    if (c2 == NULL) return false;

    c2->SetExtent(1,
        this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back(),
        this->bbox.Right(), this->bbox.Top(), this->bbox.Front());
    c2->SetDataHash(this->datahash);

    return true;
}
