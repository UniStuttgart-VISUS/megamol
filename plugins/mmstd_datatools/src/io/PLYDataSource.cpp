/*
 * PLYDataSource.cpp
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "io/PLYDataSource.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include <fstream>
#include <string>
#include "mmcore/moldyn/MultiParticleDataCall.h"

using namespace megamol;
using namespace megamol::stdplugin::datatools;

bool icompare_pred(unsigned char a, unsigned char b) { return std::tolower(a) == std::tolower(b); }

bool icompare(std::string const& a, std::string const& b) {
    if (a.length() == b.length()) {
        return std::equal(b.begin(), b.end(), a.begin(), icompare_pred);
    } else {
        return false;
    }
}

const char theUndef[] = "<undef>";

io::PLYDataSource::PLYDataSource(void) : core::Module(),
        filename("filename", "The path to the PLY file to load."),
        vertElemSlot("vertex element", "which element to get the vertex info from"),
        faceElemSlot("face element", "which element to get the face info from"),
        xPropSlot("x property", "which property to get the x position from"),
        yPropSlot("y property", "which property to get the y position from"),
        zPropSlot("z property", "which property to get the z position from"),
        nxPropSlot("nx property", "which property to get the normal x component from"),
        nyPropSlot("ny property", "which property to get the normal y component from"),
        nzPropSlot("nz property", "which property to get the normal z component from"),
        rPropSlot("r property", "which property to get the red component from"),
        gPropSlot("g property", "which property to get the green component from"),
        bPropSlot("b property", "which property to get the blue component from"),
        iPropSlot("i property", "which property to get the intensity from"),
        indexPropSlot("index property", "which property to get the vertex indices from"),
        getData("getdata", "Slot to request data from this data source."),
        file(nullptr), data_hash(0) {

    this->filename.SetParameter(new core::param::FilePathParam(""));
    this->filename.SetUpdateCallback(&PLYDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    this->vertElemSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->vertElemSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->vertElemSlot);
    this->faceElemSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->faceElemSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->faceElemSlot);

    this->xPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->xPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->xPropSlot);
    this->yPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->yPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->yPropSlot);
    this->zPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->zPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->zPropSlot);
    this->nxPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->nxPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->nxPropSlot);
    this->nyPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->nyPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->nyPropSlot);
    this->nzPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->nzPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->nzPropSlot);
    this->rPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->rPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->rPropSlot);
    this->gPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->gPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->gPropSlot);
    this->bPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->bPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->bPropSlot);
    this->iPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->iPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->iPropSlot);
    this->indexPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->indexPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->indexPropSlot);

    this->getData.SetCallback("MultiParticleDataCall", "GetData", &PLYDataSource::getDataCallback);
    this->getData.SetCallback("MultiParticleDataCall", "GetExtent", &PLYDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getData);

    //this->setFrameCount(1);
    //this->initFrameCache(1);
}

io::PLYDataSource::~PLYDataSource(void) {
    Release();
}

bool io::PLYDataSource::create(void) {
    // intentionally empty
    return true;
}

void io::PLYDataSource::release(void) {
    //this->resetFrameCache();
    //if (file != nullptr) {
    //    vislib::sys::File *f = file;
    //    file = nullptr;
    //    f->Close();
    //    delete f;
    //}
    //frameIdx.clear();
}

bool io::PLYDataSource::assertData() {

    if (!instream.is_open()) return false;
    if (vertices != nullptr) return true; // there is a modicum of data present, we have read the file before

    auto& vertElem = this->vertElemSlot.Param<core::param::FlexEnumParam>()->Value();

    if (vertElem != theUndef) {
        if (this->xPropSlot.Param<core::param::FlexEnumParam>()->Value() != theUndef &&
            this->yPropSlot.Param<core::param::FlexEnumParam>()->Value() != theUndef &&
            this->zPropSlot.Param<core::param::FlexEnumParam>()->Value() != theUndef) {
            try {
                vertices = plf.request_properties_from_element(
                    vertElem,
                        {this->xPropSlot.Param<core::param::FlexEnumParam>()->Value(),
                            this->yPropSlot.Param<core::param::FlexEnumParam>()->Value(),
                            this->zPropSlot.Param<core::param::FlexEnumParam>()->Value()});
            } catch (const std::exception& e) {
                vislib::sys::Log::DefaultLog.WriteError("tinyply exception reading vertices: %s", e.what());
            }
        }
        if (this->nxPropSlot.Param<core::param::FlexEnumParam>()->Value() != theUndef &&
            this->nyPropSlot.Param<core::param::FlexEnumParam>()->Value() != theUndef &&
            this->nzPropSlot.Param<core::param::FlexEnumParam>()->Value() != theUndef) {
            try {
                normals = plf.request_properties_from_element(
                    vertElem,
                        {this->nxPropSlot.Param<core::param::FlexEnumParam>()->Value(),
                            this->nyPropSlot.Param<core::param::FlexEnumParam>()->Value(),
                            this->nzPropSlot.Param<core::param::FlexEnumParam>()->Value()});
            } catch (const std::exception& e) {
                vislib::sys::Log::DefaultLog.WriteError("tinyply exception reading normals: %s", e.what());
            }
        }
        if (this->rPropSlot.Param<core::param::FlexEnumParam>()->Value() != theUndef &&
            this->gPropSlot.Param<core::param::FlexEnumParam>()->Value() != theUndef &&
            this->bPropSlot.Param<core::param::FlexEnumParam>()->Value() != theUndef) {
            try {
                colors = plf.request_properties_from_element(
                    vertElem,
                        {this->rPropSlot.Param<core::param::FlexEnumParam>()->Value(),
                            this->gPropSlot.Param<core::param::FlexEnumParam>()->Value(),
                            this->bPropSlot.Param<core::param::FlexEnumParam>()->Value()});
            } catch (const std::exception& e) {
                vislib::sys::Log::DefaultLog.WriteError("tinyply exception reading colors: %s", e.what());
            }
        }
    }
    if (this->faceElemSlot.Param<core::param::FlexEnumParam>()->Value() != theUndef) {
        if (this->indexPropSlot.Param<core::param::FlexEnumParam>()->Value() != theUndef) {
            try {
                faces = plf.request_properties_from_element(this->faceElemSlot.Param<core::param::FlexEnumParam>()->Value(),
                        {this->indexPropSlot.Param<core::param::FlexEnumParam>()->Value()});
            } catch (const std::exception& e) {
                vislib::sys::Log::DefaultLog.WriteError("tinyply exception reading faces: %s", e.what());
            }
        }
    }
    plf.read(instream);
}

bool io::PLYDataSource::filenameChanged(core::param::ParamSlot& slot) {

    using vislib::sys::Log;

    instream.open(filename.Param<core::param::FilePathParam>()->Value(), std::ios::binary);
    if (instream.fail()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to open PLY File \"%s\".", vislib::StringA(filename.Param<core::param::FilePathParam>()->Value()).PeekBuffer());
        // TODO ?
        return true;
    }

    this->vertElemSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->faceElemSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->xPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->yPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->zPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->nxPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->nyPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->nzPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->rPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->gPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->bPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->iPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->indexPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();

    guessedVertices = "";
    guessedFaces = "";
    guessedIndices = "";
    guessedPos.assign(3, "");
    guessedNormal.assign(3, "");
    guessedColor.assign(3, "");

    vertices = normals = colors = faces = nullptr;

    plf.parse_header(instream);

    for (auto e: plf.get_elements()) {
        this->vertElemSlot.Param<core::param::FlexEnumParam>()->AddValue(e.name);
        this->faceElemSlot.Param<core::param::FlexEnumParam>()->AddValue(e.name);
        if (icompare(e.name,"vertex")) {
            guessedVertices = e.name;
        }
        if (icompare(e.name, "face")) {
            guessedFaces = e.name;
        }

        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "element: %s, %u bytes", e.name.c_str(), e.size);
        for (auto p: e.properties) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "    property: %s %s", p.name.c_str(), tinyply::PropertyTable[p.propertyType].str.c_str());
            this->xPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->yPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->zPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->nxPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->nyPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->nzPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->rPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->gPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->bPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->iPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->indexPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);

            if (icompare(p.name, "x")) {
                guessedPos[0] = p.name;
            }
            if (icompare(p.name, "y")) {
                guessedPos[1] = p.name;
            }
            if (icompare(p.name, "z")) {
                guessedPos[2] = p.name;
            }
            if (icompare(p.name, "nx")) {
                guessedNormal[0] = p.name;
            }
            if (icompare(p.name, "ny")) {
                guessedNormal[1] = p.name;
            }
            if (icompare(p.name, "nz")) {
                guessedNormal[2] = p.name;
            }
            if (icompare(p.name, "r")) {
                guessedColor[0] = p.name;
            }
            if (icompare(p.name, "g")) {
                guessedColor[1] = p.name;
            }
            if (icompare(p.name, "b")) {
                guessedColor[2] = p.name;
            }
            if (icompare(p.name, "indices")){ // || icompare(p.name, "vertex_indices")) {
                guessedIndices = p.name;
            }
        }
    }

    if (!guessedVertices.empty()) {
        this->vertElemSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedVertices);
        if (std::none_of(guessedPos.begin(), guessedPos.end(), [](std::string s) { return s.empty(); })) {
            this->xPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedPos[0]);
            this->yPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedPos[1]);
            this->zPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedPos[2]);
        }
        if (std::none_of(guessedNormal.begin(), guessedNormal.end(), [](std::string s) { return s.empty(); })) {
            this->nxPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedNormal[0]);
            this->nyPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedNormal[1]);
            this->nzPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedNormal[2]);
        }
        if (std::none_of(guessedColor.begin(), guessedColor.end(), [](std::string s) { return s.empty(); })) {
            this->rPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedColor[0]);
            this->gPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedColor[1]);
            this->bPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedColor[2]);
        }
    }
    if (!guessedFaces.empty()) {
        this->faceElemSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedFaces);
        if (!guessedIndices.empty()) {
            this->indexPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedIndices);
        }
    }

//    using vislib::sys::Log;
//    using vislib::sys::File;
//    this->resetFrameCache();
//    this->data_hash++;
//
//    if (file == nullptr) {
//        file = new vislib::sys::FastFile();
//    } else {
//        file->Close();
//    }
//    assert(filename.Param<core::param::FilePathParam>() != nullptr);
//    if (!file->Open(filename.Param<core::param::FilePathParam>()->Value(), File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY)) {
//        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to open MMGDD-File \"%s\".", vislib::StringA(filename.Param<core::param::FilePathParam>()->Value()).PeekBuffer());
//        SAFE_DELETE(file);
//        this->setFrameCount(1);
//        this->initFrameCache(1);
//        return true;
//    }
//
//#define _ERROR_OUT(MSG) Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, MSG); \
//        SAFE_DELETE(this->file); \
//        this->setFrameCount(1); \
//        this->initFrameCache(1); \
//        return true;
//#define _ASSERT_READFILE(BUFFER, BUFFERSIZE) if (this->file->Read((BUFFER), (BUFFERSIZE)) != (BUFFERSIZE)) { \
//        _ERROR_OUT("Unable to read MMPLD file header"); \
//        }
//
//    char magicid[6];
//    _ASSERT_READFILE(magicid, 6);
//    if (::memcmp(magicid, "MMGDD", 6) != 0) {
//        _ERROR_OUT("MMGDD file header id wrong");
//    }
//    unsigned short ver;
//    _ASSERT_READFILE(&ver, 2);
//    if (ver != 100) {
//        _ERROR_OUT("MMGDD file header version wrong");
//    }
//
//    uint32_t frmCnt = 0;
//    _ASSERT_READFILE(&frmCnt, 4);
//    if (frmCnt == 0) {
//        _ERROR_OUT("MMGDD file does not contain any frame information");
//    }
//
//    frameIdx.resize(frmCnt + 1);
//    _ASSERT_READFILE(frameIdx.data(), frameIdx.size() * 8);
//
//    double memHere = static_cast<double>(vislib::sys::SystemInformation::AvailableMemorySize());
//    memHere *= 0.25; // only use max 25% of the memory of this data
//    Log::DefaultLog.WriteInfo("Memory available: %u MB\n", static_cast<uint32_t>(memHere / (1024.0 * 1024.0)));
//    double memWant = static_cast<double>(frameIdx.back() - frameIdx.front());
//    Log::DefaultLog.WriteInfo("Memory required: %u MB for %u frames total\n", static_cast<uint32_t>(memWant / (1024.0 * 1024.0)), static_cast<uint32_t>(frameIdx.size()));
//    uint32_t cacheSize = static_cast<uint32_t>((memHere / memWant) * static_cast<double>(frameIdx.size()) + 0.5);
//    Log::DefaultLog.WriteInfo("Cache set to %u frames\n", cacheSize);
//
//    this->setFrameCount(frmCnt);
//    this->initFrameCache(cacheSize);
//
//#undef _ASSERT_READFILE
//#undef _ERROR_OUT
//
//    return true;
}

bool io::PLYDataSource::anyEnumChanged(core::param::ParamSlot& slot) {
    this->vertices = nullptr;
    return false;
}

bool io::PLYDataSource::getDataCallback(core::Call& caller) {
    auto c2 = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&caller);
    if (c2 == nullptr) return false;

    if (!assertData()) return false;
    //Frame *f = nullptr;
    //if (c2 != nullptr) {
    //    f = dynamic_cast<Frame *>(this->requestLockedFrame(c2->FrameID(), true));
    //    if (f == nullptr) return false;
    //    c2->SetUnlocker(new Unlocker(*f));
    //    c2->SetFrameID(f->FrameNumber());
    //    c2->SetDataHash(this->data_hash);
    //    f->SetData(*c2);
    //}

    return true;
}

bool io::PLYDataSource::getExtentCallback(core::Call& caller) {
    auto c2 = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&caller);

    if (!assertData()) return false;

    if (c2 != nullptr) {
        c2->SetFrameCount(1);
        c2->SetDataHash(this->data_hash);
        return true;
    }

    return true;
}
