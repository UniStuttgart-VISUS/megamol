/*
 * PLYDataSource.cpp
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "io/PLYDataSource.h"
#include "mmcore/param/FilePathParam.h"
#include "geometry_calls/CallTriMeshData.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include <fstream>
#include "tinyply.h"
#include <chrono>
#include <iostream>

using namespace megamol;
using namespace megamol::stdplugin::datatools;
using namespace megamol::geocalls;
using namespace megamol::core::moldyn;

/*
 * io::PLYDataSource::PLYDataSource
 */
io::PLYDataSource::PLYDataSource(void) : core::Module(),
        filename("filename", "The path to the MMPLD file to load."),
        getDataMesh("getdatamesh", "Slot to request the mesh data from this data source."),
        getDataSpheres("getdataspheres", "Slot to request the vertex data from this data source."),
        file(nullptr), data_hash(0) {

    this->filename.SetParameter(new core::param::FilePathParam(""));
    this->filename.SetUpdateCallback(&PLYDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);
    
    this->getDataMesh.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(0), &PLYDataSource::getMeshDataCallback);
    this->getDataMesh.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(1), &PLYDataSource::getMeshExtentCallback);
    this->MakeSlotAvailable(&this->getDataMesh);

    this->getDataSpheres.SetCallback(MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(0), &PLYDataSource::getSphereDataCallback);
    this->getDataSpheres.SetCallback(MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(1), &PLYDataSource::getSphereExtentCallback);
    this->MakeSlotAvailable(&this->getDataSpheres);

    this->clearAllFields();

    //this->setFrameCount(1);
    //this->initFrameCache(1);
}

/*
 * io::PLYDataSource::~PLYDataSource
 */
io::PLYDataSource::~PLYDataSource(void) {
    Release();
}

/*
 * io::PLYDataSource::create
 */
bool io::PLYDataSource::create(void) {
    // intentionally empty
    return true;
}

/*
 * io::PLYDataSource::release
 */
void io::PLYDataSource::release(void) {
    
}

/*
 * io::PLYDataSource::filenameChanged
 */
bool io::PLYDataSource::filenameChanged(core::param::ParamSlot& slot) {

    using vislib::sys::Log;

    this->clearAllFields();

    std::string fname = filename.Param<core::param::FilePathParam>()->Value();
    std::ifstream instream(fname, std::ios::binary);
    if (instream.fail()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to open Ply-File \"%s\".", vislib::StringA(filename.Param<core::param::FilePathParam>()->Value()).PeekBuffer());
        // TODO ?
        return true;
    }
    tinyply::PlyFile plf;
    plf.parse_header(instream);
    for (auto e: plf.get_elements()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "element: %s, %u bytes", e.name.c_str(), e.size);
        for (auto p: e.properties) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "    property: %s %s %i", p.name.c_str(), tinyply::PropertyTable[p.propertyType].str.c_str(),
                tinyply::PropertyTable[p.propertyType].stride);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    int32_t vertIdx;
    vertIdx = -1;
    std::vector<int32_t> faceIndices;

    for (int32_t i = 0; i < plf.get_elements().size(); i++) {
        std::string name = plf.get_elements()[i].name;
        if (name.compare("vertex") == 0) {
            if (vertIdx >= 0) {
                Log::DefaultLog.WriteError("Multiple vertex definitions found.");
                return false;
            }
            vertIdx = i;
        } else if (name.compare("face") == 0) {
            faceIndices.push_back(i);
        }
    }

    if (vertIdx < 0) {
        Log::DefaultLog.WriteError("No vertex definition found");
        return false;
    }

    // search for the positions of the different paramers
    int32_t vx, vy, vz;
    vx = vy = vz = -1;

    int32_t nx, ny, nz;
    nx = ny = nz = -1;

    int32_t r, g, b, a;
    r = g = b = a = -1;

    uint32_t vertSize = 0;
    std::vector<uint32_t> sizes;
    
    for (int32_t i = 0; i < plf.get_elements()[vertIdx].properties.size(); i++) {
        std::string name = plf.get_elements()[vertIdx].properties[i].name;
        if (name.compare("x") == 0) {
            vx = i;
        } else if (name.compare("y") == 0) {
            vy = i;
        } else if (name.compare("z") == 0) {
            vz = i;
        } else if (name.compare("nx") == 0) {
            nx = i;
        } else if (name.compare("ny") == 0) {
            ny = i;
        } else if (name.compare("nz") == 0) {
            nz = i;
        }

        switch (plf.get_elements()[vertIdx].properties[i].propertyType) {
        case tinyply::Type::INT8:
        case tinyply::Type::UINT8:
            vertSize += 1;
            sizes.push_back(1);
            break;
        case tinyply::Type::INT16:
        case tinyply::Type::UINT16:
            vertSize += 2;
            sizes.push_back(2);
            break;
        case tinyply::Type::INT32:
        case tinyply::Type::UINT32:
        case tinyply::Type::FLOAT32:
            vertSize += 4;
            sizes.push_back(4);
            break;
        case tinyply::Type::FLOAT64:
            vertSize += 8;
            sizes.push_back(8);
            break;
        case tinyply::Type::INVALID:
        default:
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    Log::DefaultLog.WriteInfo("Parsing of %s took %f s", fname.c_str(), duration.count() / 1000.0f);

    this->data_hash++;
    return true;
}

/*
 * io::PLYDataSource::getMeshDataCallback
 */
bool io::PLYDataSource::getMeshDataCallback(core::Call& caller) {
    return true;
}

/*
 * io::PLYDataSource::getMeshExtentCallback
 */
bool io::PLYDataSource::getMeshExtentCallback(core::Call& caller) {
    return false;
}

/*
 * io::PLYDataSource::getSphereDataCallback
 */
bool io::PLYDataSource::getSphereDataCallback(core::Call& caller) {
    return true;
}

/*
 * io::PLYDataSource::getSphereExtentCallback
 */
bool io::PLYDataSource::getSphereExtentCallback(core::Call& caller) {
    return true;
}

/*
 * io::PLYDataSource::clearAllFields
 */
void io::PLYDataSource::clearAllFields(void) {
    if (posPointers.pos_float != nullptr) {
        delete[] posPointers.pos_float;
        posPointers.pos_float = nullptr;
    }
    if (posPointers.pos_double != nullptr) {
        delete[] posPointers.pos_double;
        posPointers.pos_double = nullptr;
    }
    if (colorPointers.col_uchar != nullptr) {
        delete[] colorPointers.col_uchar;
        colorPointers.col_uchar = nullptr;
    }
    if (colorPointers.col_float != nullptr) {
        delete[] colorPointers.col_float;
        colorPointers.col_float = nullptr;
    }
    if (colorPointers.col_double != nullptr) {
        delete[] colorPointers.col_double;
        colorPointers.col_double = nullptr;
    }
    if (normalPointers.norm_float != nullptr) {
        delete[] normalPointers.norm_float;
        normalPointers.norm_float = nullptr;
    }
    if (normalPointers.norm_double != nullptr) {
        delete[] normalPointers.norm_double;
        normalPointers.norm_double = nullptr;
    }
    if (facePointers.face_uchar != nullptr) {
        delete[] facePointers.face_uchar;
        facePointers.face_uchar = nullptr;
    }
    if (facePointers.face_u16 != nullptr) {
        delete[] facePointers.face_u16;
        facePointers.face_u16 = nullptr;
    }
    if (facePointers.face_u32 != nullptr) {
        delete[] facePointers.face_u32;
        facePointers.face_u32 = nullptr;
    }
}