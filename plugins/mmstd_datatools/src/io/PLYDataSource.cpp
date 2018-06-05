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
        file(nullptr), data_hash(0), vertexData(nullptr), 
        vertexNormalData(nullptr), vertexColorData(nullptr), 
        faceData(nullptr), texCoordData(nullptr) {

    this->filename.SetParameter(new core::param::FilePathParam(""));
    this->filename.SetUpdateCallback(&PLYDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);
    
    this->getDataMesh.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(0), &PLYDataSource::getMeshDataCallback);
    this->getDataMesh.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(1), &PLYDataSource::getMeshExtentCallback);
    this->MakeSlotAvailable(&this->getDataMesh);

    this->getDataSpheres.SetCallback(MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(0), &PLYDataSource::getSphereDataCallback);
    this->getDataSpheres.SetCallback(MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(1), &PLYDataSource::getSphereExtentCallback);
    this->MakeSlotAvailable(&this->getDataSpheres);

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
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "    property: %s %s", p.name.c_str(), tinyply::PropertyTable[p.propertyType].str.c_str());
        }
    }

    try {
        this->vertexData = plf.request_properties_from_element("vertex", {"x", "y", "z"});
    } catch (std::exception& e) {
        Log::DefaultLog.WriteError("Error during vertex coordinate parsing");
    }
    try {
        this->vertexNormalData = plf.request_properties_from_element("vertex", { "nx", "ny", "nz" });
    } catch (std::exception& e) {
        Log::DefaultLog.WriteError("Error during vertex normal parsing");
    }
    try {
        this->vertexColorData = plf.request_properties_from_element("vertex", { "red", "green", "blue", "alpha" });
    } catch (std::exception& e) {
        Log::DefaultLog.WriteError("Error during vertex color parsing");
    }
    try {
        this->faceData = plf.request_properties_from_element("face", { "vertex_indices" });
    } catch (std::exception& e) {
        Log::DefaultLog.WriteError("Error during face parsing");
    }
    try {
        this->texCoordData = plf.request_properties_from_element("face", { "texcoord" });
    } catch (std::exception& e) {
        Log::DefaultLog.WriteError("Error during texture coordinate parsing");
    }

    auto start = std::chrono::high_resolution_clock::now();
    plf.read(instream);
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