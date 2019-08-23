#include "stdafx.h"

#include "WavefrontObjLoader.h"

#include "mmcore/param/FilePathParam.h"

megamol::mesh::WavefrontObjLoader::WavefrontObjLoader()
    : core::Module()
    , m_update_hash(0)
    , m_filename_slot("Wavefront OBJ filename", "The name of the obj file to load")
    , m_getData_slot("CallMesh", "The slot publishing the loaded data") {
    this->m_getData_slot.SetCallback(CallMesh::ClassName(), "GetData", &WavefrontObjLoader::getDataCallback);
    this->m_getData_slot.SetCallback(CallMesh::ClassName(), "GetMetaData", &WavefrontObjLoader::getDataCallback);
    this->MakeSlotAvailable(&this->m_getData_slot);

    this->m_filename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_filename_slot);
}

megamol::mesh::WavefrontObjLoader::~WavefrontObjLoader() {}

bool megamol::mesh::WavefrontObjLoader::create(void) { return false; }

bool megamol::mesh::WavefrontObjLoader::getDataCallback(core::Call& caller) { return false; }

bool megamol::mesh::WavefrontObjLoader::getMetaDataCallback(core::Call& caller) { return false; }

void megamol::mesh::WavefrontObjLoader::release() {}
