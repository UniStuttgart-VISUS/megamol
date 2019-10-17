#include "stdafx.h"

#include "glTFFileLoader.h"

#include "mmcore/param/FilePathParam.h"

#ifndef TINYGLTF_IMPLEMENTATION
#    define TINYGLTF_IMPLEMENTATION
#endif // !TINYGLTF_IMPLEMENTATION
#ifndef STB_IMAGE_IMPLEMENTATION
#    define STB_IMAGE_IMPLEMENTATION
#endif // !STB_IMAGE_IMPLEMENTATION
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#    define STB_IMAGE_WRITE_IMPLEMENTATION
#endif // !STB_IMAGE_WRITE_IMPLEMENTATION

#include "tiny_gltf.h"

megamol::mesh::GlTFFileLoader::GlTFFileLoader()
    : core::Module()
    , m_update_hash(0)
    , m_glTFFilename_slot("glTF filename", "The name of the gltf file to load")
    , m_getData_slot("CallGlTFData", "The slot publishing the loaded data") {
    this->m_getData_slot.SetCallback(CallGlTFData::ClassName(), "GetData", &GlTFFileLoader::getDataCallback);
    this->m_getData_slot.SetCallback(CallGlTFData::ClassName(), "GetMetaData", &GlTFFileLoader::getDataCallback);
    this->MakeSlotAvailable(&this->m_getData_slot);

    this->m_glTFFilename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_glTFFilename_slot);
}

megamol::mesh::GlTFFileLoader::~GlTFFileLoader() { this->Release(); }

bool megamol::mesh::GlTFFileLoader::create(void) {
    // intentionally empty ?
    return true;
}

bool megamol::mesh::GlTFFileLoader::getDataCallback(core::Call& caller) {
    CallGlTFData* cd = dynamic_cast<CallGlTFData*>(&caller);

    if (cd == NULL) return false;

    if (this->m_glTFFilename_slot.IsDirty()) {
        m_glTFFilename_slot.ResetDirty();

        auto vislib_filename = m_glTFFilename_slot.Param<core::param::FilePathParam>()->Value();
        std::string filename(vislib_filename.PeekBuffer());

        m_gltf_model = std::make_shared<tinygltf::Model>();
        tinygltf::TinyGLTF loader;
        std::string err;
        std::string war;

        bool ret = loader.LoadASCIIFromFile(&*m_gltf_model, &err, &war, filename);
        if (!err.empty()) {
            vislib::sys::Log::DefaultLog.WriteError("Err: %s\n", err.c_str());
        }

        if (!ret) {
            vislib::sys::Log::DefaultLog.WriteError("Failed to parse glTF\n");
        }

        ++m_update_hash;
    }

    cd->setMetaData({m_update_hash});
    cd->setData(m_gltf_model);

    return true;
}

bool megamol::mesh::GlTFFileLoader::getMetaDataCallback(core::Call& caller)
{
    CallGlTFData* cd = dynamic_cast<CallGlTFData*>(&caller);

    if (cd == NULL) return false;

    cd->setMetaData({m_update_hash});

    return true;
}

void megamol::mesh::GlTFFileLoader::release() {
    // intentionally empty ?
}
