#include "stdafx.h"

#include "glTFFileLoader.h"

#include "mmcore/param/FilePathParam.h"

#include "tiny_gltf.h"

megamol::ngmesh::GlTFFileLoader::GlTFFileLoader()
	: core::Module(),
	m_glTFFilename_slot("glTF filename", "The name of the gltf file to load"),
	m_getData_slot("getData","The slot publishing the loaded data")
{
	this->m_getData_slot.SetCallback(GlTFDataCall::ClassName(), "GetData", &GlTFFileLoader::getDataCallback);
	this->MakeSlotAvailable(&this->m_getData_slot);

	this->m_glTFFilename_slot << new core::param::FilePathParam("");
	this->MakeSlotAvailable(&this->m_glTFFilename_slot);
}

megamol::ngmesh::GlTFFileLoader::~GlTFFileLoader()
{
	this->Release();
}

bool megamol::ngmesh::GlTFFileLoader::create(void)
{
	// intentionally empty ?
	return true;
}

bool megamol::ngmesh::GlTFFileLoader::getDataCallback(core::Call & caller)
{
	GlTFDataCall* cd = dynamic_cast<GlTFDataCall*>(&caller);

	if (cd == NULL)
		return false;

	cd->clearUpdateFlag();
	m_update_flag = std::max(0, m_update_flag-1);

	if (this->m_glTFFilename_slot.IsDirty())
	{
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

		m_update_flag = std::min(2, m_update_flag + 2);
	}

	cd->setGlTFModel(m_gltf_model);
	if(m_update_flag > 0)
		cd->setUpdateFlag();
	
	return true;
}

void megamol::ngmesh::GlTFFileLoader::release()
{
	// intentionally empty ?
}
