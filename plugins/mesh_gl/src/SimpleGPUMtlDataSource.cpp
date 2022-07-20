#include "SimpleGPUMtlDataSource.h"

#include "mesh_gl/MeshCalls_gl.h"

#include "mmcore/param/FilePathParam.h"

megamol::mesh_gl::SimpleGPUMtlDataSource::SimpleGPUMtlDataSource()
        : m_version(0)
        , m_vert_shdr_filepath_slot("Vert shader", "Filepath of the vertex shader.")
        , m_geom_shdr_filepath_slot("Geom shader", "Filepath of the geometry shader. Optional.")
        , m_tessCtrl_shdr_filepath_slot("Tess-Ctrl shader", "Filepath of the tessellation control shader. Optional")
        , m_tessEval_shdr_filepath_slot("Tess-Eval shader", "Filepath of the tessellation evaluation shader. Optional")
        , m_frag_shdr_filepath_slot("Frag shader", "Filepath of the fragement shader.") {
    this->m_vert_shdr_filepath_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_vert_shdr_filepath_slot);
    this->m_geom_shdr_filepath_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_geom_shdr_filepath_slot);
    this->m_tessCtrl_shdr_filepath_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_tessCtrl_shdr_filepath_slot);
    this->m_tessEval_shdr_filepath_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_tessEval_shdr_filepath_slot);
    this->m_frag_shdr_filepath_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_frag_shdr_filepath_slot);
}

megamol::mesh_gl::SimpleGPUMtlDataSource::~SimpleGPUMtlDataSource() {}

bool megamol::mesh_gl::SimpleGPUMtlDataSource::getDataCallback(core::Call& caller) {
    CallGPUMaterialData* lhs_mtl_call = dynamic_cast<CallGPUMaterialData*>(&caller);
    CallGPUMaterialData* rhs_mtl_call = this->m_mtl_callerSlot.CallAs<CallGPUMaterialData>();

    if (lhs_mtl_call == nullptr) {
        return false;
    }

    auto gpu_mtl_collections = std::make_shared<std::vector<std::shared_ptr<GPUMaterialCollection>>>();
    // if there is a material connection to the right, issue callback
    if (rhs_mtl_call != nullptr) {
        (*rhs_mtl_call)(0);
        if (rhs_mtl_call->hasUpdate()) {
            ++m_version;
        }
        gpu_mtl_collections = rhs_mtl_call->getData();
    }
    gpu_mtl_collections->push_back(m_material_collection.first);

    bool something_has_changed = m_vert_shdr_filepath_slot.IsDirty() || m_geom_shdr_filepath_slot.IsDirty() ||
                                 m_tessCtrl_shdr_filepath_slot.IsDirty() || m_tessEval_shdr_filepath_slot.IsDirty() ||
                                 m_frag_shdr_filepath_slot.IsDirty();

    if (something_has_changed) {
        m_vert_shdr_filepath_slot.ResetDirty();
        m_geom_shdr_filepath_slot.ResetDirty();
        m_tessCtrl_shdr_filepath_slot.ResetDirty();
        m_tessEval_shdr_filepath_slot.ResetDirty();
        m_frag_shdr_filepath_slot.ResetDirty();

        ++m_version;

        //TODO only clear
        clearMaterialCollection();

        bool vert_available = false;
        bool frag_available = false;
        std::vector<std::filesystem::path> shader_filepaths;

        if (!m_vert_shdr_filepath_slot.Param<core::param::FilePathParam>()->Value().empty()) {
            shader_filepaths.emplace_back(m_vert_shdr_filepath_slot.Param<core::param::FilePathParam>()->Value());
            vert_available = true;
        }
        if (!m_geom_shdr_filepath_slot.Param<core::param::FilePathParam>()->Value().empty()) {
            shader_filepaths.emplace_back(m_geom_shdr_filepath_slot.Param<core::param::FilePathParam>()->Value());
        }
        if (!m_tessCtrl_shdr_filepath_slot.Param<core::param::FilePathParam>()->Value().empty()) {
            shader_filepaths.emplace_back(m_tessCtrl_shdr_filepath_slot.Param<core::param::FilePathParam>()->Value());
        }
        if (!m_tessEval_shdr_filepath_slot.Param<core::param::FilePathParam>()->Value().empty()) {
            shader_filepaths.emplace_back(m_tessEval_shdr_filepath_slot.Param<core::param::FilePathParam>()->Value());
        }
        if (!m_frag_shdr_filepath_slot.Param<core::param::FilePathParam>()->Value().empty()) {
            shader_filepaths.emplace_back(m_frag_shdr_filepath_slot.Param<core::param::FilePathParam>()->Value());
            frag_available = true;
        }

        if (vert_available && frag_available) {
            try {
                m_material_collection.first->addMaterial(
                    this->instance(), std::string(this->FullName()), shader_filepaths);
                m_material_collection.second.push_back(std::string(this->FullName()));
            } catch (std::runtime_error const& exc) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Error during shader program creation of \"%s\": %s. [%s, %s, line %d]\n",
                    this->FullName().PeekBuffer(), exc.what(), __FILE__, __FUNCTION__, __LINE__);
            }
        }
    }

    lhs_mtl_call->setData(gpu_mtl_collections, m_version);

    return true;
}

bool megamol::mesh_gl::SimpleGPUMtlDataSource::getMetaDataCallback(core::Call& caller) {
    return true;
}
