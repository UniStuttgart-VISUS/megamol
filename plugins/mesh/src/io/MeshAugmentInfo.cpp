#include "MeshAugmentInfo.h"

#include <algorithm>
#include <filesystem>
#include <fstream>

#include "mmcore/param/FilePathParam.h"

#include "TFUtils.h"


megamol::mesh::io::MeshAugmentInfo::MeshAugmentInfo()
        : mesh_out_slot_("meshOut", "")
        , info_out_slot_("infoOut", "")
        , mesh_in_slot_("meshIn", "")
        , tf_in_slot_("tfIn", "")
        , filename_slot_("filename", "") {
    mesh_out_slot_.SetCallback(CallMesh::ClassName(), CallMesh::FunctionName(0), &MeshAugmentInfo::get_mesh_data_cb);
    mesh_out_slot_.SetCallback(CallMesh::ClassName(), CallMesh::FunctionName(1), &MeshAugmentInfo::get_mesh_extent_cb);
    MakeSlotAvailable(&mesh_out_slot_);

    info_out_slot_.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(0), &MeshAugmentInfo::get_info_data_cb);
    info_out_slot_.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(1), &MeshAugmentInfo::get_info_extent_cb);
    MakeSlotAvailable(&info_out_slot_);

    mesh_in_slot_.SetCompatibleCall<CallMeshDescription>();
    MakeSlotAvailable(&mesh_in_slot_);

    tf_in_slot_.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    MakeSlotAvailable(&tf_in_slot_);

    filename_slot_ << new core::param::FilePathParam("");
    MakeSlotAvailable(&filename_slot_);
}


megamol::mesh::io::MeshAugmentInfo::~MeshAugmentInfo() {
    this->Release();
}


bool megamol::mesh::io::MeshAugmentInfo::create() {
    return true;
}


void megamol::mesh::io::MeshAugmentInfo::release() {}


bool megamol::mesh::io::MeshAugmentInfo::get_mesh_data_cb(core::Call& c) {
    auto out_mesh = dynamic_cast<CallMesh*>(&c);
    if (out_mesh == nullptr)
        return false;
    auto in_mesh = mesh_in_slot_.CallAs<CallMesh>();
    if (in_mesh == nullptr)
        return false;
    core::view::CallGetTransferFunction* cgtf = tf_in_slot_.CallAs<core::view::CallGetTransferFunction>();
    if (cgtf == nullptr)
        return false;
    if (!(*cgtf)())
        return false;

    auto out_meta_data = out_mesh->getMetaData();
    auto in_meta_data = in_mesh->getMetaData();
    in_meta_data.m_frame_ID = out_meta_data.m_frame_ID;

    in_mesh->setMetaData(in_meta_data);
    if (!(*in_mesh)(1))
        return false;
    if (!(*in_mesh)(0))
        return false;

    bool success = true;

    if (in_mesh->hasUpdate() || is_dirty() || cgtf->IsDirty()) {
        success = assert_data(*in_mesh, *cgtf);
        reset_dirty();
        cgtf->ResetDirty();
    }

    if (success) {
        auto meta_data = in_mesh->getMetaData();
        out_mesh->setMetaData(meta_data);
        out_mesh->setData(meshes_, out_data_hash_);
    }

    return success;
}


bool megamol::mesh::io::MeshAugmentInfo::get_mesh_extent_cb(core::Call& c) {
    auto out_mesh = dynamic_cast<mesh::CallMesh*>(&c);
    if (out_mesh == nullptr)
        return false;
    auto in_mesh = mesh_in_slot_.CallAs<CallMesh>();
    if (in_mesh == nullptr)
        return false;
    auto out_meta_data = out_mesh->getMetaData();
    auto in_meta_data = in_mesh->getMetaData();
    in_meta_data.m_frame_ID = out_meta_data.m_frame_ID;
    in_mesh->setMetaData(in_meta_data);
    if (!(*in_mesh)(1))
        return false;

    in_meta_data = in_mesh->getMetaData();
    out_mesh->setMetaData(in_meta_data);

    return true;
}


bool megamol::mesh::io::MeshAugmentInfo::get_info_data_cb(core::Call& c) {
    auto out_table = dynamic_cast<stdplugin::datatools::table::TableDataCall*>(&c);
    if (out_table == nullptr)
        return false;
    auto in_mesh = mesh_in_slot_.CallAs<CallMesh>();
    if (in_mesh == nullptr)
        return false;
    core::view::CallGetTransferFunction* cgtf = tf_in_slot_.CallAs<core::view::CallGetTransferFunction>();
    if (cgtf == nullptr)
        return false;
    if (!(*cgtf)())
        return false;

    auto in_meta_data = in_mesh->getMetaData();
    in_meta_data.m_frame_ID = out_table->GetFrameID();
    if (!(*in_mesh)(1))
        return false;
    if (!(*in_mesh)(0))
        return false;

    bool success = true;

    if (in_mesh->hasUpdate() || is_dirty() || cgtf->IsDirty()) {
        success = assert_data(*in_mesh, *cgtf);
        reset_dirty();
        cgtf->ResetDirty();
    }

    if (success) {
        out_table->Set(1, infos_.size(), &column_info_, infos_.data());
        out_table->SetDataHash(out_data_hash_);
    }

    return true;
}


bool megamol::mesh::io::MeshAugmentInfo::get_info_extent_cb(core::Call& c) {
    auto out_table = dynamic_cast<stdplugin::datatools::table::TableDataCall*>(&c);
    if (out_table == nullptr)
        return false;
    auto in_mesh = mesh_in_slot_.CallAs<CallMesh>();
    if (in_mesh == nullptr)
        return false;

    auto in_meta_data = in_mesh->getMetaData();
    in_meta_data.m_frame_ID = out_table->GetFrameID();
    if (!(*in_mesh)(1))
        return false;

    in_meta_data = in_mesh->getMetaData();

    out_table->SetFrameCount(in_meta_data.m_frame_cnt);
    out_table->SetFrameID(in_meta_data.m_frame_ID);

    return true;
}


bool megamol::mesh::io::MeshAugmentInfo::assert_data(CallMesh& mesh, core::view::CallGetTransferFunction& tf) {
    auto const filename =
        std::filesystem::path(filename_slot_.Param<core::param::FilePathParam>()->Value().PeekBuffer());

    if (!std::filesystem::exists(filename)) {
        core::utility::log::Log::DefaultLog.WriteError("[MeshAugmentInfo] File %s does not exist", filename.c_str());
        return false;
    }

    {
        auto ifs = std::ifstream(filename, std::ios::binary);
        uint64_t data_size = 0;
        ifs.read(reinterpret_cast<char*>(&data_size), sizeof(uint64_t));
        infos_.resize(data_size / sizeof(float));
        ifs.read(reinterpret_cast<char*>(infos_.data()), data_size);
    }

    auto const mesh_data = mesh.getData();

    auto const minmax_el = std::minmax_element(infos_.begin(), infos_.end());
    auto const min_i = *minmax_el.first;
    auto const max_i = *minmax_el.second;
    auto const fac_i = 1.0f / (max_i - min_i + 1e-8f);

    tf.SetRange({min_i, max_i});
    tf(0);

    column_info_.SetName("mesh_info");
    column_info_.SetType(stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE);
    column_info_.SetMinimumValue(min_i);
    column_info_.SetMaximumValue(max_i);

    auto const color_tf_size = tf.TextureSize();
    auto const color_tf = tf.GetTextureData();

    colors_.clear();

    meshes_ = std::make_shared<MeshDataAccessCollection>();

    for (auto const& [name, mesh_el] : mesh_data->accessMeshes()) {
        if (mesh_el.primitive_type != MeshDataAccessCollection::PrimitiveType::TRIANGLES) {
            core::utility::log::Log::DefaultLog.WriteError("[MeshAugmentInfo] Only triangles are supported");
            return false;
        }

        auto const num_triangles =
            mesh_el.indices.byte_size / MeshDataAccessCollection::getByteSize(mesh_el.indices.type) / 3;

        if (infos_.size() != num_triangles * 3) {
            core::utility::log::Log::DefaultLog.WriteError(
                "[MeshAugmentInfo] Num triangles and info buffer size do not match");
            return false;
        }

        auto const start_idx = colors_.size();
        // samples transferfunction
        colors_.reserve(colors_.size() + infos_.size() * 4);

        std::for_each(infos_.begin(), infos_.end(), [&min_i, &fac_i, &color_tf_size, &color_tf, this](auto info) {
            auto const val = (info - min_i) * fac_i * static_cast<float>(color_tf_size);
            std::remove_const_t<decltype(val)> main = 0;
            auto rest = std::modf(val, &main);
            rest = static_cast<int>(main) >= 0 && static_cast<int>(main) < color_tf_size ? rest : 0.0f;
            main = std::clamp<int>(static_cast<int>(main), 0, color_tf_size - 1);
            auto const col = stdplugin::datatools::sample_tf(color_tf, color_tf_size, static_cast<int>(main), rest);
            colors_.push_back(col.r);
            colors_.push_back(col.g);
            colors_.push_back(col.b);
            colors_.push_back(col.a);
        });

        auto attributes = mesh_el.attributes;
        auto const attr_fit = std::find_if(attributes.begin(), attributes.end(),
            [](auto const& el) { return el.semantic == mesh::MeshDataAccessCollection::COLOR; });
        if (attr_fit != attributes.end()) {
            attributes.erase(attr_fit);
        }
        attributes.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{
            reinterpret_cast<uint8_t*>(colors_.data() + start_idx), infos_.size() * 4 * sizeof(float), 4,
            mesh::MeshDataAccessCollection::FLOAT, sizeof(float) * 4, 0, mesh::MeshDataAccessCollection::COLOR});
        meshes_->addMesh(name, attributes, mesh_el.indices);
    }

    ++out_data_hash_;

    return true;
}
