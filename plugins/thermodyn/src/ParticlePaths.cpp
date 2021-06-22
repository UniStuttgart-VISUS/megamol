#include "ParticlePaths.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"


megamol::thermodyn::ParticlePaths::ParticlePaths()
        : data_out_slot_("dataOut", "")
        , data_in_slot_("dataIn", "")
        , all_frames_slot_("frames::all", "")
        , min_frame_slot_("frames::min", "")
        , max_frame_slot_("frames::max", "") {
    data_out_slot_.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &ParticlePaths::get_data_cb);
    data_out_slot_.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &ParticlePaths::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    data_in_slot_.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&data_in_slot_);

    all_frames_slot_ << new core::param::BoolParam(true);
    MakeSlotAvailable(&all_frames_slot_);

    min_frame_slot_ << new core::param::IntParam(0, 0);
    MakeSlotAvailable(&min_frame_slot_);

    max_frame_slot_ << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&max_frame_slot_);
}


megamol::thermodyn::ParticlePaths::~ParticlePaths() {
    this->Release();
}


bool megamol::thermodyn::ParticlePaths::create() {
    return true;
}


void megamol::thermodyn::ParticlePaths::release() {}


bool megamol::thermodyn::ParticlePaths::get_data_cb(core::Call& c) {
    auto data_out = dynamic_cast<mesh::CallMesh*>(&c);
    if (data_out == nullptr)
        return false;

    auto data_in = data_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (data_in == nullptr)
        return false;

    auto meta = data_out->getMetaData();
    data_in->SetFrameID(meta.m_frame_ID);
    if (!(*data_in)(0))
        return false;

    if (data_in->DataHash() != in_data_hash_ || is_dirty()) {
        if (!assert_data(*data_in))
            return false;

        in_data_hash_ = data_in->DataHash();
        ++out_data_hash_;

        mesh_col_ = std::make_shared<mesh::MeshDataAccessCollection>();

        for (auto& lines_col : lines_) {
            for (auto& [id, line] : lines_col) {
                mesh::MeshDataAccessCollection::IndexData mesh_indices;
                mesh_indices.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
                mesh_indices.byte_size = line.second.size() * sizeof(uint32_t);
                mesh_indices.data = reinterpret_cast<uint8_t*>(line.second.data());
                std::vector<mesh::MeshDataAccessCollection::VertexAttribute> mesh_attributes;
                mesh_attributes.emplace_back(
                    mesh::MeshDataAccessCollection::VertexAttribute{reinterpret_cast<uint8_t*>(line.first.data()),
                        line.first.size() * sizeof(glm::vec4), 4, mesh::MeshDataAccessCollection::FLOAT,
                        sizeof(glm::vec4), 0, mesh::MeshDataAccessCollection::POSITION});
                auto const ident = std::to_string(id);
                mesh_col_->addMesh(ident, mesh_attributes, mesh_indices, mesh::MeshDataAccessCollection::LINE_STRIP);
            }
        }

        reset_dirty();
    }

    data_out->setData(mesh_col_, out_data_hash_);
    meta.m_frame_ID = frame_id_;
    data_out->setMetaData(meta);

    return true;
}


bool megamol::thermodyn::ParticlePaths::get_extent_cb(core::Call& c) {
    auto data_out = dynamic_cast<mesh::CallMesh*>(&c);
    if (data_out == nullptr)
        return false;

    auto data_in = data_in_slot_.CallAs<core::moldyn::MultiParticleDataCall>();
    if (data_in == nullptr)
        return false;

    auto meta = data_out->getMetaData();
    data_in->SetFrameID(meta.m_frame_ID);
    if (!(*data_in)(1))
        return false;
    if (!(*data_in)(0))
        return false;

    if (data_in->DataHash() != in_data_hash_ || is_dirty()) {
        if (!assert_data(*data_in))
            return false;

        in_data_hash_ = data_in->DataHash();
        ++out_data_hash_;

        mesh_col_ = std::make_shared<mesh::MeshDataAccessCollection>();

        for (auto& lines_col : lines_) {
            for (auto& [id, line] : lines_col) {
                mesh::MeshDataAccessCollection::IndexData mesh_indices;
                mesh_indices.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
                mesh_indices.byte_size = line.second.size() * sizeof(uint32_t);
                mesh_indices.data = reinterpret_cast<uint8_t*>(line.second.data());
                std::vector<mesh::MeshDataAccessCollection::VertexAttribute> mesh_attributes;
                mesh_attributes.emplace_back(
                    mesh::MeshDataAccessCollection::VertexAttribute{reinterpret_cast<uint8_t*>(line.first.data()),
                        line.first.size() * sizeof(glm::vec3), 3, mesh::MeshDataAccessCollection::FLOAT,
                        sizeof(glm::vec3), 0, mesh::MeshDataAccessCollection::POSITION});
                auto const ident = std::to_string(id);
                mesh_col_->addMesh(ident, mesh_attributes, mesh_indices, mesh::MeshDataAccessCollection::LINES);
            }
        }

        reset_dirty();
    }

    meta.m_frame_cnt = frame_count_;
    meta.m_frame_ID = frame_id_;
    meta.m_bboxs.SetBoundingBox(data_in->AccessBoundingBoxes().ObjectSpaceBBox());
    meta.m_bboxs.SetClipBox(data_in->AccessBoundingBoxes().ObjectSpaceClipBox());

    data_out->setMetaData(meta);

    return true;
}


bool megamol::thermodyn::ParticlePaths::assert_data(core::moldyn::MultiParticleDataCall& in_parts) {
    auto f_count = in_parts.FrameCount();
    auto min_f = 0;
    auto max_f = f_count == 0 ? 0 : f_count - 1;

    if (!all_frames_slot_.Param<core::param::BoolParam>()->Value()) {
        min_f = min_frame_slot_.Param<core::param::IntParam>()->Value();
        max_f = max_frame_slot_.Param<core::param::IntParam>()->Value();
        auto new_f_count = max_f - min_f;
        if (new_f_count > f_count)
            return false;
        f_count = new_f_count;
    }

    frame_count_ = f_count;
    frame_id_ = min_f;

    if (f_count == 0)
        return true;

    auto const pl_count = in_parts.GetParticleListCount();

    lines_.clear();
    lines_.resize(pl_count);

    for (std::decay_t<decltype(f_count)> f_idx = min_f; f_idx < max_f; ++f_idx) {
        bool got_data = false;
        do {
            in_parts.SetFrameID(f_idx, true);
            got_data = in_parts(0);
        } while (in_parts.FrameID() != f_idx && !got_data);

        for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
            auto const& parts = in_parts.AccessParticles(pl_idx);
            auto& lines = lines_[pl_idx];

            auto const p_count = parts.GetCount();

            lines.reserve(p_count);

            auto const id_acc = parts.GetParticleStore().GetIDAcc();
            auto const x_acc = parts.GetParticleStore().GetXAcc();
            auto const y_acc = parts.GetParticleStore().GetYAcc();
            auto const z_acc = parts.GetParticleStore().GetZAcc();

            for (std::decay_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
                auto& line_pos = lines[id_acc->Get_u64(p_idx)];
                line_pos.first.reserve(f_count);
                line_pos.first.push_back(
                    glm::vec3(x_acc->Get_f(p_idx), y_acc->Get_f(p_idx), z_acc->Get_f(p_idx)));
                /*line_pos.second.reserve(2 * f_count);
                if (line_pos.second.empty()) {
                    line_pos.second.push_back(0);
                    line_pos.second.push_back(1);
                } else {
                    line_pos.second.push_back(line_pos.second.back());
                    line_pos.second.push_back(line_pos.first.size() - 1);
                }*/
            }
        }

    }

    for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& parts = in_parts.AccessParticles(pl_idx);
        auto& lines = lines_[pl_idx];

        auto const p_count = parts.GetCount();

        for (auto& [key, value] : lines) {
            value.second.reserve(2 * f_count);
            //std::iota(value.second.begin(), value.second.end(), 0);
            for (std::decay_t<decltype(f_count)> f_idx = 0; f_idx < f_count - 1; ++f_idx) {
                value.second.push_back(f_idx);
                value.second.push_back(f_idx + 1);
            }
        }
    }

    return true;
}
