#include "PathReader.h"

#include <fstream>

#include "mmcore/param/FilePathParam.h"


megamol::thermodyn::PathReader::PathReader() : data_out_slot_("dataOut", ""), filename_slot_("filename", "") {
    data_out_slot_.SetCallback(mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &PathReader::get_data_cb);
    data_out_slot_.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &PathReader::get_extent_cb);
    MakeSlotAvailable(&data_out_slot_);

    filename_slot_ << new core::param::FilePathParam("path.dump");
    MakeSlotAvailable(&filename_slot_);
}


megamol::thermodyn::PathReader::~PathReader() {
    this->Release();
}


bool megamol::thermodyn::PathReader::create() {
    return true;
}


void megamol::thermodyn::PathReader::release() {}


bool megamol::thermodyn::PathReader::get_data_cb(core::Call& c) {
    auto data_out = dynamic_cast<mesh::CallMesh*>(&c);
    if (data_out == nullptr)
        return false;

    auto meta = data_out->getMetaData();

    if (meta.m_frame_ID != frame_id_ || filename_slot_.IsDirty()) {
        if (!assert_data(meta.m_frame_ID))
            return false;
        frame_id_ = meta.m_frame_ID;
        filename_slot_.ResetDirty();
    }

    data_out->setData(mesh_col_, out_data_hash_);

    return true;
}


bool megamol::thermodyn::PathReader::get_extent_cb(core::Call& c) {
    auto data_out = dynamic_cast<mesh::CallMesh*>(&c);
    if (data_out == nullptr)
        return false;

    auto meta = data_out->getMetaData();

    if (meta.m_frame_ID != frame_id_ || filename_slot_.IsDirty()) {
        if (!assert_data(meta.m_frame_ID))
            return false;
        frame_id_ = meta.m_frame_ID;
        filename_slot_.ResetDirty();
    }

    meta.m_frame_cnt = frame_count_;
    meta.m_bboxs.SetBoundingBox(bbox_);
    meta.m_bboxs.SetClipBox(cbox_);

    data_out->setMetaData(meta);

    return true;
}


bool megamol::thermodyn::PathReader::assert_data(int frame_id) {
    auto const filename =
        std::filesystem::path(filename_slot_.Param<core::param::FilePathParam>()->Value().PeekBuffer());

    mesh_col_ = std::make_shared<mesh::MeshDataAccessCollection>();

    indices_.clear();
    positions_.clear();
    colors_.clear();

    auto ifile = std::ifstream(filename, std::ios::binary);

    std::array<float, 6> tmp_box;
    ifile.read(reinterpret_cast<char*>(tmp_box.data()), 6 * sizeof(float));
    bbox_.Set(tmp_box[0], tmp_box[1], tmp_box[2], tmp_box[3], tmp_box[4], tmp_box[5]);
    ifile.read(reinterpret_cast<char*>(tmp_box.data()), 6 * sizeof(float));
    cbox_.Set(tmp_box[0], tmp_box[1], tmp_box[2], tmp_box[3], tmp_box[4], tmp_box[5]);

    uint32_t f_count = 0;
    ifile.read(reinterpret_cast<char*>(&f_count), sizeof(uint32_t));
    frame_count_ = f_count;
    /*auto const frame_offset_pos = ifile.tellg();
    std::vector<uint64_t> frame_offset_list(f_count);
    ifile.read(reinterpret_cast<char*>(frame_offset_list.data()), f_count * sizeof(uint64_t));

    uint64_t frame_offset = ifile.tellg();
    if (frame_id != 0) {
        frame_offset = frame_offset_list[frame_id - 1];
    }

    ifile.seekg(frame_offset);*/

    uint64_t mesh_count = 0;
    ifile.read(reinterpret_cast<char*>(&mesh_count), sizeof(uint64_t));
    std::vector<uint64_t> mesh_offset_list(mesh_count);
    ifile.read(reinterpret_cast<char*>(mesh_offset_list.data()), mesh_count * sizeof(uint64_t));

    //mesh_count = 400;

    indices_.resize(mesh_count);
    positions_.resize(mesh_count);
    colors_.resize(mesh_count);

    for (uint64_t m_idx = 0; m_idx < mesh_count; ++m_idx) {
        auto& indices = indices_[m_idx];
        auto& colors = colors_[m_idx];
        auto& positions = positions_[m_idx];

        uint64_t idx = 0;
        ifile.read(reinterpret_cast<char*>(&idx), sizeof(uint64_t));

        uint64_t indices_size = 0;
        ifile.read(reinterpret_cast<char*>(&indices_size), sizeof(uint64_t));
        indices.clear();
        indices.reserve(indices_size / sizeof(uint32_t));
        // ifile.read(reinterpret_cast<char*>(indices.data()), indices_size);
        for (uint64_t f_idx = 0; f_idx < (indices_size / sizeof(uint32_t) / 2) - 1; ++f_idx) {
            indices.push_back(f_idx);
            indices.push_back(f_idx + 1);
        }

        uint64_t num_pos = 0;
        ifile.read(reinterpret_cast<char*>(&num_pos), sizeof(uint64_t));
        positions.resize(num_pos);
        ifile.read(reinterpret_cast<char*>(positions.data()), num_pos * sizeof(glm::vec3));

        colors.resize(num_pos, glm::vec3(1.f));

        mesh::MeshDataAccessCollection::IndexData index_data;
        index_data.type = mesh::MeshDataAccessCollection::UNSIGNED_INT;
        index_data.byte_size = indices.size() * sizeof(uint32_t);
        index_data.data = reinterpret_cast<uint8_t*>(indices.data());

        std::vector<mesh::MeshDataAccessCollection::VertexAttribute> attributes;
        attributes.emplace_back(
            mesh::MeshDataAccessCollection::VertexAttribute{reinterpret_cast<uint8_t*>(positions.data()),
                num_pos * sizeof(glm::vec3), 3, mesh::MeshDataAccessCollection::FLOAT, sizeof(glm::vec3), 0,
                mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION});
        attributes.emplace_back(
            mesh::MeshDataAccessCollection::VertexAttribute{reinterpret_cast<uint8_t*>(colors.data()),
                num_pos * sizeof(glm::vec3), 3, mesh::MeshDataAccessCollection::FLOAT, sizeof(glm::vec3), 0,
                mesh::MeshDataAccessCollection::AttributeSemanticType::COLOR});

        mesh_col_->addMesh(std::to_string(idx), attributes, index_data, mesh::MeshDataAccessCollection::LINES);
    }

    ++out_data_hash_;

    return true;
}
