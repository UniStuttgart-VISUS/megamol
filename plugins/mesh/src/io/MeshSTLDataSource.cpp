#include "MeshSTLDataSource.h"

#include <filesystem>
#include <fstream>

#include "mmcore/param/FilePathParam.h"


megamol::mesh::io::MeshSTLDataSource::MeshSTLDataSource() {}


megamol::mesh::io::MeshSTLDataSource::~MeshSTLDataSource() {
    this->Release();
}


bool megamol::mesh::io::MeshSTLDataSource::get_extent_callback(core::Call& caller) {
    if (this->filename_slot.IsDirty()) {
        this->filename_slot.ResetDirty();

        // Read data
        const auto& vislib_filename = this->filename_slot.Param<core::param::FilePathParam>()->Value();
        const std::string filename(vislib_filename.PeekBuffer());
        auto const color_filename = std::filesystem::path(filename).replace_extension(".col");

        try {
            read(filename);
            colors_.clear();
            if (std::filesystem::exists(color_filename)) {
                // auto const file_size = std::filesystem::file_size(color_filename);
                auto ifs = std::ifstream(color_filename, std::ios::binary);
                uint64_t data_size = 0;
                ifs.read(reinterpret_cast<char*>(&data_size), sizeof(uint64_t));
                auto const num_floats = data_size / sizeof(float);
                colors_.resize(num_floats);
                ifs.read(reinterpret_cast<char*>(colors_.data()), data_size);
            }
        } catch (const std::runtime_error& ex) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Request for extent of plugin 'STL Reader' failed: %s", ex.what());

            return false;
        }

        // Extract extent information
        this->min_x = this->min_y = this->min_z = std::numeric_limits<float>::max();
        this->max_x = this->max_y = this->max_z = std::numeric_limits<float>::lowest();

        const float* vertices = reinterpret_cast<float*>(&this->vertex_normal_buffer[2 * sizeof(uint32_t)]);

        for (std::size_t value_index = 0; value_index < static_cast<std::size_t>(this->num_triangles * 9);) {
            const float x = vertices[value_index++];
            const float y = vertices[value_index++];
            const float z = vertices[value_index++];

            this->min_x = std::min(this->min_x, x);
            this->min_y = std::min(this->min_y, y);
            this->min_z = std::min(this->min_z, z);

            this->max_x = std::max(this->max_x, x);
            this->max_y = std::max(this->max_y, y);
            this->max_z = std::max(this->max_z, z);
        }

        megamol::core::utility::log::Log::DefaultLog.WriteInfo("Extent: [%.2f, %.2f, %.2f] x [%.2f, %.2f, %.2f]",
            this->min_x, this->min_y, this->min_z, this->max_x, this->max_y, this->max_z);
    }

    auto& call = dynamic_cast<CallMesh&>(caller);

    auto meta = call.getMetaData();
    meta.m_bboxs.SetBoundingBox(this->min_x, this->min_y, this->min_z, this->max_x, this->max_y, this->max_z);
    meta.m_bboxs.SetClipBox(this->min_x, this->min_y, this->min_z, this->max_x, this->max_y, this->max_z);
    meta.m_frame_cnt = 1;
    meta.m_frame_ID = 0;

    call.setMetaData(meta);

    return true;
}


bool megamol::mesh::io::MeshSTLDataSource::get_mesh_data_callback(core::Call& caller) {
    // Get mesh call
    auto& call = dynamic_cast<CallMesh&>(caller);

    if (call.version() != static_cast<SIZE_T>(hash())) {
        // Read data if necessary
        if (this->filename_slot.IsDirty()) {
            if (!get_extent_callback(caller)) {
                return false;
            }
        }

        data_ = std::make_shared<MeshDataAccessCollection>();

        std::vector<MeshDataAccessCollection::VertexAttribute> attributes;
        attributes.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{
            &this->vertex_normal_buffer.data()[2 * sizeof(uint32_t)], this->num_triangles * 9 * sizeof(float), 3,
            mesh::MeshDataAccessCollection::FLOAT, sizeof(float) * 3, 0, mesh::MeshDataAccessCollection::POSITION});
        attributes.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{
            &this->vertex_normal_buffer.data()[2 * sizeof(uint32_t) + 9 * this->num_triangles * sizeof(float)],
            this->num_triangles * 9 * sizeof(float), 3, mesh::MeshDataAccessCollection::FLOAT, sizeof(float) * 3, 0,
            mesh::MeshDataAccessCollection::NORMAL});
        if (!colors_.empty()) {
            attributes.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{
                reinterpret_cast<uint8_t*>(colors_.data()), colors_.size() * sizeof(float), 4,
                mesh::MeshDataAccessCollection::FLOAT, sizeof(float) * 4, 0, mesh::MeshDataAccessCollection::COLOR});
        }

        MeshDataAccessCollection::IndexData indices;
        indices.byte_size = sizeof(unsigned int) * index_buffer.size();
        indices.data = reinterpret_cast<uint8_t*>(this->index_buffer.data());
        indices.type = MeshDataAccessCollection::ValueType::UNSIGNED_INT;

        data_->addMesh("stl0", attributes, indices);

        call.setData(data_, hash());
    }

    return true;
}
