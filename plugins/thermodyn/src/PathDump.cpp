#include "PathDump.h"

#include <filesystem>
#include <fstream>

#include "mmcore/param/FilePathParam.h"


megamol::thermodyn::PathDump::PathDump() : data_in_slot_("dataIn", ""), filename_slot_("filename", "") {
    data_in_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&data_in_slot_);

    filename_slot_ << new core::param::FilePathParam("paths.dump");
    MakeSlotAvailable(&filename_slot_);
}


megamol::thermodyn::PathDump::~PathDump() {
    this->Release();
}


bool megamol::thermodyn::PathDump::run() {
    auto const filename =
        std::filesystem::path(filename_slot_.Param<core::param::FilePathParam>()->Value().PeekBuffer());

    auto data_in = data_in_slot_.CallAs<mesh::CallMesh>();
    if (data_in == nullptr)
        return false;

    if (!(*data_in)(1))
        return false;

    if (!(*data_in)(0))
        return false;

    auto const meta = data_in->getMetaData();

    uint32_t const f_count = meta.m_frame_cnt;

    auto ofile = std::ofstream(filename, std::ios::binary);

    ofile.write(reinterpret_cast<char const*>(meta.m_bboxs.BoundingBox().PeekBounds()), 6 * sizeof(float));
    ofile.write(reinterpret_cast<char const*>(meta.m_bboxs.ClipBox().PeekBounds()), 6 * sizeof(float));

    ofile.write(reinterpret_cast<char const*>(&f_count), sizeof(uint32_t));
    /*auto const frame_offset_pos = ofile.tellp();
    ofile.seekp(f_count * sizeof(uint64_t), std::ios::cur);

    std::vector<uint64_t> frame_offset_list;
    frame_offset_list.reserve(f_count);*/

    /* for (std::decay_t<decltype(f_count)> f_idx = 0; f_idx < f_count; ++f_idx)*/ {
        /*auto req_meta = meta;
        bool got_data = false;
        do {
            req_meta.m_frame_ID = f_idx;
            data_in->setMetaData(req_meta);
            got_data = (*data_in)(0);
            req_meta = data_in->getMetaData();
        } while (req_meta.m_frame_ID != f_idx && !got_data);*/

        auto const data = data_in->getData();

        auto const& mesh_col = data->accessMeshes();

        uint64_t const mesh_count = mesh_col.size();

        ofile.write(reinterpret_cast<char const*>(&mesh_count), sizeof(uint64_t));
        auto const mesh_offset_pos = ofile.tellp();
        ofile.seekp(mesh_count * sizeof(uint64_t), std::ios::cur);

        std::vector<uint64_t> offset_list;
        offset_list.reserve(mesh_count);

        for (auto const& [ident, mesh] : mesh_col) {
            uint64_t const idx = std::stoull(ident);

            ofile.write(reinterpret_cast<char const*>(&idx), sizeof(uint64_t));

            uint64_t const indices_size = mesh.indices.byte_size;

            ofile.write(reinterpret_cast<char const*>(&indices_size), sizeof(uint64_t));
            // ofile.write(reinterpret_cast<char const*>(mesh.indices.data), indices_size);

            auto const fit = std::find_if(mesh.attributes.begin(), mesh.attributes.end(),
                [](auto const& el) { return el.semantic == mesh::MeshDataAccessCollection::POSITION; });

            uint64_t const num_pos =
                fit->byte_size / fit->component_cnt / mesh::MeshDataAccessCollection::getByteSize(fit->component_type);

            ofile.write(reinterpret_cast<char const*>(&num_pos), sizeof(uint64_t));
            ofile.write(reinterpret_cast<char const*>(fit->data), fit->byte_size);

            offset_list.push_back(ofile.tellp());
        }
        /*auto const frame_end_pos = ofile.tellp();
        frame_offset_list.push_back(frame_end_pos);*/

        ofile.seekp(mesh_offset_pos);
        ofile.write(reinterpret_cast<char const*>(offset_list.data()), offset_list.size() * sizeof(uint64_t));
        // ofile.seekp(frame_end_pos);
    }

    /*ofile.seekp(frame_offset_pos);
    ofile.write(reinterpret_cast<char const*>(frame_offset_list.data()), frame_offset_list.size() * sizeof(uint64_t));*/

    return true;
}


bool megamol::thermodyn::PathDump::getCapabilities(core::DataWriterCtrlCall& call) {
    call.SetAbortable(true);
    return true;
}


bool megamol::thermodyn::PathDump::create() {
    return true;
}


void megamol::thermodyn::PathDump::release() {}
