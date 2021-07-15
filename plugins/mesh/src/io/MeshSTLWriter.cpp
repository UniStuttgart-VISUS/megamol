#include "stdafx.h"
#include "MeshSTLWriter.h"

#include "mmcore/AbstractGetData3DCall.h"

#include "mmcore/utility/log/Log.h"

#include <stdexcept>
#include <filesystem>
#include <fstream>

namespace megamol::mesh::io {
MeshSTLWriter::MeshSTLWriter()
        : AbstractSTLWriter(CallMesh::ClassName(), CallMesh::FunctionName(0), CallMesh::FunctionName(1)) {}

MeshSTLWriter::~MeshSTLWriter() {}

bool MeshSTLWriter::create() {
    return true;
}

bool MeshSTLWriter::copy_info_upstream(core::Call& caller, core::Call& callee) {
    auto& incoming_call = dynamic_cast<CallMesh&>(caller);
    auto& outgoing_call = dynamic_cast<CallMesh&>(callee);

    outgoing_call.setMetaData(incoming_call.getMetaData());

    return true;
}

bool MeshSTLWriter::copy_info_downstream(core::Call& caller, core::Call& callee) {
    auto& incoming_call = dynamic_cast<CallMesh&>(caller);
    auto& outgoing_call = dynamic_cast<CallMesh&>(callee);

    incoming_call.setMetaData(outgoing_call.getMetaData());

    return true;
}

bool MeshSTLWriter::check_update(core::Call& caller, core::Call& callee) {
    auto& incoming_call = dynamic_cast<CallMesh&>(caller);
    auto& outgoing_call = dynamic_cast<CallMesh&>(callee);

    if (!outgoing_call.hasUpdate())
        return false;

    incoming_call.setMetaData(outgoing_call.getMetaData());

    return true;
}

bool MeshSTLWriter::copy_data(core::Call& caller, core::Call& callee) {
    auto& incoming_call = dynamic_cast<CallMesh&>(caller);
    auto& outgoing_call = dynamic_cast<CallMesh&>(callee);

    // incoming_call.SetObjects(outgoing_call.Count(), outgoing_call.Objects());
    incoming_call.setData(outgoing_call.getData(), outgoing_call.version());

    return true;
}

bool MeshSTLWriter::write_data(core::Call& callee) {
    auto& outgoing_call = dynamic_cast<CallMesh&>(callee);

    // Get data and save it to file
    try {
        auto const meta = outgoing_call.getMetaData();
        auto const mesh_data = outgoing_call.getData();

        auto const& meshes = mesh_data->accessMeshes();

        for (auto const& [name, mesh] : meshes) {
            auto const mesh_type = mesh.primitive_type;

            if (mesh_type != mesh::MeshDataAccessCollection::PrimitiveType::TRIANGLES) {
                throw std::runtime_error("Illegal primitive type");
            }

            auto const& indices = mesh.indices;
            auto const& attributes = mesh.attributes;

            auto const vertices = std::find_if(attributes.begin(), attributes.end(), [](auto const& el) {
                return el.semantic == mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION;
            });
            auto const normals = std::find_if(attributes.begin(), attributes.end(), [](auto const& el) {
                return el.semantic == mesh::MeshDataAccessCollection::AttributeSemanticType::NORMAL;
            });
            auto const colors = std::find_if(attributes.begin(), attributes.end(), [](auto const& el) {
                return el.semantic == mesh::MeshDataAccessCollection::AttributeSemanticType::COLOR;
            });

            if (vertices == attributes.cend()) {
                throw std::runtime_error("[MeshSTLWriter] No vertex sttribute available");
            }
            if (normals == attributes.cend()) {
                throw std::runtime_error("[MeshSTLWriter] No normal sttribute available");
            }

            if (indices.type == mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT) {
                if (vertices->component_type == mesh::MeshDataAccessCollection::ValueType::FLOAT &&
                    normals->component_type == mesh::MeshDataAccessCollection::ValueType::FLOAT) {
                    auto const num_triangles = indices.byte_size / (sizeof(unsigned int) * 3);
                    AbstractSTLWriter::write(num_triangles, meta.m_frame_ID, reinterpret_cast<float*>(vertices->data),
                        reinterpret_cast<float*>(normals->data), reinterpret_cast<unsigned int*>(indices.data));
                }
            } else if (indices.type == mesh::MeshDataAccessCollection::ValueType::INT) {
                if (vertices->component_type == mesh::MeshDataAccessCollection::ValueType::FLOAT &&
                    normals->component_type == mesh::MeshDataAccessCollection::ValueType::FLOAT) {
                    auto const num_triangles = indices.byte_size / (sizeof(int) * 3);
                    AbstractSTLWriter::write(num_triangles, meta.m_frame_ID, reinterpret_cast<float*>(vertices->data),
                        reinterpret_cast<float*>(normals->data), reinterpret_cast<int*>(indices.data));
                }
            }

            if (colors != attributes.cend()) {
                // write colors in separate file
                auto filename = std::filesystem::path(current_filename);
                filename.replace_extension(".col");
                auto ofs = std::ofstream(filename, std::ios::binary);
                uint64_t data_size = colors->byte_size;
                ofs.write(reinterpret_cast<char*>(&data_size), sizeof(uint64_t));
                ofs.write(reinterpret_cast<char*>(colors->data), colors->byte_size);
            }

            break;
        }

        //// Sanity check
        // if (outgoing_call.Objects()->GetVertexDataType() == geocalls::CallTriMeshData::Mesh::DT_NONE) {
        //    throw std::runtime_error("Illegal vertex data type");
        //}
        // if (outgoing_call.Objects()->GetNormalDataType() == geocalls::CallTriMeshData::Mesh::DT_NONE) {
        //    throw std::runtime_error("Illegal normal data type");
        //}

        //// Write file
        // const std::size_t num_triangles = static_cast<std::size_t>(outgoing_call.Objects()->GetTriCount());

        // if (outgoing_call.Objects()->GetVertexDataType() == geocalls::CallTriMeshData::Mesh::DT_FLOAT) {
        //    if (outgoing_call.Objects()->GetNormalDataType() == geocalls::CallTriMeshData::Mesh::DT_FLOAT) {
        //        if (outgoing_call.Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_BYTE) {
        //            AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
        //                outgoing_call.Objects()->GetVertexPointerFloat(),
        //                outgoing_call.Objects()->GetNormalPointerFloat(),
        //                outgoing_call.Objects()->GetTriIndexPointerByte());
        //        } else if (outgoing_call.Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT16) {
        //            AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
        //                outgoing_call.Objects()->GetVertexPointerFloat(),
        //                outgoing_call.Objects()->GetNormalPointerFloat(),
        //                outgoing_call.Objects()->GetTriIndexPointerUInt16());
        //        } else if (outgoing_call.Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT32) {
        //            AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
        //                outgoing_call.Objects()->GetVertexPointerFloat(),
        //                outgoing_call.Objects()->GetNormalPointerFloat(),
        //                outgoing_call.Objects()->GetTriIndexPointerUInt32());
        //        } else {
        //            AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
        //                outgoing_call.Objects()->GetVertexPointerFloat(),
        //                outgoing_call.Objects()->GetNormalPointerFloat());
        //        }
        //    } else {
        //        if (outgoing_call.Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_BYTE) {
        //            AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
        //                outgoing_call.Objects()->GetVertexPointerFloat(),
        //                outgoing_call.Objects()->GetNormalPointerDouble(),
        //                outgoing_call.Objects()->GetTriIndexPointerByte());
        //        } else if (outgoing_call.Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT16) {
        //            AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
        //                outgoing_call.Objects()->GetVertexPointerFloat(),
        //                outgoing_call.Objects()->GetNormalPointerDouble(),
        //                outgoing_call.Objects()->GetTriIndexPointerUInt16());
        //        } else if (outgoing_call.Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT32) {
        //            AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
        //                outgoing_call.Objects()->GetVertexPointerFloat(),
        //                outgoing_call.Objects()->GetNormalPointerDouble(),
        //                outgoing_call.Objects()->GetTriIndexPointerUInt32());
        //        } else {
        //            AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
        //                outgoing_call.Objects()->GetVertexPointerFloat(),
        //                outgoing_call.Objects()->GetNormalPointerDouble());
        //        }
        //    }
        //} else {
        //    if (outgoing_call.Objects()->GetNormalDataType() == geocalls::CallTriMeshData::Mesh::DT_FLOAT) {
        //        if (outgoing_call.Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_BYTE) {
        //            AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
        //                outgoing_call.Objects()->GetVertexPointerDouble(),
        //                outgoing_call.Objects()->GetNormalPointerFloat(),
        //                outgoing_call.Objects()->GetTriIndexPointerByte());
        //        } else if (outgoing_call.Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT16) {
        //            AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
        //                outgoing_call.Objects()->GetVertexPointerDouble(),
        //                outgoing_call.Objects()->GetNormalPointerFloat(),
        //                outgoing_call.Objects()->GetTriIndexPointerUInt16());
        //        } else if (outgoing_call.Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT32) {
        //            AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
        //                outgoing_call.Objects()->GetVertexPointerDouble(),
        //                outgoing_call.Objects()->GetNormalPointerFloat(),
        //                outgoing_call.Objects()->GetTriIndexPointerUInt32());
        //        } else {
        //            AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
        //                outgoing_call.Objects()->GetVertexPointerDouble(),
        //                outgoing_call.Objects()->GetNormalPointerFloat());
        //        }
        //    } else {
        //        if (outgoing_call.Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_BYTE) {
        //            AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
        //                outgoing_call.Objects()->GetVertexPointerDouble(),
        //                outgoing_call.Objects()->GetNormalPointerDouble(),
        //                outgoing_call.Objects()->GetTriIndexPointerByte());
        //        } else if (outgoing_call.Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT16) {
        //            AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
        //                outgoing_call.Objects()->GetVertexPointerDouble(),
        //                outgoing_call.Objects()->GetNormalPointerDouble(),
        //                outgoing_call.Objects()->GetTriIndexPointerUInt16());
        //        } else if (outgoing_call.Objects()->GetTriDataType() == geocalls::CallTriMeshData::Mesh::DT_UINT32) {
        //            AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
        //                outgoing_call.Objects()->GetVertexPointerDouble(),
        //                outgoing_call.Objects()->GetNormalPointerDouble(),
        //                outgoing_call.Objects()->GetTriIndexPointerUInt32());
        //        } else {
        //            AbstractSTLWriter::write(num_triangles, outgoing_call.FrameID(),
        //                outgoing_call.Objects()->GetVertexPointerDouble(),
        //                outgoing_call.Objects()->GetNormalPointerDouble());
        //        }
        //    }
        //}
    } catch (const std::runtime_error& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Request for writing to STL file failed: %s", ex.what());

        return false;
    }

    return true;
}

void MeshSTLWriter::release() {}
} // namespace megamol::mesh::io
