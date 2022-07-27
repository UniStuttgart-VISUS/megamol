/*
 * GPUMeshCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef GPU_MESH_DATA_STORAGE_H_INCLUDED
#define GPU_MESH_DATA_STORAGE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <algorithm>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "glowl/Mesh.hpp"

#include "mesh/MeshDataAccessCollection.h"

namespace megamol {
namespace mesh_gl {

class GPUMeshCollection {
public:
    template<typename T>
    using IteratorPair = std::pair<T, T>;

    struct BatchedMeshes {
        BatchedMeshes()
                : mesh(nullptr)
                , vertices_allocated(0)
                , vertices_used(0)
                , indices_allocated(0)
                , indices_used(0) {}

        BatchedMeshes(unsigned int vertices_allocated, unsigned int indices_allocated)
                : mesh(nullptr)
                , vertices_allocated(vertices_allocated)
                , vertices_used(0)
                , indices_allocated(indices_allocated)
                , indices_used(0) {}

        std::shared_ptr<glowl::Mesh> mesh;
        unsigned int vertices_allocated;
        unsigned int vertices_used;
        unsigned int indices_allocated;
        unsigned int indices_used;
    };

    struct SubMeshData {
        SubMeshData() : mesh(nullptr), sub_mesh_draw_command(), vertex_cnt(0), index_cnt(0) {}

        /** OpenGL mesh object that contain the data for this submesh */
        std::shared_ptr<BatchedMeshes> mesh;
        /** Draw command to render this submesh */
        glowl::DrawElementsCommand sub_mesh_draw_command;

        /** Number of vertices used by this submesh */
        unsigned int vertex_cnt;
        /** Number of indices used by this submesh */
        unsigned int index_cnt;
    };

    GPUMeshCollection() = default;
    ~GPUMeshCollection() = default;

    template<typename VertexBufferIterator, typename IndexBufferIterator>
    void addMesh(std::string const& identifier, std::vector<glowl::VertexLayout> const& vertex_descriptor,
        std::vector<IteratorPair<VertexBufferIterator>> const& vertex_buffers,
        IteratorPair<IndexBufferIterator> index_buffer, GLenum index_type, GLenum usage, GLenum primitive_type,
        bool store_seperate = false);

    void addMesh(std::string const& identifier, SubMeshData submesh);

    void addMesh(std::string const& identifier, mesh::MeshDataAccessCollection::Mesh const& mesh);

    void addMeshes(mesh::MeshDataAccessCollection const& mesh_data_access_collection);

    template<typename VertexBufferIterator, typename IndexBufferIterator>
    void updateSubMesh(std::string const& identifier, std::vector<glowl::VertexLayout> const& vertex_descriptor,
        std::vector<IteratorPair<VertexBufferIterator>> const& vertex_buffers,
        IteratorPair<IndexBufferIterator> index_buffer, GLenum index_type, GLenum primitive_type);

    void deleteSubMesh(std::string const& identifier);

    // Future work: defragmentation of GPU mesh data after several deletions
    //void defrag();

    void clear();

    SubMeshData getSubMesh(std::string const& identifier);

    std::vector<std::shared_ptr<BatchedMeshes>> const& getMeshes();

    std::unordered_map<std::string, SubMeshData> const& getSubMeshData();

private:
    std::vector<std::shared_ptr<BatchedMeshes>> m_batched_meshes;
    std::unordered_map<std::string, SubMeshData> m_sub_mesh_data;
};

template<typename VertexBufferIterator, typename IndexBufferIterator>
inline void GPUMeshCollection::addMesh(std::string const& identifier,
    std::vector<glowl::VertexLayout> const& vertex_descriptor,
    std::vector<IteratorPair<VertexBufferIterator>> const& vertex_buffers,
    IteratorPair<IndexBufferIterator> index_buffer, GLenum index_type, GLenum usage, GLenum primitive_type,
    bool store_seperate) {
    typedef typename std::iterator_traits<IndexBufferIterator>::value_type IndexBufferType;
    typedef typename std::iterator_traits<VertexBufferIterator>::value_type VertexBufferType;

    std::vector<size_t> vb_attrib_byte_sizes;
    for (auto& vertex_layout : vertex_descriptor) {
        vb_attrib_byte_sizes.push_back(vertex_layout.stride);
    }

    // get vertex buffer data pointers and byte sizes
    std::vector<GLvoid*> vb_data;
    std::vector<size_t> vb_byte_sizes;
    for (auto& vb : vertex_buffers) {
        vb_data.push_back(reinterpret_cast<GLvoid*>(&(*std::get<0>(vb))));
        vb_byte_sizes.push_back(sizeof(VertexBufferType) * std::distance(std::get<0>(vb), std::get<1>(vb)));
    }
    // compute overall byte size of index buffer
    size_t ib_byte_size =
        sizeof(VertexBufferType) * std::distance(std::get<0>(index_buffer), std::get<1>(index_buffer));

    // computer number of requested vertices and indices
    size_t req_vertex_cnt = vb_byte_sizes.front() / vb_attrib_byte_sizes.front();
    size_t req_index_cnt = ib_byte_size / glowl::computeByteSize(index_type);

    auto query = std::find_if(m_batched_meshes.begin(), m_batched_meshes.end(),
        [&vertex_descriptor, index_type, req_vertex_cnt, req_index_cnt](
            std::shared_ptr<BatchedMeshes> const& batched_meshes) {
            bool retval = true;
            if (vertex_descriptor.size() == batched_meshes->mesh->getVertexLayouts().size()) {
                for (int i = 0; i < vertex_descriptor.size(); ++i) {
                    retval &= vertex_descriptor[i] == batched_meshes->mesh->getVertexLayouts()[i];
                }
            } else {
                retval = false;
            }
            retval &= index_type == batched_meshes->mesh->getIndexType();

            auto ava_vertex_cnt = batched_meshes->vertices_allocated - batched_meshes->vertices_used;
            auto ava_index_cnt = batched_meshes->indices_allocated - batched_meshes->indices_used;
            retval &= ((req_vertex_cnt < ava_vertex_cnt) && (req_index_cnt < ava_index_cnt));
            return retval;
        });

    std::shared_ptr<BatchedMeshes> batched_mesh = nullptr;

    // if either no batch was found or mesh is requested to be stored seperated, create a new GPU mesh batch
    if (query == m_batched_meshes.end() || store_seperate) {
        size_t new_allocation_vertex_cnt = store_seperate ? req_vertex_cnt : std::max<size_t>(req_vertex_cnt, 800000);
        size_t new_allocation_index_cnt = store_seperate ? req_index_cnt : std::max<size_t>(req_index_cnt, 3200000);

        m_batched_meshes.push_back(
            std::make_shared<BatchedMeshes>(new_allocation_vertex_cnt, new_allocation_index_cnt));
        batched_mesh = m_batched_meshes.back();
        batched_mesh->vertices_used = 0;
        batched_mesh->indices_used = 0;

        std::vector<GLvoid const*> alloc_data(vertex_buffers.size(), nullptr);
        std::vector<size_t> alloc_vb_byte_sizes;
        for (size_t attrib_byte_size : vb_attrib_byte_sizes) {
            alloc_vb_byte_sizes.push_back(attrib_byte_size * new_allocation_vertex_cnt);
        }

        batched_mesh->mesh = std::make_shared<glowl::Mesh>(alloc_data, alloc_vb_byte_sizes, vertex_descriptor, nullptr,
            new_allocation_index_cnt * glowl::computeByteSize(index_type), index_type, primitive_type, usage);

    } else {
        batched_mesh = (*query);
    }

    auto new_sub_mesh = SubMeshData();
    new_sub_mesh.mesh = batched_mesh;
    new_sub_mesh.sub_mesh_draw_command.first_idx = batched_mesh->indices_used;
    new_sub_mesh.sub_mesh_draw_command.base_vertex = batched_mesh->vertices_used;
    new_sub_mesh.sub_mesh_draw_command.cnt = req_index_cnt;
    new_sub_mesh.sub_mesh_draw_command.instance_cnt = 1;
    new_sub_mesh.sub_mesh_draw_command.base_instance = 0;
    new_sub_mesh.vertex_cnt = req_vertex_cnt;
    new_sub_mesh.index_cnt = req_index_cnt;
    m_sub_mesh_data.insert({identifier, new_sub_mesh});

    // upload data to GPU
    for (size_t i = 0; i < vb_data.size(); ++i) {
        // at this point, it should be guaranteed that it points at a mesh with matching vertex layout,
        // hence it's legal to multiply requested attrib byte sizes with vertex used count
        batched_mesh->mesh->bufferVertexSubData(
            i, vb_data[i], vb_byte_sizes[i], vb_attrib_byte_sizes[i] * (batched_mesh->vertices_used));
    }

    batched_mesh->mesh->bufferIndexSubData(reinterpret_cast<GLvoid*>(&*std::get<0>(index_buffer)), ib_byte_size,
        glowl::computeByteSize(index_type) * batched_mesh->indices_used);

    // updated vertices and indices used
    batched_mesh->vertices_used += req_vertex_cnt;
    batched_mesh->indices_used += req_index_cnt;
}

template<typename VertexBufferIterator, typename IndexBufferIterator>
inline void GPUMeshCollection::updateSubMesh(std::string const& identifier,
    std::vector<glowl::VertexLayout> const& vertex_descriptor,
    std::vector<IteratorPair<VertexBufferIterator>> const& vertex_buffers,
    IteratorPair<IndexBufferIterator> index_buffer, GLenum index_type, GLenum primitive_type) {

    auto query = m_sub_mesh_data.find(identifier);

    if (query != m_sub_mesh_data.end()) {

        if (vertex_descriptor == query->second.mesh->mesh->getVertexLayouts() &&
            index_type == query->second.mesh->mesh->getIndexType() &&
            primitive_type == query->second.mesh->mesh->getPrimitiveType()) {
            typedef typename std::iterator_traits<IndexBufferIterator>::value_type IndexBufferType;
            typedef typename std::iterator_traits<VertexBufferIterator>::value_type VertexBufferType;

            std::vector<size_t> vb_attrib_byte_sizes;
            for (auto& vertex_layout : vertex_descriptor) {
                vb_attrib_byte_sizes.push_back(vertex_layout.stride);
            }

            // get vertex buffer data pointers and byte sizes
            std::vector<GLvoid*> vb_data;
            std::vector<size_t> vb_byte_sizes;
            for (auto& vb : vertex_buffers) {
                vb_data.push_back(reinterpret_cast<GLvoid*>(&(*std::get<0>(vb))));
                vb_byte_sizes.push_back(sizeof(VertexBufferType) * std::distance(std::get<0>(vb), std::get<1>(vb)));
            }
            // compute overall byte size of index buffer
            size_t ib_byte_size =
                sizeof(VertexBufferType) * std::distance(std::get<0>(index_buffer), std::get<1>(index_buffer));

            // computer number of requested vertices and indices
            size_t req_vertex_cnt = vb_byte_sizes.front() / vb_attrib_byte_sizes.front();
            size_t req_index_cnt = ib_byte_size / glowl::computeByteSize(index_type);

            if (query->second.vertex_cnt == req_vertex_cnt && query->second.index_cnt == req_index_cnt) {
                size_t index_byte_offset =
                    query->second.sub_mesh_draw_command.first_idx * glowl::computeByteSize(index_type);
                query->second.mesh->mesh->bufferIndexSubData(
                    reinterpret_cast<GLvoid*>(&*std::get<0>(index_buffer)), ib_byte_size, index_byte_offset);

                int vb_idx = 0;
                for (auto& vb : vertex_buffers) {
                    size_t vb_stride = 0;
                    for (auto& attrib : vertex_descriptor[vb_idx].attributes) {
                        vb_stride += glowl::computeAttributeByteSize(attrib);
                    }
                    size_t vertex_byte_offset = query->second.sub_mesh_draw_command.base_vertex * vb_stride;
                    query->second.mesh->mesh->bufferVertexSubData(
                        vb_idx, vb_data[vb_idx], vb_byte_sizes[vb_idx], vertex_byte_offset);

                    vb_idx++;
                }
            }
        }

    } else {
        // mesh for update not found, do nothing?
    }
}

inline void GPUMeshCollection::addMesh(std::string const& identifier, SubMeshData submesh) {

    auto query = std::find(m_batched_meshes.begin(), m_batched_meshes.end(), submesh.mesh);

    if (query == m_batched_meshes.end()) {
        m_batched_meshes.push_back(submesh.mesh);
    }

    m_sub_mesh_data.insert({identifier, submesh});
}

inline void GPUMeshCollection::addMesh(
    std::string const& identifier, mesh::MeshDataAccessCollection::Mesh const& mesh) {

    // check if primtives type
    GLenum primitive_type = GL_NONE;
    switch (mesh.primitive_type) {
    case 0:
        primitive_type = GL_TRIANGLES;
        break;
    case 1:
        primitive_type = GL_PATCHES;
        break;
    case 2:
        primitive_type = GL_LINES;
        break;
    case 3:
        primitive_type = GL_LINE_STRIP;
        break;
    case 4:
        primitive_type = GL_TRIANGLE_FAN;
        break;
    default:
        core::utility::log::Log::DefaultLog.WriteError(
            "GPUMeshCollection::addMeshes - No matching primitive type found!");
    }

    std::vector<glowl::VertexLayout> vb_layouts;
    std::vector<std::pair<uint8_t*, uint8_t*>> vb_iterators;
    std::pair<uint8_t*, uint8_t*> ib_iterators;

    ib_iterators = {mesh.indices.data, mesh.indices.data + mesh.indices.byte_size};

    auto formated_attrib_indices = mesh.getFormattedAttributeIndices();

    for (auto vb_attribs : formated_attrib_indices) {

        // for each set of attribute indices, create a vertex buffer layout and set data pointers
        // using the first attribute (could be any from the set, data pointer and stride should be equal)
        auto first_attrib = mesh.attributes[vb_attribs.front()];
        vb_layouts.push_back(glowl::VertexLayout(first_attrib.stride, {}));
        vb_iterators.push_back({first_attrib.data, first_attrib.data + first_attrib.byte_size});

        // for each attribute in the set, add it to the attributes of the vertex buffer layout
        for (auto const& attrib_idx : vb_attribs) {
            auto const& attrib = mesh.attributes[attrib_idx];
            vb_layouts.back().attributes.push_back(glowl::VertexLayout::Attribute(attrib.component_cnt,
                mesh::MeshDataAccessCollection::convertToGLType(attrib.component_type), GL_FALSE /*ToDO*/,
                attrib.offset));
        }
    }

    try {
        addMesh(identifier, vb_layouts, vb_iterators, ib_iterators,
            mesh::MeshDataAccessCollection::convertToGLType(mesh.indices.type), GL_STATIC_DRAW, primitive_type, true);
    } catch (glowl::MeshException const& exc) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Failed to add GPU mesh \"%s\": %s. [%s, %s, line %d]\n", identifier.c_str(), exc.what(), __FILE__,
            __FUNCTION__, __LINE__);

    } catch (glowl::BufferObjectException const& exc) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Failed to add GPU mesh \"%s\": %s. [%s, %s, line %d]\n", identifier.c_str(), exc.what(), __FILE__,
            __FUNCTION__, __LINE__);
    }
}

inline void GPUMeshCollection::addMeshes(mesh::MeshDataAccessCollection const& mesh_data_access_collection) {
    for (auto const& mesh_itr : mesh_data_access_collection.accessMeshes()) {
        addMesh(mesh_itr.first, mesh_itr.second);
    }
}

inline void GPUMeshCollection::deleteSubMesh(std::string const& identifier) {

    auto query = m_sub_mesh_data.find(identifier);

    if (query != m_sub_mesh_data.end()) {
        m_sub_mesh_data.erase(query);
    }

    // (Memory) Optimization: check for batch meshes without reference by any submesh,
    // i.e., shared_ptr ref cnt == 1, and delete it from vector
    for (size_t i = 0; i < m_batched_meshes.size(); ++i) {
        if (m_batched_meshes[i].use_count() < 2) {
            m_batched_meshes.erase(m_batched_meshes.begin() + i);
        }
    }
}

inline void GPUMeshCollection::clear() {
    m_batched_meshes.clear();
    m_sub_mesh_data.clear();
}

inline GPUMeshCollection::SubMeshData GPUMeshCollection::getSubMesh(std::string const& identifier) {

    auto query = m_sub_mesh_data.find(identifier);

    if (query != m_sub_mesh_data.end()) {
        return query->second;
    }

    return GPUMeshCollection::SubMeshData();
}

inline std::vector<std::shared_ptr<GPUMeshCollection::BatchedMeshes>> const& GPUMeshCollection::getMeshes() {
    return m_batched_meshes;
}

inline std::unordered_map<std::string, GPUMeshCollection::SubMeshData> const& GPUMeshCollection::getSubMeshData() {
    return m_sub_mesh_data;
}

} // namespace mesh_gl
} // namespace megamol

#endif // !GPU_MESH_DATA_STORAGE_H_INCLUDED
