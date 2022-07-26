/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "FEMMeshDataSource.h"

#include <glowl/VertexLayout.hpp>

#include "ArchVisCalls.h"
#include "mesh_gl/MeshCalls_gl.h"

megamol::archvis_gl::FEMMeshDataSource::FEMMeshDataSource()
        : m_fem_callerSlot("getFEMFile", "Connects the data source with loaded FEM data")
        , m_version(0) {
    this->m_fem_callerSlot.SetCompatibleCall<FEMModelCallDescription>();
    this->MakeSlotAvailable(&this->m_fem_callerSlot);
}

megamol::archvis_gl::FEMMeshDataSource::~FEMMeshDataSource() {}

bool megamol::archvis_gl::FEMMeshDataSource::getDataCallback(core::Call& caller) {
    mesh_gl::CallGPUMeshData* lhs_mesh_call = dynamic_cast<mesh_gl::CallGPUMeshData*>(&caller);
    mesh_gl::CallGPUMeshData* rhs_mesh_call = this->m_mesh_rhs_slot.CallAs<mesh_gl::CallGPUMeshData>();

    if (lhs_mesh_call == nullptr) {
        return false;
    }

    auto gpu_mesh_collection = std::make_shared<std::vector<std::shared_ptr<mesh_gl::GPUMeshCollection>>>();
    // if there is a mesh connection to the right, pass on the mesh collection
    if (rhs_mesh_call != nullptr) {
        if (!(*rhs_mesh_call)(0)) {
            return false;
        }
        if (rhs_mesh_call->hasUpdate()) {
            ++m_version;
        }
        gpu_mesh_collection = rhs_mesh_call->getData();
    }
    gpu_mesh_collection->push_back(m_mesh_collection.first);


    CallFEMModel* fem_call = this->m_fem_callerSlot.CallAs<CallFEMModel>();
    if (fem_call == nullptr) {
        return false;
    }

    if (!(*fem_call)(0)) {
        return false;
    }

    if (fem_call->hasUpdate()) {
        ++m_version;

        clearMeshCollection();

        auto fem_data = fem_call->getData();

        // TODO generate vertex and index data

        // Create std-container for holding vertex data
        std::vector<std::vector<float>> vbs(1);
        vbs[0].reserve(fem_data->getNodes().size() * 3);
        for (auto& node : fem_data->getNodes()) {
            vbs[0].push_back(node.X()); // position data buffer
            vbs[0].push_back(node.Y());
            vbs[0].push_back(node.Z());
        }
        // Create std-container holding vertex attribute descriptions
        std::vector<glowl::VertexLayout::Attribute> attribs = {
            glowl::VertexLayout::Attribute(3, GL_FLOAT, GL_FALSE, 0)};
        glowl::VertexLayout vertex_descriptor(0, attribs);

        // Create std-container holding index data
        std::vector<uint32_t> indices;
        std::vector<size_t> node_indices;

        for (auto& element : fem_data->getElements()) {
            switch (element.getType()) {
            case FEMModel::CUBE:

                // TODO indices for a cube....
                node_indices = element.getNodeIndices();

                indices.insert(indices.end(),
                    {// front
                        static_cast<uint32_t>(node_indices[0] - 1), static_cast<uint32_t>(node_indices[1] - 1),
                        static_cast<uint32_t>(node_indices[2] - 1), static_cast<uint32_t>(node_indices[2] - 1),
                        static_cast<uint32_t>(node_indices[3] - 1), static_cast<uint32_t>(node_indices[0] - 1),
                        // right
                        static_cast<uint32_t>(node_indices[1] - 1), static_cast<uint32_t>(node_indices[5] - 1),
                        static_cast<uint32_t>(node_indices[6] - 1), static_cast<uint32_t>(node_indices[6] - 1),
                        static_cast<uint32_t>(node_indices[2] - 1), static_cast<uint32_t>(node_indices[1] - 1),
                        // back
                        static_cast<uint32_t>(node_indices[7] - 1), static_cast<uint32_t>(node_indices[6] - 1),
                        static_cast<uint32_t>(node_indices[5] - 1), static_cast<uint32_t>(node_indices[5] - 1),
                        static_cast<uint32_t>(node_indices[4] - 1), static_cast<uint32_t>(node_indices[7] - 1),
                        // left
                        static_cast<uint32_t>(node_indices[4] - 1), static_cast<uint32_t>(node_indices[0] - 1),
                        static_cast<uint32_t>(node_indices[3] - 1), static_cast<uint32_t>(node_indices[3] - 1),
                        static_cast<uint32_t>(node_indices[7] - 1), static_cast<uint32_t>(node_indices[4] - 1),
                        // bottom
                        static_cast<uint32_t>(node_indices[4] - 1), static_cast<uint32_t>(node_indices[5] - 1),
                        static_cast<uint32_t>(node_indices[1] - 1), static_cast<uint32_t>(node_indices[1] - 1),
                        static_cast<uint32_t>(node_indices[0] - 1), static_cast<uint32_t>(node_indices[4] - 1),
                        // top
                        static_cast<uint32_t>(node_indices[3] - 1), static_cast<uint32_t>(node_indices[2] - 1),
                        static_cast<uint32_t>(node_indices[6] - 1), static_cast<uint32_t>(node_indices[6] - 1),
                        static_cast<uint32_t>(node_indices[7] - 1), static_cast<uint32_t>(node_indices[3] - 1)});

                break;
            default:
                break;
            }
        }

        std::vector<glowl::VertexLayout> vb_layouts = {vertex_descriptor};
        std::vector<std::pair<std::vector<float>::iterator, std::vector<float>::iterator>> vb_iterators = {
            {vbs[0].begin(), vbs[0].end()}};
        std::pair<std::vector<uint32_t>::iterator, std::vector<uint32_t>::iterator> ib_iterators = {
            indices.begin(), indices.end()};

        std::string identifier = std::string(this->FullName());

        try {
            m_mesh_collection.first->addMesh(identifier, vb_layouts, vb_iterators, ib_iterators, GL_UNSIGNED_INT,
                GL_STATIC_DRAW, GL_TRIANGLES, true);
            m_mesh_collection.second.push_back(identifier);
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

    lhs_mesh_call->setData(gpu_mesh_collection, m_version);

    return true;
}

bool megamol::archvis_gl::FEMMeshDataSource::getMetaDataCallback(core::Call& caller) {
    return false;
}
