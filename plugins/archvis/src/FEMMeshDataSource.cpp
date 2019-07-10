#include "FEMMeshDataSource.h"

#include "FEMDataCall.h"
#include "mesh/CallGPUMeshData.h"

megamol::archvis::FEMMeshDataSource::FEMMeshDataSource()
    : m_fem_callerSlot("getFEMFile", "Connects the data source with loaded FEM data"), m_FEM_model_hash(0) {
    this->m_fem_callerSlot.SetCompatibleCall<FEMDataCallDescription>();
    this->MakeSlotAvailable(&this->m_fem_callerSlot);
}

megamol::archvis::FEMMeshDataSource::~FEMMeshDataSource() {}

bool megamol::archvis::FEMMeshDataSource::create() {
    m_gpu_meshes = std::make_shared<mesh::GPUMeshCollection>();

    m_bbox = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

    return true;
}

bool megamol::archvis::FEMMeshDataSource::getDataCallback(core::Call& caller) {
    mesh::CallGPUMeshData* mc = dynamic_cast<mesh::CallGPUMeshData*>(&caller);
    if (mc == NULL) return false;

    FEMDataCall* fem_call = this->m_fem_callerSlot.CallAs<FEMDataCall>();
    if (fem_call == NULL) return false;

    if (!(*fem_call)(0)) return false;

    if (this->m_FEM_model_hash == fem_call->DataHash()) {
        return true;
    }

    m_gpu_meshes->clear();

    m_bbox[0] = std::numeric_limits<float>::max();
    m_bbox[1] = std::numeric_limits<float>::max();
    m_bbox[2] = std::numeric_limits<float>::max();
    m_bbox[3] = std::numeric_limits<float>::min();
    m_bbox[4] = std::numeric_limits<float>::min();
    m_bbox[5] = std::numeric_limits<float>::min();

    auto fem_data = fem_call->getFEMData();

    // TODO generate vertex and index data

    // Create std-container for holding vertex data
    std::vector<std::vector<float>> vbs(1);
    vbs[0].reserve(fem_data->getNodes().size() * 3);
    for (auto& node : fem_data->getNodes()) {
        vbs[0].push_back(node.X()); // position data buffer
        vbs[0].push_back(node.Y());
        vbs[0].push_back(node.Z());

        m_bbox[0] = std::min(m_bbox[0], static_cast<float>(node.X()));
        m_bbox[1] = std::min(m_bbox[1], static_cast<float>(node.Y()));
        m_bbox[2] = std::min(m_bbox[2], static_cast<float>(node.Z()));
        m_bbox[3] = std::max(m_bbox[3], static_cast<float>(node.X()));
        m_bbox[4] = std::max(m_bbox[4], static_cast<float>(node.Y()));
        m_bbox[5] = std::max(m_bbox[5], static_cast<float>(node.Z()));
    }
    // Create std-container holding vertex attribute descriptions
    std::vector<mesh::VertexLayout::Attribute> attribs = {
        mesh::VertexLayout::Attribute(3, GL_FLOAT, GL_FALSE, 0)};
    mesh::VertexLayout vertex_descriptor(0, attribs);

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

    std::vector<std::pair<std::vector<float>::iterator, std::vector<float>::iterator>> vb_iterators = {
        {vbs[0].begin(), vbs[0].end()}};
    std::pair<std::vector<uint32_t>::iterator, std::vector<uint32_t>::iterator> ib_iterators = {
        indices.begin(), indices.end()};

    m_gpu_meshes->addMesh(
        vertex_descriptor, vb_iterators, ib_iterators, GL_UNSIGNED_INT, GL_STATIC_DRAW, GL_TRIANGLES);

    mc->setGPUMeshes(m_gpu_meshes);
    
    this->m_FEM_model_hash = fem_call->DataHash();

    return true;
}
