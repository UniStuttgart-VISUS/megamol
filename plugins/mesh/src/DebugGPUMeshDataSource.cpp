#include "DebugGPUMeshDataSource.h"
#include "mesh/CallGPUMeshData.h"

megamol::mesh::DebugGPUMeshDataSource::DebugGPUMeshDataSource() {}

megamol::mesh::DebugGPUMeshDataSource::~DebugGPUMeshDataSource() {}

bool megamol::mesh::DebugGPUMeshDataSource::create() {
    m_gpu_meshes = std::make_shared<GPUMeshCollection>();

    m_bbox = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

    return load();
}

bool megamol::mesh::DebugGPUMeshDataSource::getDataCallback(core::Call& caller) {
    CallGPUMeshData* mc = dynamic_cast<CallGPUMeshData*>(&caller);
    if (mc == NULL) return false;

    mc->setGPUMeshes(m_gpu_meshes);

    return true;
}

bool megamol::mesh::DebugGPUMeshDataSource::load() {
    // Create std-container for holding vertex data
    std::vector<std::vector<float>> vbs = {{0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f}, // normal data buffer
        {-0.5f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f}}; // position data buffer
    // Create std-container holding vertex attribute descriptions
    std::vector<VertexLayout::Attribute> attribs = {
        VertexLayout::Attribute(3, GL_FLOAT, GL_FALSE, 0), VertexLayout::Attribute(3, GL_FLOAT, GL_FALSE, 0)};

    VertexLayout vertex_descriptor(0, attribs);

    // Create std-container holding index data
    std::vector<uint32_t> indices = {0, 1, 2};

    std::vector<std::pair<std::vector<float>::iterator, std::vector<float>::iterator>> vb_iterators = {
        {vbs[0].begin(), vbs[0].end()}, {vbs[1].begin(), vbs[1].end()}};
    std::pair<std::vector<uint32_t>::iterator, std::vector<uint32_t>::iterator> ib_iterators = {
        indices.begin(), indices.end()};

    m_gpu_meshes->addMesh(vertex_descriptor, vb_iterators, ib_iterators, GL_UNSIGNED_INT, GL_STATIC_DRAW, GL_TRIANGLES);

    return true;
}
