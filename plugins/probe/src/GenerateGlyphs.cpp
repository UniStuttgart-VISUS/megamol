/*
 * GenerateGlyphs.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "GenerateGlyphs.h"
#include "mesh/MeshCalls.h"
#include "ProbeCalls.h"
#include "DrawTextureUtility.h"

namespace megamol {
namespace probe {


GenerateGlyphs::GenerateGlyphs() : 
    Module()
    , _deploy_texture("deployTexture", "") 
    , _deploy_mesh("deployMesh", "") 
    , _get_probes("getProbes", "")
{

    this->_deploy_mesh.SetCallback(mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &GenerateGlyphs::getMesh);
    this->_deploy_mesh.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &GenerateGlyphs::getMetaData);
    this->MakeSlotAvailable(&this->_deploy_mesh);

    this->_deploy_texture.SetCallback(
        mesh::CallImage::ClassName(), mesh::CallImage::FunctionName(0), &GenerateGlyphs::getTexture);
    this->_deploy_texture.SetCallback(
        mesh::CallImage::ClassName(), mesh::CallImage::FunctionName(1), &GenerateGlyphs::getTextureMetaData);
    this->MakeSlotAvailable(&this->_deploy_texture);

    this->_get_probes.SetCompatibleCall<CallProbesDescription>();
    this->MakeSlotAvailable(&this->_get_probes);

}

GenerateGlyphs::~GenerateGlyphs() {
    this->Release();
}


bool GenerateGlyphs::doGlyphGeneration() {

    this->_mesh_data = std::make_shared<mesh::MeshDataAccessCollection>();
    this->_tex_data = std::make_shared<mesh::ImageDataAccessCollection>();

    _dtu.resize(this->_probe_data->getProbeCount());
    _generated_mesh.resize(this->_probe_data->getProbeCount() * 4);

    this->_generated_mesh_indices = {0, 1, 3, 0, 3, 2};
    this->_generated_texture_coordinates[0] = {0.0f, 0.0f};
    this->_generated_texture_coordinates[1] = {0.0f, 1.0f};
    this->_generated_texture_coordinates[2] = {1.0f, 0.0f};
    this->_generated_texture_coordinates[3] = {1.0f, 1.0f};

//#pragma omp parallel for
    for (int i = 0; i < this->_probe_data->getProbeCount(); i++) {

        // get probe
        auto probe = this->_probe_data->getProbe<FloatProbe>(i);
        // read samples of probe
        auto samples = probe.getSamplingResult();

        if (samples->samples.empty()) {
            vislib::sys::Log::DefaultLog.WriteError("[GenerateGlyphs] Probes have not been sampled.");
            return false;
        }

        // calc vertices
        auto dir = probe.m_direction;
        auto smallest_normal_index = std::distance(dir.begin(), std::min_element(dir.begin(), dir.end()));
        dir[smallest_normal_index] = 1.0f;
        //auto second_smallest_normal_index = std::distance(dir.begin(), std::min_element(dir.begin(), dir.end()));
        //auto largest_normal_index = std::distance(
        //    probe.m_direction.begin(), std::max_element(probe.m_direction.begin(), probe.m_direction.end()));

        std::array<float,3> axis0 = {0.0f, 0.0f, 0.0f};
        //if (smallest_normal_index == 1) smallest_normal_index = second_smallest_normal_index;
        axis0[smallest_normal_index] = 1.0f;
        std::array<float,3> plane_vec_1;
        plane_vec_1[0] = probe.m_direction[1] * axis0[2] - probe.m_direction[2] * axis0[1];
        plane_vec_1[1] = probe.m_direction[2] * axis0[0] - probe.m_direction[0] * axis0[2];
        plane_vec_1[2] = probe.m_direction[0] * axis0[1] - probe.m_direction[1] * axis0[0];

        std::array<float, 3> plane_vec_2;
        plane_vec_2[0] = probe.m_direction[1] * plane_vec_1[2] - probe.m_direction[2] * plane_vec_1[1];
        plane_vec_2[1] = probe.m_direction[2] * plane_vec_1[0] - probe.m_direction[0] * plane_vec_1[2];
        plane_vec_2[2] = probe.m_direction[0] * plane_vec_1[1] - probe.m_direction[1] * plane_vec_1[0];

        float plane_vec_1_length = std::sqrt(
        plane_vec_1[0] * plane_vec_1[0] + plane_vec_1[1] * plane_vec_1[1] + plane_vec_1[2] * plane_vec_1[2]);
        plane_vec_1[0] /= plane_vec_1_length;
        plane_vec_1[1] /= plane_vec_1_length;
        plane_vec_1[2] /= plane_vec_1_length;

        float plane_vec_2_length = std::sqrt(
        plane_vec_2[0] * plane_vec_2[0] + plane_vec_2[1] * plane_vec_2[1] + plane_vec_2[2] * plane_vec_2[2]);
        plane_vec_2[0] /= plane_vec_2_length;
        plane_vec_2[1] /= plane_vec_2_length;
        plane_vec_2[2] /= plane_vec_2_length;

        std::array<float,3> middle;
        middle[0] = probe.m_position[0] + probe.m_direction[0] * probe.m_begin;
        middle[1] = probe.m_position[1] + probe.m_direction[1] * probe.m_begin;
        middle[2] = probe.m_position[2] + probe.m_direction[2] * probe.m_begin;

        std::array<float, 3> vertex1;
        vertex1[0] = middle[0] + scale / 2 * plane_vec_1[0] + scale / 2 * plane_vec_2[0];
        vertex1[1] = middle[1] + scale / 2 * plane_vec_1[1] + scale / 2 * plane_vec_2[1];
        vertex1[2] = middle[2] + scale / 2 * plane_vec_1[2] + scale / 2 * plane_vec_2[2];

        std::array<float, 3> vertex2;
        vertex2[0] = middle[0] - scale / 2 * plane_vec_1[0] + scale / 2 * plane_vec_2[0];
        vertex2[1] = middle[1] - scale / 2 * plane_vec_1[1] + scale / 2 * plane_vec_2[1];
        vertex2[2] = middle[2] - scale / 2 * plane_vec_1[2] + scale / 2 * plane_vec_2[2];

        std::array<float, 3> vertex3;
        vertex3[0] = middle[0] + scale / 2 * plane_vec_1[0] - scale / 2 * plane_vec_2[0];
        vertex3[1] = middle[1] + scale / 2 * plane_vec_1[1] - scale / 2 * plane_vec_2[1];
        vertex3[2] = middle[2] + scale / 2 * plane_vec_1[2] - scale / 2 * plane_vec_2[2];

        std::array<float, 3> vertex4;
        vertex4[0] = middle[0] - scale / 2 * plane_vec_1[0] - scale / 2 * plane_vec_2[0];
        vertex4[1] = middle[1] - scale / 2 * plane_vec_1[1] - scale / 2 * plane_vec_2[1];
        vertex4[2] = middle[2] - scale / 2 * plane_vec_1[2] - scale / 2 * plane_vec_2[2];

        this->_generated_mesh[i * 4 + 0] = vertex1; 
        this->_generated_mesh[i * 4 + 1] = vertex2;    
        this->_generated_mesh[i * 4 + 2] = vertex3;
        this->_generated_mesh[i * 4 + 3] = vertex4;

        std::vector<mesh::MeshDataAccessCollection::VertexAttribute> vertex_attributes(2);
        mesh::MeshDataAccessCollection::VertexAttribute pos_attrib;
        pos_attrib.data = reinterpret_cast<uint8_t*>(&this->_generated_mesh[i * 4 + 0]);
        pos_attrib.stride = sizeof(std::array<float, 3>);
        pos_attrib.byte_size = pos_attrib.stride * 4;
        pos_attrib.component_cnt = 3;
        pos_attrib.component_type = mesh::MeshDataAccessCollection::FLOAT;
        pos_attrib.offset = 0;
        pos_attrib.semantic = mesh::MeshDataAccessCollection::POSITION;
        vertex_attributes[0] = pos_attrib;

        mesh::MeshDataAccessCollection::VertexAttribute texco_attrib;
        texco_attrib.data = reinterpret_cast<uint8_t*>(this->_generated_texture_coordinates.data());
        texco_attrib.stride = sizeof(float) * 2;
        texco_attrib.byte_size = texco_attrib.stride * 4;
        texco_attrib.component_cnt = 2;
        texco_attrib.component_type = mesh::MeshDataAccessCollection::FLOAT;
        texco_attrib.offset = 0;
        texco_attrib.semantic = mesh::MeshDataAccessCollection::TEXCOORD;
        vertex_attributes[1] = texco_attrib;

        mesh::MeshDataAccessCollection::IndexData index_data;
        index_data.data = reinterpret_cast<uint8_t*>(this->_generated_mesh_indices.data());
        index_data.byte_size = sizeof(this->_generated_mesh_indices);
        index_data.type = mesh::MeshDataAccessCollection::UNSIGNED_INT;

        this->_mesh_data->addMesh(vertex_attributes, index_data);

        _dtu[i].setResolution(300, 300); // should be changeable
        _dtu[i].setGraphType(DrawTextureUtility::GLYPH); // should be changeable

        auto tex_ptr = _dtu[i].draw(samples->samples, samples->min_value, samples->max_value);
        this->_tex_data->addImage(mesh::ImageDataAccessCollection::RGBA8, _dtu[i].getPixelWidth(),
            _dtu[i].getPixelHeight(), tex_ptr, 4 * _dtu[i].getPixelWidth() *
            _dtu[i].getPixelHeight());

    } // end for probe count

    return true;
}


bool GenerateGlyphs::getMesh(core::Call& call) {

    auto cprobes = this->_get_probes.CallAs<CallProbes>();
    auto cm = dynamic_cast<mesh::CallMesh*>(&call);

    if (cprobes == nullptr) return false;
    if (cm == nullptr) return false;

    if (!(*cprobes)(0)) return false;

    auto probe_meta_data = cprobes->getMetaData();
    auto mesh_meta_data = cm->getMetaData();

    mesh_meta_data.m_bboxs = probe_meta_data.m_bboxs;

    cm->setMetaData(mesh_meta_data);

    if (cprobes->hasUpdate()){
        ++_version;
        this->_probe_data = cprobes->getData();
        if (this->scale <= 0.0) this->scale = probe_meta_data.m_bboxs.BoundingBox().LongestEdge() * 8e-3;
        doGlyphGeneration();
    }

    cm->setData(this->_mesh_data,_version);

    return true;
}

bool GenerateGlyphs::getMetaData(core::Call& call) {

    auto cprobes = this->_get_probes.CallAs<CallProbes>();
    auto cm = dynamic_cast<mesh::CallMesh*>(&call);

    if (cprobes == nullptr) return false;
    if (cm == nullptr) return false;

    auto probe_meta_data = cprobes->getMetaData();
    auto mesh_meta_data = cm->getMetaData();

    probe_meta_data.m_frame_ID = mesh_meta_data.m_frame_ID;
    cprobes->setMetaData(probe_meta_data);

    if (!(*cprobes)(1)) return false;

    probe_meta_data = cprobes->getMetaData();

    mesh_meta_data.m_frame_cnt = probe_meta_data.m_frame_cnt;
    mesh_meta_data.m_bboxs = probe_meta_data.m_bboxs;
    cm->setMetaData(mesh_meta_data);

    return true;
}

bool GenerateGlyphs::getTexture(core::Call& call) {

    auto cprobes = this->_get_probes.CallAs<CallProbes>();
    auto ctex = dynamic_cast<mesh::CallImage*>(&call);

    if (cprobes == nullptr) return false;
    if (ctex == nullptr) return false;

    if (!(*cprobes)(0)) return false;

    auto probe_meta_data = cprobes->getMetaData();


    if (cprobes->hasUpdate()) {
        ++_version;
        this->_probe_data = cprobes->getData();
        if (this->scale < 0) this->scale = probe_meta_data.m_bboxs.BoundingBox().LongestEdge() * 2e-2;
        if(!doGlyphGeneration()) return false;
    }

    ctex->setData(this->_tex_data,_version);

    return true;
}

bool GenerateGlyphs::getTextureMetaData(core::Call& call) {
    auto cprobes = this->_get_probes.CallAs<CallProbes>();
    auto ctex = dynamic_cast<mesh::CallImage*>(&call);

    if (cprobes == nullptr) return false;
    if (ctex == nullptr) return false;

    if (!(*cprobes)(1)) return false;

    auto probe_meta_data = cprobes->getMetaData();
    auto tex_meta_data = ctex->getMetaData();

    ctex->setMetaData(tex_meta_data);

    return true;
}
} // namespace probe
} // namespace megamol