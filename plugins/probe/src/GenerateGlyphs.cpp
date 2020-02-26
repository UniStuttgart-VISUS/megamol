/*
 * GenerateGlyphs.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include <variant>

#include "GenerateGlyphs.h"
#include "DrawTextureUtility.h"
#include "ProbeCalls.h"
#include "mesh/MeshCalls.h"

namespace megamol {
namespace probe {

template <typename T> static bool approxEq(T a, T b) { return std::abs(a - b) < std::numeric_limits<T>::epsilon(); }

GenerateGlyphs::GenerateGlyphs()
    : Module(), _deploy_texture("deployTexture", ""), _deploy_mesh("deployMesh", ""), _get_probes("getProbes", "") {

    this->_deploy_mesh.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &GenerateGlyphs::getMesh);
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

GenerateGlyphs::~GenerateGlyphs() { this->Release(); }


bool GenerateGlyphs::doScalarGlyphGeneration(FloatProbe& probe) {

    // get probe
    // auto probe = this->_probe_data->getProbe<FloatProbe>(i);
    // read samples of probe
    auto samples = probe.getSamplingResult();

    if (samples->samples.empty()) {
        vislib::sys::Log::DefaultLog.WriteError("[GenerateGlyphs] Probes have not been sampled.");
        return false;
    }

    bool skip = false;
    if (approxEq(samples->min_value, samples->max_value)) {
        // if ( i <= 0.5* this->_probe_data->getProbeCount()) {
        skip = true;
    }

    // calc vertices
    auto dir = probe.m_direction;
    auto smallest_normal_index = std::distance(dir.begin(), std::min_element(dir.begin(), dir.end()));
    dir[smallest_normal_index] = 1.0f;
    // auto second_smallest_normal_index = std::distance(dir.begin(), std::min_element(dir.begin(), dir.end()));
    // auto largest_normal_index = std::distance(
    //    probe.m_direction.begin(), std::max_element(probe.m_direction.begin(), probe.m_direction.end()));

    std::array<float, 3> axis0 = {0.0f, 0.0f, 0.0f};
    // if (smallest_normal_index == 1) smallest_normal_index = second_smallest_normal_index;
    axis0[smallest_normal_index] = 1.0f;
    std::array<float, 3> plane_vec_1;
    plane_vec_1[0] = probe.m_direction[1] * axis0[2] - probe.m_direction[2] * axis0[1];
    plane_vec_1[1] = probe.m_direction[2] * axis0[0] - probe.m_direction[0] * axis0[2];
    plane_vec_1[2] = probe.m_direction[0] * axis0[1] - probe.m_direction[1] * axis0[0];

    std::array<float, 3> plane_vec_2;
    plane_vec_2[0] = probe.m_direction[1] * plane_vec_1[2] - probe.m_direction[2] * plane_vec_1[1];
    plane_vec_2[1] = probe.m_direction[2] * plane_vec_1[0] - probe.m_direction[0] * plane_vec_1[2];
    plane_vec_2[2] = probe.m_direction[0] * plane_vec_1[1] - probe.m_direction[1] * plane_vec_1[0];

    float plane_vec_1_length =
        std::sqrt(plane_vec_1[0] * plane_vec_1[0] + plane_vec_1[1] * plane_vec_1[1] + plane_vec_1[2] * plane_vec_1[2]);
    plane_vec_1[0] /= plane_vec_1_length;
    plane_vec_1[1] /= plane_vec_1_length;
    plane_vec_1[2] /= plane_vec_1_length;

    float plane_vec_2_length =
        std::sqrt(plane_vec_2[0] * plane_vec_2[0] + plane_vec_2[1] * plane_vec_2[1] + plane_vec_2[2] * plane_vec_2[2]);
    plane_vec_2[0] /= plane_vec_2_length;
    plane_vec_2[1] /= plane_vec_2_length;
    plane_vec_2[2] /= plane_vec_2_length;

    std::array<float, 3> middle;
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

    size_t offset = _generated_mesh_vertices.size();
    this->_generated_mesh_vertices.push_back(vertex1);
    this->_generated_mesh_vertices.push_back(vertex2);
    this->_generated_mesh_vertices.push_back(vertex3);
    this->_generated_mesh_vertices.push_back(vertex4);

    std::vector<mesh::MeshDataAccessCollection::VertexAttribute> vertex_attributes(2);
    mesh::MeshDataAccessCollection::VertexAttribute pos_attrib;
    pos_attrib.data = reinterpret_cast<uint8_t*>(&this->_generated_mesh_vertices[offset]);
    pos_attrib.stride = sizeof(std::array<float, 3>);
    pos_attrib.byte_size = pos_attrib.stride * 4;
    pos_attrib.component_cnt = 3;
    pos_attrib.component_type = mesh::MeshDataAccessCollection::FLOAT;
    pos_attrib.offset = 0;
    pos_attrib.semantic = mesh::MeshDataAccessCollection::POSITION;
    vertex_attributes[0] = pos_attrib;

    mesh::MeshDataAccessCollection::VertexAttribute texco_attrib;
    texco_attrib.data = reinterpret_cast<uint8_t*>(this->_generated_billboard_texture_coordinates.data());
    texco_attrib.stride = sizeof(float) * 2;
    texco_attrib.byte_size = texco_attrib.stride * 4;
    texco_attrib.component_cnt = 2;
    texco_attrib.component_type = mesh::MeshDataAccessCollection::FLOAT;
    texco_attrib.offset = 0;
    texco_attrib.semantic = mesh::MeshDataAccessCollection::TEXCOORD;
    vertex_attributes[1] = texco_attrib;

    mesh::MeshDataAccessCollection::IndexData index_data;
    index_data.data = reinterpret_cast<uint8_t*>(this->_generated_billboard_mesh_indices.data());
    index_data.byte_size = sizeof(this->_generated_billboard_mesh_indices);
    index_data.type = mesh::MeshDataAccessCollection::UNSIGNED_INT;

    if (!skip) {
        this->_mesh_data->addMesh(vertex_attributes, index_data);

        _dtu.push_back(DrawTextureUtility());
        _dtu.back().setResolution(300, 300);                 // should be changeable
        _dtu.back().setGraphType(DrawTextureUtility::GLYPH); // should be changeable

        auto tex_ptr = _dtu.back().draw(samples->samples, samples->min_value, samples->max_value);
        this->_tex_data->addImage(mesh::ImageDataAccessCollection::RGBA8, _dtu.back().getPixelWidth(),
            _dtu.back().getPixelHeight(), tex_ptr, 4 * _dtu.back().getPixelWidth() * _dtu.back().getPixelHeight());
    }
    return true;
}

bool GenerateGlyphs::doVectorRibbonGlyphGeneration(Vec4Probe& probe) {

    // get probe
    // auto probe = this->_probe_data->getProbe<FloatProbe>(i);
    // read samples of probe
    auto samples = probe.getSamplingResult();

    if (samples->samples.empty()) {
        vislib::sys::Log::DefaultLog.WriteError("[GenerateGlyphs] Probes have not been sampled.");
        return false;
    }

    // calculate vertices & indices

    // create first pair of vertices at the base of the ribbon
    float ribbon_width = 0.0001f;
    std::array<float, 3> ribbon_base;
    ribbon_base[0] = probe.m_position[0] + probe.m_begin * probe.m_direction[0];
    ribbon_base[1] = probe.m_position[1] + probe.m_begin * probe.m_direction[1];
    ribbon_base[2] = probe.m_position[2] + probe.m_begin * probe.m_direction[2];

    std::array<float, 3> vertex1;
    vertex1[0] = ribbon_base[0] + ribbon_width * samples->samples.front()[0];
    vertex1[1] = ribbon_base[1] + ribbon_width * samples->samples.front()[1];
    vertex1[2] = ribbon_base[2] + ribbon_width * samples->samples.front()[2];

    std::array<float, 3> vertex2;
    vertex2[0] = ribbon_base[0] - ribbon_width * samples->samples.front()[0];
    vertex2[1] = ribbon_base[1] - ribbon_width * samples->samples.front()[1];
    vertex2[2] = ribbon_base[2] - ribbon_width * samples->samples.front()[2];

    // update ribbon base
    std::array<float, 3> sample_vector = {
        samples->samples.front()[0], samples->samples.front()[1], samples->samples.front()[2]};

    float sample_vector_length = std::sqrt(sample_vector[0] * sample_vector[0] + sample_vector[1] * sample_vector[1] +
                                     sample_vector[2] * sample_vector[2]);
    sample_vector[0] /= sample_vector_length;
    sample_vector[1] /= sample_vector_length;
    sample_vector[2] /= sample_vector_length;

    std::array<float, 3> offset_direction; // TODO
    offset_direction[0] = probe.m_direction[1] * sample_vector[2] - probe.m_direction[2] * sample_vector[1];
    offset_direction[1] = probe.m_direction[2] * sample_vector[0] - probe.m_direction[0] * sample_vector[2];
    offset_direction[2] = probe.m_direction[0] * sample_vector[1] - probe.m_direction[1] * sample_vector[0];

    ribbon_base[0] = ribbon_base[0] + ribbon_width * 2.0f * offset_direction[0];
    ribbon_base[1] = ribbon_base[1] + ribbon_width * 2.0f * offset_direction[1];
    ribbon_base[2] = ribbon_base[2] + ribbon_width * 2.0f * offset_direction[2];

    // compute first two normal vectors
    std::array<float, 3> normal1;
    normal1[0] = offset_direction[1] * sample_vector[2] - offset_direction[2] * sample_vector[1];
    normal1[1] = offset_direction[2] * sample_vector[0] - offset_direction[0] * sample_vector[2];
    normal1[2] = offset_direction[0] * sample_vector[1] - offset_direction[1] * sample_vector[0];

    std::array<float, 3> normal2;
    normal2[0] = offset_direction[1] * sample_vector[2] - offset_direction[2] * sample_vector[1];
    normal2[1] = offset_direction[2] * sample_vector[0] - offset_direction[0] * sample_vector[2];
    normal2[2] = offset_direction[0] * sample_vector[1] - offset_direction[1] * sample_vector[0];


    size_t base_vertex = _generated_mesh_vertices.size();
    size_t base_index = _generated_mesh_indices.size();

    for (int i = 1; i < samples->samples.size(); ++i) {

        std::array<float, 3> sample_vector = {samples->samples[i][0], samples->samples[i][1], samples->samples[i][2]};
        float sample_vector_length = std::sqrt(
            sample_vector[0] * sample_vector[0] + 
            sample_vector[1] * sample_vector[1] +
            sample_vector[2] * sample_vector[2]);
        sample_vector[0] /= sample_vector_length;
        sample_vector[1] /= sample_vector_length;
        sample_vector[2] /= sample_vector_length;
    
        // compute vertices 3 and 4 of "quad" section of the ribbon
        std::array<float, 3> vertex3;
        vertex3[0] = ribbon_base[0] + ribbon_width * sample_vector[0];
        vertex3[1] = ribbon_base[1] + ribbon_width * sample_vector[1];
        vertex3[2] = ribbon_base[2] + ribbon_width * sample_vector[2];

        std::array<float, 3> vertex4;
        vertex4[0] = ribbon_base[0] - ribbon_width * sample_vector[0];
        vertex4[1] = ribbon_base[1] - ribbon_width * sample_vector[1];
        vertex4[2] = ribbon_base[2] - ribbon_width * sample_vector[2];

        size_t index_offset = _generated_mesh_vertices.size();
        this->_generated_mesh_vertices.push_back(vertex1);
        this->_generated_mesh_vertices.push_back(vertex2);
        this->_generated_mesh_vertices.push_back(vertex3);
        this->_generated_mesh_vertices.push_back(vertex4);

        // compute indices
        this->_generated_mesh_indices.push_back(index_offset + 0);
        this->_generated_mesh_indices.push_back(index_offset + 1);
        this->_generated_mesh_indices.push_back(index_offset + 2);
        this->_generated_mesh_indices.push_back(index_offset + 2);
        this->_generated_mesh_indices.push_back(index_offset + 1);
        this->_generated_mesh_indices.push_back(index_offset + 3);

        // TODO update offset direction and base

        std::array<float, 3> normal3;
        normal3[0] = offset_direction[1] * sample_vector[2] - offset_direction[2] * sample_vector[1];
        normal3[1] = offset_direction[2] * sample_vector[0] - offset_direction[0] * sample_vector[2];
        normal3[2] = offset_direction[0] * sample_vector[1] - offset_direction[1] * sample_vector[0];

        std::array<float, 3> normal4;
        normal4[0] = offset_direction[1] * sample_vector[2] - offset_direction[2] * sample_vector[1];
        normal4[1] = offset_direction[2] * sample_vector[0] - offset_direction[0] * sample_vector[2];
        normal4[2] = offset_direction[0] * sample_vector[1] - offset_direction[1] * sample_vector[0];

        this->_generated_mesh_normals.push_back(normal1);
        this->_generated_mesh_normals.push_back(normal2);
        this->_generated_mesh_normals.push_back(normal3);
        this->_generated_mesh_normals.push_back(normal4);

        // assign v3/4 to v1/2 for next loop iteration
        vertex1 = vertex3;
        vertex2 = vertex4;
    }

    std::vector<mesh::MeshDataAccessCollection::VertexAttribute> vertex_attributes(2);
    mesh::MeshDataAccessCollection::VertexAttribute pos_attrib;
    pos_attrib.data = reinterpret_cast<uint8_t*>(&this->_generated_mesh_vertices[base_vertex]);
    pos_attrib.stride = sizeof(std::array<float, 3>);
    pos_attrib.byte_size = pos_attrib.stride * samples->samples.size();
    pos_attrib.component_cnt = 3;
    pos_attrib.component_type = mesh::MeshDataAccessCollection::FLOAT;
    pos_attrib.offset = 0;
    pos_attrib.semantic = mesh::MeshDataAccessCollection::POSITION;
    vertex_attributes[0] = pos_attrib;

    mesh::MeshDataAccessCollection::VertexAttribute normal_attrib;
    normal_attrib.data = reinterpret_cast<uint8_t*>(&this->_generated_mesh_normals[base_vertex]);
    normal_attrib.stride = sizeof(std::array<float, 3>);
    normal_attrib.byte_size = normal_attrib.stride * samples->samples.size();
    normal_attrib.component_cnt = 3;
    normal_attrib.component_type = mesh::MeshDataAccessCollection::FLOAT;
    normal_attrib.offset = 0;
    normal_attrib.semantic = mesh::MeshDataAccessCollection::NORMAL;
    vertex_attributes[1] = normal_attrib;

    //mesh::MeshDataAccessCollection::VertexAttribute texco_attrib;
    //texco_attrib.data = reinterpret_cast<uint8_t*>(this->_generated_billboard_texture_coordinates.data());
    //texco_attrib.stride = sizeof(float) * 2;
    //texco_attrib.byte_size = texco_attrib.stride * 4;
    //texco_attrib.component_cnt = 2;
    //texco_attrib.component_type = mesh::MeshDataAccessCollection::FLOAT;
    //texco_attrib.offset = 0;
    //texco_attrib.semantic = mesh::MeshDataAccessCollection::TEXCOORD;
    //vertex_attributes[1] = texco_attrib;

    mesh::MeshDataAccessCollection::IndexData index_data;
    index_data.data = reinterpret_cast<uint8_t*>(&this->_generated_mesh_indices[base_index]);
    index_data.byte_size = sizeof(uint32_t) * 6 * samples->samples.size() -1;
    index_data.type = mesh::MeshDataAccessCollection::UNSIGNED_INT;

    this->_mesh_data->addMesh(vertex_attributes, index_data);

    return false; 
}

bool GenerateGlyphs::doVectorRadarGlyphGeneration(Vec4Probe& probe) {

    // get probe
    // auto probe = this->_probe_data->getProbe<FloatProbe>(i);
    // read samples of probe
    auto samples = probe.getSamplingResult();

    if (samples->samples.empty()) {
        vislib::sys::Log::DefaultLog.WriteError("[GenerateGlyphs] Probes have not been sampled.");
        return false;
    }

    {
        // calc vertices
        auto dir = probe.m_direction;
        auto smallest_normal_index = std::distance(dir.begin(), std::min_element(dir.begin(), dir.end()));
        dir[smallest_normal_index] = 1.0f;
        // auto second_smallest_normal_index = std::distance(dir.begin(), std::min_element(dir.begin(), dir.end()));
        // auto largest_normal_index = std::distance(
        //    probe.m_direction.begin(), std::max_element(probe.m_direction.begin(), probe.m_direction.end()));

        std::array<float, 3> axis0 = {0.0f, 0.0f, 0.0f};
        // if (smallest_normal_index == 1) smallest_normal_index = second_smallest_normal_index;
        axis0[smallest_normal_index] = 1.0f;
        std::array<float, 3> plane_vec_1;
        plane_vec_1[0] = probe.m_direction[1] * axis0[2] - probe.m_direction[2] * axis0[1];
        plane_vec_1[1] = probe.m_direction[2] * axis0[0] - probe.m_direction[0] * axis0[2];
        plane_vec_1[2] = probe.m_direction[0] * axis0[1] - probe.m_direction[1] * axis0[0];

        std::array<float, 3> plane_vec_2;
        plane_vec_2[0] = probe.m_direction[1] * plane_vec_1[2] - probe.m_direction[2] * plane_vec_1[1];
        plane_vec_2[1] = probe.m_direction[2] * plane_vec_1[0] - probe.m_direction[0] * plane_vec_1[2];
        plane_vec_2[2] = probe.m_direction[0] * plane_vec_1[1] - probe.m_direction[1] * plane_vec_1[0];

        float plane_vec_1_length =
            std::sqrt(plane_vec_1[0] * plane_vec_1[0] + plane_vec_1[1] * plane_vec_1[1] + plane_vec_1[2] * plane_vec_1[2]);
        plane_vec_1[0] /= plane_vec_1_length;
        plane_vec_1[1] /= plane_vec_1_length;
        plane_vec_1[2] /= plane_vec_1_length;

        float plane_vec_2_length =
            std::sqrt(plane_vec_2[0] * plane_vec_2[0] + plane_vec_2[1] * plane_vec_2[1] + plane_vec_2[2] * plane_vec_2[2]);
        plane_vec_2[0] /= plane_vec_2_length;
        plane_vec_2[1] /= plane_vec_2_length;
        plane_vec_2[2] /= plane_vec_2_length;

        std::array<float, 3> middle;
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

        size_t offset = _generated_mesh_vertices.size();
        this->_generated_mesh_vertices.push_back(vertex1);
        this->_generated_mesh_vertices.push_back(vertex2);
        this->_generated_mesh_vertices.push_back(vertex3);
        this->_generated_mesh_vertices.push_back(vertex4);

        std::vector<mesh::MeshDataAccessCollection::VertexAttribute> vertex_attributes(2);
        mesh::MeshDataAccessCollection::VertexAttribute pos_attrib;
        pos_attrib.data = reinterpret_cast<uint8_t*>(&this->_generated_mesh_vertices[offset]);
        pos_attrib.stride = sizeof(std::array<float, 3>);
        pos_attrib.byte_size = pos_attrib.stride * 4;
        pos_attrib.component_cnt = 3;
        pos_attrib.component_type = mesh::MeshDataAccessCollection::FLOAT;
        pos_attrib.offset = 0;
        pos_attrib.semantic = mesh::MeshDataAccessCollection::POSITION;
        vertex_attributes[0] = pos_attrib;

        mesh::MeshDataAccessCollection::VertexAttribute texco_attrib;
        texco_attrib.data = reinterpret_cast<uint8_t*>(this->_generated_billboard_texture_coordinates.data());
        texco_attrib.stride = sizeof(float) * 2;
        texco_attrib.byte_size = texco_attrib.stride * 4;
        texco_attrib.component_cnt = 2;
        texco_attrib.component_type = mesh::MeshDataAccessCollection::FLOAT;
        texco_attrib.offset = 0;
        texco_attrib.semantic = mesh::MeshDataAccessCollection::TEXCOORD;
        vertex_attributes[1] = texco_attrib;

        mesh::MeshDataAccessCollection::IndexData index_data;
        index_data.data = reinterpret_cast<uint8_t*>(this->_generated_billboard_mesh_indices.data());
        index_data.byte_size = sizeof(this->_generated_billboard_mesh_indices);
        index_data.type = mesh::MeshDataAccessCollection::UNSIGNED_INT;

        this->_mesh_data->addMesh(vertex_attributes, index_data);
    }

    _dtu.push_back(DrawTextureUtility());
    _dtu.back().setResolution(400, 400);                 // should be changeable
    _dtu.back().setGraphType(DrawTextureUtility::RADARGLYPH); // should be changeable

    auto tex_ptr = _dtu.back().draw(samples->samples, probe.m_direction);
    this->_tex_data->addImage(mesh::ImageDataAccessCollection::RGBA8, _dtu.back().getPixelWidth(),
        _dtu.back().getPixelHeight(), tex_ptr, 4 * _dtu.back().getPixelWidth() * _dtu.back().getPixelHeight());

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

    if (cprobes->hasUpdate()) {
        ++_version;
        
        if (this->scale <= 0.0) this->scale = probe_meta_data.m_bboxs.BoundingBox().LongestEdge() * 8e-3;

        this->_probe_data = cprobes->getData();

        //TODO visitor pattern

        this->_mesh_data = std::make_shared<mesh::MeshDataAccessCollection>();
        this->_tex_data = std::make_shared<mesh::ImageDataAccessCollection>();

        _dtu.reserve(this->_probe_data->getProbeCount());
        // reserve memory because reallocation will invalidate pointers to memory later on ! 
        _generated_mesh_vertices.reserve(this->_probe_data->getProbeCount() * 10 * 2);
        _generated_mesh_normals.reserve(this->_probe_data->getProbeCount() * 10 * 2);
        _generated_mesh_indices.reserve(this->_probe_data->getProbeCount() * 10 * 6);

        this->_generated_billboard_mesh_indices = {0, 1, 3, 0, 3, 2};
        this->_generated_billboard_texture_coordinates[0] = {0.0f, 0.0f};
        this->_generated_billboard_texture_coordinates[1] = {0.0f, 1.0f};
        this->_generated_billboard_texture_coordinates[2] = {1.0f, 0.0f};
        this->_generated_billboard_texture_coordinates[3] = {1.0f, 1.0f};

        //#pragma omp parallel for
        for (int i = 0; i < this->_probe_data->getProbeCount(); i++) {

            auto generic_probe = this->_probe_data->getGenericProbe(i);

            auto visitor = [this](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, FloatProbe>) {
                    doScalarGlyphGeneration(arg);
                } else if constexpr (std::is_same_v<T, IntProbe>) {
                    // TODO
                } else if constexpr (std::is_same_v<T, Vec4Probe>) {
                    doVectorRadarGlyphGeneration(arg);
                } else {
                    // unknown probe type, throw error? do nothing?
                }
            };

            std::visit(visitor, generic_probe);
        } // end for probe count

    }

    cm->setData(this->_mesh_data, _version);

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

        if (this->scale <= 0.0) this->scale = probe_meta_data.m_bboxs.BoundingBox().LongestEdge() * 8e-3;

        this->_probe_data = cprobes->getData();
        this->_mesh_data = std::make_shared<mesh::MeshDataAccessCollection>();
        this->_tex_data = std::make_shared<mesh::ImageDataAccessCollection>();

        _dtu.reserve(this->_probe_data->getProbeCount());
        _generated_mesh_vertices.reserve(this->_probe_data->getProbeCount() * 4);

        this->_generated_billboard_mesh_indices = {0, 1, 3, 0, 3, 2};
        this->_generated_billboard_texture_coordinates[0] = {0.0f, 0.0f};
        this->_generated_billboard_texture_coordinates[1] = {0.0f, 1.0f};
        this->_generated_billboard_texture_coordinates[2] = {1.0f, 0.0f};
        this->_generated_billboard_texture_coordinates[3] = {1.0f, 1.0f};

        //#pragma omp parallel for
        for (int i = 0; i < this->_probe_data->getProbeCount(); i++) {

            auto generic_probe = this->_probe_data->getGenericProbe(i);

            auto visitor =
                [this](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, FloatProbe>) {
                    doScalarGlyphGeneration(arg);
                }
                else if constexpr (std::is_same_v<T, IntProbe>) {
                    //TODO
                }
                else if constexpr (std::is_same_v<T, Vec4Probe>) {
                    doVectorRadarGlyphGeneration(arg);
                } else {
                    // unknown probe type, throw error? do nothing?
                }
            };

            std::visit(visitor, generic_probe);

        } // end for probe count
    }

    ctex->setData(this->_tex_data, _version);

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