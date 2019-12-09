/*
 * ExtractMesh.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "ExtractMesh.h"
#include <limits>
#include "CallKDTree.h"
#include "adios_plugin/CallADIOSData.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "normal_3d_omp.h"


namespace megamol {
namespace probe {


ExtractMesh::ExtractMesh()
    : Module()
    , m_version(0)
    , _getDataCall("getData", "")
    , _deployMeshCall("deployMesh", "")
    , _deployLineCall("deployCenterline", "")
    , _deploySpheresCall("deploySpheres", "")
    , _deployFullDataTree("deployFullDataTree", "")
    , _algorithmSlot("algorithm", "")
    , _xSlot("x", "")
    , _ySlot("y", "")
    , _zSlot("z", "")
    , _xyzSlot("xyz", "")
    , _formatSlot("format", "")
    , _alphaSlot("alpha", "") {

    this->_alphaSlot << new core::param::FloatParam(1.0f);
    this->_alphaSlot.SetUpdateCallback(&ExtractMesh::alphaChanged);
    this->MakeSlotAvailable(&this->_alphaSlot);

    core::param::EnumParam* ep = new core::param::EnumParam(0);
    ep->SetTypePair(0, "alpha_shape");
    this->_algorithmSlot << ep;
    this->MakeSlotAvailable(&this->_algorithmSlot);

    core::param::EnumParam* fp = new core::param::EnumParam(0);
    fp->SetTypePair(0, "separated");
    fp->SetTypePair(1, "interleaved");
    this->_formatSlot << fp;
    this->_formatSlot.SetUpdateCallback(&ExtractMesh::toggleFormat);
    this->MakeSlotAvailable(&this->_formatSlot);

    core::param::FlexEnumParam* xEp = new core::param::FlexEnumParam("undef");
    this->_xSlot << xEp;
    this->MakeSlotAvailable(&this->_xSlot);

    core::param::FlexEnumParam* yEp = new core::param::FlexEnumParam("undef");
    this->_ySlot << yEp;
    this->MakeSlotAvailable(&this->_ySlot);

    core::param::FlexEnumParam* zEp = new core::param::FlexEnumParam("undef");
    this->_zSlot << zEp;
    this->MakeSlotAvailable(&this->_zSlot);

    core::param::FlexEnumParam* xyzEp = new core::param::FlexEnumParam("undef");
    this->_xyzSlot << xyzEp;
    xyzEp->SetGUIVisible(false);
    this->MakeSlotAvailable(&this->_xyzSlot);

    this->_deployMeshCall.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &ExtractMesh::getData);
    this->_deployMeshCall.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &ExtractMesh::getMetaData);
    this->MakeSlotAvailable(&this->_deployMeshCall);

    this->_deployLineCall.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &ExtractMesh::getCenterlineData);
    this->_deployLineCall.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &ExtractMesh::getMetaData);
    this->MakeSlotAvailable(&this->_deployLineCall);

    this->_deployFullDataTree.SetCallback(
        CallKDTree::ClassName(), CallKDTree::FunctionName(0), &ExtractMesh::getKDData);
    this->_deployFullDataTree.SetCallback(
        CallKDTree::ClassName(), CallKDTree::FunctionName(1), &ExtractMesh::getKDMetaData);
    this->MakeSlotAvailable(&this->_deployFullDataTree);

    this->_getDataCall.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->_getDataCall);

    this->_deploySpheresCall.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &ExtractMesh::getParticleData);
    this->_deploySpheresCall.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &ExtractMesh::getParticleMetaData);
    this->MakeSlotAvailable(&this->_deploySpheresCall);
}

ExtractMesh::~ExtractMesh() { this->Release(); }

bool ExtractMesh::create() { return true; }

void ExtractMesh::release() {}

bool ExtractMesh::InterfaceIsDirty() { return (this->_alphaSlot.IsDirty() || this->_formatSlot.IsDirty()); }

bool ExtractMesh::flipNormalsWithCenterLine(pcl::PointCloud<pcl::PointNormal>& point_cloud) {

    for (uint32_t i = 0; i < _cl_indices_per_slice.size(); i++) {
        for (auto& center_idx : _cl_indices_per_slice[i]) {

            auto cl_x = _centerline[i][0];
            auto cl_y = _centerline[i][1];
            auto cl_z = _centerline[i][2];

            cl_x -= point_cloud.points[center_idx].x;
            cl_y -= point_cloud.points[center_idx].y;
            cl_z -= point_cloud.points[center_idx].z;

            // Projection of the normal on the center line point
            const float cos_theta =
                (cl_x * point_cloud.points[center_idx].normal_x + cl_y * point_cloud.points[center_idx].normal_y +
                    cl_z * point_cloud.points[center_idx].normal_z);

            // Flip the plane normal away from center line point
            if (cos_theta > 0) {
                point_cloud.points[center_idx].normal_x *= -1;
                point_cloud.points[center_idx].normal_y *= -1;
                point_cloud.points[center_idx].normal_z *= -1;
            }
        }
    }

    return true;
}

bool ExtractMesh::flipNormalsWithCenterLine_distanceBased(pcl::PointCloud<pcl::PointNormal>& point_cloud) {

    for (uint32_t i = 0; i < point_cloud.points.size(); i++) {

        std::vector<float> distances(_centerline.size());
        for (uint32_t j = 0; j < _centerline.size(); j++) {
            // std::array<float, 3> diffvec = {
            //    vertex_accessor[vertex_step * i + 0] - centerline_accessor[centerline_step * j + 0],
            //    vertex_accessor[vertex_step * i + 1] - centerline_accessor[centerline_step * j + 1],
            //    vertex_accessor[vertex_step * i + 2] - centerline_accessor[centerline_step * j + 2]};
            // distances[j] = std::sqrt(diffvec[0] * diffvec[0] + diffvec[1] * diffvec[1] + diffvec[2] * diffvec[2]);
            std::array<float, 3> diff;
            diff[0] = point_cloud.points[i].x - _centerline[j][0];
            diff[1] = point_cloud.points[i].y - _centerline[j][1];
            diff[2] = point_cloud.points[i].z - _centerline[j][2];
            distances[j] = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
        }

        auto min_iter = std::min_element(distances.begin(), distances.end());
        auto min_index = std::distance(distances.begin(), min_iter);

        auto cl_x = _centerline[min_index][0];
        auto cl_y = _centerline[min_index][1];
        auto cl_z = _centerline[min_index][2];

        cl_x -= point_cloud.points[i].x;
        cl_y -= point_cloud.points[i].y;
        cl_z -= point_cloud.points[i].z;

        // Projection of the normal on the center line point
        const float cos_theta = (cl_x * point_cloud.points[i].normal_x + cl_y * point_cloud.points[i].normal_y +
                                 cl_z * point_cloud.points[i].normal_z);

        // Flip the plane normal away from center line point
        if (cos_theta > 0) {
            point_cloud.points[i].normal_x *= -1;
            point_cloud.points[i].normal_y *= -1;
            point_cloud.points[i].normal_z *= -1;
        }
    }

    return true;
}

bool ExtractMesh::extractCenterLine(pcl::PointCloud<pcl::PointNormal>& point_cloud) {

    std::array<float, 3> whd = {this->_bbox.ObjectSpaceBBox().Width(), this->_bbox.ObjectSpaceBBox().Height(),
        this->_bbox.ObjectSpaceBBox().Depth()};
    const auto longest_edge_index = std::distance(whd.begin(), std::max_element(whd.begin(), whd.end()));

    const uint32_t num_steps = 20;
    const auto step_size = whd[longest_edge_index] / (num_steps + 2); // without begin and end
    const auto step_epsilon = step_size / 2;
    float offset = 0.0f;
    if (longest_edge_index == 0) {
        offset = std::min(this->_bbox.ObjectSpaceBBox().GetLeft(), this->_bbox.ObjectSpaceBBox().GetRight());
    } else if (longest_edge_index == 1) {
        offset = std::min(this->_bbox.ObjectSpaceBBox().GetBottom(), this->_bbox.ObjectSpaceBBox().GetTop());
    } else if (longest_edge_index == 2) {
        offset = std::min(this->_bbox.ObjectSpaceBBox().GetFront(), this->_bbox.ObjectSpaceBBox().GetBack());
    }
    _cl_indices_per_slice.clear();
    _centerline.clear();
    _centerline.resize(num_steps);
    _cl_indices_per_slice.resize(num_steps);


    for (uint32_t i = 0; i < _centerline.size(); i++) {
        const auto slice = offset + (i + 1) * step_size;
        const auto slice_min = slice - step_epsilon;
        const auto slice_max = slice + step_epsilon;
        float slice_dim1_mean = 0.0f;
        float slice_dim2_mean = 0.0f;
        for (uint32_t n = 0; n < point_cloud.points.size(); n++) {
            if (longest_edge_index == 0) {
                if (point_cloud.points[n].x >= slice_min && point_cloud.points[n].x < slice_max) {
                    _cl_indices_per_slice[i].emplace_back(n);
                    slice_dim1_mean += point_cloud.points[n].y;
                    slice_dim2_mean += point_cloud.points[n].z;
                    // std::array<float, 2> tmp_slice_point = {point_cloud.points[n].y, point_cloud.points[n].z};
                    // slice_vertices.emplace_back(tmp_slice_point);
                }
            } else if (longest_edge_index == 1) {
                if (point_cloud.points[n].y >= slice_min && point_cloud.points[n].y < slice_max) {
                    _cl_indices_per_slice[i].emplace_back(n);
                    slice_dim1_mean += point_cloud.points[n].x;
                    slice_dim2_mean += point_cloud.points[n].z;
                }
            } else if (longest_edge_index == 2) {
                if (point_cloud.points[n].z >= slice_min && point_cloud.points[n].z < slice_max) {
                    _cl_indices_per_slice[i].emplace_back(n);
                    slice_dim1_mean += point_cloud.points[n].x;
                    slice_dim2_mean += point_cloud.points[n].y;
                }
            }
        }
        slice_dim1_mean /= _cl_indices_per_slice[i].size();
        slice_dim2_mean /= _cl_indices_per_slice[i].size();
        if (longest_edge_index == 0) {
            _centerline[i] = {slice, slice_dim1_mean, slice_dim2_mean, 1.0f};
        } else if (longest_edge_index == 1) {
            _centerline[i] = {slice_dim1_mean, slice, slice_dim2_mean, 1.0f};
        } else if (longest_edge_index == 2) {
            _centerline[i] = {slice_dim1_mean, slice_dim2_mean, slice, 1.0f};
        }
    }

    return true;
}

void ExtractMesh::applyMeshCorrections() {

    for (auto& point : _poissonCloud.points) {

        std::vector<uint32_t> k_indices;
        std::vector<float> k_distances;

        pcl::PointXYZ p;
        p.x = point.x;
        p.y = point.y;
        p.z = point.z;

        auto neighbors = this->_alpha_hull_tree->nearestKSearch(p, 1, k_indices, k_distances);

        std::array<float, 3> diffvec;
        float distance = 0.0f;
        if (neighbors > 0) {
            diffvec[0] = _alphaHullCloud->points[k_indices[0]].x - point.x;
            diffvec[1] = _alphaHullCloud->points[k_indices[0]].y - point.y;
            diffvec[2] = _alphaHullCloud->points[k_indices[0]].z - point.z;
            // distance = std::sqrt(diffvec[0] * diffvec[0] + diffvec[1] * diffvec[1] + diffvec[2] * diffvec[2]);

            // point.x -= _resultNormalCloud->points[k_indices[0]].normal_x * distance;
            // point.y -= _resultNormalCloud->points[k_indices[0]].normal_y * distance;
            // point.z -= _resultNormalCloud->points[k_indices[0]].normal_z * distance;

            // point.x += diffvec[0];
            // point.y += diffvec[1];
            // point.z += diffvec[2];

            // move along normal
            float d = std::sqrt(diffvec[0] * _resultNormalCloud->points[k_indices[0]].normal_x +
                                diffvec[1] * _resultNormalCloud->points[k_indices[0]].normal_y +
                                diffvec[2] * _resultNormalCloud->points[k_indices[0]].normal_z);

            point.x -= d * _resultNormalCloud->points[k_indices[0]].normal_x;
            point.y -= d * _resultNormalCloud->points[k_indices[0]].normal_y;
            point.z -= d * _resultNormalCloud->points[k_indices[0]].normal_z;
        }
    }
}


void ExtractMesh::calculateAlphaShape() {

    // Calculate the alpha hull of the initial point cloud
    pcl::ConcaveHull<pcl::PointXYZ> hull;
    hull.setAlpha(this->_alphaSlot.Param<core::param::FloatParam>()->Value());
    hull.setDimension(3);
    _inputCloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(_cloud);
    hull.setInputCloud(_inputCloud);
    // hull.setDoFiltering(true);
    pcl::PointCloud<pcl::PointXYZ> resultCloud;
    hull.reconstruct(resultCloud, _polygons);
    _alphaHullCloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(resultCloud);

    // Extract the kd tree for easy sampling of the data
    this->_full_data_tree = std::make_shared<pcl::KdTreeFLANN<pcl::PointXYZ>>();
    this->_full_data_tree->setInputCloud(_inputCloud, nullptr);

    this->_alpha_hull_tree = std::make_shared<pcl::KdTreeFLANN<pcl::PointXYZ>>();
    this->_alpha_hull_tree->setInputCloud(_alphaHullCloud, nullptr);


    // Estimate normals of the alpha shape vertices.
    // Correct normals are needed for the Poisson surface reconstruction
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::PointNormal> normal_est;
    normal_est.setInputCloud(_alphaHullCloud);
    pcl::PointCloud<pcl::PointNormal> normal_cloud;
    normal_est.setRadiusSearch(_bbox.ObjectSpaceBBox().LongestEdge() * 1e-2);
    // normal_est.setKSearch(40);
    normal_est.compute(normal_cloud);

    // The estimated normals are not oriented correctly
    // Calculate center line and use it to flip normals
    this->extractCenterLine(normal_cloud);
    // this->flipNormalsWithCenterLine(normal_cloud);
    this->flipNormalsWithCenterLine_distanceBased(normal_cloud);

    _resultNormalCloud = std::make_shared<pcl::PointCloud<pcl::PointNormal>>(normal_cloud);

    if (usePoisson) {

        pcl::Poisson<pcl::PointNormal> surface;

        // Perform a Poisson surface reconstruction
        _polygons.clear();
        surface.setInputCloud(_resultNormalCloud);
        surface.setDepth(9);
        surface.setSamplesPerNode(2.0f);
        // surface.setConfidence(true);

        surface.reconstruct(_poissonCloud, _polygons);

        // Estimate normals of the reconstructed surface
        pcl::NormalEstimationOMP<pcl::PointNormal, pcl::PointNormal> new_normal_est;
        new_normal_est.setKSearch(40);
        std::shared_ptr<const pcl::PointCloud<pcl::PointNormal>> surface_shared =
            std::make_shared<const pcl::PointCloud<pcl::PointNormal>>(_poissonCloud);
        new_normal_est.setInputCloud(surface_shared);
        new_normal_est.compute(_poissonCloud);

        this->flipNormalsWithCenterLine_distanceBased(_poissonCloud);
    }

    // this->applyMeshCorrections();
}

bool ExtractMesh::createPointCloud(std::vector<std::string>& vars) {

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    if (vars.empty()) return false;

    const auto count = cd->getData(vars[0])->size();

    _cloud.points.resize(count);

    for (auto var : vars) {
        if (this->_formatSlot.Param<core::param::EnumParam>()->Value() == 0) {
            auto x =
                cd->getData(std::string(this->_xSlot.Param<core::param::FlexEnumParam>()->ValueString()))->GetAsFloat();
            auto y =
                cd->getData(std::string(this->_ySlot.Param<core::param::FlexEnumParam>()->ValueString()))->GetAsFloat();
            auto z =
                cd->getData(std::string(this->_zSlot.Param<core::param::FlexEnumParam>()->ValueString()))->GetAsFloat();

            auto xminmax = std::minmax_element(x.begin(), x.end());
            auto yminmax = std::minmax_element(y.begin(), y.end());
            auto zminmax = std::minmax_element(z.begin(), z.end());
            _bbox.SetObjectSpaceBBox(
                *xminmax.first, *yminmax.first, *zminmax.second, *xminmax.second, *yminmax.second, *zminmax.first);

            for (unsigned long long i = 0; i < count; i++) {
                _cloud.points[i].x = x[i];
                _cloud.points[i].y = y[i];
                _cloud.points[i].z = z[i];
            }

        } else {
            // auto xyz = cd->getData(std::string(this->_xyzSlot.Param<core::param::FlexEnumParam>()->ValueString()))
            //               ->GetAsFloat();
            int coarse_factor = 30;
            auto xyz = cd->getData(std::string(this->_xyzSlot.Param<core::param::FlexEnumParam>()->ValueString()))
                           ->GetAsDouble();
            float xmin = std::numeric_limits<float>::max();
            float xmax = std::numeric_limits<float>::min();
            float ymin = std::numeric_limits<float>::max();
            float ymax = std::numeric_limits<float>::min();
            float zmin = std::numeric_limits<float>::max();
            float zmax = std::numeric_limits<float>::min();

            _cloud.points.resize(count / coarse_factor);
            for (unsigned long long i = 0; i < count / (3 * coarse_factor); i++) {
                _cloud.points[i].x = xyz[3 * (i * coarse_factor) + 0];
                _cloud.points[i].y = xyz[3 * (i * coarse_factor) + 1];
                _cloud.points[i].z = xyz[3 * (i * coarse_factor) + 2];

                xmin = std::min(xmin, _cloud.points[i].x);
                xmax = std::max(xmax, _cloud.points[i].x);
                ymin = std::min(ymin, _cloud.points[i].y);
                ymax = std::max(ymax, _cloud.points[i].y);
                zmin = std::min(zmin, _cloud.points[i].z);
                zmax = std::max(zmax, _cloud.points[i].z);
            }
            _bbox.SetObjectSpaceBBox(xmin, ymin, zmax, xmax, ymax, zmin);
        }
    }
    return true;
}

void ExtractMesh::convertToMesh() {

    if (!usePoisson) {
        _mesh_attribs.resize(1);
        _mesh_attribs[0].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
        _mesh_attribs[0].byte_size = _alphaHullCloud->points.size() * sizeof(pcl::PointXYZ);
        _mesh_attribs[0].component_cnt = 3;
        _mesh_attribs[0].stride = sizeof(pcl::PointXYZ);
        _mesh_attribs[0].data = reinterpret_cast<uint8_t*>(const_cast<pcl::PointXYZ*>(_alphaHullCloud->points.data()));
    } else {
        _mesh_attribs.resize(2);
        _vertex_data.resize(3 * _poissonCloud.points.size());
        _normal_data.resize(3 * _poissonCloud.points.size());
#pragma omp parallel for
        for (auto i = 0; i < _poissonCloud.points.size(); i++) {
            //
            _vertex_data[3 * i + 0] = _poissonCloud.points[i].x;
            _vertex_data[3 * i + 1] = _poissonCloud.points[i].y;
            _vertex_data[3 * i + 2] = _poissonCloud.points[i].z;
            //
            _normal_data[3 * i + 0] = _poissonCloud.points[i].normal_x;
            _normal_data[3 * i + 1] = _poissonCloud.points[i].normal_y;
            _normal_data[3 * i + 2] = _poissonCloud.points[i].normal_z;
        }

        //
        _mesh_attribs[0].semantic = mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION;
        _mesh_attribs[0].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
        _mesh_attribs[0].byte_size = sizeof(float) * _vertex_data.size();
        _mesh_attribs[0].component_cnt = 3;
        _mesh_attribs[0].stride = 3 * sizeof(float);
        _mesh_attribs[0].data = reinterpret_cast<uint8_t*>(_vertex_data.data());

        //
        _mesh_attribs[1].semantic = mesh::MeshDataAccessCollection::AttributeSemanticType::NORMAL;
        _mesh_attribs[1].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
        _mesh_attribs[1].byte_size = sizeof(float) * _normal_data.size();
        _mesh_attribs[1].component_cnt = 3;
        _mesh_attribs[1].stride = 3 * sizeof(float);
        _mesh_attribs[1].data = reinterpret_cast<uint8_t*>(_normal_data.data());
    }

    _mesh_indices.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
    _mesh_indices.byte_size = 3 * _polygons.size() * sizeof(uint32_t);
    _mesh_indices.data = reinterpret_cast<uint8_t*>(_polygons.data());
}

bool ExtractMesh::getData(core::Call& call) {

    bool something_changed = _recalc;

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr) return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    std::vector<std::string> toInq;
    toInq.clear();
    if (this->_formatSlot.Param<core::param::EnumParam>()->Value() == 0) {
        toInq.emplace_back(std::string(this->_xSlot.Param<core::param::FlexEnumParam>()->ValueString()));
        toInq.emplace_back(std::string(this->_ySlot.Param<core::param::FlexEnumParam>()->ValueString()));
        toInq.emplace_back(std::string(this->_zSlot.Param<core::param::FlexEnumParam>()->ValueString()));
    } else {
        toInq.emplace_back(std::string(this->_xyzSlot.Param<core::param::FlexEnumParam>()->ValueString()));
    }

    // get data from adios
    for (auto var : toInq) {
        if (!cd->inquire(var)) return false;
    }

    if (cd->getDataHash() != _old_datahash) {
        if (!(*cd)(0)) return false;
        something_changed = true;
    }

    if (something_changed) {

        if (!this->createPointCloud(toInq)) return false;

        this->calculateAlphaShape();

        // this->filterResult();
        // this->filterByIndex();
    }

    if (_mesh_attribs.empty() || something_changed) {
        this->convertToMesh();

        ++m_version;
    }

    if (cm->version() < m_version) {
        auto meta_data = cm->getMetaData();
        meta_data.m_bboxs = _bbox;
        cm->setMetaData(meta_data);

        // put data in mesh
        mesh::MeshDataAccessCollection mesh;

        mesh.addMesh(_mesh_attribs, _mesh_indices);
        cm->setData(std::make_shared<mesh::MeshDataAccessCollection>(std::move(mesh)), m_version);
    }

    _old_datahash = cd->getDataHash();
    _recalc = false;

    return true;
}

bool ExtractMesh::getMetaData(core::Call& call) {

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr) return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    auto meta_data = cm->getMetaData();

    // get metadata from adios
    cd->setFrameIDtoLoad(meta_data.m_frame_ID);
    if (!(*cd)(1)) return false;
    auto vars = cd->getAvailableVars();
    for (auto var : vars) {
        this->_xSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_ySlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_zSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_xyzSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
    }

    if (cd->getDataHash() == _old_datahash && !_recalc) return true;

    // put metadata in mesh call
    meta_data.m_frame_cnt = cd->getFrameCount();
    cm->setMetaData(meta_data);

    return true;
}

bool ExtractMesh::getParticleData(core::Call& call) {
    auto cm = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
    if (cm == nullptr) return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    // if (cd->getDataHash() == _old_datahash && cm->FrameID() == cd->getFrameIDtoLoad() && cm->DataHash() ==
    // _recalc_hash)
    //    return true;

    std::vector<std::string> toInq;
    toInq.clear();
    if (this->_formatSlot.Param<core::param::EnumParam>()->Value() == 0) {
        toInq.emplace_back(std::string(this->_xSlot.Param<core::param::FlexEnumParam>()->ValueString()));
        toInq.emplace_back(std::string(this->_ySlot.Param<core::param::FlexEnumParam>()->ValueString()));
        toInq.emplace_back(std::string(this->_zSlot.Param<core::param::FlexEnumParam>()->ValueString()));
    } else {
        toInq.emplace_back(std::string(this->_xyzSlot.Param<core::param::FlexEnumParam>()->ValueString()));
    }

    // get data from adios
    for (auto var : toInq) {
        if (!cd->inquire(var)) return false;
    }
    if (cd->getDataHash() != _old_datahash)
        if (!(*cd)(0)) return false;


    if (!this->createPointCloud(toInq)) return false;


    cm->AccessBoundingBoxes().SetObjectSpaceBBox(_bbox.ObjectSpaceBBox());

    if (cd->getDataHash() != _old_datahash || _recalc) this->calculateAlphaShape();

    // this->filterResult();
    // this->filterByIndex();

    cm->SetParticleListCount(1);
    // cm->AccessParticles(0).SetGlobalRadius(0.02f);
    cm->AccessParticles(0).SetGlobalRadius(_bbox.ObjectSpaceBBox().LongestEdge() * 1e-3);
    cm->AccessParticles(0).SetCount(_alphaHullCloud->points.size());
    cm->AccessParticles(0).SetVertexData(core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ,
        reinterpret_cast<uint8_t*>(&_resultNormalCloud->points[0].x), sizeof(pcl::PointNormal));
    cm->AccessParticles(0).SetDirData(core::moldyn::MultiParticleDataCall::Particles::DIRDATA_FLOAT_XYZ,
        &_resultNormalCloud->points[0].normal_x, sizeof(pcl::PointNormal));

    _old_datahash = cd->getDataHash();
    _recalc = false;

    return true;
}

bool ExtractMesh::getParticleMetaData(core::Call& call) {
    auto cm = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
    if (cm == nullptr) return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    if (cd->getDataHash() == _old_datahash && cm->FrameID() == cd->getFrameIDtoLoad()) return true;

    // get metadata from adios
    if (!(*cd)(1)) return false;
    auto vars = cd->getAvailableVars();
    for (auto var : vars) {
        this->_xSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_ySlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_zSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_xyzSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
    }


    cm->SetFrameCount(cd->getFrameCount());
    cd->setFrameIDtoLoad(cm->FrameID());
    cm->SetDataHash(_old_datahash);

    return true;
}

bool ExtractMesh::getCenterlineData(core::Call& call) {
    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr) return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    // if (cd->getDataHash() == _old_datahash && meta_data.m_frame_ID == cd->getFrameIDtoLoad() &&
    //    meta_data.m_data_hash == _recalc_hash)
    //    return true;

    std::vector<std::string> toInq;
    toInq.clear();
    if (this->_formatSlot.Param<core::param::EnumParam>()->Value() == 0) {
        toInq.emplace_back(std::string(this->_xSlot.Param<core::param::FlexEnumParam>()->ValueString()));
        toInq.emplace_back(std::string(this->_ySlot.Param<core::param::FlexEnumParam>()->ValueString()));
        toInq.emplace_back(std::string(this->_zSlot.Param<core::param::FlexEnumParam>()->ValueString()));
    } else {
        toInq.emplace_back(std::string(this->_xyzSlot.Param<core::param::FlexEnumParam>()->ValueString()));
    }

    // get data from adios
    for (auto var : toInq) {
        if (!cd->inquire(var)) return false;
    }

    if (cd->getDataHash() != _old_datahash)
        if (!(*cd)(0)) return false;

    bool something_has_changed = (cd->getDataHash() != _old_datahash);

    if (something_has_changed) {
        ++m_version;

        if (!this->createPointCloud(toInq)) return false;

        this->calculateAlphaShape();
    }

    if (cm->version() < m_version) {

        _line_attribs.resize(1);
        _line_attribs[0].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
        _line_attribs[0].byte_size = _centerline.size() * sizeof(std::array<float, 4>);
        _line_attribs[0].component_cnt = 3;
        _line_attribs[0].stride = sizeof(std::array<float, 4>);
        _line_attribs[0].data = reinterpret_cast<uint8_t*>(_centerline.data());

        _cl_indices.resize(_centerline.size() - 1);
        std::generate(_cl_indices.begin(), _cl_indices.end(), [n = 0]() mutable { return n++; });

        _line_indices.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
        _line_indices.byte_size = _cl_indices.size() * sizeof(uint32_t);
        _line_indices.data = reinterpret_cast<uint8_t*>(_cl_indices.data());

        // put data in line
        mesh::MeshDataAccessCollection line;
        line.addMesh(_line_attribs, _line_indices);
        cm->setData(std::make_shared<mesh::MeshDataAccessCollection>(std::move(line)), m_version);

        auto meta_data = cm->getMetaData();
        meta_data.m_bboxs = _bbox;
        cm->setMetaData(meta_data);
    }

    _old_datahash = cd->getDataHash();
    _recalc = false;

    return true;
}

bool ExtractMesh::getKDMetaData(core::Call& call) {

    auto cm = dynamic_cast<CallKDTree*>(&call);
    if (cm == nullptr) return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    auto meta_data = cm->getMetaData();
    if (cd->getDataHash() == _old_datahash && meta_data.m_frame_ID == cd->getFrameIDtoLoad() && !_recalc) return true;

    // get metadata from adios

    cd->setFrameIDtoLoad(meta_data.m_frame_ID);
    if (!(*cd)(1)) return false;
    auto vars = cd->getAvailableVars();
    for (auto var : vars) {
        this->_xSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_ySlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_zSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        this->_xyzSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
    }

    // put metadata in mesh call
    meta_data.m_frame_cnt = cd->getFrameCount();
    cm->setMetaData(meta_data);

    return true;
}

bool ExtractMesh::getKDData(core::Call& call) {

    auto cm = dynamic_cast<CallKDTree*>(&call);
    if (cm == nullptr) return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    
    // if (cd->getDataHash() == _old_datahash && meta_data.m_frame_ID == cd->getFrameIDtoLoad() &&
    //    meta_data.m_data_hash == _recalc_hash)
    //    return true;

    std::vector<std::string> toInq;
    toInq.clear();
    if (this->_formatSlot.Param<core::param::EnumParam>()->Value() == 0) {
        toInq.emplace_back(std::string(this->_xSlot.Param<core::param::FlexEnumParam>()->ValueString()));
        toInq.emplace_back(std::string(this->_ySlot.Param<core::param::FlexEnumParam>()->ValueString()));
        toInq.emplace_back(std::string(this->_zSlot.Param<core::param::FlexEnumParam>()->ValueString()));
    } else {
        toInq.emplace_back(std::string(this->_xyzSlot.Param<core::param::FlexEnumParam>()->ValueString()));
    }

    // get data from adios
    for (auto var : toInq) {
        if (!cd->inquire(var)) return false;
    }

    if (cd->getDataHash() != _old_datahash)
        if (!(*cd)(0)) return false;

    if (cd->getDataHash() != _old_datahash || _recalc) {
        ++m_version;

        if (!this->createPointCloud(toInq)) return false;

        this->calculateAlphaShape();
    }

    if (cm->version() < m_version) {
        cm->setData(this->_full_data_tree, m_version);

        auto meta_data = cm->getMetaData();
        meta_data.m_bboxs = _bbox;
        cm->setMetaData(meta_data);
    }

    _old_datahash = cd->getDataHash();
    _recalc = false;

    return true;
}

bool ExtractMesh::toggleFormat(core::param::ParamSlot& p) {

    if (this->_formatSlot.Param<core::param::EnumParam>()->Value() == 0) {
        this->_xSlot.Param<core::param::FlexEnumParam>()->SetGUIVisible(true);
        this->_ySlot.Param<core::param::FlexEnumParam>()->SetGUIVisible(true);
        this->_zSlot.Param<core::param::FlexEnumParam>()->SetGUIVisible(true);
        this->_xyzSlot.Param<core::param::FlexEnumParam>()->SetGUIVisible(false);
    } else {
        this->_xSlot.Param<core::param::FlexEnumParam>()->SetGUIVisible(false);
        this->_ySlot.Param<core::param::FlexEnumParam>()->SetGUIVisible(false);
        this->_zSlot.Param<core::param::FlexEnumParam>()->SetGUIVisible(false);
        this->_xyzSlot.Param<core::param::FlexEnumParam>()->SetGUIVisible(true);
    }

    return true;
}

bool ExtractMesh::alphaChanged(core::param::ParamSlot& p) {

    _recalc = true;

    return true;
}

bool ExtractMesh::filterResult() {

    const int factor = 3;

    std::vector<pcl::Vertices> new_polygon;
    new_polygon.reserve(_polygons.size());

    for (auto& polygon : _polygons) {
        auto v1 = _alphaHullCloud->points[polygon.vertices[0]];
        auto v2 = _alphaHullCloud->points[polygon.vertices[1]];
        auto v3 = _alphaHullCloud->points[polygon.vertices[2]];

        auto a = v1 - v2;
        auto length_a = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);

        auto b = v1 - v3;
        auto length_b = std::sqrt(b.x * b.x + b.y * b.y + b.z * b.z);

        auto c = v2 - v3;
        auto length_c = std::sqrt(c.x * c.x + c.y * c.y + c.z * c.z);

        auto l1 = length_b > length_a ? length_b / length_a : length_a / length_b;
        auto l2 = length_c > length_a ? length_c / length_a : length_a / length_c;
        auto l3 = length_b > length_c ? length_b / length_c : length_c / length_b;

        if (l1 >= factor || l2 >= factor || l3 >= factor) {
            vislib::sys::Log::DefaultLog.WriteInfo("[ExtractMesh] Found bad polygon.");
        } else {
            new_polygon.emplace_back(polygon);
        }
    }
    new_polygon.shrink_to_fit();

    _polygons = std::move(new_polygon);

    return true;
}

bool ExtractMesh::filterByIndex() {

    size_t max_distance = 10;

    std::vector<pcl::Vertices> new_polygon;
    new_polygon.reserve(_polygons.size());

    for (auto polygon : _polygons) {
        auto dist_a = std::abs(static_cast<int>(polygon.vertices[0]) - static_cast<int>(polygon.vertices[1]));
        auto dist_b = std::abs(static_cast<int>(polygon.vertices[2]) - static_cast<int>(polygon.vertices[1]));
        auto dist_c = std::abs(static_cast<int>(polygon.vertices[0]) - static_cast<int>(polygon.vertices[2]));

        if (!(dist_a >= max_distance || dist_b >= max_distance || dist_c >= max_distance)) {
            new_polygon.emplace_back(polygon);
        }
    }
    new_polygon.shrink_to_fit();
    _polygons = std::move(new_polygon);

    return true;
}

} // namespace probe
} // namespace megamol
