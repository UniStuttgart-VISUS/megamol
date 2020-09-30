/*
 * ConstructKDTree.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "ConstructKDTree.h"
#include <limits>
#include "CallKDTree.h"
#include "adios_plugin/CallADIOSData.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "normal_3d_omp.h"
#include <atomic>


namespace megamol {
namespace probe {

ConstructKDTree::ConstructKDTree()
    : Module()
    , _getDataCall("getData", "")
    , _deployFullDataTree("deployFullDataTree", "")
    , _xSlot("x", "")
    , _ySlot("y", "")
    , _zSlot("z", "")
    , _xyzSlot("xyz", "")
    , _formatSlot("format", "") {

    core::param::EnumParam* fp = new core::param::EnumParam(0);
    fp->SetTypePair(0, "separated");
    fp->SetTypePair(1, "interleaved");
    this->_formatSlot << fp;
    this->_formatSlot.SetUpdateCallback(&ConstructKDTree::toggleFormat);
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

     this->_deployFullDataTree.SetCallback(CallKDTree::ClassName(), CallKDTree::FunctionName(0), &ConstructKDTree::getData);
    this->_deployFullDataTree.SetCallback(
        CallKDTree::ClassName(), CallKDTree::FunctionName(1), &ConstructKDTree::getMetaData);
    this->MakeSlotAvailable(&this->_deployFullDataTree);

    this->_getDataCall.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->_getDataCall);
}

ConstructKDTree::~ConstructKDTree() { this->Release(); }

bool ConstructKDTree::create() { return true; }

void ConstructKDTree::release() {}

bool ConstructKDTree::InterfaceIsDirty() { return this->_formatSlot.IsDirty(); }

bool ConstructKDTree::createPointCloud(std::vector<std::string>& vars) {

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
            _bbox.SetBoundingBox(
                *xminmax.first, *yminmax.first, *zminmax.second, *xminmax.second, *yminmax.second, *zminmax.first);

            for (unsigned long long i = 0; i < count; i++) {
                _cloud.points[i].x = x[i];
                _cloud.points[i].y = y[i];
                _cloud.points[i].z = z[i];
            }

        } else {
            //auto xyz = cd->getData(std::string(this->_xyzSlot.Param<core::param::FlexEnumParam>()->ValueString()))
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

            _cloud.points.resize(count/coarse_factor);
            for (unsigned long long i = 0; i < count/(3*coarse_factor); i++) {
                _cloud.points[i].x = xyz[3 * (i*coarse_factor) + 0];
                _cloud.points[i].y = xyz[3 * (i*coarse_factor) + 1];
                _cloud.points[i].z = xyz[3 * (i*coarse_factor) + 2];

                xmin = std::min(xmin, _cloud.points[i].x);
                xmax = std::max(xmax, _cloud.points[i].x);
                ymin = std::min(ymin, _cloud.points[i].y);
                ymax = std::max(ymax, _cloud.points[i].y);
                zmin = std::min(zmin, _cloud.points[i].z);
                zmax = std::max(zmax, _cloud.points[i].z);
            }
            _bbox.SetBoundingBox(xmin, ymin, zmax, xmax, ymax, zmin);
        }
    }
    return true;
}

bool ConstructKDTree::getMetaData(core::Call& call) {

    auto ct = dynamic_cast<CallKDTree*>(&call);
    if (ct == nullptr) return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    auto meta_data = ct->getMetaData();
    if (cd->getDataHash() == _old_datahash && meta_data.m_frame_ID == cd->getFrameIDtoLoad()) return true;

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
    ct->setMetaData(meta_data);

    return true;
}

bool ConstructKDTree::getData(core::Call& call) {

    auto ct = dynamic_cast<CallKDTree*>(&call);
    if (ct == nullptr) return false;

    auto cd = this->_getDataCall.CallAs<adios::CallADIOSData>();
    if (cd == nullptr) return false;

    auto meta_data = ct->getMetaData();

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

        if (!this->createPointCloud(toInq)) return false;

        meta_data.m_bboxs = _bbox;
        ct->setMetaData(meta_data);

        // Extract the kd tree for easy sampling of the data
        _inputCloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(_cloud);
        this->_full_data_tree = std::make_shared<pcl::KdTreeFLANN<pcl::PointXYZ>>();
        this->_full_data_tree->setInputCloud(_inputCloud, nullptr);
        this->_version++;
        ct->setData(this->_full_data_tree, this->_version);
        _old_datahash = cd->getDataHash();
    }
    return true;
}

bool ConstructKDTree::toggleFormat(core::param::ParamSlot& p) {

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

} // namespace probe
} // namespace megamol
