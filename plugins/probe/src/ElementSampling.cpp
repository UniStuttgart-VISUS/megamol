/*
 * ElementSampling.cpp
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "ElementSampling.h"
#include "glm/glm.hpp"
#include "mmadios/CallADIOSData.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "probe/CallKDTree.h"
#include "probe/ProbeCalls.h"


namespace megamol {
namespace probe {

ElementSampling::ElementSampling()
        : Module()
        , _version(0)
        , _old_datahash(0)
        , _probe_lhs_slot("deployProbe", "")
        , _elements_rhs_slot("getElements", "")
        , _adios_rhs_slot("getData", "")
        , _parameter_to_sample_slot("ParameterToSample", "")
        , _xSlot("x", "")
        , _ySlot("y", "")
        , _zSlot("z", "")
        , _xyzSlot("xyz", "")
        , _formatSlot("format", "") {

    core::param::EnumParam* fp = new core::param::EnumParam(0);
    fp->SetTypePair(0, "separated");
    fp->SetTypePair(1, "interleaved");
    this->_formatSlot << fp;
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
    this->MakeSlotAvailable(&this->_xyzSlot);

    this->_probe_lhs_slot.SetCallback(CallProbes::ClassName(), CallProbes::FunctionName(0), &ElementSampling::getData);
    this->_probe_lhs_slot.SetCallback(
        CallProbes::ClassName(), CallProbes::FunctionName(1), &ElementSampling::getMetaData);
    this->MakeSlotAvailable(&this->_probe_lhs_slot);

    this->_elements_rhs_slot.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->_elements_rhs_slot);
    this->_elements_rhs_slot.SetNecessity(megamol::core::AbstractCallSlotPresentation::SLOT_REQUIRED);


    this->_adios_rhs_slot.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->_adios_rhs_slot);
    this->_adios_rhs_slot.SetNecessity(megamol::core::AbstractCallSlotPresentation::SLOT_REQUIRED);

    core::param::FlexEnumParam* paramEnum = new core::param::FlexEnumParam("undef");
    this->_parameter_to_sample_slot << paramEnum;
    this->_parameter_to_sample_slot.SetUpdateCallback(&ElementSampling::paramChanged);
    this->MakeSlotAvailable(&this->_parameter_to_sample_slot);
}

ElementSampling::~ElementSampling() {
    this->Release();
}

bool ElementSampling::create() {
    _probes = std::make_shared<ProbeCol>();
    return true;
}

void ElementSampling::release() {}

bool ElementSampling::getData(core::Call& call) {

    bool something_has_changed = false;
    auto cp = dynamic_cast<CallProbes*>(&call);
    if (cp == nullptr)
        return false;

    // query adios data
    auto cd = this->_adios_rhs_slot.CallAs<adios::CallADIOSData>();
    if (cd == nullptr)
        return false;
    auto celements = this->_elements_rhs_slot.CallAs<adios::CallADIOSData>();
    if (celements == nullptr)
        return false;


    std::vector<std::string> toInq;


    core::Spatial3DMetaData meta_data = cp->getMetaData();
    std::string var_str =
        std::string(this->_parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->ValueString());
    toInq.emplace_back(var_str);

    if (this->_formatSlot.Param<core::param::EnumParam>()->Value() == 0) {
        toInq.emplace_back(std::string(this->_xSlot.Param<core::param::FlexEnumParam>()->ValueString()));
        toInq.emplace_back(std::string(this->_ySlot.Param<core::param::FlexEnumParam>()->ValueString()));
        toInq.emplace_back(std::string(this->_zSlot.Param<core::param::FlexEnumParam>()->ValueString()));
    } else {
        toInq.emplace_back(std::string(this->_xyzSlot.Param<core::param::FlexEnumParam>()->ValueString()));
    }

    // get data from adios
    for (auto var : toInq) {
        if (!cd->inquireVar(var))
            return false;
    }

    if (cd->getDataHash() != _old_datahash || _trigger_recalc) {
        if (!(*cd)(0))
            return false;
    }

    if (celements->getDataHash() != _elements_cached_hash || _trigger_recalc) {
        if (!readElements())
            return false;
        auto bbox = celements->getData("bbox")->GetAsFloat();
        if (bbox.size() == 6) {
            _bbox.SetBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
        }
    }
    something_has_changed = something_has_changed || (cd->getDataHash() != _old_datahash) ||
                            (celements->getDataHash() != _elements_cached_hash);


    if (something_has_changed) {
        ++_version;
        std::vector<float> raw_positions;
        if (this->_formatSlot.Param<core::param::EnumParam>()->Value() == 0) {
            auto x =
                cd->getData(std::string(this->_xSlot.Param<core::param::FlexEnumParam>()->ValueString()))->GetAsFloat();
            auto y =
                cd->getData(std::string(this->_ySlot.Param<core::param::FlexEnumParam>()->ValueString()))->GetAsFloat();
            auto z =
                cd->getData(std::string(this->_zSlot.Param<core::param::FlexEnumParam>()->ValueString()))->GetAsFloat();
            assert(x.size() == y.size());
            assert(y.size() == z.size());
            raw_positions.resize(x.size() * 3);
            for (int i = 0; i < x.size(); ++i) {
                raw_positions[3 * i + 0] = x[i];
                raw_positions[3 * i + 1] = y[i];
                raw_positions[3 * i + 2] = z[i];
            }
        } else {
            const std::string varname = std::string(_xyzSlot.Param<core::param::FlexEnumParam>()->ValueString());
            raw_positions = cd->getData(varname)->GetAsFloat();
        }

        //if (cd->getData(var_str)->getType() == "double") {
        //    std::vector<double> data = cd->getData(var_str)->GetAsDouble();
        //        this->doScalarSampling<double>(_elements, data, raw_positions);

        //} else
        //if (cd->getData(var_str)->getType() == "float") {
        std::vector<float> data = cd->getData(var_str)->GetAsFloat();
        placeProbes(_elements);
        doScalarSampling(_elements, data, raw_positions);
        //}
    }


    // put data into probes
    _old_datahash = cd->getDataHash();
    _elements_cached_hash = celements->getDataHash();
    meta_data.m_bboxs = _bbox;

    cp->setMetaData(meta_data);
    cp->setData(_probes, _version);
    _trigger_recalc = false;


    return true;
}

bool ElementSampling::getMetaData(core::Call& call) {

    auto cp = dynamic_cast<CallProbes*>(&call);
    if (cp == nullptr)
        return false;

    auto cd = this->_adios_rhs_slot.CallAs<adios::CallADIOSData>();
    auto celements = this->_elements_rhs_slot.CallAs<adios::CallADIOSData>();
    if (celements == nullptr)
        return false;

    auto meta_data = cp->getMetaData();
    // if (cd->getDataHash() == _old_datahash && meta_data.m_frame_ID == cd->getFrameIDtoLoad() && !_trigger_recalc)
    //    return true;

    if (cd != nullptr && celements != nullptr) {
        cd->setFrameIDtoLoad(meta_data.m_frame_ID);
        celements->setFrameIDtoLoad(meta_data.m_frame_ID);
        if (!(*cd)(1))
            return false;
        if (!(*celements)(1))
            return false;
        meta_data.m_frame_cnt = cd->getFrameCount();

        // get adios meta data
        auto vars = cd->getAvailableVars();
        for (auto var : vars) {
            this->_parameter_to_sample_slot.Param<core::param::FlexEnumParam>()->AddValue(var);
        }
    } else {
        return false;
    }

    // put metadata in mesh call
    cp->setMetaData(meta_data);

    return true;
}

bool ElementSampling::readElements() {
    auto celements = _elements_rhs_slot.CallAs<adios::CallADIOSData>();
    if (celements == nullptr)
        return false;

    celements->setFrameIDtoLoad(0); // TODO: maybe support more frames in the future
    if (!(*celements)(1))
        return false;

    auto vars = celements->getAvailableVars();

    // get data from adios
    for (auto var : vars) {
        if (!celements->inquireVar(var)) {
            core::utility::log::Log::DefaultLog.WriteError(
                (std::string("[ReconstructSurface] Could not inquire ") + var).c_str());
            return false;
        }
    }

    if (!(*celements)(0))
        return false;

    auto elements = celements->getData("elements")->GetAsChar();
    auto elements_offsets = celements->getData("elements_offsets")->GetAsUInt64();
    auto elements_shape = celements->getData("elements_offsets")->getShape();

    _elements.clear();
    _elements.resize(elements_shape[0]);
    for (int i = 0; i < elements_shape[0]; ++i) {
        _elements[i].resize(elements_shape[1]);
#pragma omp parallel for
        for (int j = 0; j < elements_shape[1]; ++j) {
            std::string current_element;
            if ((i == (elements_shape[0] - 1)) && (j == (elements_shape[1] - 1))) {
                current_element =
                    std::string(elements.begin() + elements_offsets[i * elements_shape[1] + j], elements.end());
            } else {
                current_element = std::string(elements.begin() + elements_offsets[i * elements_shape[1] + j],
                    elements.begin() + elements_offsets[i * elements_shape[1] + j + 1]);
            }
            std::stringstream(current_element) >> _elements[i][j];
        }
    }

    return true;
}

void ElementSampling::placeProbes(const std::vector<std::vector<Surface_mesh>>& elements) {

    if (elements.empty()) {
        core::utility::log::Log::DefaultLog.WriteError(
            "[ElementSampling] placeProbes exited because elements are empty.");
        return;
    }
    if (elements[0].empty()) {
        core::utility::log::Log::DefaultLog.WriteError(
            "[ElementSampling] placeProbes exited because elements are empty.");
        return;
    }

    auto longest_edge = _bbox.BoundingBox().LongestEdge();
    std::vector<std::string> geom_ids(elements.size());

    //select Element
    for (int j = 0; j < elements[0].size(); ++j) {

        geom_ids.clear();
        geom_ids.resize(elements.size());
        // generate flat geomety indices
        for (int k = 0; k < elements.size(); ++k) {
            geom_ids[k] = "element_mesh_" + std::to_string(k) + "," + std::to_string(j);
        }

        BaseProbe new_probe;
        glm::vec3 start = glm::vec3(0);
        for (auto spoint : elements[0][j].points()) {
            start.x += spoint.x();
            start.y += spoint.y();
            start.z += spoint.z();
        }
        const auto num_spoints = elements[0][j].num_vertices();
        start /= num_spoints;
        new_probe.m_position = {start.x, start.y, start.z};

        glm::vec3 end = glm::vec3(0);
        for (auto epoint : elements[elements.size() - 1][j].points()) {
            end.x += epoint.x();
            end.y += epoint.y();
            end.z += epoint.z();
        }
        const auto num_epoints = elements[elements.size() - 1][j].num_vertices();
        end /= num_epoints;
        auto dir = glm::normalize(end - start);
        new_probe.m_direction = {dir.x, dir.y, dir.z};
        new_probe.m_end = glm::length(end - start);
        new_probe.m_geo_ids = geom_ids;
        new_probe.m_begin = -0.2 * new_probe.m_end;

        _probes->addProbe(new_probe);
    }
}

bool ElementSampling::paramChanged(core::param::ParamSlot& p) {

    _trigger_recalc = true;
    return true;
}

} // namespace probe
} // namespace megamol
