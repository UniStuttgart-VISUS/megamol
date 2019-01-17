/*
 * MultiParticletoADIOS.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MultiParticletoADIOS.h"
#include "CallADIOSData.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/EnumParam.h"
#include "vislib/sys/Log.h"

namespace megamol {
namespace adios {

MultiParticletoADIOS::MultiParticletoADIOS(void)
    : core::Module()
    , mpSlot("mpSlot", "Slot to request multi particle data.")
    , adiosSlot("adiosSlot", "Slot to send ADIOS IO")
    , orderSlot("order", "Sets xyz parameter order") {

    this->adiosSlot.SetCallback(
        CallADIOSData::ClassName(), CallADIOSData::FunctionName(0), &MultiParticletoADIOS::getDataCallback);
    this->adiosSlot.SetCallback(
        CallADIOSData::ClassName(), CallADIOSData::FunctionName(1), &MultiParticletoADIOS::getHeaderCallback);
    this->MakeSlotAvailable(&this->adiosSlot);

    auto tmpEnum = new core::param::EnumParam(0);
    tmpEnum->SetTypePair(0, "separated");
    tmpEnum->SetTypePair(1, "interleaved");
    this->orderSlot << tmpEnum;
    this->MakeSlotAvailable(&this->orderSlot);


    this->mpSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->mpSlot);
}

MultiParticletoADIOS::~MultiParticletoADIOS(void) { this->Release(); }

bool MultiParticletoADIOS::create(void) { return true; }

void MultiParticletoADIOS::release(void) {}

bool MultiParticletoADIOS::getDataCallback(core::Call& call) {
    CallADIOSData* cad = dynamic_cast<CallADIOSData*>(&call);
    if (cad == nullptr) return false;

    core::moldyn::MultiParticleDataCall* mpdc = this->mpSlot.CallAs<core::moldyn::MultiParticleDataCall>();
    if (mpdc == nullptr) return false;

    if (!(*mpdc)(1)) return false;

    // set frame to load from view
    mpdc->SetFrameID(cad->getFrameIDtoLoad());

    if (!(*mpdc)(0)) return false;

    auto availVars = cad->getAvailableVars();

    // mandatory
    auto fc_x = std::make_shared<FloatContainer>(FloatContainer());
    auto fc_y = std::make_shared<FloatContainer>(FloatContainer());
    auto fc_z = std::make_shared<FloatContainer>(FloatContainer());
    auto fc_box = std::make_shared<FloatContainer>(FloatContainer());
    auto ic_pcount = std::make_shared<IntContainer>(IntContainer());
    // optional
    auto fc_r = std::make_shared<FloatContainer>(FloatContainer());
    auto fc_g = std::make_shared<FloatContainer>(FloatContainer());
    auto fc_b = std::make_shared<FloatContainer>(FloatContainer());
    auto fc_a = std::make_shared<FloatContainer>(FloatContainer());
    auto fc_radius = std::make_shared<FloatContainer>(FloatContainer());
    auto ic_id = std::make_shared<IntContainer>(IntContainer());

    std::vector<float>& tmp_x = fc_x->getVec();
    std::vector<float>& tmp_y = fc_y->getVec();
    std::vector<float>& tmp_z = fc_z->getVec();
    std::vector<float>& tmp_box = fc_box->getVec();
    std::vector<int>& tmp_pcount = ic_pcount->getVec();

    std::vector<float>& tmp_r = fc_r->getVec();
    std::vector<float>& tmp_g = fc_g->getVec();
    std::vector<float>& tmp_b = fc_b->getVec();
    std::vector<float>& tmp_a = fc_a->getVec();
    std::vector<float>& tmp_radius = fc_radius->getVec();
    std::vector<int>& tmp_id = ic_id->getVec();

    vislib::math::Cuboid<float> bbox = mpdc->GetBoundingBoxes().ObjectSpaceBBox();
    tmp_box.resize(6);
    tmp_box = {bbox.GetLeft(), bbox.GetBottom(), bbox.GetBack(), bbox.GetRight(), bbox.GetTop(), bbox.GetFront()};

    for (auto i = 0; i < mpdc->GetParticleListCount(); i++) {
        core::moldyn::MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(i);

        // global radius
        if (cad->isInVars("global_radius")) {
            tmp_radius.push_back(parts.GetParticleStore().GetRAcc()->Get_f(0));
        }

        // global color
        if (cad->isInVars("global_r")) {
            tmp_r.push_back(static_cast<float>(parts.GetParticleStore().GetCRAcc()->Get_u8(0)) / 255.0f);
            tmp_g.push_back(static_cast<float>(parts.GetParticleStore().GetCGAcc()->Get_u8(0)) / 255.0f);
            tmp_b.push_back(static_cast<float>(parts.GetParticleStore().GetCBAcc()->Get_u8(0)) / 255.0f);
            tmp_a.push_back(static_cast<float>(parts.GetParticleStore().GetCAAcc()->Get_u8(0)) / 255.0f);
        }


        const size_t num = parts.GetCount();

        if (this->orderSlot.Param<core::param::EnumParam>()->Value() == 0) {
            tmp_x.reserve(tmp_x.size() + num);
            tmp_y.reserve(tmp_y.size() + num);
            tmp_z.reserve(tmp_z.size() + num);
        } else {
            tmp_x.reserve(tmp_x.size() + 3 * num);
        }

        if (cad->isInVars("radius")) {
            tmp_radius.reserve(tmp_radius.size() + num);
        }
        if (std::find(availVars.begin(), availVars.end(), "r") != availVars.end()) {
            tmp_r.reserve(tmp_r.size() + num);
            tmp_g.reserve(tmp_g.size() + num);
            tmp_b.reserve(tmp_b.size() + num);
            tmp_a.reserve(tmp_a.size() + num);
        }
        if (parts.HasID()) {
            tmp_id.reserve(tmp_id.size() + num);
        }

        for (auto j = 0; j < num; j++) {

            if (this->orderSlot.Param<core::param::EnumParam>()->Value() == 0) {
                tmp_x.push_back(parts.GetParticleStore().GetXAcc()->Get_f(j));
                tmp_y.push_back(parts.GetParticleStore().GetYAcc()->Get_f(j));
                tmp_z.push_back(parts.GetParticleStore().GetZAcc()->Get_f(j));
            } else {
                tmp_x.push_back(parts.GetParticleStore().GetXAcc()->Get_f(j));
                tmp_x.push_back(parts.GetParticleStore().GetYAcc()->Get_f(j));
                tmp_x.push_back(parts.GetParticleStore().GetZAcc()->Get_f(j));
            }

            // radius
            if (cad->isInVars("radius")) {
                tmp_radius.push_back(parts.GetParticleStore().GetRAcc()->Get_f(j));
            }

            // color
            if (cad->isInVars("r")) {
                tmp_r.push_back(parts.GetParticleStore().GetCRAcc()->Get_f(j));
                tmp_g.push_back(parts.GetParticleStore().GetCGAcc()->Get_f(j));
                tmp_b.push_back(parts.GetParticleStore().GetCBAcc()->Get_f(j));
                tmp_a.push_back(parts.GetParticleStore().GetCAAcc()->Get_f(j));
            }

            // id
            if (parts.HasID()) {
                tmp_id.push_back(parts.GetParticleStore().GetIDAcc()->Get_u64(j));
            }

        } // end for each particle
    }     // end for each particle lists

    tmp_pcount.resize(1);
	if (this->orderSlot.Param<core::param::EnumParam>()->Value() == 0) {	
		tmp_pcount[0] = tmp_x.size();
	} else {
		tmp_pcount[0] = tmp_x.size()/3;
	}


    if (this->orderSlot.Param<core::param::EnumParam>()->Value() == 0) {
        dataMap["x"] = std::move(fc_x);
        dataMap["y"] = std::move(fc_y);
        dataMap["z"] = std::move(fc_z);
    } else {
        dataMap["xyz"] = std::move(fc_x);
    }
    dataMap["box"] = std::move(fc_box);
    dataMap["p_count"] = std::move(ic_pcount);

    if (std::find(availVars.begin(), availVars.end(), "global_radius") != availVars.end()) {
        dataMap["global_radius"] = std::move(fc_radius);
    } else if (std::find(availVars.begin(), availVars.end(), "radius") != availVars.end()) {
        dataMap["radius"] = std::move(fc_radius);
    }

    if (std::find(availVars.begin(), availVars.end(), "global_r") != availVars.end()) {
        dataMap["global_r"] = std::move(fc_r);
        dataMap["global_g"] = std::move(fc_g);
        dataMap["global_b"] = std::move(fc_b);
        dataMap["global_a"] = std::move(fc_a);
    } else if (std::find(availVars.begin(), availVars.end(), "r") != availVars.end()) {
        dataMap["r"] = std::move(fc_r);
        dataMap["g"] = std::move(fc_g);
        dataMap["b"] = std::move(fc_b);
        dataMap["a"] = std::move(fc_a);
    }

    if (std::find(availVars.begin(), availVars.end(), "id") != availVars.end()) {
        dataMap["id"] = std::move(ic_id);
    }

    // set stuff in call
    cad->setData(std::make_shared<adiosDataMap>(dataMap));
    cad->setDataHash(mpdc->DataHash());

    return true;
}

bool MultiParticletoADIOS::getHeaderCallback(core::Call& call) {

    CallADIOSData* cad = dynamic_cast<CallADIOSData*>(&call);
    if (cad == nullptr) return false;

    core::moldyn::MultiParticleDataCall* mpdc = this->mpSlot.CallAs<core::moldyn::MultiParticleDataCall>();
    if (mpdc == nullptr) return false;

    if (!(*mpdc)(1)) return false;
    if (!(*mpdc)(0)) return false;

    core::moldyn::MultiParticleDataCall::Particles& parts = mpdc->AccessParticles(0);

    // get total frame cound from data source
    cad->setFrameCount(mpdc->FrameCount());
    // DEBUG
    //cad->setFrameCount(1);

    // set available vars
    std::vector<std::string> availVars = {"box", "p_count"};

    if (this->orderSlot.Param<core::param::EnumParam>()->Value() == 0) {
        availVars.push_back("x");
        availVars.push_back("y");
        availVars.push_back("z");
    } else {
        availVars.push_back("xyz");
    }

    if (parts.GetColourDataType() != core::moldyn::SimpleSphericalParticles::COLDATA_NONE) {
        availVars.push_back("r");
        availVars.push_back("g");
        availVars.push_back("b");
        availVars.push_back("a");
    } else {
        if (mpdc->GetParticleListCount() > 1) {
            availVars.push_back("r");
            availVars.push_back("g");
            availVars.push_back("b");
            availVars.push_back("a");
        } else {
            availVars.push_back("global_r");
            availVars.push_back("global_g");
            availVars.push_back("global_b");
            availVars.push_back("global_a");
        }
    }

    if (parts.GetVertexDataType() == core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR) {
        availVars.push_back("radius");
    } else {
        if (mpdc->GetParticleListCount() > 1) {
            availVars.push_back("radius");
        } else {
            availVars.push_back("global_radius");
        }
    }

    if (parts.HasID()) {
        availVars.push_back("id");
    }

    cad->setAvailableVars(availVars);

    return true;
}

} // end namespace adios
} // end namespace megamol