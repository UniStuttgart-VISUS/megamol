/*
 * ls1ParticleFormat.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ls1ParticleFormat.h"
#include "adios_plugin/CallADIOSData.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include <numeric>


namespace megamol {
namespace adios {

ls1ParticleFormat::ls1ParticleFormat(void)
    : core::Module()
    , mpSlot("mpSlot", "Slot to send multi particle data.")
    , adiosSlot("adiosSlot", "Slot to request ADIOS IO")
    , representationSlot("representation", "Chose between displaying molecules or atoms")
    , transferfunctionSlot("transferfunctionSlot", "") {

    this->mpSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &ls1ParticleFormat::getDataCallback);
    this->mpSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &ls1ParticleFormat::getExtentCallback);
    this->MakeSlotAvailable(&this->mpSlot);

    this->adiosSlot.SetCompatibleCall<CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->adiosSlot);
    this->adiosSlot.SetNecessity(megamol::core::AbstractCallSlotPresentation::SLOT_REQUIRED);

    core::param::EnumParam* displayEnum = new core::param::EnumParam(0);
    displayEnum->SetTypePair(0, "Molecules");
    displayEnum->SetTypePair(1, "Atoms");
    this->representationSlot << displayEnum;
    this->representationSlot.SetUpdateCallback(&ls1ParticleFormat::representationChanged);
    this->MakeSlotAvailable(&this->representationSlot);


    this->transferfunctionSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->transferfunctionSlot);
    this->transferfunctionSlot.SetNecessity(megamol::core::AbstractCallSlotPresentation::SLOT_REQUIRED);

}

ls1ParticleFormat::~ls1ParticleFormat(void) { this->Release(); }

bool ls1ParticleFormat::create(void) { return true; }

void ls1ParticleFormat::release(void) {}

bool ls1ParticleFormat::getDataCallback(core::Call& call) {
    core::moldyn::MultiParticleDataCall* mpdc = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
    if (mpdc == nullptr) return false;

    CallADIOSData* cad = this->adiosSlot.CallAs<CallADIOSData>();
    if (cad == nullptr) return false;

    if (!(*cad)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[ls1ParticleFormat]: Error during GetHeader");
        return false;
    }
    bool datahashChanged = (mpdc->DataHash() != cad->getDataHash());
    if ((mpdc->FrameID() != currentFrame) || datahashChanged || representationDirty) {
        representationDirty = false;

        cad->setFrameIDtoLoad(mpdc->FrameID());

        try {
            auto availAttributes = cad->getAvailableAttributes();
            for (auto attr : availAttributes) {
                cad->inquireAttr(attr);
            }

            auto availVars = cad->getAvailableVars();
            cad->inquireVar("rx");
            cad->inquireVar("ry");
            cad->inquireVar("rz");
            cad->inquireVar("component_id");
            cad->inquireVar("qw");
            cad->inquireVar("qx");
            cad->inquireVar("qy");
            cad->inquireVar("qz");

            cad->inquireVar("global_box");

            if (!(*cad)(0)) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("[ls1ParticleFormat]: Error during GetData");
                return false;
            }

            auto qw = cad->getData("qw")->GetAsDouble();
            auto qx = cad->getData("qx")->GetAsDouble();
            auto qy = cad->getData("qy")->GetAsDouble();
            auto qz = cad->getData("qz")->GetAsDouble();

            auto comp_id = cad->getData("component_id")->GetAsUInt64();

            stride = 0;
            auto X = cad->getData("rx")->GetAsUChar();
            auto Y = cad->getData("ry")->GetAsUChar();
            auto Z = cad->getData("rz")->GetAsUChar();
            stride += 3 * cad->getData("rx")->getTypeSize();
            uint64_t p_count = X.size() / cad->getData("rx")->getTypeSize();

            int pos_size = 0;
            if (cad->getData("rx")->getTypeSize() == 4) {
                vertType = core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ;
                pos_size = sizeof(float);
            } else {
                vertType = core::moldyn::SimpleSphericalParticles::VERTDATA_DOUBLE_XYZ;
                pos_size = sizeof(double);
            }
            bbox = cad->getData("global_box")->GetAsFloat();


            int num_atoms_total = 0;
            auto num_components = cad->getData("num_components")->GetAsInt32()[0];
            std::vector<int> atoms_per_component(num_components);
            std::vector<int> component_offset(num_components);
            std::vector<double> comp_sigmas(num_components);
            std::vector<std::vector<double>> comp_centers(num_components);
            for (int n = 0; n < num_components; ++n) {
                std::string sigma_string = std::string("component_") + std::to_string(n) + std::string("_sigma");
                comp_sigmas[n] = cad->getData(sigma_string)->GetAsDouble()[0];

                std::string centers_string = std::string("component_") + std::to_string(n) + std::string("_centers");
                comp_centers[n] = cad->getData(centers_string)->GetAsDouble();
                component_offset[n] = num_atoms_total;
                num_atoms_total += comp_centers[n].size() / 3;
                atoms_per_component[n] = comp_centers[n].size() / 3;
            }

            plist_count.clear();
            mix.clear();
            num_plists = 0;
            if (this->representationSlot.Param<core::param::EnumParam>()->Value() == 0) {
                num_plists = num_components;
                mix.resize(num_plists);
                plist_count.resize(num_plists,0);

                for (int i = 0; i < p_count; ++i) {
                    mix[comp_id[i]].insert(
                        mix[comp_id[i]].end(), X.begin() + pos_size * i, X.begin() + pos_size * (i + 1));
                    mix[comp_id[i]].insert(
                        mix[comp_id[i]].end(), Y.begin() + pos_size * i, Y.begin() + pos_size * (i + 1));
                    mix[comp_id[i]].insert(
                        mix[comp_id[i]].end(), Z.begin() + pos_size * i, Z.begin() + pos_size * (i + 1));
                    plist_count[comp_id[i]] += 1;
                }

            } else {
                num_plists = num_atoms_total;
                mix.resize(num_plists);
                plist_count.resize(num_plists, 0);

                for (int i = 0; i < p_count; ++i) {

                    for (int j = 0; j < atoms_per_component[comp_id[i]]; ++j) {

                        if (pos_size  == sizeof(float)) {
                            auto com_x = static_cast<float>(X[pos_size * i]);
                            auto com_y = static_cast<float>(Y[pos_size * i]);
                            auto com_z = static_cast<float>(Z[pos_size * i]);
                            auto a_x = comp_centers[comp_id[i]][3 * j + 0];
                            auto a_y = comp_centers[comp_id[i]][3 * j + 1];
                            auto a_z = comp_centers[comp_id[i]][3 * j + 2];

                            auto pos = calcAtomPos(com_x, com_y, com_z, a_x, a_y, a_z, qw[i], qx[i], qy[i], qz[i]);
                            auto uchar_pos = reinterpret_cast<std::vector<unsigned char>&>(pos);
                            mix[component_offset[comp_id[i]] + j].insert(mix[component_offset[comp_id[i]] + j].end(), uchar_pos.begin(), uchar_pos.end());
                        } else {
                            auto com_x = static_cast<double>(X[pos_size * i]);
                            auto com_y = static_cast<double>(Y[pos_size * i]);
                            auto com_z = static_cast<double>(Z[pos_size * i]);
                            auto a_x = comp_centers[comp_id[i]][3 * j + 0];
                            auto a_y = comp_centers[comp_id[i]][3 * j + 1];
                            auto a_z = comp_centers[comp_id[i]][3 * j + 2];

                            auto pos = calcAtomPos(com_x, com_y, com_z, a_x, a_y, a_z, qw[i], qx[i], qy[i], qz[i]);
                            auto uchar_pos = reinterpret_cast<std::vector<unsigned char>&>(pos);
                            mix[component_offset[comp_id[i]] + j].insert(mix[component_offset[comp_id[i]] + j].end(), uchar_pos.begin(), uchar_pos.end());
                        }
                        plist_count[component_offset[comp_id[i]] + j] += 1;
                    }
                }
            }
        } catch (std::exception ex) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ls1ParticleFormat]: exception while trying to use data: %s", ex.what());
        }
    }

    // set number of particle lists
    mpdc->SetParticleListCount(num_plists);
    // Set bounding box
    const vislib::math::Cuboid<float> cubo(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
    mpdc->AccessBoundingBoxes().SetObjectSpaceBBox(cubo);
    mpdc->AccessBoundingBoxes().SetObjectSpaceClipBox(cubo);

    // transferfunction stuff
    core::view::CallGetTransferFunction* ctf = transferfunctionSlot.CallAs<core::view::CallGetTransferFunction>();
    if (ctf != nullptr) {
        std::array<float, 2> range = {0, num_plists};
        ctf->SetRange(range);
        if (!(*ctf)()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ls1ParticleFormat]: Error in transfer function callback." );
            return false;
        }
        if (ctf->GetTextureData() != nullptr) {
            auto texture_size = ctf->TextureSize();
            auto texture_step = (texture_size - 1) / (num_plists - 1);
            list_colors.resize(num_plists);
            for (int i = 0; i < num_plists; ++i) {
                ctf->CopyColor(i * texture_step, list_colors[i].data(), 4 * sizeof(float));
            }
        }
    }

    for (auto k = 0; k < mix.size(); k++) {
        if (!list_colors.empty()) {
            mpdc->AccessParticles(k).SetGlobalColour(
                list_colors[k][0] * 255, list_colors[k][1] * 255, list_colors[k][2] * 255, list_colors[k][3] * 255);
        } else {
            auto step = 255 / (num_plists - 1);
            mpdc->AccessParticles(k).SetGlobalColour(k * step, k * step, k * step);
        }
        // Set particles
        mpdc->AccessParticles(k).SetCount(plist_count[k]);

        mpdc->AccessParticles(k).SetVertexData(vertType, mix[k].data(), stride);

        // add id and velocity?
        //         mpdc->AccessParticles(k).SetColourData(
        //    colType, mix[k].data() + core::moldyn::SimpleSphericalParticles::VertexDataSize[vertType], stride);
        //mpdc->AccessParticles(k).SetIDData(idType,
        //    mix[k].data() + core::moldyn::SimpleSphericalParticles::VertexDataSize[vertType] +
        //        core::moldyn::SimpleSphericalParticles::ColorDataSize[colType],
        //    stride);

    }

    mpdc->SetFrameCount(cad->getFrameCount());
    mpdc->SetDataHash(cad->getDataHash());
    currentFrame = mpdc->FrameID();

    return true;
}

bool ls1ParticleFormat::getExtentCallback(core::Call& call) {

    core::moldyn::MultiParticleDataCall* mpdc = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
    if (mpdc == nullptr) return false;

    CallADIOSData* cad = this->adiosSlot.CallAs<CallADIOSData>();
    if (cad == nullptr) return false;

    if (!this->getDataCallback(call)) return false;

    return true;
}

bool ls1ParticleFormat::representationChanged(core::param::ParamSlot& p) {
    representationDirty = true;
    return true;
}

} // end namespace adios
} // end namespace megamol
