/*
 * ls1ParticleFormat.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "ls1ParticleFormat.h"
#include "MinSphereWrapper.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmadios/CallADIOSData.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmstd/renderer/CallGetTransferFunction.h"
#include <numeric>


namespace megamol::adios {

ls1ParticleFormat::ls1ParticleFormat()
        : core::Module()
        , mpSlot("mpSlot", "Slot to send multi particle data.")
        , adiosSlot("adiosSlot", "Slot to request ADIOS IO")
        , representationSlot("representation", "Chose between displaying molecules or atoms")
        , forceFloatSlot("force float", "")
        , transferfunctionSlot("transferfunctionSlot", "") {

    this->mpSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(0), &ls1ParticleFormat::getDataCallback);
    this->mpSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(1), &ls1ParticleFormat::getExtentCallback);
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

    forceFloatSlot << new core::param::BoolParam(false);
    MakeSlotAvailable(&forceFloatSlot);


    this->transferfunctionSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->transferfunctionSlot);
    this->transferfunctionSlot.SetNecessity(megamol::core::AbstractCallSlotPresentation::SLOT_REQUIRED);
}

ls1ParticleFormat::~ls1ParticleFormat() {
    this->Release();
}

bool ls1ParticleFormat::create() {
    return true;
}

void ls1ParticleFormat::release() {}

bool ls1ParticleFormat::getDataCallback(core::Call& call) {
    geocalls::MultiParticleDataCall* mpdc = dynamic_cast<geocalls::MultiParticleDataCall*>(&call);
    if (mpdc == nullptr)
        return false;

    CallADIOSData* cad = this->adiosSlot.CallAs<CallADIOSData>();
    if (cad == nullptr)
        return false;

    if (!(*cad)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[ls1ParticleFormat]: Error during GetHeader");
        return false;
    }
    bool datahashChanged = (datahash != cad->getDataHash());
    if ((mpdc->FrameID() != currentFrame) || datahashChanged || representationDirty) {
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
            cad->inquireVar("molecule_id");
            cad->inquireVar("vx");
            cad->inquireVar("vy");
            cad->inquireVar("vz");
            if (cad->isInVars("qw")) {
                cad->inquireVar("qw");
                cad->inquireVar("qx");
                cad->inquireVar("qy");
                cad->inquireVar("qz");
            }

            cad->inquireVar("global_box");

            if (!(*cad)(0)) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("[ls1ParticleFormat]: Error during GetData");
                return false;
            }

            std::vector<float> qw;
            std::vector<float> qx;
            std::vector<float> qy;
            std::vector<float> qz;

            if (cad->isInVars("qw")) {
                qw = cad->getData("qw")->GetAsFloat();
                qx = cad->getData("qx")->GetAsFloat();
                qy = cad->getData("qy")->GetAsFloat();
                qz = cad->getData("qz")->GetAsFloat();
            }


            auto comp_id = cad->getData("component_id")->GetAsUInt64();
            auto m_id = cad->getData("molecule_id")->GetAsUInt64();

            auto X = cad->getData("rx")->GetAsFloat();
            auto Y = cad->getData("ry")->GetAsFloat();
            auto Z = cad->getData("rz")->GetAsFloat();
            uint64_t p_count = X.size();

            auto VX = cad->getData("vx")->GetAsFloat();
            auto VY = cad->getData("vy")->GetAsFloat();
            auto VZ = cad->getData("vz")->GetAsFloat();

            int pos_size = 0;
            int dir_size = 0;

            bbox = cad->getData("global_box")->GetAsFloat();

            int num_atoms_total = 0;
            auto num_components = cad->getData("num_components")->GetAsInt32()[0];
            std::vector<int> atoms_per_component(num_components);
            std::vector<int> component_offset(num_components);
            std::vector<std::vector<float>> comp_sigmas(num_components);
            std::vector<std::vector<float>> comp_centers(num_components);
            std::vector<std::vector<std::string>> comp_element_names(num_components);
            for (int n = 0; n < num_components; ++n) {
                std::string sigma_string = std::string("component_") + std::to_string(n) + std::string("_sigma");
                comp_sigmas[n] = cad->getData(sigma_string)->GetAsFloat();
                std::string element_string =
                    std::string("component_") + std::to_string(n) + std::string("_element_names");
                auto element_name_data = cad->getData(element_string)->GetAsString()[0];
                comp_element_names[n] = splitElementString(element_name_data);

                std::string centers_string = std::string("component_") + std::to_string(n) + std::string("_centers");
                comp_centers[n] = cad->getData(centers_string)->GetAsFloat();
                component_offset[n] = num_atoms_total;
                num_atoms_total += comp_centers[n].size() / 3;
                atoms_per_component[n] = comp_centers[n].size() / 3;
            }

            plist_count.clear();
            mix.clear();
            dirs.clear();
            num_plists = 0;
            if (this->representationSlot.Param<core::param::EnumParam>()->Value() == 0 || qw.empty()) {
                num_plists = num_components;
                mix.clear();
                mix.resize(num_plists);
                dirs.clear();
                dirs.resize(num_plists);
                ids.clear();
                ids.resize(num_plists);
                list_radii.clear();
                list_radii.resize(num_plists);
                plist_count.resize(num_plists, 0);

                for (uint64_t pl_idx = 0; pl_idx < num_plists; ++pl_idx) {
                    auto const count = std::count(comp_id.begin(), comp_id.end(), pl_idx);
                    plist_count[pl_idx] = count;

                    mix[pl_idx].reserve(count * 3);
                    dirs[pl_idx].reserve(count * 3);
                    ids[pl_idx].reserve(count);
                }

                for (int i = 0; i < p_count; ++i) {
                    mix[comp_id[i]].push_back(X[i]);
                    mix[comp_id[i]].push_back(Y[i]);
                    mix[comp_id[i]].push_back(Z[i]);
                    dirs[comp_id[i]].push_back(VX[i]);
                    dirs[comp_id[i]].push_back(VY[i]);
                    dirs[comp_id[i]].push_back(VZ[i]);
                    ids[comp_id[i]].push_back(m_id[i]);
                }

                // calc circumsphere
                for (int j = 0; j < num_components; ++j) {
                    std::vector<double> spheres;
                    spheres.reserve(comp_sigmas[j].size() * 4);
                    for (int n = 0; n < comp_sigmas[j].size(); ++n) {
                        spheres.push_back(comp_centers[j][3 * n + 0]);
                        spheres.push_back(comp_centers[j][3 * n + 1]);
                        spheres.push_back(comp_centers[j][3 * n + 2]);
                        spheres.push_back(comp_sigmas[j][n]);
                    }
                    auto minSphere = getMinSphere(spheres);
                    list_radii[j] = minSphere[3];
                }


            } else {
                num_plists = num_atoms_total;
                mix.clear();
                mix.resize(num_plists);
                dirs.clear();
                dirs.resize(num_plists);
                ids.clear();
                ids.resize(num_plists);
                list_radii.clear();
                list_radii.reserve(num_plists);
                plist_count.resize(num_plists, 0);

                for (uint64_t pl_idx = 0; pl_idx < num_plists; ++pl_idx) {
                    auto const count = std::count(comp_id.begin(), comp_id.end(), pl_idx);
                    plist_count[pl_idx] = count;

                    mix[pl_idx].reserve(count * 3);
                    dirs[pl_idx].reserve(count * 3);
                    ids[pl_idx].reserve(count);
                }

                for (int j = 0; j < comp_sigmas.size(); ++j) {
                    for (int k = 0; k < comp_sigmas[j].size(); ++k) {
                        list_radii.emplace_back(comp_sigmas[j][k] * 0.5);
                    }
                }


                for (int i = 0; i < p_count; ++i) {

                    for (int j = 0; j < atoms_per_component[comp_id[i]]; ++j) {
                        auto com_x = X[i];
                        auto com_y = Y[i];
                        auto com_z = Z[i];
                        auto a_x = comp_centers[comp_id[i]][3 * j + 0];
                        auto a_y = comp_centers[comp_id[i]][3 * j + 1];
                        auto a_z = comp_centers[comp_id[i]][3 * j + 2];

                        auto pos = calcAtomPos(com_x, com_y, com_z, a_x, a_y, a_z, qw[i], qx[i], qy[i], qz[i]);
                        mix[component_offset[comp_id[i]] + j].insert(
                            mix[component_offset[comp_id[i]] + j].end(), pos.begin(), pos.end());

                        dirs[component_offset[comp_id[i]] + j].push_back(VX[i]);
                        dirs[component_offset[comp_id[i]] + j].push_back(VY[i]);
                        dirs[component_offset[comp_id[i]] + j].push_back(VZ[i]);
                        ids[component_offset[comp_id[i]] + j].push_back(m_id[i]);
                    }
                }
            }
        } catch (std::exception ex) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ls1ParticleFormat]: exception while trying to use data: %s", ex.what());
        }
        if (datahashChanged || representationDirty)
            version++;
        representationDirty = false;
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
        std::array<float, 2> range = {0.0f, static_cast<float>(num_plists - 1)};
        ctf->SetRange(range);
        if (!(*ctf)()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ls1ParticleFormat]: Error in transfer function callback.");
            return false;
        }
        auto tf_dirty = ctf->IsDirty();
        if (tf_dirty && ctf->GetTextureData() != nullptr) {
            auto texture_size = ctf->TextureSize();
            auto texture_step = num_plists > 1 ? (texture_size - 1) / (num_plists - 1) : texture_size;
            list_colors.resize(num_plists);
            for (int i = 0; i < num_plists; ++i) {
                ctf->CopyColor(i * texture_step, list_colors[i].data(), 4 * sizeof(float));
            }
            version++;
            ctf->ResetDirty();
        }
    }

    for (auto k = 0; k < mix.size(); k++) {
        auto& parts = mpdc->AccessParticles(k);
        if (!list_colors.empty()) {
            parts.SetGlobalColour(
                list_colors[k][0] * 255, list_colors[k][1] * 255, list_colors[k][2] * 255, list_colors[k][3] * 255);
        } else {
            auto step = 255 / (num_plists - 1);
            parts.SetGlobalColour(k * step, k * step, k * step);
        }
        // Set particles
        parts.SetCount(plist_count[k]);

        if (mix[k].data() == nullptr) {
            parts.SetVertexData(geocalls::SimpleSphericalParticles::VERTDATA_NONE, nullptr);
        } else {
            parts.SetVertexData(geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, mix[k].data());
        }

        if (dirs[k].data() == nullptr) {
            parts.SetDirData(geocalls::SimpleSphericalParticles::DIRDATA_NONE, nullptr);
        } else {
            parts.SetDirData(geocalls::SimpleSphericalParticles::DIRDATA_FLOAT_XYZ, dirs[k].data());
        }

        if (ids[k].data() == nullptr) {
            parts.SetIDData(geocalls::SimpleSphericalParticles::IDDATA_NONE, nullptr);
        } else {
            parts.SetIDData(geocalls::SimpleSphericalParticles::IDDATA_UINT64, ids[k].data());
        }
        parts.SetGlobalRadius(list_radii[k]);

        // add id and velocity?
        //         mpdc->AccessParticles(k).SetColourData(
        //    colType, mix[k].data() + geocalls::SimpleSphericalParticles::VertexDataSize[vertType], stride);
        //mpdc->AccessParticles(k).SetIDData(idType,
        //    mix[k].data() + geocalls::SimpleSphericalParticles::VertexDataSize[vertType] +
        //        geocalls::SimpleSphericalParticles::ColorDataSize[colType],
        //    stride);
    }

    mpdc->SetFrameCount(cad->getFrameCount());
    mpdc->SetDataHash(version);
    currentFrame = mpdc->FrameID();
    datahash = cad->getDataHash();

    return true;
}

bool ls1ParticleFormat::getExtentCallback(core::Call& call) {

    geocalls::MultiParticleDataCall* mpdc = dynamic_cast<geocalls::MultiParticleDataCall*>(&call);
    if (mpdc == nullptr)
        return false;

    CallADIOSData* cad = this->adiosSlot.CallAs<CallADIOSData>();
    if (cad == nullptr)
        return false;

    if (!this->getDataCallback(call))
        return false;

    return true;
}

std::vector<std::string> ls1ParticleFormat::splitElementString(std::string& elements) {
    std::stringstream input;
    input << elements;
    std::string s;
    std::vector<std::string> result;
    while (std::getline(input, s, ',')) {
        result.push_back(s);
    }
    return result;
}

bool ls1ParticleFormat::representationChanged(core::param::ParamSlot& p) {
    representationDirty = true;
    return true;
}

} // namespace megamol::adios
