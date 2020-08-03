/*
 * ParticleInstantiator.cpp
 *
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "ParticleInstantiator.h"
#include "mmcore/param/Vector3fParam.h"
#include "glm/glm.hpp"

using namespace megamol;
using namespace megamol::stdplugin;


datatools::ParticleInstantiator::ParticleInstantiator(void)
        : AbstractParticleManipulator("outData", "indata")
    , numInstancesParam("instances", "number of dataset replications in X, Y, Z direction")
    , instanceOffsetParam("instOffset", "offset per instance in X, Y, Z") {

    this->numInstancesParam << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 1.0f, 1.0f), vislib::math::Vector<float, 3>(1.0f, 1.0f, 1.0f));
    this->MakeSlotAvailable(&this->numInstancesParam);

    this->instanceOffsetParam << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->MakeSlotAvailable(&this->instanceOffsetParam);
}

datatools::ParticleInstantiator::~ParticleInstantiator(void) {
    this->Release();
}

bool datatools::ParticleInstantiator::InterfaceIsDirty() {
    return this->numInstancesParam.IsDirty() || this->instanceOffsetParam.IsDirty();
}

void datatools::ParticleInstantiator::InterfaceResetDirty() {
    this->numInstancesParam.ResetDirty();
    this->instanceOffsetParam.ResetDirty();
}

bool datatools::ParticleInstantiator::manipulateData(
        megamol::core::moldyn::MultiParticleDataCall& outData,
        megamol::core::moldyn::MultiParticleDataCall& inData) {
    using megamol::core::moldyn::MultiParticleDataCall;

    outData = inData; // also transfers the unlocker to 'outData'
    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    unsigned int plc = inData.GetParticleListCount();
    auto ni_temp = this->numInstancesParam.Param<core::param::Vector3fParam>()->Value();
    glm::ivec3 numInstances = glm::ivec3(static_cast<int>(ni_temp.X()), static_cast<int>(ni_temp.Y()), static_cast<int>(ni_temp.Z()));
    auto io_temp = this->instanceOffsetParam.Param<core::param::Vector3fParam>()->Value();
    glm::vec3 instOffsets = glm::vec3(io_temp.X(), io_temp.Y(), io_temp.Z());
    int numTotalInstances = numInstances.x * numInstances.y * numInstances.z;

    if (InterfaceIsDirty() || (hash != inData.DataHash()) || (inData.DataHash() == 0) || (frameID != inData.FrameID())) {
        // Update data
        hash = inData.DataHash();
        frameID = inData.FrameID();
        InterfaceResetDirty();

        vertData.resize(plc);
        colData.resize(plc);
        dirData.resize(plc);
        for (auto i = 0; i < plc; ++i) {
            // first copy everything into a defined format
            // XYZR, RGBA, VXVYVZ, no IDs (might waste space for radii, but we are wasting space here bigtime anyway)
            const auto& p = inData.AccessParticles(i);
            const auto& s = p.GetParticleStore();

            vertData[i].resize(p.GetCount() * 4 * numTotalInstances);
            colData[i].resize(p.GetCount() * 4 * numTotalInstances);
            dirData[i].resize(p.GetCount() * 3 * numTotalInstances);

            const auto& xacc = s.GetXAcc();
            const auto& yacc = s.GetYAcc();
            const auto& zacc = s.GetZAcc();
            const auto& racc = s.GetRAcc();
            const auto& cracc = s.GetCRAcc();
            const auto& cgacc = s.GetCGAcc();
            const auto& cbacc = s.GetCBAcc();
            const auto& caacc = s.GetCAAcc();

            for (uint64_t j = 0; j < p.GetCount(); ++j) {
                //vertData[i][j * 4 + 0] = xacc->Get_f(j);
                //vertData[i][j * 4 + 1] = yacc->Get_f(j);
                //vertData[i][j * 4 + 2] = zacc->Get_f(j);
                //vertData[i][j * 4 + 3] = racc->Get_f(j);

                colData[i][j * 4 + 0] = cracc->Get_f(j) * 255;
                colData[i][j * 4 + 1] = cgacc->Get_f(j) * 255;
                colData[i][j * 4 + 2] = cbacc->Get_f(j) * 255;
                colData[i][j * 4 + 3] = caacc->Get_f(j) * 255;
            }

            if (p.GetDirData() != nullptr) {
                const auto& dxacc = s.GetDXAcc();
                const auto& dyacc = s.GetDYAcc();
                const auto& dzacc = s.GetDZAcc();

                for (uint64_t j = 0; j < p.GetCount(); ++j) {
                    dirData[i][j * 3 + 0] = dxacc->Get_f(j);
                    dirData[i][j * 3 + 1] = dyacc->Get_f(j);
                    dirData[i][j * 3 + 2] = dzacc->Get_f(j);
                }
            }

            // then make copies of that data
            for (auto instX = 0; instX < numInstances.x; instX++) {
                for (auto instY = 0; instY < numInstances.y; instY++) {
                    for (auto instZ = 0; instZ < numInstances.z; instZ++) {
                        const auto instSize = p.GetCount() * 4;
                        const auto offset = (instSize * instX) + (instSize * numInstances.x * instY) + (instSize * numInstances.x * numInstances.y * instZ);
                        for (uint64_t j = 0; j < p.GetCount(); ++j) {
                            vertData[i][offset + j * 4 + 0] = xacc->Get_f(j) + instOffsets.x * static_cast<float>(instX);
                            vertData[i][offset + j * 4 + 1] = yacc->Get_f(j) + instOffsets.y * static_cast<float>(instY);
                            vertData[i][offset + j * 4 + 2] = zacc->Get_f(j) + instOffsets.z * static_cast<float>(instZ);
                            vertData[i][offset + j * 4 + 3] = racc->Get_f(j);
                        }
                        if (instX + instY + instZ > 0) {
                            const auto colInstSize = p.GetCount() * 4;
                            const auto coloffset = (colInstSize * instX) + (colInstSize * numInstances.x * instY) + (colInstSize * numInstances.x * numInstances.y * instZ);
                            memcpy(&colData[i][coloffset], &colData[i][0], colInstSize * sizeof(uint8_t));

                            if (p.GetDirData() != nullptr) {
                                const auto dirInstSize = p.GetCount() * 3;
                                const auto diroffset = (dirInstSize * instX) + (dirInstSize * numInstances.x * instY) + (dirInstSize * numInstances.x * numInstances.y * instZ);
                                memcpy(&dirData[i][diroffset], &dirData[i][0], dirInstSize * sizeof(float));
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto i = 0; i < plc; ++i) {
        const auto& p = inData.AccessParticles(i);
        outData.AccessParticles(i).SetCount(p.GetCount() * numTotalInstances);
        outData.AccessParticles(i).SetVertexData(core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR, vertData[i].data(), 0);
        outData.AccessParticles(i).SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_UINT8_RGBA, colData[i].data(), 0);
        if (p.GetDirData() != nullptr) {
            outData.AccessParticles(i).SetDirData(core::moldyn::SimpleSphericalParticles::DIRDATA_FLOAT_XYZ, dirData[i].data(), 0);
        } else {
            outData.AccessParticles(i).SetDirData(core::moldyn::SimpleSphericalParticles::DIRDATA_NONE, nullptr, 0);
        }
    }

    return true;
}

bool datatools::ParticleInstantiator::manipulateExtent(core::moldyn::MultiParticleDataCall &outData,
    core::moldyn::MultiParticleDataCall &inData) {

    auto ni_temp = this->numInstancesParam.Param<core::param::Vector3fParam>()->Value();
    glm::ivec3 numInstances = glm::ivec3(static_cast<int>(ni_temp.X()), static_cast<int>(ni_temp.Y()), static_cast<int>(ni_temp.Z()));
    auto io_temp = this->instanceOffsetParam.Param<core::param::Vector3fParam>()->Value();
    glm::vec3 instOffsets = glm::vec3(io_temp.X(), io_temp.Y(), io_temp.Z());
    int numTotalInstances = numInstances.x * numInstances.y * numInstances.z;

    auto obbox = inData.GetBoundingBoxes().ObjectSpaceBBox();
    auto ocbox = inData.GetBoundingBoxes().ObjectSpaceClipBox();
    obbox.SetRight(obbox.Left() + (numInstances.x - 1) * instOffsets.x + obbox.Width());
    ocbox.SetRight(ocbox.Left() + (numInstances.x - 1) * instOffsets.x + ocbox.Width()); // TODO + some radius, meh
    obbox.SetTop(obbox.Bottom() + (numInstances.y - 1) * instOffsets.y + obbox.Height());
    ocbox.SetTop(ocbox.Bottom() + (numInstances.y - 1) * instOffsets.y + ocbox.Height()); // TODO + some radius, meh
    obbox.SetFront(obbox.Back() + (numInstances.z - 1) * instOffsets.z + obbox.Depth());
    ocbox.SetFront(ocbox.Back() + (numInstances.z - 1) * instOffsets.z + ocbox.Depth()); // TODO + some radius, meh
    core::BoundingBoxes boxes;
    boxes.SetObjectSpaceBBox(obbox);
    boxes.SetObjectSpaceClipBox(ocbox);
    outData.SetExtent(inData.FrameCount(), boxes);

    return true;
}
