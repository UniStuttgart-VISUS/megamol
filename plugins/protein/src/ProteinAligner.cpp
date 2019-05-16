/*
 * ProteinAligner.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * Author: Karsten Schatz
 * All rights reserved.
 */

#include "stdafx.h"
#include "ProteinAligner.h"
#include "RMS.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "mmcore/param/BoolParam.h"

using namespace megamol;
using namespace megamol::protein;
using namespace megamol::protein_calls;

/*
 * ProteinAligner::ProteinAligner
 */
ProteinAligner::ProteinAligner(void)
    : core::Module()
    , dataOutSlot("dataOut", "Output protein slot")
    , inputProteinSlot("inputProtein", "Input protein that will be moved and rotated to match the reference")
    , referenceProteinSlot("referenceProtein", "Reference protein slot")
    , isActiveSlot("isActive", "Activates and deactivates the effect of this module") {

    // callee slot
    this->dataOutSlot.SetCallback(
        MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(0), &ProteinAligner::getData);
    this->dataOutSlot.SetCallback(
        MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(1), &ProteinAligner::getExtents);
    this->MakeSlotAvailable(&this->dataOutSlot);

    // caller slots
    this->inputProteinSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->inputProteinSlot);

    this->referenceProteinSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->referenceProteinSlot);

    // param slots
    this->isActiveSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->isActiveSlot);
}

/*
 * ProteinAligner::~ProteinAligner
 */
ProteinAligner::~ProteinAligner(void) { this->Release(); }

/*
 * ProteinAligner::create
 */
bool ProteinAligner::create(void) {
    // intentionally empty
    return true;
}

/*
 * ProteinAligner::release
 */
void ProteinAligner::release(void) {
    // intentionally empty
}

/*
 * ProteinAligner::getData
 */
bool ProteinAligner::getData(core::Call& call) {
    MolecularDataCall* out = dynamic_cast<MolecularDataCall*>(&call);
    if (out == nullptr) return false;

    if (this->isActiveSlot.Param<core::param::BoolParam>()->Value() && this->alignedPositions.data() != nullptr) {
        out->SetAtomPositions(this->alignedPositions.data());
        out->AccessBoundingBoxes().Clear();
        out->AccessBoundingBoxes().SetObjectSpaceBBox(this->boundingBox);
        out->AccessBoundingBoxes().SetObjectSpaceClipBox(this->boundingBox);
    }

    return true;
}

/*
 * ProteinAligner::getExtents
 */
bool ProteinAligner::getExtents(core::Call& call) {
    MolecularDataCall* out = dynamic_cast<MolecularDataCall*>(&call);
    if (out == nullptr) return false;

    MolecularDataCall* input = this->inputProteinSlot.CallAs<MolecularDataCall>();
    if (input == nullptr) return false;

    MolecularDataCall* ref = this->referenceProteinSlot.CallAs<MolecularDataCall>();
    if (ref == nullptr) return false;

    // call all the data
    if (!(*input)(1)) return false;
    if (!(*input)(0)) return false;

    if (!(*ref)(1)) return false;
    if (!(*ref)(0)) return false;

    out->operator=(*input); // deep copy

    this->alignPositions(*input, *ref);

    if (this->isActiveSlot.Param<core::param::BoolParam>()->Value() && this->alignedPositions.data() != nullptr) {
        out->AccessBoundingBoxes().Clear();
        out->AccessBoundingBoxes().SetObjectSpaceBBox(this->boundingBox);
        out->AccessBoundingBoxes().SetObjectSpaceClipBox(this->boundingBox);
    }

    return true;
}

/*
 * ProteinAligner::alignPositions
 */
bool ProteinAligner::alignPositions(const MolecularDataCall& input, const MolecularDataCall& ref) {
    std::vector<float> inputCAlphas, refCAlphas;
    this->getCAlphaPosList(input, inputCAlphas);
    this->getCAlphaPosList(ref, refCAlphas);
    auto atomCount = std::min(inputCAlphas.size() / 3, refCAlphas.size() / 3);

    std::vector<float> mass(atomCount, 1.0f);
    std::vector<int> mask(atomCount, 1);
    float rotation[3][3], translation[3];
    auto result = CalculateRMS(static_cast<unsigned int>(atomCount), true, 2, mass.data(), mask.data(),
        inputCAlphas.data(), refCAlphas.data(), rotation, translation);

    this->alignedPositions.resize(static_cast<size_t>(input.AtomCount()) * 3);
    std::memcpy(this->alignedPositions.data(), input.AtomPositions(), this->alignedPositions.size() * sizeof(float));

    glm::mat3 rotmat = glm::mat3(glm::make_vec3(rotation[0]), glm::make_vec3(rotation[1]), glm::make_vec3(rotation[2]));
    glm::vec3 trans = glm::make_vec3(translation);

    // the rotation and translation has to be performed in a two step process as there are some problems with the values
    // we have now

    // the rotation matrix is fixed by transposing it:
    rotmat = glm::transpose(rotmat);

    for (size_t i = 0; i < input.AtomCount(); ++i) {
        glm::vec3 pos = glm::make_vec3(&this->alignedPositions[3 * i]);
        pos = rotmat * pos;
        std::memcpy(&this->alignedPositions[3 * i], &pos.x, sizeof(float) * 3);
    }

    // as the resulting transpose vector of the CalculateRMS-method is complete and utter bullcrap, we calculate the
    // midpoint of each of the two proteins and move the onto each other
    glm::vec3 inputCenter(0.0f, 0.0f, 0.0f), refCenter(0.0f, 0.0f, 0.0f);
    for (size_t i = 0; i < input.AtomCount(); ++i) {
        glm::vec3 pos = glm::make_vec3(&alignedPositions[3 * i]);
        inputCenter += pos;
    }
    for (size_t i = 0; i < ref.AtomCount(); ++i) {
        glm::vec3 pos = glm::make_vec3(&ref.AtomPositions()[3 * i]);
        refCenter += pos;
    }
    inputCenter /= static_cast<float>(input.AtomCount());
    refCenter /= static_cast<float>(ref.AtomCount());
    this->boundingBox.Set(refCenter.x, refCenter.y, refCenter.z, refCenter.x, refCenter.y, refCenter.z);
    for (size_t i = 0; i < input.AtomCount(); ++i) {
        glm::vec3 pos = glm::make_vec3(&this->alignedPositions[3 * i]);
        pos += refCenter - inputCenter;
        std::memcpy(&this->alignedPositions[3 * i], &pos.x, sizeof(float) * 3);
        this->boundingBox.GrowToPoint(vislib::math::Point<float, 3>(pos.x, pos.y, pos.z));
    }
    this->boundingBox.Grow(3.0f);

    return true;
}

/*
 * ProteinAligner::getCAlphaPosList
 */
void ProteinAligner::getCAlphaPosList(const MolecularDataCall& input, std::vector<float>& cAlphaPositions) {
    auto resCnt = input.ResidueCount();
    cAlphaPositions.clear();
    for (unsigned int res = 0; res < resCnt; ++res) {
        MolecularDataCall::AminoAcid* amino;
        if (input.Residues()[res]->Identifier() == MolecularDataCall::Residue::AMINOACID) {
            amino = (MolecularDataCall::AminoAcid*)(input.Residues()[res]);
        } else {
            continue;
        }
        auto ca = amino->CAlphaIndex();
        cAlphaPositions.push_back(input.AtomPositions()[3 * ca + 0]);
        cAlphaPositions.push_back(input.AtomPositions()[3 * ca + 1]);
        cAlphaPositions.push_back(input.AtomPositions()[3 * ca + 2]);
    }
}
