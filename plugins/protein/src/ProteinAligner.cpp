/*
 * ProteinAligner.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * Author: Karsten Schatz
 * All rights reserved.
 */

#include "ProteinAligner.h"
#include "protein_calls/RMSD.h"

#include "mmcore/param/BoolParam.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

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
ProteinAligner::~ProteinAligner(void) {
    this->Release();
}

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
    if (out == nullptr)
        return false;

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
    if (out == nullptr)
        return false;

    MolecularDataCall* input = this->inputProteinSlot.CallAs<MolecularDataCall>();
    if (input == nullptr)
        return false;

    MolecularDataCall* ref = this->referenceProteinSlot.CallAs<MolecularDataCall>();
    if (ref == nullptr)
        return false;

    // call all the data
    if (!(*input)(1))
        return false;
    if (!(*input)(0))
        return false;

    if (!(*ref)(1))
        return false;
    if (!(*ref)(0))
        return false;

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
    std::vector<glm::vec3> inputCAlphas, refCAlphas;
    this->getCAlphaPosList(input, inputCAlphas);
    this->getCAlphaPosList(ref, refCAlphas);
    auto atomCount = std::min(inputCAlphas.size() / 3, refCAlphas.size() / 3);

    const auto rmsres = CalculateRMSD(inputCAlphas, refCAlphas, protein_calls::RMSDMode::RMSD_CALC_MATRICES);

    this->alignedPositions.resize(static_cast<size_t>(input.AtomCount()) * 3);
    std::memcpy(this->alignedPositions.data(), input.AtomPositions(), this->alignedPositions.size() * sizeof(float));

    glm::mat3 rotmat = rmsres.rotationMatrix;
    glm::vec3 transvec = rmsres.translationVector;
    glm::vec3 refcenter = rmsres.referenceCenter;
    glm::vec3 center = rmsres.toFitCenter;

    this->boundingBox.Set(refcenter.x, refcenter.y, refcenter.z, refcenter.x, refcenter.y, refcenter.z);
    for (size_t i = 0; i < input.AtomCount(); ++i) {
        glm::vec3 pos = glm::make_vec3(&this->alignedPositions[3 * i]);
        pos -= center;
        pos = rotmat * pos;
        pos += refcenter;
        std::memcpy(&this->alignedPositions[3 * i], &pos.x, sizeof(float) * 3);
        this->boundingBox.GrowToPoint(vislib::math::Point<float, 3>(pos.x, pos.y, pos.z));
    }
    this->boundingBox.Grow(3.0f);

    return true;
}

/*
 * ProteinAligner::getCAlphaPosList
 */
void ProteinAligner::getCAlphaPosList(const MolecularDataCall& input, std::vector<glm::vec3>& cAlphaPositions) {
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
        cAlphaPositions.push_back(glm::make_vec3(&input.AtomPositions()[3 * ca]));
    }
}
