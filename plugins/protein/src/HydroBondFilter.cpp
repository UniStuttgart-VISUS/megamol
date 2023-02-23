/*
 * HydroBondFilter.cpp
 *
 * Copyright (C) 2016 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#include "HydroBondFilter.h"
#include "protein/GridNeighbourFinder.h"

#include "protein_calls/MolecularDataCall.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include "mmcore/utility/log/Log.h"
#include "vislib/math/ShallowVector.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;

/*
 * HydroBondFilter::~HydroBondFilter
 */
HydroBondFilter::HydroBondFilter()
        : Module()
        , inDataSlot("dataIn", "Molecular data source (usually PDBLoader)")
        , outDataSlot("dataOut", "The slot providing the molecular data including the generated hydrogen bonds")
        , hBondDonorAcceptorDistance(
              "Bond::hBondDonorAcceptorDistance", "Maximal distance between donor and acceptor of a hydrogen bond")
        , alphaHelixHBonds("Allowed::alphaHelixHBonds", "Shall the H-Bonds inside the alpha helices be computed?")
        , betaSheetHBonds("Allowed::betaSheetHBonds", "Shall the H-Bonds between two beta sheets be computed?")
        , otherHBonds("Allowed::otherHBonds", "Shall all other H-Bonds be computed?")
        , cAlphaHBonds("cAlphaHBonds", "Fake hydrogen bonds as bonds between two C-alpha atoms") {

    this->inDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

    this->outDataSlot.SetCallback(MolecularDataCall::ClassName(),
        MolecularDataCall::FunctionName(MolecularDataCall::CallForGetData), &HydroBondFilter::getData);
    this->outDataSlot.SetCallback(MolecularDataCall::ClassName(),
        MolecularDataCall::FunctionName(MolecularDataCall::CallForGetExtent), &HydroBondFilter::getExtent);
    this->MakeSlotAvailable(&this->outDataSlot);

    // distance between donor and acceptor of the hydrogen bonds
    this->hBondDonorAcceptorDistance.SetParameter(new param::FloatParam(10.0f, 0.0f));
    this->MakeSlotAvailable(&this->hBondDonorAcceptorDistance);

    this->alphaHelixHBonds.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->alphaHelixHBonds);

    this->betaSheetHBonds.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->betaSheetHBonds);

    this->otherHBonds.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->otherHBonds);

    this->cAlphaHBonds.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->cAlphaHBonds);

    this->lastDataHash = 0;
    this->dataHashOffset = 0;
}

/*
 * HydroBondFilter::~HydroBondFilter
 */
HydroBondFilter::~HydroBondFilter() {
    this->Release();
}

/*
 * HydroBondFilter::create
 */
bool HydroBondFilter::create() {
    return true;
}

/*
 * HydroBondFilter::fillSecStructVector
 */
void HydroBondFilter::fillSecStructVector(MolecularDataCall& mdc) {

    this->secStructPerAtom.clear();
    this->secStructPerAtom.resize(mdc.AtomCount(), MolecularDataCall::SecStructure::ElementType::TYPE_COIL);
    this->cAlphaIndicesPerAtom.clear();
    this->cAlphaIndicesPerAtom.resize(mdc.AtomCount(), 0);

    unsigned int firstResIdx = 0;
    unsigned int lastResIdx = 0;
    unsigned int firstAtomIdx = 0;
    unsigned int lastAtomIdx = 0;
    unsigned int atomTypeIdx = 0;
    unsigned int firstSecIdx = 0;
    unsigned int lastSecIdx = 0;
    unsigned int firstAAIdx = 0;
    unsigned int lastAAIdx = 0;

    unsigned int molCount = mdc.MoleculeCount();

    for (unsigned int molIdx = 0; molIdx < molCount; molIdx++) {

        MolecularDataCall::Molecule chain = mdc.Molecules()[molIdx];

        if (mdc.Residues()[chain.FirstResidueIndex()]->Identifier() != MolecularDataCall::Residue::AMINOACID) {
            continue;
        }

        firstSecIdx = chain.FirstSecStructIndex();
        lastSecIdx = firstSecIdx + chain.SecStructCount();

        for (unsigned int secIdx = firstSecIdx; secIdx < lastSecIdx; secIdx++) {

            firstAAIdx = mdc.SecondaryStructures()[secIdx].FirstAminoAcidIndex();
            lastAAIdx = firstAAIdx + mdc.SecondaryStructures()[secIdx].AminoAcidCount();

            auto secStructType = mdc.SecondaryStructures()[secIdx].Type();

            for (unsigned int aaIdx = firstAAIdx; aaIdx < lastAAIdx; aaIdx++) {

                MolecularDataCall::AminoAcid* acid;

                // is the current residue really an aminoacid?
                if (mdc.Residues()[aaIdx]->Identifier() == MolecularDataCall::Residue::AMINOACID)
                    acid = (MolecularDataCall::AminoAcid*)(mdc.Residues()[aaIdx]);
                else
                    continue;

                firstAtomIdx = acid->FirstAtomIndex();
                lastAtomIdx = firstAtomIdx + acid->AtomCount();

                auto calphaIndex = acid->CAlphaIndex();

                for (unsigned int atomIdx = firstAtomIdx; atomIdx < lastAtomIdx; atomIdx++) {
                    this->secStructPerAtom[atomIdx] = secStructType;
                    this->cAlphaIndicesPerAtom[atomIdx] = calphaIndex;
                }
            }
        }
    }
}

/*
 * HydroBondFilter::getData
 */
bool HydroBondFilter::getData(Call& call) {
    MolecularDataCall* outCall = dynamic_cast<MolecularDataCall*>(&call);
    MolecularDataCall* inCall = this->inDataSlot.CallAs<MolecularDataCall>();

    if (!outCall || !inCall) {
        return false;
    }
    inCall->SetFrameID(outCall->FrameID());
    if (!(*inCall)(MolecularDataCall::CallForGetData)) {
        return false;
    }

    *outCall = *inCall; // deep copy

    // create secondary structure arrays if the incoming data has changed
    if (lastDataHash != inCall->DataHash()) {
        this->fillSecStructVector(*outCall);
    }

    if (this->alphaHelixHBonds.IsDirty() || this->betaSheetHBonds.IsDirty() || this->otherHBonds.IsDirty() ||
        this->cAlphaHBonds.IsDirty() || this->hBondDonorAcceptorDistance.IsDirty() ||
        lastDataHash != inCall->DataHash()) {

        this->filterHBonds(*outCall);

        this->lastDataHash = inCall->DataHash();

        this->alphaHelixHBonds.ResetDirty();
        this->betaSheetHBonds.ResetDirty();
        this->otherHBonds.ResetDirty();
        this->cAlphaHBonds.ResetDirty();
        this->hBondDonorAcceptorDistance.ResetDirty();
    }

    outCall->SetAtomHydrogenBondDistance(this->hBondDonorAcceptorDistance.Param<param::FloatParam>()->Value());
    outCall->SetAtomHydrogenBondStatistics(this->hBondStatistics.data());
    outCall->SetAtomHydrogenBondsFake(this->cAlphaHBonds.Param<param::BoolParam>()->Value());
    outCall->SetHydrogenBonds(
        this->hydrogenBondsFiltered.data(), static_cast<unsigned int>(this->hydrogenBondsFiltered.size() / 2));
    outCall->SetDataHash(this->lastDataHash + this->dataHashOffset);

    return true;
}

/*
 * HydroBondFilter::getExtent
 */
bool HydroBondFilter::getExtent(Call& call) {
    MolecularDataCall* outCall = dynamic_cast<MolecularDataCall*>(&call);
    MolecularDataCall* inCall = this->inDataSlot.CallAs<MolecularDataCall>();

    if (!outCall || !inCall) {
        return false;
    }

    if (!(*inCall)(MolecularDataCall::CallForGetExtent))
        return false;

    // increment local data hash if data will change
    if (this->hBondDonorAcceptorDistance.IsDirty() || this->alphaHelixHBonds.IsDirty() ||
        this->betaSheetHBonds.IsDirty() || this->otherHBonds.IsDirty() || this->cAlphaHBonds.IsDirty()) {

        this->dataHashOffset++;
    }

    outCall->AccessBoundingBoxes().Clear();
    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inCall->AccessBoundingBoxes().ObjectSpaceBBox());
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inCall->AccessBoundingBoxes().ObjectSpaceClipBox());
    outCall->SetFrameCount(inCall->FrameCount());
    outCall->SetDataHash(inCall->DataHash() + dataHashOffset);
    return true;
}

/*
 * HydroBondFilter::isValidHBond
 */
bool HydroBondFilter::isValidHBond(unsigned int donorIndex, unsigned int acceptorIndex, MolecularDataCall& mdc) {

    if (donorIndex == acceptorIndex)
        return false;

    // shallow vectors do not work because of constness
    vislib::math::Vector<float, 3> donorPos = vislib::math::Vector<float, 3>(&mdc.AtomPositions()[donorIndex * 3]);
    vislib::math::Vector<float, 3> acceptorPos =
        vislib::math::Vector<float, 3>(&mdc.AtomPositions()[acceptorIndex * 3]);
    vislib::math::Vector<float, 3> DToA(acceptorPos - donorPos);

    // the distance between acceptor and donator has to be below a threshold
    return DToA.Length() <= this->hBondDonorAcceptorDistance.Param<param::FloatParam>()->Value();
}

/*
 * HydroBondFilter::filterHBonds
 */
void HydroBondFilter::filterHBonds(MolecularDataCall& mdc) {

    std::vector<bool> copyVector(mdc.HydrogenBondCount(), false);
    unsigned int copyCount = 0;

    bool copyAlpha = this->alphaHelixHBonds.Param<param::BoolParam>()->Value();
    bool copyBeta = this->betaSheetHBonds.Param<param::BoolParam>()->Value();
    bool copyOther = this->otherHBonds.Param<param::BoolParam>()->Value();
    bool fake = this->cAlphaHBonds.Param<param::BoolParam>()->Value();

    // determine which H-Bonds have to be copied
    for (unsigned int i = 0; i < mdc.HydrogenBondCount(); i++) {
        unsigned int donorIdx = mdc.GetHydrogenBonds()[i * 2 + 0];
        unsigned int acceptorIdx = mdc.GetHydrogenBonds()[i * 2 + 1];

        auto secStructDonor = this->secStructPerAtom[donorIdx];
        auto secStructAcceptor = this->secStructPerAtom[acceptorIdx];

        bool copy = false;

        if (secStructDonor == secStructAcceptor) { // inside of a secondary structure element
            if (secStructDonor == MolecularDataCall::SecStructure::ElementType::TYPE_SHEET) {
                // beta sheets are always copied, if allowed
                copy = copyBeta;
            } else if (secStructDonor == MolecularDataCall::SecStructure::ElementType::TYPE_HELIX) {
                // alpha sheets are only copied if it is the same alpha sheet
                unsigned int min = std::min(donorIdx, acceptorIdx);
                unsigned int max = std::max(donorIdx, acceptorIdx);

                bool isSame = true;
                for (unsigned int j = min; j <= max; j++) {
                    if (this->secStructPerAtom[j] != MolecularDataCall::SecStructure::ElementType::TYPE_HELIX) {
                        // not the same if there happens a change in secondary structure between the two atoms
                        isSame = false;
                        break;
                    }
                }

                if (copyAlpha && isSame) {
                    copy = true;
                }
                if (copyOther && !isSame) {
                    copy = true;
                }
            } else {
                copy = copyOther;
            }

        } else { // random coil or between different elements
            copy = copyOther;
        }

        if (!isValidHBond(donorIdx, acceptorIdx, mdc)) {
            copy = false;
        }

        copyVector[i] = copy;
        if (copy) {
            copyCount++;
        }
    }

    this->hBondStatistics.resize(mdc.AtomCount(), 0);
    this->hydrogenBondsFiltered.resize(copyCount * 2, 0);

    // copy the H-Bond statistics and hydrogen bonds
    unsigned int copied = 0;
    for (unsigned int i = 0; i < mdc.HydrogenBondCount() * 2; i = i + 2) {
        if (copyVector[i / 2]) {
            this->hydrogenBondsFiltered[copied] = mdc.GetHydrogenBonds()[i];
            this->hydrogenBondsFiltered[copied + 1] = mdc.GetHydrogenBonds()[i + 1];
            copied += 2;
        }
    }
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        "%u hydrogen bonds out of %u survived the filtering.\n", copyCount, mdc.HydrogenBondCount());

    for (unsigned int i = 0; i < static_cast<unsigned int>(this->hydrogenBondsFiltered.size()); i++) {
        if (fake) {
            this->hydrogenBondsFiltered[i] = this->cAlphaIndicesPerAtom[this->hydrogenBondsFiltered[i]];
        }
        this->hBondStatistics[this->hydrogenBondsFiltered[i]]++;
    }
}

/*
 * HydroBondFilter::release
 */
void HydroBondFilter::release() {}
