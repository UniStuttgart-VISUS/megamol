/*
 * ProteinExploder.cpp
 *
 * Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 * Author: Karsten Schatz
 * All rights reserved.
 */

#include "ProteinExploder.h"
#include "AtomWeights.h"
#include "geometry_calls/LinesDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmstd/data/AbstractGetData3DCall.h"

#include "vislib/MD5HashProvider.h"
#include "vislib/SHA1HashProvider.h"
#include "vislib/math/Matrix.h"
#include "vislib/math/ShallowVector.h"
#include "vislib/math/Vector.h"

#include <algorithm>
#include <cfloat>

#include "mmcore/BoundingBoxes_2.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;
using namespace megamol::geocalls;

using namespace std::chrono;

/*
 * ProteinExploder::ProteinExploder
 */
ProteinExploder::ProteinExploder()
        : Module()
        , getDataSlot("getData", "Calls molecular data")
        , dataOutSlot("dataOut", "Provides the exploded molecular data")
        , groupOutSlot("groupOut", "Provides the grouping information for the molecules")
        , explosionModeParam("explosionMode", "The mode of the performed explosion")
        , minDistanceParam("minDistance", "Minimal distance between two exploded components in angstrom")
        , maxExplosionFactorParam("maxExplosionFactor", "Maximal displacement factor")
        , explosionFactorParam("explosionFactor", "Current displacement factor")
        , midPointParam("midpoint", "The middle point of the explosion. Only used when forceMidpoint is activated")
        , forceMidPointParam("forceMidpoint", "Should the explosion center be forced to a certain position?")
        , playParam("animation::play", "Should the animation be played?")
        , togglePlayParam("animation::togglePlay", "Button to toggle animation")
        , replayParam("animation::replay", "Restart animation after end")
        , playDurationParam("animation::duration", "Animation duration in seconds")
        , resetButtonParam("animation::reset", "Resets the animation into the start state")
        , useMassCenterParam("useMassCenter", "Use the mass center for explosion instead of the average position")
        , lineCenterParam("lineCenter", "Connect the output-lines with the common center") {

    // caller slot
    this->getDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    // callee slot
    this->dataOutSlot.SetCallback(
        MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(0), &ProteinExploder::getData);
    this->dataOutSlot.SetCallback(
        MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(1), &ProteinExploder::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->groupOutSlot.SetCallback(
        LinesDataCall::ClassName(), LinesDataCall::FunctionName(0), &ProteinExploder::getGroupData);
    this->groupOutSlot.SetCallback(
        LinesDataCall::ClassName(), LinesDataCall::FunctionName(1), &ProteinExploder::getGroupExtent);
    this->MakeSlotAvailable(&this->groupOutSlot);

    // other parameters
    param::EnumParam* emParam = new param::EnumParam(int(ExplosionMode::SPHERICAL));
    ExplosionMode eMode;
    for (int i = 0; i < getModeNumber(); i++) {
        eMode = getModeByIndex(i);
        emParam->SetTypePair(eMode, getModeName(eMode).c_str());
    }
    this->explosionModeParam << emParam;
    this->MakeSlotAvailable(&this->explosionModeParam);

    this->minDistanceParam.SetParameter(new param::FloatParam(0.0f, 0.0f, 10.0f));
    this->MakeSlotAvailable(&this->minDistanceParam);

    float maxExplosionFactor = 2.0f;

    this->explosionFactorParam.SetParameter(new param::FloatParam(0.0f, 0.0f, 10.0f));
    this->MakeSlotAvailable(&this->explosionFactorParam);

    this->maxExplosionFactorParam.SetParameter(new param::FloatParam(maxExplosionFactor, 0.0f, 10.0f));
    this->MakeSlotAvailable(&this->maxExplosionFactorParam);

    this->guiMidpoint = vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f);
    this->midPointParam.SetParameter(new param::Vector3fParam(guiMidpoint));
    this->MakeSlotAvailable(&this->midPointParam);

    this->forceMidPointParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->forceMidPointParam);

    this->playParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->playParam);

    this->togglePlayParam << new param::ButtonParam(core::view::Key::KEY_P);
    this->togglePlayParam.SetUpdateCallback(this, &ProteinExploder::onPlayToggleButton);
    this->MakeSlotAvailable(&this->togglePlayParam);

    this->resetButtonParam << new param::ButtonParam(core::view::Key::KEY_O);
    this->resetButtonParam.SetUpdateCallback(this, &ProteinExploder::onResetAnimationButton);
    this->MakeSlotAvailable(&this->resetButtonParam);

    this->replayParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->replayParam);

    this->playDurationParam.SetParameter(new param::FloatParam(3.0f, 1.0f, 20.0f));
    this->MakeSlotAvailable(&this->playDurationParam);

    this->useMassCenterParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->useMassCenterParam);

    this->lineCenterParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->lineCenterParam);

    this->atomPositions = NULL;
    this->atomPositionsSize = 0;

    this->playDone = true;
    lastTime = high_resolution_clock::now();
    this->firstRequest = true;
    this->timeAccum = 0.0f;
    this->midpoint = vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f);
    this->massMidpoint = vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f);

    mainDirections.resize(3);
    eigenValues.resize(3);
}

/*
 * ProteinExploder::~ProteinExploder
 */
ProteinExploder::~ProteinExploder() {
    this->Release();
}

/*
 * ProteinExploder::create
 */
bool ProteinExploder::create() {
    return true;
}

/*
 * ProteinExploder::release
 */
void ProteinExploder::release() {
    if (this->atomPositions != NULL) {
        delete[] this->atomPositions;
        this->atomPositions = NULL;
        this->atomPositionsSize = 0;
    }
}

/*
 * ProteinExploder::computeMainDirectionPCA
 */
void ProteinExploder::computeMainDirectionPCA(MolecularDataCall& call) {

    vislib::math::Matrix<float, 3, vislib::math::ROW_MAJOR> covMat;
    covMat.SetNull();

    // compute covariance matrix
    for (unsigned int x = 0; x < 3; x++) {
        for (unsigned int y = 0; y < 3; y++) {
            for (unsigned int k = 0; k < call.AtomCount(); k++) {
                vislib::math::ShallowVector<const float, 3> p1(&call.AtomPositions()[k * 3]);
                covMat(x, y) += (p1[x] - midpoint[x]) * (p1[y] - midpoint[y]);
            }
            covMat(x, y) /= static_cast<float>(call.AtomCount() - 1);
        }
    }
    //covMat.Dump(std::cout);

    float eigenVals[3];
    vislib::math::Vector<float, 3> eigenVectors[3];
    covMat.FindEigenvalues(eigenVals, eigenVectors, 3);
    std::vector<unsigned int> indexVec = {0, 1, 2};

    /*printf("%f %f %f\n", eigenVals[0], eigenVals[1], eigenVals[2]);
    printf("v1: %f %f %f , %f\n", eigenVectors[0][0], eigenVectors[0][1], eigenVectors[0][2], eigenVectors[0].Length());
    printf("v2: %f %f %f , %f\n", eigenVectors[1][0], eigenVectors[1][1], eigenVectors[1][2], eigenVectors[1].Length());
    printf("v3: %f %f %f , %f\n", eigenVectors[2][0], eigenVectors[2][1], eigenVectors[2][2], eigenVectors[2].Length());*/

    std::sort(indexVec.begin(), indexVec.end(),
        [&eigenVals](const unsigned int& a, const unsigned int& b) { return eigenVals[a] > eigenVals[b]; });

    /*printf("%u %u %u\n", indexVec[0], indexVec[1], indexVec[2]);
    printf("sorted: %f %f %f\n", eigenVals[indexVec[0]], eigenVals[indexVec[1]], eigenVals[indexVec[2]]);*/

    for (int i = 0; i < 3; i++) {
        mainDirections[i] = eigenVectors[indexVec[i]];
        mainDirections[i].Normalise();
        eigenValues[i] = eigenVals[indexVec[i]];
    }
}

/*
 * ProteinExploder::computeMidPoint
 */
void ProteinExploder::computeMidPoint(MolecularDataCall& call) {

    midpoint = vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f);
    massMidpoint = vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f);

    for (unsigned int i = 0; i < call.AtomCount(); i++) {
        midpoint.SetX(midpoint.GetX() + call.AtomPositions()[i * 3 + 0]);
        midpoint.SetY(midpoint.GetY() + call.AtomPositions()[i * 3 + 1]);
        midpoint.SetZ(midpoint.GetZ() + call.AtomPositions()[i * 3 + 2]);
    }
    midpoint /= (float) call.AtomCount();

    if (useMassCenterParam.Param<param::BoolParam>()->Value()) {
        float weightSum = 0.0f;
        for (unsigned int i = 0; i < call.AtomCount(); i++) {
            float myWeight = getElementWeightBySymbolString(call.AtomTypes()[call.AtomTypeIndices()[i]].Element());
            massMidpoint.SetX(massMidpoint.GetX() + call.AtomPositions()[i * 3 + 0] * myWeight);
            massMidpoint.SetY(massMidpoint.GetY() + call.AtomPositions()[i * 3 + 1] * myWeight);
            massMidpoint.SetZ(massMidpoint.GetZ() + call.AtomPositions()[i * 3 + 2] * myWeight);
            weightSum += myWeight;
        }
        massMidpoint /= weightSum;
    } else {
        massMidpoint = midpoint;
    }
}

/*
 * ProteinExploder::computeSimilarities
 */
void ProteinExploder::computeSimilarities(MolecularDataCall& call) {

    unsigned int firstResIdx = 0;
    unsigned int lastResIdx = 0;
    unsigned int molAtomCount = 0;
    std::vector<BYTE*> hashPerMol;
    std::vector<unsigned int> hashSizes;

    // compute hash values for each molecule
    for (unsigned int molIdx = 0; molIdx < call.MoleculeCount(); molIdx++) { // for each molecule
        firstResIdx = call.Molecules()[molIdx].FirstResidueIndex();
        lastResIdx = firstResIdx + call.Molecules()[molIdx].ResidueCount();

        unsigned int firstAtom = call.Residues()[firstResIdx]->FirstAtomIndex();
        molAtomCount = 0;

        for (unsigned int resIdx = firstResIdx; resIdx < lastResIdx; resIdx++) { // for each residue in the molecule
            molAtomCount += call.Residues()[resIdx]->AtomCount();
        }

        vislib::MD5HashProvider provider;
        SIZE_T hashSize = 16; // this is the same as the hardcoded one in the vislib
        BYTE* outHash = new BYTE[hashSize];

        provider.ComputeHash(outHash, hashSize, reinterpret_cast<const BYTE*>(&call.AtomTypeIndices()[firstAtom]),
            molAtomCount * sizeof(unsigned int));

        hashSizes.push_back(static_cast<unsigned int>(hashSize));
        hashPerMol.push_back(outHash);
    }

    groups.clear();
    groups.push_back(std::vector<int>(1, 0));

    // build groups by comparing the hash values. Everything with the same hash value belongs to the same group
    for (unsigned int i = 1; i < hashPerMol.size(); i++) {
        BYTE* h1 = hashPerMol[i];
        unsigned int size1 = hashSizes[i];
        bool found = false;
        for (unsigned int j = 0; j < groups.size(); j++) {
            BYTE* h2 = hashPerMol[groups[j][0]];
            unsigned int size2 = hashSizes[groups[j][0]];

            if (isHashEqual(h1, size1, h2, size2)) {
                groups[j].push_back(i);
                found = true;
            }
        }
        if (!found) {
            groups.push_back(std::vector<int>(1, i));
        }
    }

    for (unsigned int i = 0; i < groups.size(); i++) {
        groups[i].shrink_to_fit();
    }
    groups.shrink_to_fit();

    /*for (int i = 0; i < groups.size(); i++) {
        for (unsigned int j = 0; j < groups[i].size(); j++) {
            std::cout << groups[i][j] << " ";
        }
        std::cout << std::endl;
    }*/

    // cleanup
    for (unsigned int i = 0; i < hashPerMol.size(); i++) {
        delete[] hashPerMol[i];
        hashPerMol[i] = nullptr;
    }
}

/*
 * ProteinExploder::explodeMolecule
 */
void ProteinExploder::explodeMolecule(
    MolecularDataCall& call, ProteinExploder::ExplosionMode mode, float exFactor, bool computeBoundingBox) {

    if (atomPositions != NULL) {
        delete[] this->atomPositions;
        this->atomPositions = NULL;
        this->atomPositionsSize = 0;
    }

    atomPositions = new float[call.AtomCount() * 3];
    atomPositionsSize = call.AtomCount() * 3;

    unsigned int firstResIdx = 0;
    unsigned int lastResIdx = 0;
    unsigned int firstAtomIdx = 0;
    unsigned int lastAtomIdx = 0;
    unsigned int molAtomCount = 0;

    moleculeMiddles.clear();
    displacedMoleculeMiddles.clear();
    bool useMass = useMassCenterParam.Param<param::BoolParam>()->Value();

    // compute middle point for each molecule
    for (unsigned int molIdx = 0; molIdx < call.MoleculeCount(); molIdx++) { // for each molecule
        firstResIdx = call.Molecules()[molIdx].FirstResidueIndex();
        lastResIdx = firstResIdx + call.Molecules()[molIdx].ResidueCount();
        molAtomCount = 0;
        float weightSum = 0.0f;

        vislib::math::Vector<float, 3> molMiddle(0.0f, 0.0f, 0.0f);

        for (unsigned int resIdx = firstResIdx; resIdx < lastResIdx; resIdx++) { // for each residue in the molecule
            firstAtomIdx = call.Residues()[resIdx]->FirstAtomIndex();
            lastAtomIdx = firstAtomIdx + call.Residues()[resIdx]->AtomCount();
            molAtomCount += call.Residues()[resIdx]->AtomCount();

            for (unsigned int atomIdx = firstAtomIdx; atomIdx < lastAtomIdx;
                 atomIdx++) { // for each atom in the residue
                vislib::math::Vector<float, 3> curPos;
                curPos.SetX(call.AtomPositions()[3 * atomIdx + 0]);
                curPos.SetY(call.AtomPositions()[3 * atomIdx + 1]);
                curPos.SetZ(call.AtomPositions()[3 * atomIdx + 2]);
                float weight = 1.0f;
                if (useMass)
                    weight =
                        getElementWeightBySymbolString(call.AtomTypes()[call.AtomTypeIndices()[atomIdx]].Element());
                molMiddle += curPos * weight;
                weightSum += weight;
            }
        }

        molMiddle /= weightSum;
        //printf("middle %u:  %f %f %f\n", molIdx, molMiddle.GetX(), molMiddle.GetY(), molMiddle.GetZ());
        moleculeMiddles.push_back(molMiddle);
    }

    std::vector<vislib::math::Vector<float, 3>> displaceDirections = moleculeMiddles;
    float maxFactor = maxExplosionFactorParam.Param<param::FloatParam>()->Value();
    displacedMoleculeMiddles = moleculeMiddles;

    auto myMid = midpoint;
    if (useMassCenterParam.Param<param::BoolParam>()->Value())
        myMid = massMidpoint;

    if (mode == ExplosionMode::SPHERICAL) {

        // compute the direction for each molecule in which it should be displaced
        for (unsigned int i = 0; i < moleculeMiddles.size(); i++) {
            displaceDirections[i] = moleculeMiddles[i] - myMid;
            displacedMoleculeMiddles[i] = moleculeMiddles[i] + exFactor * displaceDirections[i];
        }

        // displace all atoms by the relevant vector
        for (unsigned int molIdx = 0; molIdx < call.MoleculeCount(); molIdx++) { // for each molecule
            firstResIdx = call.Molecules()[molIdx].FirstResidueIndex();
            lastResIdx = firstResIdx + call.Molecules()[molIdx].ResidueCount();

            for (unsigned int resIdx = firstResIdx; resIdx < lastResIdx; resIdx++) { // for each residue in the molecule
                firstAtomIdx = call.Residues()[resIdx]->FirstAtomIndex();
                lastAtomIdx = firstAtomIdx + call.Residues()[resIdx]->AtomCount();

                for (unsigned int atomIdx = firstAtomIdx; atomIdx < lastAtomIdx;
                     atomIdx++) { // for each atom in the residue
                    this->atomPositions[3 * atomIdx + 0] =
                        call.AtomPositions()[3 * atomIdx + 0] + exFactor * displaceDirections[molIdx].GetX();
                    this->atomPositions[3 * atomIdx + 1] =
                        call.AtomPositions()[3 * atomIdx + 1] + exFactor * displaceDirections[molIdx].GetY();
                    this->atomPositions[3 * atomIdx + 2] =
                        call.AtomPositions()[3 * atomIdx + 2] + exFactor * displaceDirections[molIdx].GetZ();
                }
            }
        }
    } else if (mode == ExplosionMode::MAIN_DIRECTION) {

        for (unsigned int i = 0; i < moleculeMiddles.size(); i++) {
            displaceDirections[i] = displaceDirections[i].Dot(mainDirections[0]) * mainDirections[0];
            displacedMoleculeMiddles[i] = moleculeMiddles[i] + exFactor * displaceDirections[i];
        }

        // displace all atoms by the relevant vector
        for (unsigned int molIdx = 0; molIdx < call.MoleculeCount(); molIdx++) { // for each molecule
            firstResIdx = call.Molecules()[molIdx].FirstResidueIndex();
            lastResIdx = firstResIdx + call.Molecules()[molIdx].ResidueCount();

            for (unsigned int resIdx = firstResIdx; resIdx < lastResIdx; resIdx++) { // for each residue in the molecule
                firstAtomIdx = call.Residues()[resIdx]->FirstAtomIndex();
                lastAtomIdx = firstAtomIdx + call.Residues()[resIdx]->AtomCount();

                for (unsigned int atomIdx = firstAtomIdx; atomIdx < lastAtomIdx;
                     atomIdx++) { // for each atom in the residue
                    this->atomPositions[3 * atomIdx + 0] =
                        call.AtomPositions()[3 * atomIdx + 0] + exFactor * displaceDirections[molIdx].GetX();
                    this->atomPositions[3 * atomIdx + 1] =
                        call.AtomPositions()[3 * atomIdx + 1] + exFactor * displaceDirections[molIdx].GetY();
                    this->atomPositions[3 * atomIdx + 2] =
                        call.AtomPositions()[3 * atomIdx + 2] + exFactor * displaceDirections[molIdx].GetZ();
                }
            }
        }
    } else if (mode == ExplosionMode::MAIN_DIRECTION_CIRCULAR) {
        for (unsigned int i = 0; i < moleculeMiddles.size(); i++) {
            displaceDirections[i] = displaceDirections[i].Dot(mainDirections[0]) * mainDirections[0];
            displacedMoleculeMiddles[i] =
                moleculeMiddles[i] + std::min(2.0f * exFactor, maxFactor) * displaceDirections[i];
        }

        // displace all atoms by the relevant vector
        for (unsigned int molIdx = 0; molIdx < call.MoleculeCount(); molIdx++) { // for each molecule
            firstResIdx = call.Molecules()[molIdx].FirstResidueIndex();
            lastResIdx = firstResIdx + call.Molecules()[molIdx].ResidueCount();

            for (unsigned int resIdx = firstResIdx; resIdx < lastResIdx; resIdx++) { // for each residue in the molecule
                firstAtomIdx = call.Residues()[resIdx]->FirstAtomIndex();
                lastAtomIdx = firstAtomIdx + call.Residues()[resIdx]->AtomCount();

                for (unsigned int atomIdx = firstAtomIdx; atomIdx < lastAtomIdx;
                     atomIdx++) { // for each atom in the residue
                    this->atomPositions[3 * atomIdx + 0] =
                        call.AtomPositions()[3 * atomIdx + 0] +
                        std::min(2.0f * exFactor, maxFactor) * displaceDirections[molIdx].GetX();
                    this->atomPositions[3 * atomIdx + 1] =
                        call.AtomPositions()[3 * atomIdx + 1] +
                        std::min(2.0f * exFactor, maxFactor) * displaceDirections[molIdx].GetY();
                    this->atomPositions[3 * atomIdx + 2] =
                        call.AtomPositions()[3 * atomIdx + 2] +
                        std::min(2.0f * exFactor, maxFactor) * displaceDirections[molIdx].GetZ();
                }
            }
        }

        for (unsigned int i = 0; i < moleculeMiddles.size(); i++) {
            displaceDirections[i] = moleculeMiddles[i] -
                                    (myMid + ((moleculeMiddles[i] - myMid).Dot(mainDirections[0]) * mainDirections[0]));
            displacedMoleculeMiddles[i] +=
                std::max(0.0f, std::min(2.0f * exFactor - maxFactor, maxFactor)) * displaceDirections[i];
        }

        for (unsigned int molIdx = 0; molIdx < call.MoleculeCount(); molIdx++) { // for each molecule
            firstResIdx = call.Molecules()[molIdx].FirstResidueIndex();
            lastResIdx = firstResIdx + call.Molecules()[molIdx].ResidueCount();

            for (unsigned int resIdx = firstResIdx; resIdx < lastResIdx; resIdx++) { // for each residue in the molecule
                firstAtomIdx = call.Residues()[resIdx]->FirstAtomIndex();
                lastAtomIdx = firstAtomIdx + call.Residues()[resIdx]->AtomCount();

                for (unsigned int atomIdx = firstAtomIdx; atomIdx < lastAtomIdx;
                     atomIdx++) { // for each atom in the residue
                    this->atomPositions[3 * atomIdx + 0] +=
                        std::max(0.0f, std::min(2.0f * exFactor - maxFactor, maxFactor)) *
                        displaceDirections[molIdx].GetX();
                    this->atomPositions[3 * atomIdx + 1] +=
                        std::max(0.0f, std::min(2.0f * exFactor - maxFactor, maxFactor)) *
                        displaceDirections[molIdx].GetY();
                    this->atomPositions[3 * atomIdx + 2] +=
                        std::max(0.0f, std::min(2.0f * exFactor - maxFactor, maxFactor)) *
                        displaceDirections[molIdx].GetZ();
                }
            }
        }
    }

    /*float v1 = std::min(3.0f * exFactor, maxFactor);
    float v2 = std::max(0.0f, std::min(3.0f * exFactor - maxFactor, maxFactor));
    float v3 = std::max(0.0f, std::min(3.0f * exFactor - 2.0f * maxFactor, maxFactor));

    printf("%f %f %f\n", v1, v2, v3);*/

    vislib::math::Cuboid<float> newBB = currentBoundingBox;

    if (computeBoundingBox) {
        vislib::math::Point<float, 3> bbMin(FLT_MAX, FLT_MAX, FLT_MAX);
        vislib::math::Point<float, 3> bbMax(FLT_MIN, FLT_MIN, FLT_MIN);

        // compute new bounding box
        for (unsigned int i = 0; i < atomPositionsSize / 3; i++) {
            if (atomPositions[i * 3 + 0] < bbMin.X())
                bbMin.SetX(atomPositions[i * 3 + 0]);
            if (atomPositions[i * 3 + 1] < bbMin.Y())
                bbMin.SetY(atomPositions[i * 3 + 1]);
            if (atomPositions[i * 3 + 2] < bbMin.Z())
                bbMin.SetZ(atomPositions[i * 3 + 2]);
            if (atomPositions[i * 3 + 0] > bbMax.X())
                bbMax.SetX(atomPositions[i * 3 + 0]);
            if (atomPositions[i * 3 + 1] > bbMax.Y())
                bbMax.SetY(atomPositions[i * 3 + 1]);
            if (atomPositions[i * 3 + 2] > bbMax.Z())
                bbMax.SetZ(atomPositions[i * 3 + 2]);
        }

        newBB.Set(bbMin.X(), bbMin.Y(), bbMin.Z(), bbMax.X(), bbMax.Y(), bbMax.Z());
        newBB.Grow(3.0f); // add 3 angstrom to each side for some renderers

        // when the growing moves everything to one direction this is necessary
        newBB.Union(call.AccessBoundingBoxes().ObjectSpaceBBox());
    }

    currentBoundingBox = newBB;
}

/*
 * ProteinExploder::explosionFunction
 */
float ProteinExploder::explosionFunction(float exVal) {
    return exVal;
    //return static_cast<float>(pow(tanh(exVal / (0.1f * 3.1413f)), 4.0f)) * exVal;
}

/*
 * ProteinExploder::getData
 */
bool ProteinExploder::getData(core::Call& call) {
    MolecularDataCall* outCall = dynamic_cast<MolecularDataCall*>(&call);
    if (outCall == NULL)
        return false;

    outCall->SetAtomPositions(this->atomPositions);

    return true;
}

/*
 * ProteinExploder::getExtent
 */
bool ProteinExploder::getExtent(core::Call& call) {
    MolecularDataCall* agdc = dynamic_cast<MolecularDataCall*>(&call);
    if (agdc == NULL)
        return false;

    MolecularDataCall* mdc = this->getDataSlot.CallAs<MolecularDataCall>();
    if (mdc == NULL)
        return false;
    mdc->SetCalltime(agdc->Calltime());
    if (!(*mdc)(1))
        return false;
    if (!(*mdc)(0))
        return false;

    agdc->operator=(*mdc); // deep copy

    float theParam = std::min(this->explosionFactorParam.Param<param::FloatParam>()->Value(),
        this->maxExplosionFactorParam.Param<param::FloatParam>()->Value());

    bool play = this->playParam.Param<param::BoolParam>()->Value();
    bool replay = this->replayParam.Param<param::BoolParam>()->Value();
    high_resolution_clock::time_point curTime = high_resolution_clock::now();

    if (firstRequest) {
        lastTime = curTime;
        lastDataHash = 0;
    }

    float dur = static_cast<float>(duration_cast<duration<double>>(curTime - lastTime).count());

    if (play)
        timeAccum += dur;

    float timeVal = timeAccum / this->playDurationParam.Param<param::FloatParam>()->Value();

    if (timeVal > 1.0f && play) {
        playDone = true;
        timeVal = 1.0f;

        if (replay) {
            playDone = false;
            lastTime = high_resolution_clock::now();
            timeAccum = 0.0f;
        } else {
            this->playParam.Param<param::BoolParam>()->SetValue(false);
        }
    }

    if (this->playParam.Param<param::BoolParam>()->Value()) {
        //printf("%f %f %f\n", timeAccum, timeVal, dur);
        float maxVal = this->maxExplosionFactorParam.Param<param::FloatParam>()->Value();
        theParam = timeVal * maxVal;
        this->explosionFactorParam.Param<param::FloatParam>()->SetValue(theParam);
    }
    lastTime = curTime;

    if (!forceMidPointParam.Param<param::BoolParam>()->Value()) {
        computeMidPoint(*mdc);

        if (useMassCenterParam.Param<param::BoolParam>()->Value())
            guiMidpoint = massMidpoint;
        else
            guiMidpoint = midpoint;

        midPointParam.Param<param::Vector3fParam>()->SetValue(guiMidpoint);
    } else {
        midpoint = midPointParam.Param<param::Vector3fParam>()->Value();
        massMidpoint = midPointParam.Param<param::Vector3fParam>()->Value();
    }

    if (firstRequest || lastDataHash != mdc->DataHash() || maxExplosionFactorParam.IsDirty() ||
        forceMidPointParam.IsDirty() || midPointParam.IsDirty() || explosionModeParam.IsDirty()) {

        computeSimilarities(*mdc);

        // compute the main direction of the data set
        computeMainDirectionPCA(*mdc);

        // compute the bounding box for the maximal explosion factor
        explodeMolecule(*agdc, getModeByIndex(this->explosionModeParam.Param<param::EnumParam>()->Value()),
            this->maxExplosionFactorParam.Param<param::FloatParam>()->Value(), true);

        lastDataHash = mdc->DataHash();
        maxExplosionFactorParam.ResetDirty();
        forceMidPointParam.ResetDirty();
        midPointParam.ResetDirty();
        explosionModeParam.ResetDirty();
        firstRequest = false;
    }

    explodeMolecule(*agdc, getModeByIndex(this->explosionModeParam.Param<param::EnumParam>()->Value()),
        explosionFunction(theParam));

    agdc->SetFrameCount(mdc->FrameCount());
    agdc->AccessBoundingBoxes().Clear();
    agdc->AccessBoundingBoxes().SetObjectSpaceBBox(currentBoundingBox);
    agdc->AccessBoundingBoxes().SetObjectSpaceClipBox(currentBoundingBox);

    return true;
}

/*
 * ProteinExploder::getGroupData
 */
bool ProteinExploder::getGroupData(core::Call& call) {

    LinesDataCall* outCall = dynamic_cast<LinesDataCall*>(&call);
    if (outCall == NULL)
        return false;

    /*if (lines.size() > 0) {
        const float * arr = lines[0].VertexArrayFloat();
        std::cout << arr[0] << " " << arr[1] << " " << arr[2] << " ::: " << arr[3] << " " << arr[4] << " " << arr[5] << std::endl;
    }*/

    outCall->SetDataHash(lastDataHash);
    outCall->SetData(static_cast<unsigned int>(lines.size()), lines.data());
    outCall->SetUnlocker(NULL);

    return true;
}

/*
 * ProteinExploder::getGroupExtent
 */
bool ProteinExploder::getGroupExtent(core::Call& call) {

    LinesDataCall* outCall = dynamic_cast<LinesDataCall*>(&call);
    if (outCall == NULL)
        return false;

    MolecularDataCall* inCall = this->getDataSlot.CallAs<MolecularDataCall>();
    if (inCall == NULL)
        return false;
    inCall->SetCalltime(outCall->Time());
    if (!(*inCall)(1))
        return false;
    if (!(*inCall)(0))
        return false;

    computeSimilarities(*inCall);

    outCall->SetFrameCount(inCall->FrameCount());
    outCall->SetDataHash(inCall->DataHash());

    core::BoundingBoxes_2 boxes;
    boxes.SetBoundingBox(currentBoundingBox);
    outCall->SetExtent(1, boxes.BoundingBox().Left(), boxes.BoundingBox().Bottom(), boxes.BoundingBox().Back(),
        boxes.BoundingBox().Right(), boxes.BoundingBox().Top(), boxes.BoundingBox().Front());

    lines.clear();
    lineVertexPositions.clear();
    size_t lineID = 0;

    std::vector<vislib::math::Vector<float, 3>> groupMiddles;

    for (unsigned int i = 0; i < groups.size(); i++) {
        vislib::math::Vector<float, 3> mid = vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f);

        for (unsigned int j = 0; j < groups[i].size(); j++) {
            mid += displacedMoleculeMiddles[groups[i][j]];
        }

        mid /= static_cast<float>(groups[i].size());
        groupMiddles.push_back(mid);
    }

    if (!lineCenterParam.Param<param::BoolParam>()->Value()) {
        // Connection between every chain
        for (unsigned int i = 0; i < groups.size(); i++) {
            for (unsigned int j = 0; j < groups[i].size(); j++) {
                for (unsigned int k = j + 1; k < groups[i].size(); k++) {
                    auto startPos = displacedMoleculeMiddles[groups[i][j]];
                    auto endPos = displacedMoleculeMiddles[groups[i][k]];

                    lineVertexPositions.push_back(startPos[0]);
                    lineVertexPositions.push_back(startPos[1]);
                    lineVertexPositions.push_back(startPos[2]);
                    lineVertexPositions.push_back(endPos[0]);
                    lineVertexPositions.push_back(endPos[1]);
                    lineVertexPositions.push_back(endPos[2]);

                    LinesDataCall::Lines line;
                    line.Set(2, &lineVertexPositions[lineVertexPositions.size() - 6],
                        vislib::graphics::ColourRGBAu8(255, 255, 255, 255));
                    line.SetID(lineID);
                    lineID++;

                    lines.push_back(line);
                }
            }
        }
    } else {
        // Connection to common center
        for (unsigned int i = 0; i < groups.size(); i++) {
            for (unsigned int j = 0; j < groups[i].size(); j++) {
                auto startPos = displacedMoleculeMiddles[groups[i][j]];
                auto endPos = groupMiddles[i];

                lineVertexPositions.push_back(startPos[0]);
                lineVertexPositions.push_back(startPos[1]);
                lineVertexPositions.push_back(startPos[2]);
                lineVertexPositions.push_back(endPos[0]);
                lineVertexPositions.push_back(endPos[1]);
                lineVertexPositions.push_back(endPos[2]);

                LinesDataCall::Lines line;
                line.Set(2, &lineVertexPositions[lineVertexPositions.size() - 6],
                    vislib::graphics::ColourRGBAu8(255, 255, 255, 255));
                line.SetID(lineID);
                lineID++;

                lines.push_back(line);
            }
        }
    }

    return true;
}

/*
 * ProteinExploder::getModeName
 */
std::string ProteinExploder::getModeName(ProteinExploder::ExplosionMode mode) {
    switch (mode) {
    case SPHERICAL:
        return "Spherical";
    case MAIN_DIRECTION:
        return "Main Direction";
    case MAIN_DIRECTION_CIRCULAR:
        return "Main Direction Circular";
    default:
        return "";
    }
}

ProteinExploder::ExplosionMode ProteinExploder::getModeByIndex(unsigned int idx) {
    switch (idx) {
    case 0:
        return ExplosionMode::SPHERICAL;
    case 1:
        return ExplosionMode::MAIN_DIRECTION;
    case 2:
        return ExplosionMode::MAIN_DIRECTION_CIRCULAR;
    default:
        return ExplosionMode::SPHERICAL;
    }
}

/*
 * ProteinExploder::getModeNumber
 */
int ProteinExploder::getModeNumber() {
    return 3;
}

/*
 * ProteinExploder::isHashEqual
 */
bool ProteinExploder::isHashEqual(const BYTE* h1, unsigned int hashSize1, const BYTE* h2, unsigned int hashSize2) {
    if (hashSize1 != hashSize2)
        return false;
    for (unsigned int i = 0; i < hashSize1; i++) {
        if (h1[i] != h2[i])
            return false;
    }
    return true;
}

/*
 * ProteinExploder::onResetAnimationButton
 */
bool ProteinExploder::onResetAnimationButton(param::ParamSlot& p) {

    this->playParam.Param<param::BoolParam>()->SetValue(false);
    playDone = false;
    timeAccum = 0.0f;
    lastTime = high_resolution_clock::now();
    this->explosionFactorParam.Param<param::FloatParam>()->SetValue(0.0f);

    return true;
}

/*
 * ProteinExploder::onPlayToggleButton
 */
bool ProteinExploder::onPlayToggleButton(param::ParamSlot& p) {
    param::BoolParam* bp = this->playParam.Param<param::BoolParam>();
    bp->SetValue(!bp->Value());

    bool play = bp->Value();

    if (play && playDone) { // restart the animation, if the previous one has ended
        playDone = false;
        lastTime = high_resolution_clock::now();
        timeAccum = 0.0f;
    }

    return true;
}
