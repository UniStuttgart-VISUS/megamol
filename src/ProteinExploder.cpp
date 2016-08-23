/*
 * ProteinExploder.cpp
 *
 * Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 * Author: Karsten Schatz
 * All rights reserved.
 */

#include "stdafx.h"
#include "ProteinExploder.h"
#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/Vector3fParam.h"

#include "vislib/math/Vector.h"
#include "vislib/math/ShallowVector.h"
#include "vislib/math/Matrix.h"

#include <algorithm>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;

using namespace std::chrono;

/*
 *	ProteinExploder::ProteinExploder
 */
ProteinExploder::ProteinExploder(void) : 
		Module(),
		getDataSlot("getData", "Calls molecular data"),
		dataOutSlot("dataOut", "Provides the exploded molecular data"),
		explosionModeParam("explosionMode", "The mode of the performed explosion"),
		minDistanceParam("minDistance", "Minimal distance between two exploded components in angstrom"),
		maxExplosionFactorParam("maxExplosionFactor", "Maximal displacement factor"),
		explosionFactorParam("explosionFactor", "Current displacement factor"),
		midPointParam("midpoint", "The middle point of the explosion. Only used when forceMidpoint is activated"),
		forceMidPointParam("forceMidpoint", "Should the explosion center be forced to a certain position?"),
		playParam("animation::play","Should the animation be played?"),
		togglePlayParam("animation::togglePlay", "Button to toggle animation"),
		replayParam("animation::replay","Restart animation after end"),
		playDurationParam("animation::duration","Animation duration in seconds") {

	// caller slot
	this->getDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable(&this->getDataSlot);

	// callee slot
	this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(0), &ProteinExploder::getData);
	this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(1), &ProteinExploder::getExtent);
	this->MakeSlotAvailable(&this->dataOutSlot);

	// other parameters
	param::EnumParam * emParam = new param::EnumParam(int(ExplosionMode::SPHERICAL_MIDDLE));
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

	this->togglePlayParam << new param::ButtonParam('p');
	this->togglePlayParam.SetUpdateCallback(this, &ProteinExploder::onPlayToggleButton);
	this->MakeSlotAvailable(&this->togglePlayParam);

	this->replayParam.SetParameter(new param::BoolParam(false));
	this->MakeSlotAvailable(&this->replayParam);

	this->playDurationParam.SetParameter(new param::FloatParam(3.0f, 1.0f, 20.0f));
	this->MakeSlotAvailable(&this->playDurationParam);

	this->atomPositions = NULL;
	this->atomPositionsSize = 0;

	this->playDone = true;
	lastTime = high_resolution_clock::now();
	this->firstRequest = true;
	this->timeAccum = 0.0f;
	this->midpoint = vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f);

	mainDirections.resize(3);
	eigenValues.resize(3);
}

/*
 *	ProteinExploder::~ProteinExploder
 */
ProteinExploder::~ProteinExploder(void) {
	this->Release();
}

/*
 *	ProteinExploder::create
 */
bool ProteinExploder::create(void) {
	return true;
}

/*
 *	ProteinExploder::release
 */
void ProteinExploder::release(void) {
	if (this->atomPositions != NULL) {
		delete[] this->atomPositions;
		this->atomPositions = NULL;
		this->atomPositionsSize = 0;
	}
}

/*
 *	ProteinExploder::computeMainDirectionPCA
 */
void ProteinExploder::computeMainDirectionPCA(MolecularDataCall& call) {

	std::vector<float> colors(call.AtomCount(), 0.0f);
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
	std::vector<unsigned int> indexVec = { 0, 1, 2 };

	/*printf("%f %f %f\n", eigenVals[0], eigenVals[1], eigenVals[2]);
	printf("v1: %f %f %f , %f\n", eigenVectors[0][0], eigenVectors[0][1], eigenVectors[0][2], eigenVectors[0].Length());
	printf("v2: %f %f %f , %f\n", eigenVectors[1][0], eigenVectors[1][1], eigenVectors[1][2], eigenVectors[1].Length());
	printf("v3: %f %f %f , %f\n", eigenVectors[2][0], eigenVectors[2][1], eigenVectors[2][2], eigenVectors[2].Length());*/

	std::sort(indexVec.begin(), indexVec.end(), [&eigenVals](const unsigned int& a, const unsigned int& b) {
		return eigenVals[a] > eigenVals[b];
	});

	/*printf("%u %u %u\n", indexVec[0], indexVec[1], indexVec[2]);
	printf("sorted: %f %f %f\n", eigenVals[indexVec[0]], eigenVals[indexVec[1]], eigenVals[indexVec[2]]);*/

	for (int i = 0; i < 3; i++) {
		mainDirections[i] = eigenVectors[indexVec[i]];
		mainDirections[i].Normalise();
		eigenValues[i] = eigenVals[indexVec[i]];
	}
}

/*
 *	ProteinExploder::computeMidPoint
 */
void ProteinExploder::computeMidPoint(MolecularDataCall& call, ProteinExploder::ExplosionMode mode) {

	midpoint = vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f);

	if (mode != ExplosionMode::SPHERICAL_MASS) {
		for (unsigned int i = 0; i < call.AtomCount(); i++) {
			midpoint.SetX(midpoint.GetX() + call.AtomPositions()[i * 3 + 0]);
			midpoint.SetY(midpoint.GetY() + call.AtomPositions()[i * 3 + 1]);
			midpoint.SetZ(midpoint.GetZ() + call.AtomPositions()[i * 3 + 2]);
		}
	}
	else { // ExplosionMode::SPHERICAL_MASS
		for (unsigned int i = 0; i < call.AtomCount(); i++) {
			// TODO adjust the computation by the mass
			midpoint.SetX(midpoint.GetX() + call.AtomPositions()[i * 3 + 0]);
			midpoint.SetY(midpoint.GetY() + call.AtomPositions()[i * 3 + 1]);
			midpoint.SetZ(midpoint.GetZ() + call.AtomPositions()[i * 3 + 2]);
		}
	}

	midpoint /= (float)call.AtomCount();
}

/*
 *	ProteinExploder::explodeMolecule
 */
void ProteinExploder::explodeMolecule(MolecularDataCall& call, ProteinExploder::ExplosionMode mode, float exFactor, bool computeBoundingBox) {

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

	std::vector<vislib::math::Vector<float, 3>> moleculeMiddles;

	// compute middle point for each molecule
	for (unsigned int molIdx = 0; molIdx < call.MoleculeCount(); molIdx++) { // for each molecule
		firstResIdx = call.Molecules()[molIdx].FirstResidueIndex();
		lastResIdx = firstResIdx + call.Molecules()[molIdx].ResidueCount();
		molAtomCount = 0;

		vislib::math::Vector<float, 3> molMiddle(0.0f, 0.0f, 0.0f);

		for (unsigned int resIdx = firstResIdx; resIdx < lastResIdx; resIdx++) { // for each residue in the molecule
			firstAtomIdx = call.Residues()[resIdx]->FirstAtomIndex();
			lastAtomIdx = firstAtomIdx + call.Residues()[resIdx]->AtomCount();
			molAtomCount += call.Residues()[resIdx]->AtomCount();

			for (unsigned int atomIdx = firstAtomIdx; atomIdx < lastAtomIdx; atomIdx++) { // for each atom in the residue
				vislib::math::Vector<float, 3> curPos;
				curPos.SetX(call.AtomPositions()[3 * atomIdx + 0]);
				curPos.SetY(call.AtomPositions()[3 * atomIdx + 1]);
				curPos.SetZ(call.AtomPositions()[3 * atomIdx + 2]);
				molMiddle += curPos;
			}
		}

		molMiddle /= (float)molAtomCount;
		//printf("middle %u:  %f %f %f\n", molIdx, molMiddle.GetX(), molMiddle.GetY(), molMiddle.GetZ());
		moleculeMiddles.push_back(molMiddle);
	}

	std::vector<vislib::math::Vector<float, 3>> displaceDirections = moleculeMiddles;

#ifdef _MSC_VER
#pragma push_macro("min")
#undef min
#pragma push_macro("max")
#undef max
#endif /* _MSC_VER */
	if (mode == ExplosionMode::SPHERICAL_MIDDLE || mode == ExplosionMode::SPHERICAL_MASS) {

		// compute the direction for each molecule in which it should be displaced
		for (int i = 0; i < moleculeMiddles.size(); i++) {
			displaceDirections[i] = moleculeMiddles[i] - midpoint;
		}

		// displace all atoms by the relevant vector
		for (unsigned int molIdx = 0; molIdx < call.MoleculeCount(); molIdx++) { // for each molecule
			firstResIdx = call.Molecules()[molIdx].FirstResidueIndex();
			lastResIdx = firstResIdx + call.Molecules()[molIdx].ResidueCount();

			for (unsigned int resIdx = firstResIdx; resIdx < lastResIdx; resIdx++) { // for each residue in the molecule
				firstAtomIdx = call.Residues()[resIdx]->FirstAtomIndex();
				lastAtomIdx = firstAtomIdx + call.Residues()[resIdx]->AtomCount();

				for (unsigned int atomIdx = firstAtomIdx; atomIdx < lastAtomIdx; atomIdx++) { // for each atom in the residue
					this->atomPositions[3 * atomIdx + 0] = call.AtomPositions()[3 * atomIdx + 0] + exFactor * displaceDirections[molIdx].GetX();
					this->atomPositions[3 * atomIdx + 1] = call.AtomPositions()[3 * atomIdx + 1] + exFactor * displaceDirections[molIdx].GetY();
					this->atomPositions[3 * atomIdx + 2] = call.AtomPositions()[3 * atomIdx + 2] + exFactor * displaceDirections[molIdx].GetZ();
				}
			}
		}
	} else if (mode == ExplosionMode::MAIN_DIRECTION) {

		for (int i = 0; i < moleculeMiddles.size(); i++) {
			displaceDirections[i] = displaceDirections[i].Dot(mainDirections[0]) * mainDirections[0];
		}

		// displace all atoms by the relevant vector
		for (unsigned int molIdx = 0; molIdx < call.MoleculeCount(); molIdx++) { // for each molecule
			firstResIdx = call.Molecules()[molIdx].FirstResidueIndex();
			lastResIdx = firstResIdx + call.Molecules()[molIdx].ResidueCount();

			for (unsigned int resIdx = firstResIdx; resIdx < lastResIdx; resIdx++) { // for each residue in the molecule
				firstAtomIdx = call.Residues()[resIdx]->FirstAtomIndex();
				lastAtomIdx = firstAtomIdx + call.Residues()[resIdx]->AtomCount();

				for (unsigned int atomIdx = firstAtomIdx; atomIdx < lastAtomIdx; atomIdx++) { // for each atom in the residue
					this->atomPositions[3 * atomIdx + 0] = call.AtomPositions()[3 * atomIdx + 0] + exFactor * displaceDirections[molIdx].GetX();
					this->atomPositions[3 * atomIdx + 1] = call.AtomPositions()[3 * atomIdx + 1] + exFactor * displaceDirections[molIdx].GetY();
					this->atomPositions[3 * atomIdx + 2] = call.AtomPositions()[3 * atomIdx + 2] + exFactor * displaceDirections[molIdx].GetZ();
				}
			}
		}
	} else if (mode == ExplosionMode::MAIN_DIRECTION_CIRCULAR) {
		for (int i = 0; i < moleculeMiddles.size(); i++) {
			displaceDirections[i] = displaceDirections[i].Dot(mainDirections[0]) * mainDirections[0];
		}

		float maxFactor = maxExplosionFactorParam.Param<param::FloatParam>()->Value();

		// displace all atoms by the relevant vector
		for (unsigned int molIdx = 0; molIdx < call.MoleculeCount(); molIdx++) { // for each molecule
			firstResIdx = call.Molecules()[molIdx].FirstResidueIndex();
			lastResIdx = firstResIdx + call.Molecules()[molIdx].ResidueCount();

			for (unsigned int resIdx = firstResIdx; resIdx < lastResIdx; resIdx++) { // for each residue in the molecule
				firstAtomIdx = call.Residues()[resIdx]->FirstAtomIndex();
				lastAtomIdx = firstAtomIdx + call.Residues()[resIdx]->AtomCount();

				for (unsigned int atomIdx = firstAtomIdx; atomIdx < lastAtomIdx; atomIdx++) { // for each atom in the residue
					this->atomPositions[3 * atomIdx + 0] = call.AtomPositions()[3 * atomIdx + 0] + std::min(2.0f * exFactor, maxFactor) * displaceDirections[molIdx].GetX();
					this->atomPositions[3 * atomIdx + 1] = call.AtomPositions()[3 * atomIdx + 1] + std::min(2.0f * exFactor, maxFactor) * displaceDirections[molIdx].GetY();
					this->atomPositions[3 * atomIdx + 2] = call.AtomPositions()[3 * atomIdx + 2] + std::min(2.0f * exFactor, maxFactor) * displaceDirections[molIdx].GetZ();
				}
			}
		}

		for (int i = 0; i < moleculeMiddles.size(); i++) {
			displaceDirections[i] = moleculeMiddles[i] - (midpoint + ((moleculeMiddles[i] - midpoint).Dot(mainDirections[0]) * mainDirections[0]));
		}

		for (unsigned int molIdx = 0; molIdx < call.MoleculeCount(); molIdx++) { // for each molecule
			firstResIdx = call.Molecules()[molIdx].FirstResidueIndex();
			lastResIdx = firstResIdx + call.Molecules()[molIdx].ResidueCount();

			for (unsigned int resIdx = firstResIdx; resIdx < lastResIdx; resIdx++) { // for each residue in the molecule
				firstAtomIdx = call.Residues()[resIdx]->FirstAtomIndex();
				lastAtomIdx = firstAtomIdx + call.Residues()[resIdx]->AtomCount();

				for (unsigned int atomIdx = firstAtomIdx; atomIdx < lastAtomIdx; atomIdx++) { // for each atom in the residue
					this->atomPositions[3 * atomIdx + 0] += std::max(0.0f, std::min(2.0f * exFactor - maxFactor, maxFactor)) * displaceDirections[molIdx].GetX();
					this->atomPositions[3 * atomIdx + 1] += std::max(0.0f, std::min(2.0f * exFactor - maxFactor, maxFactor)) * displaceDirections[molIdx].GetY();
					this->atomPositions[3 * atomIdx + 2] += std::max(0.0f, std::min(2.0f * exFactor - maxFactor, maxFactor)) * displaceDirections[molIdx].GetZ();
				}
			}
		}
	}
#ifdef _MSC_VER
#pragma pop_macro("min")
#pragma pop_macro("max")
#endif /* _MSC_VER */

	vislib::math::Cuboid<float> newBB = currentBoundingBox;

	if (computeBoundingBox) {
		vislib::math::Point<float, 3> bbMin(FLT_MAX, FLT_MAX, FLT_MAX);
		vislib::math::Point<float, 3> bbMax(FLT_MIN, FLT_MIN, FLT_MIN);

		// compute new bounding box
		for (unsigned int i = 0; i < atomPositionsSize / 3; i++) {
			if (atomPositions[i * 3 + 0] < bbMin.X()) bbMin.SetX(atomPositions[i * 3 + 0]);
			if (atomPositions[i * 3 + 1] < bbMin.Y()) bbMin.SetY(atomPositions[i * 3 + 1]);
			if (atomPositions[i * 3 + 2] < bbMin.Z()) bbMin.SetZ(atomPositions[i * 3 + 2]);
			if (atomPositions[i * 3 + 0] > bbMax.X()) bbMax.SetX(atomPositions[i * 3 + 0]);
			if (atomPositions[i * 3 + 1] > bbMax.Y()) bbMax.SetY(atomPositions[i * 3 + 1]);
			if (atomPositions[i * 3 + 2] > bbMax.Z()) bbMax.SetZ(atomPositions[i * 3 + 2]);
		}

		newBB.Set(bbMin.X(), bbMin.Y(), bbMin.Z(), bbMax.X(), bbMax.Y(), bbMax.Z());
		newBB.Grow(3.0f); // add 3 angstrom to each side for some renderers

		// when the growing moves everything to one direction this is necessary
		newBB.Union(call.AccessBoundingBoxes().ObjectSpaceBBox()); 
	}

	currentBoundingBox = newBB;
}

/*
 *	ProteinExploder::explosionFunction
 */
float ProteinExploder::explosionFunction(float exVal) {
	return exVal;
	//return static_cast<float>(pow(tanh(exVal / (0.1f * 3.1413f)), 4.0f)) * exVal;
}

/*
 *	ProteinExploder::getData
 */
bool ProteinExploder::getData(core::Call& call) {
	MolecularDataCall * outCall = dynamic_cast<MolecularDataCall*>(&call);
	if (outCall == NULL) return false;

	outCall->SetAtomPositions(this->atomPositions);

	return true;
}

/*
 *	ProteinExploder::getExtent
 */
bool ProteinExploder::getExtent(core::Call& call) {
	MolecularDataCall * agdc = dynamic_cast<MolecularDataCall*>(&call);
	if (agdc == NULL) return false;

	MolecularDataCall *mdc = this->getDataSlot.CallAs<MolecularDataCall>();
	if (mdc == NULL) return false;
	mdc->SetCalltime(agdc->Calltime());
	if (!(*mdc)(1)) return false;
	if (!(*mdc)(0)) return false;

	agdc->operator=(*mdc); // deep copy

#ifdef _MSC_VER
#pragma push_macro("min")
#undef min
#pragma push_macro("max")
#undef max
#endif /* _MSC_VER */
	float theParam = std::min(this->explosionFactorParam.Param<param::FloatParam>()->Value(), 
		this->maxExplosionFactorParam.Param<param::FloatParam>()->Value());
#ifdef _MSC_VER
#pragma pop_macro("min")
#pragma pop_macro("max")
#endif /* _MSC_VER */

	bool play = this->playParam.Param<param::BoolParam>()->Value();
	bool replay = this->replayParam.Param<param::BoolParam>()->Value();
	high_resolution_clock::time_point curTime = high_resolution_clock::now();

	if (firstRequest) {
		lastTime = curTime;
		lastDataHash = 0;
	}

	float dur = static_cast<float>(duration_cast<duration<double>>(curTime - lastTime).count());

	if (play) timeAccum += dur;

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
		computeMidPoint(*mdc, getModeByIndex(this->explosionModeParam.Param<param::EnumParam>()->Value()));
		guiMidpoint = midpoint;
		midPointParam.Param<param::Vector3fParam>()->SetValue(guiMidpoint);
	} else {
		midpoint = midPointParam.Param<param::Vector3fParam>()->Value();
	}

	if (firstRequest || lastDataHash != mdc->DataHash() || maxExplosionFactorParam.IsDirty() 
		|| forceMidPointParam.IsDirty() || midPointParam.IsDirty() || explosionModeParam.IsDirty()) { // compute the bounding box for the maximal explosion factor

		computeMainDirectionPCA(*mdc);

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
 *	ProteinExploder::getModeName
 */
std::string ProteinExploder::getModeName(ProteinExploder::ExplosionMode mode)  {
		switch (mode) {
			case SPHERICAL_MIDDLE			: return "Spherical Middle";
			case SPHERICAL_MASS				: return "Spherical Mass";
			case MAIN_DIRECTION				: return "Main Direction";
			case MAIN_DIRECTION_CIRCULAR	: return "Main Direction Circular";
			default							: return "";
		}
}

ProteinExploder::ExplosionMode ProteinExploder::getModeByIndex(unsigned int idx) {
	switch (idx) {
		case 0: return ExplosionMode::SPHERICAL_MIDDLE;
		case 1: return ExplosionMode::SPHERICAL_MASS;
		case 2: return ExplosionMode::MAIN_DIRECTION;
		case 3: return ExplosionMode::MAIN_DIRECTION_CIRCULAR;
		default: return ExplosionMode::SPHERICAL_MIDDLE;
	}
}

/*
 *	ProteinExploder::getModeNumber
 */
int ProteinExploder::getModeNumber() {
	return 4;
}

/*
 *	ProteinExploder::onPlayToggleButton
 */
bool ProteinExploder::onPlayToggleButton(param::ParamSlot& p) {
	param::BoolParam *bp = this->playParam.Param<param::BoolParam>();
	bp->SetValue(!bp->Value());

	bool play = bp->Value();

	if (play && playDone) { // restart the animation, if the previous one has ended
		playDone = false;
		lastTime = high_resolution_clock::now();
		timeAccum = 0.0f;
	}

	return true;
}