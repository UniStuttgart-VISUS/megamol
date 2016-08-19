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

#include "vislib/math/Vector.h"

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

	float maxExplosionFactor = 3.0f;

	this->explosionFactorParam.SetParameter(new param::FloatParam(0.0f, 0.0f, 10.0f));
	this->MakeSlotAvailable(&this->explosionFactorParam);

	this->maxExplosionFactorParam.SetParameter(new param::FloatParam(maxExplosionFactor, 0.0f, 10.0f));
	this->MakeSlotAvailable(&this->maxExplosionFactorParam);

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
 *	ProteinExploder::explodeMolecule
 */
void ProteinExploder::explodeMolecule(MolecularDataCall& call, ProteinExploder::ExplosionMode mode, float exFactor) {

	// compute middle point
	vislib::math::Vector<float, 3> midpoint(0.0f, 0.0f, 0.0f);

	if (atomPositions != NULL) {
		delete[] this->atomPositions;
		this->atomPositions = NULL;
		this->atomPositionsSize = 0;
	}

	atomPositions = new float[call.AtomCount() * 3];
	atomPositionsSize = call.AtomCount() * 3;

	for (unsigned int i = 0; i < call.AtomCount(); i++) {
		midpoint.SetX(midpoint.GetX() + call.AtomPositions()[i * 3 + 0]);
		midpoint.SetY(midpoint.GetY() + call.AtomPositions()[i * 3 + 1]);
		midpoint.SetZ(midpoint.GetZ() + call.AtomPositions()[i * 3 + 2]);
	}

	midpoint /= (float)call.AtomCount();

	//printf("middle %f %f %f\n", midpoint.GetX(), midpoint.GetY(), midpoint.GetZ());
	
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

	// compute the direction for each molecule in which it should be displaced
	std::vector<vislib::math::Vector<float, 3>> displaceDirections = moleculeMiddles;

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

	//currentBoundingBox.Set(bbMin.X(), bbMin.Y(), bbMin.Z(), bbMax.X(), bbMax.Y(), bbMax.Z());
	//currentBoundingBox.Grow(3.0f); // add 3 angstrom to each side for some renderers

	float maxFactor = maxExplosionFactorParam.Param<param::FloatParam>()->Value();
	currentBoundingBox = call.AccessBoundingBoxes().ObjectSpaceBBox();
	currentBoundingBox.EnforcePositiveSize();

	// the +- 3.0f is there to reverse the growing of the bounding box by the PDBLoader
	currentBoundingBox.SetRight(currentBoundingBox.Right() - 3.0f + maxFactor * (currentBoundingBox.Right() - 3.0f - midpoint.X()));
	currentBoundingBox.SetLeft(currentBoundingBox.Left() + 3.0f - 3.0f + maxFactor * (currentBoundingBox.Left() + 3.0f - midpoint.X()));
	currentBoundingBox.SetTop(currentBoundingBox.Top() - 3.0f + maxFactor * (currentBoundingBox.Top() - 3.0f - midpoint.Y()));
	currentBoundingBox.SetBottom(currentBoundingBox.Bottom() + 3.0f + maxFactor * (currentBoundingBox.Bottom() + 3.0f - midpoint.Y()));
	currentBoundingBox.SetFront(currentBoundingBox.Front() - 3.0f + maxFactor * (currentBoundingBox.Front() - 3.0f - midpoint.Z()));
	currentBoundingBox.SetBack(currentBoundingBox.Back() + 3.0f + maxFactor * (currentBoundingBox.Back() + 3.0f - midpoint.Z()));
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
		firstRequest = false;
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
		//printf("%f %f\n", timeAccum, timeVal);
		float maxVal = this->maxExplosionFactorParam.Param<param::FloatParam>()->Value();
		theParam = timeVal * maxVal;
		this->explosionFactorParam.Param<param::FloatParam>()->SetValue(theParam);	
	}
	lastTime = curTime;

	explodeMolecule(*agdc, getModeByIndex(this->explosionModeParam.Param<param::EnumParam>()->Value()),
		theParam);

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