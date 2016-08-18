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

#include "vislib/math/Vector.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;

/*
 *	ProteinExploder::Frame::Frame
 */
ProteinExploder::Frame::Frame(view::AnimDataModule& owner) : view::AnimDataModule::Frame(owner), atomCount(0),
		maxBFactor(0), minBFactor(0),
		maxCharge(0), minCharge(0),
		maxOccupancy(0), minOccupancy(0) {
	// Intentionally empty
}

/*
 *	ProteinExploder::Frame::~Frame
 */
ProteinExploder::Frame::~Frame(void) {
}

/*
 *	ProteinExploder::Frame::operator==
 */
bool ProteinExploder::Frame::operator==(const ProteinExploder::Frame& rhs) {
	// TODO
	return true;
}

/*
 *	ProteinExploder::Frame::setFrameIdx
 */
void ProteinExploder::Frame::setFrameIdx(int idx) {
	this->frame = idx;
}

/*
 *	ProteinExploder::Frame::SetAtomPosition
 */
bool ProteinExploder::Frame::SetAtomPosition(unsigned int idx, float x, float y, float z) {
	if (idx >= this->atomCount) return false;
	this->atomPosition[idx * 3 + 0] = x;
	this->atomPosition[idx * 3 + 1] = y;
	this->atomPosition[idx * 3 + 2] = z;
	return true;
}

/*
 *	ProteinExploder::Frame::SetAtomBFactor
 */
bool ProteinExploder::Frame::SetAtomBFactor(unsigned int idx, float val) {
	if (idx >= this->atomCount) return false;
	this->bfactor[idx] = val;
	return true;
}

/*
 *	ProteinExploder::Frame::SetAtomCharge
 */
bool ProteinExploder::Frame::SetAtomCharge(unsigned int idx, float val) {
	if (idx >= this->atomCount) return false;
	this->charge[idx] = val;
	return true;
}

/*
 *	ProteinExploder::Frame::SetAtomOccupancy
 */
bool ProteinExploder::Frame::SetAtomOccupancy(unsigned int idx, float val) {
	if (idx >= this->atomCount) return false;
	this->occupancy[idx] = val;
	return true;
}

/*
 *	ProteinExploder::ProteinExploder
 */
ProteinExploder::ProteinExploder(void) : 
		AnimDataModule(),
		getDataSlot("getData", "Calls molecular data"),
		dataOutSlot("dataOut", "Provides the exploded molecular data"),
		explosionModeParam("explosionMode", "The mode of the performed explosion"),
		minDistanceParam("minDistance", "Minimal distance between two exploded components in angstrom") {

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

	this->atomPositions = NULL;
	this->atomPositionsSize = 0;
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
 *	ProteinExploder::constructFrame
 */
view::AnimDataModule::Frame* ProteinExploder::constructFrame(void) const {
	Frame *f = new Frame(*const_cast<ProteinExploder*>(this));
	// TODO
	return f;
}

/*
 *	ProteinExploder::loadFrame
 */
void ProteinExploder::loadFrame(view::AnimDataModule::Frame *frame, unsigned int idx) {
	ProteinExploder::Frame *fr = dynamic_cast<ProteinExploder::Frame*>(frame);
	fr->setFrameIdx(idx);

	// TODO
}

/*
 *	ProteinExploder::explodeMolecule
 */
void ProteinExploder::explodeMolecule(MolecularDataCall& call, ProteinExploder::ExplosionMode mode, float minDistance) {

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
				this->atomPositions[3 * atomIdx + 0] = call.AtomPositions()[3 * atomIdx + 0] + minDistance * displaceDirections[molIdx].GetX();
				this->atomPositions[3 * atomIdx + 1] = call.AtomPositions()[3 * atomIdx + 1] + minDistance * displaceDirections[molIdx].GetY();
				this->atomPositions[3 * atomIdx + 2] = call.AtomPositions()[3 * atomIdx + 2] + minDistance * displaceDirections[molIdx].GetZ();
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

	currentBoundingBox.Set(bbMin.X(), bbMin.Y(), bbMin.Z(), bbMax.X(), bbMax.Y(), bbMax.Z());
	currentBoundingBox.Grow(3.0f); // add 3 angstrom to each side for some renderers
}

/*
 *	ProteinExploder::getData
 */
bool ProteinExploder::getData(core::Call& call) {
	MolecularDataCall * outCall = dynamic_cast<MolecularDataCall*>(&call);
	if (outCall == NULL) return false;

	/*MolecularDataCall * inCall = this->getDataSlot.CallAs<MolecularDataCall>();
	if (inCall == NULL) return false;*/

	/*inCall->SetFrameID(outCall->FrameID());
	if (!(*inCall)(1)) return false;
	if (!(*inCall)(0)) return false;*/

	//outCall->operator=(*inCall); // deep copy

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

	explodeMolecule(*agdc, getModeByIndex(this->explosionModeParam.Param<param::EnumParam>()->Value()),
		this->minDistanceParam.Param<param::FloatParam>()->Value());

	agdc->SetFrameCount(mdc->FrameCount());
	agdc->AccessBoundingBoxes().Clear();
	agdc->AccessBoundingBoxes().SetObjectSpaceBBox(currentBoundingBox);
	agdc->AccessBoundingBoxes().SetObjectSpaceClipBox(currentBoundingBox);

	// TODO set correct extents

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