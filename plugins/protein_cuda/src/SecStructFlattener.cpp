/*
 *	SecStructFlattener.cpp
 *
 *	Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 *	All rights reserved
 */

#include "stdafx.h"
#include "SecStructFlattener.h"
#include "PlaneDataCall.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include "vislib/math/Matrix.h"
#include "vislib/math/ShallowVector.h"
#include "vislib/sys/Log.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_cuda;
using namespace megamol::protein_calls;

/*
 *	SecStructFlattener::SecStructFlattener
 */
SecStructFlattener::SecStructFlattener(void) :
	Module(),
	getDataSlot("getData", "Calls molecular data"),
	dataOutSlot("dataOut", "Provides the flattened molecular data"),
	planeOutSlot("planeOut", "Provides the necessary plane data for 2D renderings"),
	playParam("simulation::play", "Should the simulation be played?"),
	playButtonParam("simulation::playButton", "Button to toggle the simulation"),
	singleStepButtonParam("simulation::singleStep", "Button for the computation of a single timestep"),
	flatPlaneMode("plane::flatPlaneMode", "The plane the protein gets flattened to"),
	arbPlaneCenterParam("plane::planeOrigin", "The plane origin for the arbitrary plane mode"),
	arbPlaneNormalParam("plane::planeNormal", "The plane normal for the arbitrary plane mode"),
	oxygenOffsetParam("plane::preserveDirections", "Should the directions between c alpha and oxygen atoms be preserved?"),
	timestepParam("simulation::timestepSize", "The length of single time step of the force directed simulation"),
	timestepsPerFrameParam("simulation::timestepsPerFrame", "The number of simulation timesteps that are performed per frame"),
	maxTimestepParam("simulation::maxTimestep", "The index of the last performed timestep. A negative number means infinite running."),
	connectionSpringConstantParam("simulation::springs::conSpringConstant", "The spring constant for the atom connection springs."),
	connectionFrictionParam("simulation::springs::conFriction", "The friction parameter for the atom connection springs."),
	hbondSpringConstantParam("simulation::springs::hbondSpringConstant", "The spring constant for the h bond springs."),
	hbondFrictionParam("simulation::springs::hbondFriction", "The friction parameter for the h bond springs."),
	repellingForceCutoffDistanceParam("simulation::repelling::cutoffDistance", "Cutoff distance after which no repelling forces happen."),
	repellingForceStrengthFactor("simulation::repelling::strengthFactor", "Factor controlling the strength of the repelling forces."),
	forceToCenterParam("simulation::forceToCenter", "Activate a force towards the center of the bounding box."),
	forceToCenterStrengthParam("simulation::forceToCenterStrength", "The strength of the force towards the center."),
	resetButtonParam("simulation::resetButton", "Button to reset the simulation to the start state."){

	// caller slot
	this->getDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable(&this->getDataSlot);

	// callee slot
	this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(0), &SecStructFlattener::getData);
	this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(1), &SecStructFlattener::getExtent);
	this->MakeSlotAvailable(&this->dataOutSlot);

	this->planeOutSlot.SetCallback(PlaneDataCall::ClassName(), PlaneDataCall::FunctionName(0), &SecStructFlattener::getPlaneData);
	this->planeOutSlot.SetCallback(PlaneDataCall::ClassName(), PlaneDataCall::FunctionName(1), &SecStructFlattener::getPlaneExtent);
	this->MakeSlotAvailable(&this->planeOutSlot);

	this->timestepParam.SetParameter(new param::FloatParam(0.001f, 0.0f, 1.0f));
	this->MakeSlotAvailable(&this->timestepParam);

	this->timestepsPerFrameParam.SetParameter(new param::IntParam(1, 1, 100));
	this->MakeSlotAvailable(&this->timestepsPerFrameParam);

	this->maxTimestepParam.SetParameter(new param::IntParam(-1, -1, INT_MAX));
	this->MakeSlotAvailable(&this->maxTimestepParam);

	float minConstant = 0.0f;
	float maxConstant = 2000.0f;
	float minFriction = 0.0f;
	float maxFriction = 2.0f;

	this->connectionSpringConstantParam.SetParameter(new param::FloatParam(1.0f, minConstant, maxConstant));
	this->MakeSlotAvailable(&this->connectionSpringConstantParam);

	this->connectionFrictionParam.SetParameter(new param::FloatParam(0.5f, minFriction, maxFriction));
	this->MakeSlotAvailable(&this->connectionFrictionParam);

	this->hbondSpringConstantParam.SetParameter(new param::FloatParam(1.0f, minConstant, maxConstant));
	this->MakeSlotAvailable(&this->hbondSpringConstantParam);

	this->hbondFrictionParam.SetParameter(new param::FloatParam(1.0f, minFriction, maxFriction));
	this->MakeSlotAvailable(&this->hbondFrictionParam);

	this->playParam.SetParameter(new param::BoolParam(false));
	this->MakeSlotAvailable(&this->playParam);

	this->playButtonParam << new param::ButtonParam('p');
	this->playButtonParam.SetUpdateCallback(this, &SecStructFlattener::onPlayToggleButton);
	this->MakeSlotAvailable(&this->playButtonParam);

	this->singleStepButtonParam << new param::ButtonParam('o');
	this->singleStepButtonParam.SetUpdateCallback(this, &SecStructFlattener::onSingleStepButton);
	this->MakeSlotAvailable(&this->singleStepButtonParam);

	this->resetButtonParam << new param::ButtonParam('r');
	this->resetButtonParam.SetUpdateCallback(this, &SecStructFlattener::onResetButton);
	this->MakeSlotAvailable(&this->resetButtonParam);

	param::EnumParam * fpParam = new param::EnumParam(int(FlatPlane::XY_PLANE));
	FlatPlane fp;
	for (int i = 0; i < getFlatPlaneModeNumber(); i++) {
		fp = getFlatPlaneByIndex(i);
		fpParam->SetTypePair(fp, getFlatPlaneName(fp).c_str());
	}
	this->flatPlaneMode << fpParam;
	this->MakeSlotAvailable(&this->flatPlaneMode);

	const vislib::math::Vector<float, 3> orig(0.0f, 0.0f, 0.0f);
	this->arbPlaneCenterParam.SetParameter(new param::Vector3fParam(orig));
	this->MakeSlotAvailable(&this->arbPlaneCenterParam);

	const vislib::math::Vector<float, 3> orig2(0.0f, 0.0f, 1.0f);
	this->arbPlaneNormalParam.SetParameter(new param::Vector3fParam(orig2));
	this->MakeSlotAvailable(&this->arbPlaneNormalParam);

	this->oxygenOffsetParam.SetParameter(new param::BoolParam(true));
	this->MakeSlotAvailable(&this->oxygenOffsetParam);

	this->repellingForceCutoffDistanceParam.SetParameter(new param::FloatParam(6.0f, 0.0f));
	this->MakeSlotAvailable(&this->repellingForceCutoffDistanceParam);

	this->repellingForceStrengthFactor.SetParameter(new param::FloatParam(1.0f, 0.0f));
	this->MakeSlotAvailable(&this->repellingForceStrengthFactor);

	this->forceToCenterParam.SetParameter(new param::BoolParam(false));
	this->MakeSlotAvailable(&this->forceToCenterParam);

	this->forceToCenterStrengthParam.SetParameter(new param::FloatParam(1.0f, 0.0f));
	this->MakeSlotAvailable(&this->forceToCenterStrengthParam);

	this->atomPositions = NULL;
	this->atomPositionsSize = 0;

	this->lastHash = 0;
	this->hashOffset = 0;
	this->myHash = 0;
	this->planeHash = 0;
	this->firstFrame = true;
	this->forceReset = false;

	this->lastPlaneMode = XY_PLANE;

	this->flatPlaneMode.ForceSetDirty();
	this->mainDirections.resize(3);

	this->currentTimestep = 0;
}

/*
 *	SecStructFlattener::~SecStructFlattener
 */
SecStructFlattener::~SecStructFlattener(void) {
	this->Release();
}

/*
 *	SecStructFlattener::create
 */
bool SecStructFlattener::create(void) {
	return true;
}

/*
 *	SecStructFlattener::release
 */
void SecStructFlattener::release(void) {
	clearAll();
	if (this->atomPositions != NULL) {
		delete[] this->atomPositions;
		this->atomPositions = NULL;
		this->atomPositionsSize = 0;
	}
}

/*
 *	SecStructFlattener::computeMainDirectionPCA
 */
void SecStructFlattener::computeMainDirectionPCA(void) {
	vislib::math::Matrix<float, 3, vislib::math::ROW_MAJOR> covMat;
	covMat.SetNull();

	// compute the midpoint of the data set
	vislib::math::Vector<float, 3> midpoint(0.0f, 0.0f, 0.0f);
	for (unsigned int k = 0; k < cAlphaIndices.size(); k++) {
		vislib::math::ShallowVector<const float, 3> p1(&this->atomPositions[this->cAlphaIndices[k] * 3]);
		midpoint += p1;
	}
	midpoint /= static_cast<float>(cAlphaIndices.size());

	// compute covariance matrix
	for (unsigned int x = 0; x < 3; x++) {
		for (unsigned int y = 0; y < 3; y++) {
			for (unsigned int k = 0; k < cAlphaIndices.size(); k++) {
				vislib::math::ShallowVector<const float, 3> p1(&this->atomPositions[this->cAlphaIndices[k] * 3]);
				covMat(x, y) += (p1[x] - midpoint[x]) * (p1[y] - midpoint[y]);
			}
			covMat(x, y) /= static_cast<float>(cAlphaIndices.size() - 1);
		}
	}
	//covMat.Dump(std::cout);

	float eigenVals[3];
	vislib::math::Vector<float, 3> eigenVectors[3];
	covMat.FindEigenvalues(eigenVals, eigenVectors, 3);
	std::vector<unsigned int> indexVec = { 0, 1, 2 };

	std::sort(indexVec.begin(), indexVec.end(), [&eigenVals](const unsigned int& a, const unsigned int& b) {
		return eigenVals[a] > eigenVals[b];
	});

	for (int i = 0; i < 3; i++) {
		mainDirections[i] = eigenVectors[indexVec[i]];
		mainDirections[i].Normalise();
	}
}

/*
 *	SecStructFlattener::flatten
 */
void SecStructFlattener::flatten(bool transferPositions) {

	auto bbCenter = this->boundingBox.CalcCenter();
	vislib::math::Vector<float, 3> bbCenterVec(bbCenter.GetX(), bbCenter.GetY(), bbCenter.GetZ());
	vislib::math::Vector<float, 3> origin(0.0f, 0.0f, 0.0f);

	if (this->firstFrame) {
		this->arbPlaneCenterParam.Param<param::Vector3fParam>()->SetValue(origin);
		this->firstFrame = false;
	} else {
		bbCenterVec = this->arbPlaneCenterParam.Param<param::Vector3fParam>()->Value();
	}

	bool somethingDirty = false;

	if (this->flatPlaneMode.IsDirty() || this->arbPlaneCenterParam.IsDirty() || this->arbPlaneNormalParam.IsDirty() || this->oxygenOffsetParam.IsDirty() || transferPositions) {
		this->flatPlaneMode.ResetDirty();
		this->arbPlaneCenterParam.ResetDirty();
		this->arbPlaneNormalParam.ResetDirty();
		this->oxygenOffsetParam.ResetDirty();
		this->hashOffset++;

		somethingDirty = true;

		vislib::math::Vector<float, 3> n = this->arbPlaneNormalParam.Param<param::Vector3fParam>()->Value();
		n.Normalise();

		switch (this->flatPlaneMode.Param<param::EnumParam>()->Value()) {
			case XY_PLANE:
				for (unsigned int i = 0; i < this->atomPositionsSize / 3; i++) {
					this->atomPositions[i * 3 + 2] = bbCenterVec.GetZ();
				}
				break;
			case XZ_PLANE:
				for (unsigned int i = 0; i < this->atomPositionsSize / 3; i++) {
					this->atomPositions[i * 3 + 1] = bbCenterVec.GetY();
				}
				break;
			case YZ_PLANE:
				for (unsigned int i = 0; i < this->atomPositionsSize / 3; i++) {
					this->atomPositions[i * 3 + 0] = bbCenterVec.GetX();
				}
				break;
			case LEAST_COMMON:
				computeMainDirectionPCA();
				// project points onto plane
				for (unsigned int i = 0; i < this->atomPositionsSize / 3; i++) {
					vislib::math::ShallowVector<float, 3> atomPos(&this->atomPositions[i * 3]);
					vislib::math::Vector<float, 3> v(&this->atomPositions[i * 3]);
					v = v - bbCenterVec;
					float dist = v.Dot(mainDirections[2]);
					atomPos = atomPos - dist * mainDirections[2];
				}
				break;
			case ARBITRARY:
				// project points onto plane
				for (unsigned int i = 0; i < this->atomPositionsSize / 3; i++) {
					vislib::math::ShallowVector<float, 3> atomPos(&this->atomPositions[i * 3]);
					vislib::math::Vector<float, 3> v(&this->atomPositions[i * 3]);
					v = v - bbCenterVec;
					float dist = v.Dot(n);
					atomPos = atomPos - dist * n;
				}
				break;
			default:
				break;
		}

		if (this->oxygenOffsetParam.Param<param::BoolParam>()->Value()) {
			for (unsigned int i = 0; i < this->oxygenOffsets.size(); i++) {
				vislib::math::ShallowVector<float, 3> oPos(&this->atomPositions[oIndices[i] * 3]);
				vislib::math::ShallowVector<float, 3> cPos(&this->atomPositions[cAlphaIndices[i] * 3]);
				oPos = cPos + oxygenOffsets[i];
			}
		}

		// transfer the plane to the GPU if a new one is available
		vislib::math::Vector<float, 3> normal;
		switch (this->flatPlaneMode.Param<param::EnumParam>()->Value()) {
			case XY_PLANE:
				normal = vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f);
				break;
			case XZ_PLANE:
				normal = vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f);
				break;
			case YZ_PLANE:
				normal = vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f);
				break;
			case LEAST_COMMON:
				normal = this->mainDirections[2];
				normal.Normalise();
				break;
			case ARBITRARY:
				normal = this->arbPlaneNormalParam.Param<param::Vector3fParam>()->Value();
				normal.Normalise();
				break;
			default:
				normal = vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f);
				break;
		}
		auto pc = this->arbPlaneCenterParam.Param<param::Vector3fParam>()->Value();
		vislib::math::Point<float, 3> p(pc.GetX(), pc.GetY(), pc.GetZ());
		vislib::math::Plane<float> thePlane(p, normal);
		transferPlane(thePlane);

		planeHash++;
	}

	if (somethingDirty || transferPositions) {
		transferAtomData(this->atomPositions, this->atomPositionsSize / 3, this->cAlphaIndices.data(), static_cast<unsigned int>(this->cAlphaIndices.size()));
	}
}

/*
 *	SecStructFlattener::getFlatPlaneByIndex
 */
SecStructFlattener::FlatPlane SecStructFlattener::getFlatPlaneByIndex(unsigned int idx) {
	switch (idx) {
		case 0:		return FlatPlane::XY_PLANE;
		case 1:		return FlatPlane::XZ_PLANE;
		case 2:		return FlatPlane::YZ_PLANE;
		case 3:		return FlatPlane::LEAST_COMMON;
		case 4:		return FlatPlane::ARBITRARY;
		default:	return FlatPlane::XY_PLANE;
	}
}

/*
 *	SecStructFlattener::getFlatPlaneModeNumber
 */
int SecStructFlattener::getFlatPlaneModeNumber(void) {
	return 5;
}

/*
 *	SecStructFlattener::getFlatPlaneName
 */
std::string SecStructFlattener::getFlatPlaneName(SecStructFlattener::FlatPlane fp) {
	switch (fp) {
		case XY_PLANE		: return "XY Plane";
		case XZ_PLANE		: return "XZ Plane";
		case YZ_PLANE		: return "YZ Plane";
		case LEAST_COMMON	: return "Least Common";
		case ARBITRARY		: return "Arbitrary";
		default				: return "";
	}
}

/*
 *	SecStructFlattener::getData
 */
bool SecStructFlattener::getData(core::Call& call) {
	MolecularDataCall * outCall = dynamic_cast<MolecularDataCall*>(&call);
	if (outCall == NULL) return false;

	this->myHash = this->lastHash + this->hashOffset;

	outCall->SetDataHash(this->myHash);
	outCall->SetAtomPositions(this->atomPositions);

	outCall->AccessBoundingBoxes().SetObjectSpaceBBox(this->boundingBox);
	outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(this->boundingBox);

	return true;
}

/*
 *	SecStructFlattener::getExtent
 */
bool SecStructFlattener::getExtent(core::Call& call) {
	
	MolecularDataCall * agdc = dynamic_cast<MolecularDataCall*>(&call);
	if (agdc == NULL) return false;

	MolecularDataCall *mdc = this->getDataSlot.CallAs<MolecularDataCall>();
	if (mdc == NULL) return false;
	mdc->SetCalltime(agdc->Calltime());
	if (!(*mdc)(1)) return false;
	if (!(*mdc)(0)) return false;

	agdc->operator=(*mdc); // deep copy

	std::vector<float> atomRadii(mdc->AtomCount(), 0.0f);

	// transfer new spring data if necessary
	if (this->hbondFrictionParam.IsDirty() || this->hbondSpringConstantParam.IsDirty() || this->connectionFrictionParam.IsDirty()
		|| this->connectionSpringConstantParam.IsDirty() || this->repellingForceCutoffDistanceParam.IsDirty() || this->repellingForceStrengthFactor.IsDirty() 
		|| lastHash != mdc->DataHash() /*important if input data changes*/
		|| forceReset) {

		this->hbondFrictionParam.ResetDirty();
		this->hbondSpringConstantParam.ResetDirty();
		this->connectionFrictionParam.ResetDirty();
		this->connectionSpringConstantParam.ResetDirty();
		this->repellingForceCutoffDistanceParam.ResetDirty();
		this->repellingForceStrengthFactor.ResetDirty();

		cAlphaIndices.clear();
		oIndices.clear();

		// fill the vector for the molecule starts
		std::vector<unsigned int> moleculeStarts(mdc->MoleculeCount());
		for (unsigned int i = 0; i < mdc->MoleculeCount(); i++) {
			unsigned int idx = mdc->Molecules()[i].FirstResidueIndex();
			moleculeStarts[i] = mdc->Residues()[idx]->FirstAtomIndex();
		}

		for (unsigned int i = 0; i < mdc->AtomCount(); i++) {
			// check the relevant atom types
			vislib::StringA elName = mdc->AtomTypes()[mdc->AtomTypeIndices()[i]].Name();
			elName.ToLowerCase();
			elName.TrimSpaces();
			if (elName.StartsWith("ca")) cAlphaIndices.push_back(i);
			if (elName.StartsWith("o") && elName.Length() == 1) oIndices.push_back(i); // cut out all o atoms besides the first per aminoacid
		}

		transferSpringData(mdc->AtomPositions(), mdc->AtomCount(), mdc->GetHydrogenBonds(), mdc->HydrogenBondCount(), this->cAlphaIndices.data(), static_cast<unsigned int>(this->cAlphaIndices.size()), 
			this->oIndices.data(), static_cast<unsigned int>(this->oIndices.size()), this->connectionFrictionParam.Param<param::FloatParam>()->Value(),
			this->connectionSpringConstantParam.Param<param::FloatParam>()->Value(), this->hbondFrictionParam.Param<param::FloatParam>()->Value(),
			this->hbondSpringConstantParam.Param<param::FloatParam>()->Value(), moleculeStarts.data(), static_cast<unsigned int>(moleculeStarts.size()),
			this->repellingForceCutoffDistanceParam.Param<param::FloatParam>()->Value(), this->repellingForceStrengthFactor.Param<param::FloatParam>()->Value());
	}

	if (lastHash != mdc->DataHash() || this->flatPlaneMode.IsDirty() || this->arbPlaneCenterParam.IsDirty() || this->arbPlaneNormalParam.IsDirty() || forceReset) {
		lastHash = mdc->DataHash();
		this->boundingBox = mdc->AccessBoundingBoxes().ObjectSpaceBBox();

		// copy the atom positions to the array used here
		if (atomPositions != NULL) {
			delete[] this->atomPositions;
			this->atomPositions = NULL;
			this->atomPositionsSize = 0;
		}

		atomPositions = new float[mdc->AtomCount() * 3];
		atomPositionsSize = mdc->AtomCount() * 3;
		cAlphaIndices.clear();
		oIndices.clear();

		for (unsigned int i = 0; i < mdc->AtomCount(); i++) {
			atomPositions[i * 3 + 0] = mdc->AtomPositions()[i * 3 + 0];
			atomPositions[i * 3 + 1] = mdc->AtomPositions()[i * 3 + 1];
			atomPositions[i * 3 + 2] = mdc->AtomPositions()[i * 3 + 2];
			atomRadii[i] = mdc->AtomTypes()[mdc->AtomTypeIndices()[i]].Radius();

			// check the relevant atom types
			vislib::StringA elName = mdc->AtomTypes()[mdc->AtomTypeIndices()[i]].Name();
			elName.ToLowerCase();
			elName.TrimSpaces();
			if (elName.StartsWith("ca")) cAlphaIndices.push_back(i);
			if (elName.StartsWith("o") && elName.Length() == 1) oIndices.push_back(i); // cut out all o atoms besides the first per aminoacid
		}

		if (cAlphaIndices.size() != oIndices.size()) {
			vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
				"Malformed molecule (different number of c alpha and primary oxygen atoms)\n");
		}

		this->oxygenOffsets.resize(cAlphaIndices.size());
		for (unsigned int i = 0; i < cAlphaIndices.size(); i++) {
			vislib::math::ShallowVector<float, 3> cPos(&atomPositions[cAlphaIndices[i] * 3]);
			vislib::math::ShallowVector<float, 3> oPos(&atomPositions[oIndices[i] * 3]);
			this->oxygenOffsets[i] = oPos - cPos;
		}
	}

	// perform the flattening
	flatten(forceReset);
	forceReset = false;

	// run the simulation
	if (this->playParam.Param<param::BoolParam>()->Value() || this->oneStep) {
		runSimulation();
	}

	this->myHash = mdc->DataHash() + this->hashOffset;
	agdc->SetDataHash(this->myHash);

	// compute the new bounding box
	vislib::math::Cuboid<float> newbb;
	if (mdc->AtomCount() != 0) {
		vislib::math::Vector<float, 3> p(&atomPositions[0]);
		float r = atomRadii[0];
		newbb = vislib::math::Cuboid<float>(p[0] - r, p[1] - r, p[2] - r, p[0] + r, p[1] + r, p[2] + r);
	}

	for (unsigned int i = 1; i < mdc->AtomCount(); i++) {
		vislib::math::Vector<float, 3> p(&atomPositions[i * 3]);
		float r = atomRadii[i];
		vislib::math::Cuboid<float> b(p[0] - r, p[1] - r, p[2] - r, p[0] + r, p[1] + r, p[2] + r);
		newbb.Union(b);
	}

	newbb.Grow(3.0f);

	this->boundingBox.Union(newbb);
	agdc->AccessBoundingBoxes().SetObjectSpaceBBox(this->boundingBox);
	agdc->AccessBoundingBoxes().SetObjectSpaceClipBox(this->boundingBox);

	return true;
}

/*
 *	SecStructFlattener::getPlaneData
 */
bool SecStructFlattener::getPlaneData(core::Call& call) {
	PlaneDataCall * pdc = dynamic_cast<PlaneDataCall*>(&call);
	if (pdc == NULL) return false;

	vislib::math::Vector<float, 3> pointVector;
	vislib::math::Vector<float, 3> normal;
	pointVector = this->arbPlaneCenterParam.Param<param::Vector3fParam>()->Value();
	vislib::math::Point<float, 3> point(pointVector.GetX(), pointVector.GetY(), pointVector.GetZ());

	switch (this->flatPlaneMode.Param<param::EnumParam>()->Value()) {
		case XY_PLANE:
			normal = vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f);
			break;
		case XZ_PLANE:
			normal = vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f);
			break;
		case YZ_PLANE:
			normal = vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f);
			break;
		case LEAST_COMMON:
			normal = this->mainDirections[2];
			normal.Normalise();
			break;
		case ARBITRARY:
			normal = this->arbPlaneNormalParam.Param<param::Vector3fParam>()->Value();
			normal.Normalise();
			break;
		default:
			normal = vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f);
			break;
	}

	this->currentPlane = vislib::math::Plane<float>(point, normal);
	pdc->SetPlaneData(&this->currentPlane);

	return true;
}

/*
 *	SecStructFlattener::getPlaneExtent
 */
bool SecStructFlattener::getPlaneExtent(core::Call& call) {
	PlaneDataCall * pdc = dynamic_cast<PlaneDataCall*>(&call);
	if (pdc == NULL) return false;

	pdc->SetPlaneCnt(1); // always one plane
	pdc->SetDataHash(this->planeHash);

	return true;
}

/*
 *	SecStructFlattener::onPlayToggleButton
 */
bool SecStructFlattener::onPlayToggleButton(param::ParamSlot& p) {
	param::BoolParam *bp = this->playParam.Param<param::BoolParam>();
	bp->SetValue(!bp->Value());
	bool play = bp->Value();
	return true;
}

/*
 *	SecStructFlattener::onResetButton
 */
bool SecStructFlattener::onResetButton(param::ParamSlot& p) {
	this->forceReset = true;
	return true;
}

/*
 *	SecStructFlattener::onSingleStepButton
 */
bool SecStructFlattener::onSingleStepButton(param::ParamSlot& p) {
	this->oneStep = true;
	return true;
}

/*
 *	SecStructFlattener::runSimulation
 */
void SecStructFlattener::runSimulation(void) {

	unsigned int numTimesteps = static_cast<unsigned int>(this->timestepsPerFrameParam.Param<param::IntParam>()->Value());

	// case for a single timestep
	if (!this->playParam.Param<param::BoolParam>()->Value() && oneStep) {
		numTimesteps = 1;
	}

	oneStep = false;

	float delta = this->timestepParam.Param<param::FloatParam>()->Value();
	int maxTime = this->maxTimestepParam.Param<param::IntParam>()->Value();

	if (maxTime - currentTimestep + 1 < numTimesteps && maxTime >= 0) {
		numTimesteps = static_cast<unsigned int>(maxTime - currentTimestep + 1);
	}

	for (unsigned int i = 0; i < numTimesteps; i++) {
		performTimestep(delta, this->forceToCenterParam.Param<param::BoolParam>()->Value(), 
			this->forceToCenterStrengthParam.Param<param::FloatParam>()->Value());
	}

	// get result from device
	getPositions(this->atomPositions, this->atomPositionsSize / 3);

	if (numTimesteps > 0) {
		this->hashOffset++;
	}
}