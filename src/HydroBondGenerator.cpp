/*
 *	HydroBondGenerator.cpp
 *	
 *	Copyright (C) 2016 by University of Stuttgart (VISUS).
 *	All rights reserved.
 */

#include "stdafx.h"
#include "HydroBondGenerator.h"
#include "GridNeighbourFinder.h"

#include "protein_calls/MolecularDataCall.h"

#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"

#include "vislib/math/ShallowVector.h"
#include "vislib/sys/Log.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;

/*
 *	HydroBondGenerator::~HydroBondGenerator
 */
HydroBondGenerator::HydroBondGenerator() : Module(),
	inDataSlot("dataIn", "Molecular data source (usually PDBLoader)"),
	outDataSlot("dataOut", "The slot providing the molecular data including the generated hydrogen bonds"),
	hBondDistance("Bond::hBondDistance", "Maximal distance of a hydrogen bond in anstrom"),
	hBondDonorAcceptorDistance("Bond::hBondDonorAcceptorDistance", "Maximal distance between donor and acceptor of a hydrogen bond"),
	hBondDonorAcceptorAngle("Bond::hBondDonorAcceptorAngle", "Maximal angle between donor-acceptor and donor-hydrogen in a hydrogen bond"),
	alphaHelixHBonds("Allowed::alphaHelixHBonds", "Shall the H-Bonds inside the alpha helices be computed?"),
	betaSheetHBonds("Allowed::betaSheetHBonds", "Shall the H-Bonds between two beta sheets be computed?"),
	otherHBonds("Allowed::otherHBonds", "Shall all other H-Bonds be computed?"),
	maxHBondsPerAtom("maxHBondsPerAtom", "Maximum number of hydrogen bonds per atom"),
	cAlphaHBonds("cAlphaHBonds", "Fake hydrogen bonds as bonds between two C-alpha atoms") {

	this->inDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable(&this->inDataSlot);

	this->outDataSlot.SetCallback(MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetData), &HydroBondGenerator::getData);
	this->outDataSlot.SetCallback(MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetExtent), &HydroBondGenerator::getExtent);
	this->MakeSlotAvailable(&this->outDataSlot);

	// distance for hydrogen bonds
	this->hBondDistance.SetParameter(new param::FloatParam(1.9f, 0.0f));
	this->MakeSlotAvailable(&this->hBondDistance);

	// distance between donor and acceptor of the hydrogen bonds
	this->hBondDonorAcceptorDistance.SetParameter(new param::FloatParam(3.5f, 0.0f));
	this->MakeSlotAvailable(&this->hBondDonorAcceptorDistance);

	// angle between donor-acceptor and donor-hydrogen in degrees
	this->hBondDonorAcceptorAngle.SetParameter(new param::FloatParam(30.0f, 0.0f));
	this->MakeSlotAvailable(&this->hBondDonorAcceptorAngle);

	// maximal number of hydrogen bonds per atom
	this->maxHBondsPerAtom.SetParameter(new param::IntParam(1, 0, 4));
	this->MakeSlotAvailable(&this->maxHBondsPerAtom);

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
 *	HydroBondGenerator::~HydroBondGenerator
 */
HydroBondGenerator::~HydroBondGenerator(void) {
	this->Release();
}

/*
 *	HydroBondGenerator::computeHBonds
 */
void HydroBondGenerator::computeHBonds(MolecularDataCall& mdc) {

	float radius = this->hBondDonorAcceptorDistance.Param<param::FloatParam>()->Value();
	GridNeighbourFinder<float> finder;
	finder.SetPointData(mdc.AtomPositions(), mdc.AtomCount(), mdc.AccessBoundingBoxes().ObjectSpaceBBox(), radius);

	unsigned int atomCount = mdc.AtomCount();
	vislib::Array<unsigned int> resArray;
	std::vector<bool> isO(atomCount, false);
	std::vector<bool> isN(atomCount, false);
	std::vector<bool> isH(atomCount, false);

	for (unsigned int i = 0; i < atomCount; i++) {
		auto atType = mdc.AtomTypes()[mdc.AtomTypeIndices()[i]];
		isO[i] = atType.Element().Equals("O", false);
		isN[i] = atType.Element().Equals("N", false);
		isH[i] = atType.Element().Equals("H", false);
	}

	this->hydrogenBonds.clear();

	for (unsigned int i = 0; i < atomCount; i++) {
		// H-Bonds only with N and O as donor/acceptor
		if (isO[i] || isN[i]) {
			resArray.Clear();
			finder.FindNeighboursInRange(&mdc.AtomPositions()[i * 3], radius, resArray);
			unsigned int start, stop;
			bool res = this->findConnections(i, start, stop);
			if (res) {
				// go through the connected atoms and search for hydrogens
				for (unsigned int j = start; j <= stop; j++) {
					if (isH[this->connections[j].second]) {
						// if the connected atom is a hydrogen atom we search for neighboring atoms that may fit
						std::vector<HBond> resVec;
						for (unsigned int k = 0; k < resArray.Count(); k++) {
							unsigned int idx = resArray[k];
							if (idx != i && (isO[idx] || isN[idx])) {
								// possible H-Bond found
								float angle;
								if (isValidHBond(i, idx, this->connections[j].second, mdc, angle)) {
									resVec.push_back(HBond(i, idx, this->connections[j].second, angle));
								}
							}
						}

						std::sort(resVec.rbegin(), resVec.rend(), [](const HBond & a, const HBond & b) -> bool {
							return a.angle < b.angle;
						}); // sort ascending by angle

						unsigned int num = std::min(1u, static_cast<unsigned int>(resVec.size())); // TODO only one hydrogen bond per H-atom possible?
						for (unsigned int z = 0; z < num; z++) {
							this->hydrogenBonds.push_back(resVec[z]);
						}
					}
				}
			}
		}
	}

	// Delete H-bonds at atoms with too many bonds. H-Bonds with larger angles get deleted first
	unsigned int maxNumber = static_cast<unsigned int>(this->maxHBondsPerAtom.Param<param::IntParam>()->Value());
	if (maxNumber == 0) this->hydrogenBonds.clear();

	if (this->hydrogenBonds.size() > 0) {
		std::sort(this->hydrogenBonds.begin(), this->hydrogenBonds.end());
		unsigned int lastVal = this->hydrogenBonds[0].donorIdx;
		unsigned int counter = 1;
		for (unsigned int i = 1; i < this->hydrogenBonds.size(); i++) {
			if (lastVal == this->hydrogenBonds[i].donorIdx) {
				counter++;
			}
			else {
				counter = 1;
				lastVal = this->hydrogenBonds[i].donorIdx;
			}
			//this->hydrogenBonds[i].print();
			if (counter > maxNumber) {
				this->hydrogenBonds.erase(this->hydrogenBonds.begin() + i);
				i--;
			}
		}
	}
}

/*
 *	HydroBondGenerator::create
 */
bool HydroBondGenerator::create(void) {
	return true;
}

/*
 *	HydroBondGenerator::fillSecStructVector
 */
void HydroBondGenerator::fillSecStructVector(MolecularDataCall& mdc) {

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

				MolecularDataCall::AminoAcid * acid;

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
 *	HydroBondGenerator::findConnections
 */
bool HydroBondGenerator::findConnections(unsigned int atomIdx, unsigned int & firstIdx, unsigned int & lastIdx) {
	auto res = this->numConnections[atomIdx];
	if (res < 1) return false;
	firstIdx = this->connectionStart[atomIdx];
	lastIdx = this->connectionStart[atomIdx] + res - 1;
	return true;
}

/*
 *	HydroBondGenerator::getData
 */
bool HydroBondGenerator::getData(Call& call) {
	MolecularDataCall * outCall = dynamic_cast<MolecularDataCall*>(&call);
	MolecularDataCall * inCall = this->inDataSlot.CallAs<MolecularDataCall>();

	if (!outCall || !inCall) {
		return false;
	}
	inCall->SetFrameID(outCall->FrameID());
	if (!(*inCall)(MolecularDataCall::CallForGetData)) {
		return false;
	}

	*outCall = *inCall; // deep copy

	// create new connection arrays if the incoming data has changed
	if (lastDataHash != inCall->DataHash()) {
		this->connections.resize(inCall->ConnectionCount() * 2);
		this->numConnections.resize(inCall->AtomCount(), 0);
		this->connectionStart.resize(inCall->AtomCount(), 0);
		for (unsigned int i = 0; i < inCall->ConnectionCount(); i++) {
			std::pair<unsigned int, unsigned int> p(inCall->Connection()[i * 2], inCall->Connection()[i * 2 + 1]);
			std::pair<unsigned int, unsigned int> p2(inCall->Connection()[i * 2 + 1], inCall->Connection()[i * 2]);
			this->connections[i * 2] = p;
			this->connections[i * 2 + 1] = p2; // add the reverse direction, too
		}
		std::sort(this->connections.rbegin(), this->connections.rend()); // sort ascending

		// build the necessary statistics for fast search. The reverse iteration makes everything easier
		for (int i = static_cast<int>(this->connections.size() - 1); i >= 0; i--) {
			auto con = this->connections[i];
			//printf("(%u , %u)\n", con.first, con.second);
			this->numConnections[con.first]++;
			this->connectionStart[con.first] = i;
		}
	}

	bool newBonds = false;
	if (this->hBondDistance.IsDirty() || this->hBondDonorAcceptorAngle.IsDirty() || this->hBondDonorAcceptorDistance.IsDirty() ||
		lastDataHash != inCall->DataHash() || this->maxHBondsPerAtom.IsDirty()) {

		this->fillSecStructVector(*outCall);
		
		this->computeHBonds(*outCall);
		newBonds = true;

		this->lastDataHash = inCall->DataHash();
		this->hBondDistance.ResetDirty();
		this->hBondDonorAcceptorAngle.ResetDirty();
		this->hBondDonorAcceptorDistance.ResetDirty();
		this->maxHBondsPerAtom.ResetDirty();
	}

	if (newBonds || this->alphaHelixHBonds.IsDirty() || this->betaSheetHBonds.IsDirty() || this->otherHBonds.IsDirty() || 
		this->cAlphaHBonds.IsDirty()) {

		this->postProcessHBonds(*outCall);

		this->alphaHelixHBonds.ResetDirty();
		this->betaSheetHBonds.ResetDirty();
		this->otherHBonds.ResetDirty();
		this->cAlphaHBonds.ResetDirty();
	}

	outCall->SetAtomHydrogenBondDistance(this->hBondDistance.Param<param::FloatParam>()->Value());
	outCall->SetAtomHydrogenBondStatistics(this->hBondStatistics.data());
	outCall->SetAtomHydrogenBondIndices(this->hBondIndices.data());

	return true;
}

/*
 *	HydroBondGenerator::getExtent
 */
bool HydroBondGenerator::getExtent(Call& call) {
	MolecularDataCall * outCall = dynamic_cast<MolecularDataCall*>(&call);
	MolecularDataCall * inCall = this->inDataSlot.CallAs<MolecularDataCall>();

	if (!outCall || !inCall) {
		return false;
	}

	if (!(*inCall)(MolecularDataCall::CallForGetExtent))
		return false;

	// increment local data hash if data will change
	if (this->hBondDistance.IsDirty() || this->hBondDonorAcceptorAngle.IsDirty() || this->hBondDonorAcceptorDistance.IsDirty() ||
		this->alphaHelixHBonds.IsDirty() || this->betaSheetHBonds.IsDirty() || this->otherHBonds.IsDirty() || 
		this->cAlphaHBonds.IsDirty() || this->maxHBondsPerAtom.IsDirty()) {

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
 *	HydroBondGenerator::isValidHBond
 */
bool HydroBondGenerator::isValidHBond(unsigned int donorIndex, unsigned int acceptorIndex, unsigned int hydrogenIndex, MolecularDataCall& mdc, float & angle) {

	// shallow vectors do not work because of constness
	vislib::math::Vector<float, 3> donorPos = vislib::math::Vector<float, 3>(&mdc.AtomPositions()[donorIndex * 3]);
	vislib::math::Vector<float, 3> acceptorPos = vislib::math::Vector<float, 3>(&mdc.AtomPositions()[acceptorIndex * 3]);
	vislib::math::Vector<float, 3> hydrogenPos = vislib::math::Vector<float, 3>(&mdc.AtomPositions()[hydrogenIndex * 3]);

	vislib::math::Vector<float, 3> DToA(acceptorPos - donorPos);
	vislib::math::Vector<float, 3> DToH(hydrogenPos - donorPos);

	// the distance between acceptor and donator has to be below a threshold
	if (DToA.Length() > this->hBondDonorAcceptorDistance.Param<param::FloatParam>()->Value()) {
		return false;
	} 

	// the distance between hydrogen and acceptor has to be below a threshold
	if (DToH.Length() > this->hBondDistance.Param<param::FloatParam>()->Value()) {
		return false;
	}

	// the angle has to be below a threshold
	angle = vislib::math::AngleRad2Deg(DToA.Angle(DToH));
	bool reverse = DToA.Dot(DToH) < 0.0f;

	if (angle > this->hBondDonorAcceptorAngle.Param<param::FloatParam>()->Value() || reverse) {
		return false;
	}

	return true;
}

/*
 *	HydroBondGenerator::postProcessHBonds
 */
void HydroBondGenerator::postProcessHBonds(MolecularDataCall& mdc) {

	std::vector<bool> copyVector(this->hydrogenBonds.size(), false);
	unsigned int copyCount = 0;

	bool copyAlpha = this->alphaHelixHBonds.Param<param::BoolParam>()->Value();
	bool copyBeta = this->betaSheetHBonds.Param<param::BoolParam>()->Value();
	bool copyOther = this->otherHBonds.Param<param::BoolParam>()->Value();
	bool fake = this->cAlphaHBonds.Param<param::BoolParam>()->Value();

	// determine which H-Bonds have to be copied
	for (unsigned int i = 0; i < static_cast<unsigned int>(this->hydrogenBonds.size()); i++) {
		HBond bond = this->hydrogenBonds[i];
		auto secStructDonor = this->secStructPerAtom[bond.donorIdx];
		auto secStructAcceptor = this->secStructPerAtom[bond.acceptorIdx];

		bool copy = false;

		if (secStructDonor == secStructAcceptor) { // inside of a secondary structure element
			if (secStructDonor == MolecularDataCall::SecStructure::ElementType::TYPE_SHEET && copyBeta) {
				// beta sheets are always copied, if allowed
				copy = true;
			}
			if (secStructDonor == MolecularDataCall::SecStructure::ElementType::TYPE_HELIX) {
				// alpha sheets are only copied if it is the same alpha sheet
				unsigned int min = std::min(bond.donorIdx, bond.acceptorIdx);
				unsigned int max = std::max(bond.donorIdx, bond.acceptorIdx);

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
			}
		}
		else { // random coil or between different elements
			copy = copyOther;
		}

		copyVector[i] = copy;
		if (copy) {
			copyCount++;
		}
	}

	this->hBondStatistics.resize(mdc.AtomCount(), 0);
	this->hBondIndices.resize(mdc.AtomCount(), -1);

	// copy the H-Bonds
	for (unsigned int i = 0; i < static_cast<unsigned int>(this->hydrogenBonds.size()); i++) {
		if (copyVector[i]) {
			HBond bond = this->hydrogenBonds[i];
			if (fake) {
				unsigned int donor = this->cAlphaIndicesPerAtom[bond.donorIdx];
				unsigned int acceptor = this->cAlphaIndicesPerAtom[bond.acceptorIdx];
				this->hBondStatistics[acceptor]++;
				this->hBondStatistics[donor]++;
				this->hBondIndices[acceptor] = bond.hydrogenIdx;
				this->hBondIndices[donor] = bond.hydrogenIdx;
			} else {
				this->hBondStatistics[bond.acceptorIdx]++;
				this->hBondStatistics[bond.donorIdx]++;
				this->hBondIndices[bond.acceptorIdx] = bond.hydrogenIdx;
				this->hBondIndices[bond.donorIdx] = bond.hydrogenIdx;
			}
		}
	}

	vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "%u hydrogen bonds detected.\n", copyCount);
}

/*
 *	HydroBondGenerator::release
 */
void HydroBondGenerator::release(void) {
}