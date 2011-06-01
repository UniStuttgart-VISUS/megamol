/*
 * MolecularDataCall.cpp
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "MolecularDataCall.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/mathfunctions.h"
#include "vislib/OutOfRangeException.h"

using namespace megamol;
using namespace megamol::protein;


/*
 * protein::MolecularDataCall::Residue::Residue
 */
protein::MolecularDataCall::Residue::Residue(void) : atomCnt(0), 
        firstAtomIdx(0), boundingBox( 0, 0, 0, 0, 0, 0), type( 0), moleculeIndex(-1),
        filter(1) {
    // intentionally empty
}


/*
 * protein::MolecularDataCall::Residue::Residue
 */
protein::MolecularDataCall::Residue::Residue(
        const protein::MolecularDataCall::Residue& src) : atomCnt( src.atomCnt),
        firstAtomIdx( src.firstAtomIdx), boundingBox( src.boundingBox),
        type( src.type), moleculeIndex(src.moleculeIndex),filter(1) {
    // intentionally empty
}


/*
 * protein::MolecularDataCall::Residue::Residue
 */
protein::MolecularDataCall::Residue::Residue( unsigned int firstAtomIdx,
        unsigned int atomCnt, vislib::math::Cuboid<float> bbox, 
        unsigned int typeIdx, int moleculeIdx) : atomCnt(atomCnt), firstAtomIdx(firstAtomIdx), 
        boundingBox( bbox), type( typeIdx), moleculeIndex(moleculeIdx), filter(1) {
    // intentionally empty
}


/*
 * protein::MolecularDataCall::Residue::~Residue
 */
protein::MolecularDataCall::Residue::~Residue(void) {
    // intentionally empty
}


/*
 * protein::MolecularDataCall::Residue::SetPosition
 */
void protein::MolecularDataCall::Residue::SetPosition( unsigned int firstAtom,
        unsigned int atomCnt) {
    this->firstAtomIdx = firstAtom;
    this->atomCnt = atomCnt;
}


/*
 * protein::MolecularDataCall::Residue::operator=
 */
protein::MolecularDataCall::Residue& 
protein::MolecularDataCall::Residue::operator=(
        const protein::MolecularDataCall::Residue& rhs) {
    this->atomCnt = rhs.atomCnt;
    this->firstAtomIdx = rhs.firstAtomIdx;
    this->boundingBox = rhs.boundingBox;
    this->type = rhs.type;
	this->moleculeIndex = rhs.moleculeIndex;
    this->filter = rhs.filter;
    return *this;
}


/*
 * protein::MolecularDataCall::Residue::operator==
 */
bool protein::MolecularDataCall::Residue::operator==(
        const protein::MolecularDataCall::Residue& rhs) const {
    return ((this->atomCnt == rhs.atomCnt)
        && (this->firstAtomIdx == rhs.firstAtomIdx)
        && (this->boundingBox == rhs.boundingBox)
        && (this->type == rhs.type)
		&& (this->moleculeIndex == rhs.moleculeIndex));
}

// ======================================================================

/*
 * protein::MolecularDataCall::AminoAcid::AminoAcid
 */
protein::MolecularDataCall::AminoAcid::AminoAcid(void) : Residue(), 
        cAlphaIdx(0), cCarbIdx(0), nIdx(0), oIdx(0) {
    // intentionally empty
}


/*
 * protein::MolecularDataCall::AminoAcid::AminoAcid
 */
protein::MolecularDataCall::AminoAcid::AminoAcid(
        const protein::MolecularDataCall::AminoAcid& src) : Residue(
		src.firstAtomIdx, src.atomCnt, src.boundingBox, src.type, src.moleculeIndex), 
        cAlphaIdx(src.cAlphaIdx), cCarbIdx(src.cCarbIdx), 
        nIdx(src.nIdx), oIdx(src.oIdx) {
    // intentionally empty
}


/*
 * protein::MolecularDataCall::AminoAcid::AminoAcid
 */
protein::MolecularDataCall::AminoAcid::AminoAcid(unsigned int firstAtomIdx,
        unsigned int atomCnt, unsigned int cAlphaIdx, unsigned int cCarbIdx,
        unsigned int nIdx, unsigned int oIdx, 
        vislib::math::Cuboid<float> bbox, unsigned int typeIdx, int moleculeIdx) : 
        Residue( firstAtomIdx, atomCnt, bbox, typeIdx, moleculeIdx),
        cAlphaIdx(cAlphaIdx), cCarbIdx(cCarbIdx), nIdx(nIdx), oIdx(oIdx) {
    // intentionally empty
}


/*
 * protein::MolecularDataCall::AminoAcid::~AminoAcid
 */
protein::MolecularDataCall::AminoAcid::~AminoAcid(void) {
    // intentionally empty
}


/*
 * protein::MolecularDataCall::AminoAcid::SetCAlphaIndex
 */
void protein::MolecularDataCall::AminoAcid::SetCAlphaIndex(unsigned int idx) {
    this->cAlphaIdx = idx;
}


/*
 * protein::MolecularDataCall::AminoAcid::SetCCarbIndex
 */
void protein::MolecularDataCall::AminoAcid::SetCCarbIndex(unsigned int idx) {
    this->cCarbIdx = idx;
}


/*
 * protein::MolecularDataCall::AminoAcid::SetNIndex
 */
void protein::MolecularDataCall::AminoAcid::SetNIndex(unsigned int idx) {
    this->nIdx = idx;
}


/*
 * protein::MolecularDataCall::AminoAcid::SetOIndex
 */
void protein::MolecularDataCall::AminoAcid::SetOIndex(unsigned int idx) {
    this->oIdx = idx;
}


/*
 * protein::MolecularDataCall::AminoAcid::operator=
 */
protein::MolecularDataCall::AminoAcid& 
protein::MolecularDataCall::AminoAcid::operator=(
        const protein::MolecularDataCall::AminoAcid& rhs) {
    this->firstAtomIdx = rhs.firstAtomIdx;
    this->atomCnt = rhs.atomCnt;
    this->cAlphaIdx = rhs.cAlphaIdx;
    this->cCarbIdx = rhs.cCarbIdx;
    this->nIdx = rhs.nIdx;
    this->oIdx = rhs.oIdx;
    this->boundingBox = rhs.boundingBox;
    this->type = rhs.type;
    return *this;
}


/*
 * protein::MolecularDataCall::AminoAcid::operator==
 */
bool protein::MolecularDataCall::AminoAcid::operator==(
        const protein::MolecularDataCall::AminoAcid& rhs) const {
    return ((this->firstAtomIdx == rhs.firstAtomIdx)
        && (this->atomCnt == rhs.atomCnt)
        && (this->cAlphaIdx == rhs.cAlphaIdx)
        && (this->cCarbIdx == rhs.cCarbIdx)
        && (this->nIdx == rhs.nIdx)
        && (this->oIdx == rhs.oIdx)
        && (this->boundingBox == rhs.boundingBox)
        && (this->type == rhs.type));
}

// ======================================================================

/*
 * MolecularDataCall::SecStructure::SecStructure
 */
MolecularDataCall::SecStructure::SecStructure(void) : aminoAcidCnt(0),
        firstAminoAcidIdx(0), type(TYPE_COIL) {
    // intentionally empty
}


/*
 * MolecularDataCall::SecStructure::SecStructure
 */
MolecularDataCall::SecStructure::SecStructure(
        const MolecularDataCall::SecStructure& src)
        : aminoAcidCnt(src.aminoAcidCnt), 
        firstAminoAcidIdx(src.firstAminoAcidIdx), type(src.type) {
    // intentionally empty
}


/*
 * MolecularDataCall::SecStructure::~SecStructure
 */
MolecularDataCall::SecStructure::~SecStructure(void) {
    // intentionally empty
}


/*
 * MolecularDataCall::SecStructure::SetPosition
 */
void MolecularDataCall::SecStructure::SetPosition(
        unsigned int firstAminoAcidIdx, unsigned int aminoAcidCnt) {
    this->firstAminoAcidIdx = firstAminoAcidIdx;
    this->aminoAcidCnt = aminoAcidCnt;
}


/*
 * MolecularDataCall::SecStructure::SetType
 */
void MolecularDataCall::SecStructure::SetType(
        MolecularDataCall::SecStructure::ElementType type) {
    this->type = type;
}


/*
 * MolecularDataCall::SecStructure::operator=
 */
MolecularDataCall::SecStructure&
MolecularDataCall::SecStructure::operator=(
        const MolecularDataCall::SecStructure& rhs) {
    this->aminoAcidCnt = rhs.aminoAcidCnt;
    this->firstAminoAcidIdx = rhs.firstAminoAcidIdx;
    this->type = rhs.type;
    return *this;
}


/*
 * MolecularDataCall::SecStructure::operator==
 */
bool MolecularDataCall::SecStructure::operator==(const MolecularDataCall::SecStructure& rhs) const {
    return ((this->aminoAcidCnt == rhs.aminoAcidCnt)
        && (this->firstAminoAcidIdx == rhs.firstAminoAcidIdx)
        && (this->type == rhs.type));
}

// ======================================================================

/*
 * protein::MolecularDataCall::AtomType::AtomType
 */
protein::MolecularDataCall::AtomType::AtomType(void) : name(), rad(0.5f) {
    this->col[0] = this->col[1] = this->col[2] = 191;
}


/*
 * protein::MolecularDataCall::AtomType::AtomType
 */
protein::MolecularDataCall::AtomType::AtomType(const vislib::StringA& name,
        float rad, unsigned char colR, unsigned char colG, unsigned char colB)
        : name(name), rad(rad) {
    this->col[0] = colR;
    this->col[1] = colG;
    this->col[2] = colB;
}


/*
 * protein::MolecularDataCall::AtomType::AtomType
 */
protein::MolecularDataCall::AtomType::AtomType(const AtomType& src) {
    *this = src;
}


/*
 * protein::MolecularDataCall::AtomType::~AtomType
 */
protein::MolecularDataCall::AtomType::~AtomType(void) {
}


/*
 * protein::MolecularDataCall::AtomType::operator=
 */
protein::MolecularDataCall::AtomType& protein::MolecularDataCall::AtomType::operator=(
        const protein::MolecularDataCall::AtomType& rhs) {
    this->name = rhs.name;
    this->rad = rhs.rad;
    this->col[0] = rhs.col[0];
    this->col[1] = rhs.col[1];
    this->col[2] = rhs.col[2];
    return *this;
}


/*
 * protein::MolecularDataCall::AtomType::operator==
 */
bool protein::MolecularDataCall::AtomType::operator==(
        const protein::MolecularDataCall::AtomType& rhs) const {
    return vislib::math::IsEqual(this->rad, rhs.rad)
        && (this->name.Equals( rhs.name))
        && (this->col[0] == rhs.col[0])
        && (this->col[1] == rhs.col[1])
        && (this->col[2] == rhs.col[2]);
}

// ======================================================================

/*
 * protein::MolecularDataCall::Molecule::Molecule
 */
protein::MolecularDataCall::Molecule::Molecule(void) : firstResidueIndex(0), 
        residueCount(0), firstSecStructIdx( 0), secStructCount( 0),
        firstConIdx( 0), conCount( 0), chainIndex(-1), filter(1) {
    // intentionally empty
}


/*
 * protein::MolecularDataCall::Molecule::Molecule
 */
protein::MolecularDataCall::Molecule::Molecule(
    const protein::MolecularDataCall::Molecule& src) : 
    firstResidueIndex( src.firstResidueIndex), 
    residueCount( src.residueCount), 
    firstSecStructIdx( src.firstSecStructIdx),
    secStructCount( src.secStructCount),
    firstConIdx( src.firstConIdx), conCount( src.conCount), chainIndex(src.chainIndex),
    filter(1){
    // intentionally empty
}


/*
 * protein::MolecularDataCall::Molecule::Molecule
 */
protein::MolecularDataCall::Molecule::Molecule( unsigned int firstResIdx,
    unsigned int resCnt, int chainIdx) : firstResidueIndex( firstResIdx),
    residueCount(resCnt), firstSecStructIdx( 0), secStructCount( 0),
    firstConIdx( 0), conCount( 0), chainIndex(chainIdx),
    filter(1) {
    // intentionally empty
}


/*
 * protein::MolecularDataCall::Molecule::~Molecule
 */
protein::MolecularDataCall::Molecule::~Molecule(void) {
    // intentionally empty
}


/*
 * protein::MolecularDataCall::Molecule::operator=
 */
protein::MolecularDataCall::Molecule& 
protein::MolecularDataCall::Molecule::operator=(
        const protein::MolecularDataCall::Molecule& rhs) {
    this->firstResidueIndex = rhs.firstResidueIndex;
    this->residueCount = rhs.residueCount;
    this->chainIndex = rhs.chainIndex;
    this->firstSecStructIdx = rhs.firstSecStructIdx;
    this->secStructCount = rhs.secStructCount;
    this->firstConIdx = rhs.firstConIdx;
    this->conCount = rhs.conCount;
    this->filter = rhs.filter;
    return *this;
}


/*
 * protein::MolecularDataCall::Molecule::operator==
 */
bool protein::MolecularDataCall::Molecule::operator==(
    const protein::MolecularDataCall::Molecule& rhs) const {
    return ((this->firstResidueIndex == rhs.firstResidueIndex)
    && (this->residueCount == rhs.residueCount)
    && (this->chainIndex == rhs.chainIndex)
    && (this->firstSecStructIdx == rhs.firstSecStructIdx)
    && (this->secStructCount == rhs.secStructCount) 
    && (this->firstConIdx == rhs.firstConIdx)
    && (this->conCount == rhs.conCount));
}

// ======================================================================

/*
 * protein::MolecularDataCall::Chain::Chain
 */
protein::MolecularDataCall::Chain::Chain(void) : firstMoleculeIndex(0), 
    moleculeCount(0), type( UNSPECIFIC), filter(1) {
    // intentionally empty
}


/*
 * protein::MolecularDataCall::Chain::Chain
 */
protein::MolecularDataCall::Chain::Chain(
    const protein::MolecularDataCall::Chain& src) : 
    firstMoleculeIndex( src.firstMoleculeIndex), 
    moleculeCount( src.moleculeCount),
    type( src.type), filter(1) {
    // intentionally empty
}


/*
 * protein::MolecularDataCall::Chain::Chain
 */
protein::MolecularDataCall::Chain::Chain( unsigned int firstMolIdx,
    unsigned int molCnt, ChainType chainType) : firstMoleculeIndex( firstMolIdx),
    moleculeCount( molCnt), type( chainType), filter(1) {
    // intentionally empty
}


/*
 * protein::MolecularDataCall::Chain::~Chain
 */
protein::MolecularDataCall::Chain::~Chain(void) {
    // intentionally empty
}


/*
 * protein::MolecularDataCall::Chain::operator=
 */
protein::MolecularDataCall::Chain& 
protein::MolecularDataCall::Chain::operator=(
        const protein::MolecularDataCall::Chain& rhs) {
    this->firstMoleculeIndex = rhs.firstMoleculeIndex;
    this->moleculeCount = rhs.moleculeCount;
    this->type = rhs.type;
    this->filter = rhs.filter;
    return *this;
}


/*
 * protein::MolecularDataCall::Chain::operator==
 */
bool protein::MolecularDataCall::Chain::operator==(
    const protein::MolecularDataCall::Chain& rhs) const {
    return ((this->firstMoleculeIndex == rhs.firstMoleculeIndex)
    && (this->moleculeCount == rhs.moleculeCount)
    && (this->type == rhs.type));
}

// ======================================================================

/*
 * MolecularDataCall::CallForGetData
 */
const unsigned int protein::MolecularDataCall::CallForGetData = 0;


/*
 * MolecularDataCall::CallForGetExtent
 */
const unsigned int protein::MolecularDataCall::CallForGetExtent = 1;


/*
 * protein::MolecularDataCall::MolecularDataCall
 */
protein::MolecularDataCall::MolecularDataCall(void) : AbstractGetData3DCall(),
        atomCount( 0), atomPos( 0), atomTypeIdx( 0), atomResidueIdx( 0), atomHydrogenBondIdx( 0), atomHydrogenBondStatistics(0), atomSolventResCount(0),
        residues( 0), resCount( 0),
        molecules( 0), molCount( 0),
        chains( 0), chainCount( 0),
        atomTypeCount( 0), atomType( 0) {
    // intentionally empty
}


/*
 * protein::MolecularDataCall::~MolecularDataCall
 */
protein::MolecularDataCall::~MolecularDataCall(void) {
}


/*
 * Set the atom types and positions.
 */
void MolecularDataCall::SetAtoms( unsigned int atomCnt, unsigned int atomTypeCnt, 
        const unsigned int* typeIdx, const float* pos, const AtomType* types, const int *residueIdx,
        const float* bfactor, const float* charge, const float* occupancy) {
    // set all values
    this->atomCount = atomCnt;
    this->atomTypeCount = atomTypeCnt;
    this->atomTypeIdx = typeIdx;
	this->atomResidueIdx = residueIdx; // -> SetAtomResidueIndices()
    this->atomPos = pos;
    this->atomType = types;
    this->atomBFactors = bfactor;
    this->atomCharges = charge;
    this->atomOccupancies = occupancy;
}

/*
 * Set the residues.
 */
void MolecularDataCall::SetResidues( unsigned int resCnt, const Residue** res) {
    // set all values
    this->resCount = resCnt;
    this->residues = res;
}

/*
 * Set the residue type names.
 */
void MolecularDataCall::SetResidueTypeNames( unsigned int namesCnt, const vislib::StringA* names) {
    // set all values
    this->resTypeNameCnt = namesCnt;
    this->resTypeNames = names;
}

/*
 * Set the connections (bonds).
 */
void MolecularDataCall::SetConnections( unsigned int conCnt, const unsigned int* con) {
    // set all values
    this->connectionCount = conCnt;
    this->connections = con;
}

/*
 * Set the molecules.
 */
void MolecularDataCall::SetMolecules( unsigned int molCnt, const Molecule* mol) {
    // set all values
    this->molCount = molCnt;
    this->molecules = mol;
}

/*
 * Set the chains.
 */
void MolecularDataCall::SetChains( unsigned int chainCnt, const Chain* chain) {
    // set all values
    this->chainCount = chainCnt;
    this->chains = chain;
}

/*
 * Set the number of secondary structure elements.
 */
void MolecularDataCall::SetSecondaryStructureCount( unsigned int cnt) {
    this->secStruct.Clear();
    this->secStruct.SetCount( cnt);
}

/*
 * Set a secondary stucture element to the array.
 */
bool MolecularDataCall::SetSecondaryStructure( unsigned int idx, SecStructure secS) {
    if( idx < this->secStruct.Count() ) {
        this->secStruct[idx] = secS;
        return true;
    } else {
        return false;
    }
}

/*
 * Set a secondary stucture element range to the molecule.
 */
void MolecularDataCall::SetMoleculeSecondaryStructure( unsigned int molIdx, 
    unsigned int firstSecS, unsigned int secSCnt) {
	// TODO: this is very ugly!
	MolecularDataCall::Molecule *_molecules = const_cast<MolecularDataCall::Molecule*>(this->molecules);
    if( molIdx < this->molCount ) {
        _molecules[molIdx].SetSecondaryStructure( firstSecS, secSCnt);
    }
}

/*
 * Set filter information
 */
void MolecularDataCall::SetFilter(const int* atomFilter) {
    this->atomFilter = atomFilter;
}

/*
 * Get the secondary structure.
 */
const MolecularDataCall::SecStructure* MolecularDataCall::SecondaryStructures() const {
    return this->secStruct.PeekElements();
}

/*
 * Get the number of secondary structure elements.
 */
unsigned int MolecularDataCall::SecondaryStructureCount() const {
    return static_cast<unsigned int>(this->secStruct.Count());
}
