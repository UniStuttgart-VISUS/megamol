/*
 * MolecularDataCall.cpp
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#include "protein_calls/MolecularDataCall.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/math/mathfunctions.h"

using namespace megamol;
using namespace megamol::protein_calls;


/*
 * MolecularDataCall::Residue::Residue
 */
MolecularDataCall::Residue::Residue(void)
        : atomCnt(0)
        , firstAtomIdx(0)
        , boundingBox(0, 0, 0, 0, 0, 0)
        , type(0)
        , moleculeIndex(-1)
        , filter(1)
        , origResIndex(0) {
    // intentionally empty
}


/*
 * MolecularDataCall::Residue::Residue
 */
MolecularDataCall::Residue::Residue(const MolecularDataCall::Residue& src)
        : atomCnt(src.atomCnt)
        , firstAtomIdx(src.firstAtomIdx)
        , boundingBox(src.boundingBox)
        , type(src.type)
        , moleculeIndex(src.moleculeIndex)
        , origResIndex(src.origResIndex)
        , filter(1) {
    // intentionally empty
}


/*
 * MolecularDataCall::Residue::Residue
 */
MolecularDataCall::Residue::Residue(unsigned int firstAtomIdx, unsigned int atomCnt, vislib::math::Cuboid<float> bbox,
    unsigned int typeIdx, int moleculeIdx, unsigned int origResIdx)
        : atomCnt(atomCnt)
        , firstAtomIdx(firstAtomIdx)
        , boundingBox(bbox)
        , type(typeIdx)
        , moleculeIndex(moleculeIdx)
        , origResIndex(origResIdx)
        , filter(1) {
    // intentionally empty
}


/*
 * MolecularDataCall::Residue::~Residue
 */
MolecularDataCall::Residue::~Residue(void) {
    // intentionally empty
}


/*
 * MolecularDataCall::Residue::SetPosition
 */
void MolecularDataCall::Residue::SetPosition(unsigned int firstAtom, unsigned int atomCnt) {
    this->firstAtomIdx = firstAtom;
    this->atomCnt = atomCnt;
}


/*
 * MolecularDataCall::Residue::operator=
 */
MolecularDataCall::Residue& MolecularDataCall::Residue::operator=(const MolecularDataCall::Residue& rhs) {
    this->atomCnt = rhs.atomCnt;
    this->firstAtomIdx = rhs.firstAtomIdx;
    this->boundingBox = rhs.boundingBox;
    this->type = rhs.type;
    this->moleculeIndex = rhs.moleculeIndex;
    this->origResIndex = rhs.origResIndex;
    this->filter = rhs.filter;
    return *this;
}


/*
 * MolecularDataCall::Residue::operator==
 */
bool MolecularDataCall::Residue::operator==(const MolecularDataCall::Residue& rhs) const {
    return ((this->atomCnt == rhs.atomCnt) && (this->firstAtomIdx == rhs.firstAtomIdx) &&
            (this->boundingBox == rhs.boundingBox) && (this->type == rhs.type) &&
            (this->origResIndex == rhs.origResIndex) && (this->moleculeIndex == rhs.moleculeIndex));
}

// ======================================================================

/*
 * MolecularDataCall::AminoAcid::AminoAcid
 */
MolecularDataCall::AminoAcid::AminoAcid(void) : Residue(), cAlphaIdx(0), cCarbIdx(0), nIdx(0), oIdx(0) {
    // intentionally empty
}


/*
 * MolecularDataCall::AminoAcid::AminoAcid
 */
MolecularDataCall::AminoAcid::AminoAcid(const MolecularDataCall::AminoAcid& src)
        : Residue(src.firstAtomIdx, src.atomCnt, src.boundingBox, src.type, src.moleculeIndex, src.origResIndex)
        , cAlphaIdx(src.cAlphaIdx)
        , cCarbIdx(src.cCarbIdx)
        , nIdx(src.nIdx)
        , oIdx(src.oIdx) {
    // intentionally empty
}


/*
 * MolecularDataCall::AminoAcid::AminoAcid
 */
MolecularDataCall::AminoAcid::AminoAcid(unsigned int firstAtomIdx, unsigned int atomCnt, unsigned int cAlphaIdx,
    unsigned int cCarbIdx, unsigned int nIdx, unsigned int oIdx, vislib::math::Cuboid<float> bbox, unsigned int typeIdx,
    int moleculeIdx, unsigned int origResIdx)
        : Residue(firstAtomIdx, atomCnt, bbox, typeIdx, moleculeIdx, origResIdx)
        , cAlphaIdx(cAlphaIdx)
        , cCarbIdx(cCarbIdx)
        , nIdx(nIdx)
        , oIdx(oIdx) {
    // intentionally empty
}


/*
 * MolecularDataCall::AminoAcid::~AminoAcid
 */
MolecularDataCall::AminoAcid::~AminoAcid(void) {
    // intentionally empty
}


/*
 * MolecularDataCall::AminoAcid::SetCAlphaIndex
 */
void MolecularDataCall::AminoAcid::SetCAlphaIndex(unsigned int idx) {
    this->cAlphaIdx = idx;
}


/*
 * MolecularDataCall::AminoAcid::SetCCarbIndex
 */
void MolecularDataCall::AminoAcid::SetCCarbIndex(unsigned int idx) {
    this->cCarbIdx = idx;
}


/*
 * MolecularDataCall::AminoAcid::SetNIndex
 */
void MolecularDataCall::AminoAcid::SetNIndex(unsigned int idx) {
    this->nIdx = idx;
}


/*
 * MolecularDataCall::AminoAcid::SetOIndex
 */
void MolecularDataCall::AminoAcid::SetOIndex(unsigned int idx) {
    this->oIdx = idx;
}


/*
 * MolecularDataCall::AminoAcid::operator=
 */
MolecularDataCall::AminoAcid& MolecularDataCall::AminoAcid::operator=(const MolecularDataCall::AminoAcid& rhs) {
    this->firstAtomIdx = rhs.firstAtomIdx;
    this->atomCnt = rhs.atomCnt;
    this->cAlphaIdx = rhs.cAlphaIdx;
    this->cCarbIdx = rhs.cCarbIdx;
    this->nIdx = rhs.nIdx;
    this->oIdx = rhs.oIdx;
    this->boundingBox = rhs.boundingBox;
    this->type = rhs.type;
    this->moleculeIndex = rhs.moleculeIndex;
    this->origResIndex = rhs.origResIndex;
    return *this;
}


/*
 * MolecularDataCall::AminoAcid::operator==
 */
bool MolecularDataCall::AminoAcid::operator==(const MolecularDataCall::AminoAcid& rhs) const {
    return ((this->firstAtomIdx == rhs.firstAtomIdx) && (this->atomCnt == rhs.atomCnt) &&
            (this->cAlphaIdx == rhs.cAlphaIdx) && (this->cCarbIdx == rhs.cCarbIdx) && (this->nIdx == rhs.nIdx) &&
            (this->oIdx == rhs.oIdx) && (this->boundingBox == rhs.boundingBox) && (this->type == rhs.type) &&
            (this->moleculeIndex == rhs.moleculeIndex) && (this->origResIndex == rhs.origResIndex));
}

// ======================================================================

/*
 * MolecularDataCall::SecStructure::SecStructure
 */
MolecularDataCall::SecStructure::SecStructure(void) : aminoAcidCnt(0), firstAminoAcidIdx(0), type(TYPE_COIL) {
    // intentionally empty
}


/*
 * MolecularDataCall::SecStructure::SecStructure
 */
MolecularDataCall::SecStructure::SecStructure(const MolecularDataCall::SecStructure& src)
        : aminoAcidCnt(src.aminoAcidCnt)
        , firstAminoAcidIdx(src.firstAminoAcidIdx)
        , type(src.type) {
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
void MolecularDataCall::SecStructure::SetPosition(unsigned int firstAminoAcidIdx, unsigned int aminoAcidCnt) {
    this->firstAminoAcidIdx = firstAminoAcidIdx;
    this->aminoAcidCnt = aminoAcidCnt;
}


/*
 * MolecularDataCall::SecStructure::SetType
 */
void MolecularDataCall::SecStructure::SetType(MolecularDataCall::SecStructure::ElementType type) {
    this->type = type;
}


/*
 * MolecularDataCall::SecStructure::operator=
 */
MolecularDataCall::SecStructure& MolecularDataCall::SecStructure::operator=(
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
    return ((this->aminoAcidCnt == rhs.aminoAcidCnt) && (this->firstAminoAcidIdx == rhs.firstAminoAcidIdx) &&
            (this->type == rhs.type));
}

// ======================================================================

/*
 * MolecularDataCall::AtomType::AtomType
 */
MolecularDataCall::AtomType::AtomType(void) : name(), rad(0.5f) {
    this->col[0] = this->col[1] = this->col[2] = 191;
}


/*
 * MolecularDataCall::AtomType::AtomType
 */
MolecularDataCall::AtomType::AtomType(const vislib::StringA& name, float rad, unsigned char colR, unsigned char colG,
    unsigned char colB, const vislib::StringA& element)
        : name(name)
        , rad(rad) {
    this->col[0] = colR;
    this->col[1] = colG;
    this->col[2] = colB;

    // when element is empty just take the first symbol of the name
    // If the naming convention of rcsb.org does not change, this should work
    if (element.IsEmpty()) {
        this->element = name.Substring(0, 1);
    } else {
        this->element = element;
    }
}


/*
 * MolecularDataCall::AtomType::AtomType
 */
MolecularDataCall::AtomType::AtomType(const AtomType& src) {
    *this = src;
}


/*
 * MolecularDataCall::AtomType::~AtomType
 */
MolecularDataCall::AtomType::~AtomType(void) {}


/*
 * MolecularDataCall::AtomType::operator=
 */
MolecularDataCall::AtomType& MolecularDataCall::AtomType::operator=(const MolecularDataCall::AtomType& rhs) {
    this->name = rhs.name;
    this->rad = rhs.rad;
    this->col[0] = rhs.col[0];
    this->col[1] = rhs.col[1];
    this->col[2] = rhs.col[2];
    this->element = rhs.element;
    return *this;
}


/*
 * MolecularDataCall::AtomType::operator==
 */
bool MolecularDataCall::AtomType::operator==(const MolecularDataCall::AtomType& rhs) const {
    return vislib::math::IsEqual(this->rad, rhs.rad) && (this->name.Equals(rhs.name)) && (this->col[0] == rhs.col[0]) &&
           (this->col[1] == rhs.col[1]) && (this->col[2] == rhs.col[2]) && (this->element == rhs.element);
}

// ======================================================================

/*
 * MolecularDataCall::Molecule::Molecule
 */
MolecularDataCall::Molecule::Molecule(void)
        : firstResidueIndex(0)
        , residueCount(0)
        , firstSecStructIdx(0)
        , secStructCount(0)
        , firstConIdx(0)
        , conCount(0)
        , chainIndex(-1)
        , filter(1) {
    // intentionally empty
}


/*
 * MolecularDataCall::Molecule::Molecule
 */
MolecularDataCall::Molecule::Molecule(const MolecularDataCall::Molecule& src)
        : firstResidueIndex(src.firstResidueIndex)
        , residueCount(src.residueCount)
        , firstSecStructIdx(src.firstSecStructIdx)
        , secStructCount(src.secStructCount)
        , firstConIdx(src.firstConIdx)
        , conCount(src.conCount)
        , chainIndex(src.chainIndex)
        , filter(1) {
    // intentionally empty
}


/*
 * MolecularDataCall::Molecule::Molecule
 */
MolecularDataCall::Molecule::Molecule(unsigned int firstResIdx, unsigned int resCnt, int chainIdx)
        : firstResidueIndex(firstResIdx)
        , residueCount(resCnt)
        , firstSecStructIdx(0)
        , secStructCount(0)
        , firstConIdx(0)
        , conCount(0)
        , chainIndex(chainIdx)
        , filter(1) {
    // intentionally empty
}


/*
 * MolecularDataCall::Molecule::~Molecule
 */
MolecularDataCall::Molecule::~Molecule(void) {
    // intentionally empty
}


/*
 * MolecularDataCall::Molecule::operator=
 */
MolecularDataCall::Molecule& MolecularDataCall::Molecule::operator=(const MolecularDataCall::Molecule& rhs) {
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
 * MolecularDataCall::Molecule::operator==
 */
bool MolecularDataCall::Molecule::operator==(const MolecularDataCall::Molecule& rhs) const {
    return ((this->firstResidueIndex == rhs.firstResidueIndex) && (this->residueCount == rhs.residueCount) &&
            (this->chainIndex == rhs.chainIndex) && (this->firstSecStructIdx == rhs.firstSecStructIdx) &&
            (this->secStructCount == rhs.secStructCount) && (this->firstConIdx == rhs.firstConIdx) &&
            (this->conCount == rhs.conCount));
}

// ======================================================================

/*
 * MolecularDataCall::Chain::Chain
 */
MolecularDataCall::Chain::Chain(void) : firstMoleculeIndex(0), moleculeCount(0), type(UNSPECIFIC), filter(1) {
    // intentionally empty
}


/*
 * MolecularDataCall::Chain::Chain
 */
MolecularDataCall::Chain::Chain(const MolecularDataCall::Chain& src)
        : firstMoleculeIndex(src.firstMoleculeIndex)
        , moleculeCount(src.moleculeCount)
        , type(src.type)
        , filter(1) {
    // intentionally empty
}


/*
 * MolecularDataCall::Chain::Chain
 */
MolecularDataCall::Chain::Chain(unsigned int firstMolIdx, unsigned int molCnt, char name, ChainType chainType)
        : firstMoleculeIndex(firstMolIdx)
        , moleculeCount(molCnt)
        , type(chainType)
        , name(name)
        , filter(1) {
    // intentionally empty
}


/*
 * MolecularDataCall::Chain::~Chain
 */
MolecularDataCall::Chain::~Chain(void) {
    // intentionally empty
}


/*
 * MolecularDataCall::Chain::operator=
 */
MolecularDataCall::Chain& MolecularDataCall::Chain::operator=(const MolecularDataCall::Chain& rhs) {
    this->firstMoleculeIndex = rhs.firstMoleculeIndex;
    this->moleculeCount = rhs.moleculeCount;
    this->type = rhs.type;
    this->name = rhs.name;
    this->filter = rhs.filter;
    return *this;
}


/*
 * MolecularDataCall::Chain::operator==
 */
bool MolecularDataCall::Chain::operator==(const MolecularDataCall::Chain& rhs) const {
    return ((this->firstMoleculeIndex == rhs.firstMoleculeIndex) && (this->moleculeCount == rhs.moleculeCount) &&
            (this->type == rhs.type) && (this->name == rhs.name));
}

// ======================================================================

/*
 * MolecularDataCall::CallForGetData
 */
const unsigned int MolecularDataCall::CallForGetData = 0;


/*
 * MolecularDataCall::CallForGetExtent
 */
const unsigned int MolecularDataCall::CallForGetExtent = 1;


/*
 * MolecularDataCall::MolecularDataCall
 */
MolecularDataCall::MolecularDataCall(void)
        : AbstractGetData3DCall()
        , atomCount(0)
        , atomPos(0)
        , atomTypeIdx(0)
        , atomResidueIdx(0)
        , atomHydrogenBondIdx(0)
        , atomHydrogenBondStatistics(0)
        , atomSolventResCount(0)
        , residues(0)
        , resCount(0)
        , molecules(0)
        , molCount(0)
        , chains(0)
        , chainCount(0)
        , atomTypeCount(0)
        , atomType(0)
        , atomHydrogenBondsFake(false)
        , numHydrogenBonds(0) {
    this->neighborhoods = nullptr;
    this->neighborhoodSizes = nullptr;
    this->hydrogenBonds = nullptr;
}


/*
 * MolecularDataCall::~MolecularDataCall
 */
MolecularDataCall::~MolecularDataCall(void) {}


/*
 * Set the atom types and positions.
 */
void MolecularDataCall::SetAtoms(unsigned int atomCnt, unsigned int atomTypeCnt, const unsigned int* typeIdx,
    const float* pos, const AtomType* types, const int* residueIdx, const float* bfactor, const float* charge,
    const float* occupancy) {
    // set all values
    this->atomCount = atomCnt;
    this->atomTypeCount = atomTypeCnt;
    this->atomTypeIdx = typeIdx;
    this->atomResidueIdx = residueIdx; // -> SetAtomResidueIndices()
    this->atomPos = pos;
    this->atomType = types;
    this->atomBFactors = bfactor;
    this->ownsBFactorMemory = false;
    this->atomCharges = charge;
    this->atomOccupancies = occupancy;
}

/*
 * Set the residues.
 */
void MolecularDataCall::SetResidues(unsigned int resCnt, const Residue** res) {
    // set all values
    this->resCount = resCnt;
    this->residues = res;
}

/*
 * Set the residue type names.
 */
void MolecularDataCall::SetResidueTypeNames(unsigned int namesCnt, const vislib::StringA* names) {
    // set all values
    this->resTypeNameCnt = namesCnt;
    this->resTypeNames = names;
}

/*
 * Set the connections (bonds).
 */
void MolecularDataCall::SetConnections(unsigned int conCnt, const unsigned int* con) {
    // set all values
    this->connectionCount = conCnt;
    this->connections = con;
}

/*
 * Set the molecules.
 */
void MolecularDataCall::SetMolecules(unsigned int molCnt, const Molecule* mol) {
    // set all values
    this->molCount = molCnt;
    this->molecules = mol;
}

/*
 * Set the chains.
 */
void MolecularDataCall::SetChains(unsigned int chainCnt, const Chain* chain) {
    // set all values
    this->chainCount = chainCnt;
    this->chains = chain;
}

/*
 * Set the number of secondary structure elements.
 */
void MolecularDataCall::SetSecondaryStructureCount(unsigned int cnt) {
    this->secStruct.Clear();
    this->secStruct.SetCount(cnt);
}

/*
 * Set a secondary stucture element to the array.
 */
bool MolecularDataCall::SetSecondaryStructure(unsigned int idx, SecStructure secS) {
    if (idx < this->secStruct.Count()) {
        this->secStruct[idx] = secS;
        return true;
    } else {
        return false;
    }
}

/*
 * Set a secondary stucture element range to the molecule.
 */
void MolecularDataCall::SetMoleculeSecondaryStructure(
    unsigned int molIdx, unsigned int firstSecS, unsigned int secSCnt) {
    // TODO: this is very ugly!
    MolecularDataCall::Molecule* _molecules = const_cast<MolecularDataCall::Molecule*>(this->molecules);
    if (molIdx < this->molCount) {
        _molecules[molIdx].SetSecondaryStructure(firstSecS, secSCnt);
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
