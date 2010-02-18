/*
 * CallProteinData.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallProteinData.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/mathfunctions.h"
#include "vislib/OutOfRangeException.h"

using namespace megamol;


/****************************************************************************/

/*
 * protein::CallProteinData::AminoAcid::AminoAcid
 */
protein::CallProteinData::AminoAcid::AminoAcid(void) : atomCnt(0), cAlphaIdx(0),
        cCarbIdx(0), connectivity(), firstAtomIdx(0), nameIdx(0), nIdx(0),
        oIdx(0) {
    // intentionally empty
}


/*
 * protein::CallProteinData::AminoAcid::AminoAcid
 */
protein::CallProteinData::AminoAcid::AminoAcid(
        const protein::CallProteinData::AminoAcid& src) : atomCnt(src.atomCnt),
        cAlphaIdx(src.cAlphaIdx), cCarbIdx(src.cCarbIdx),
        connectivity(src.connectivity), firstAtomIdx(src.firstAtomIdx),
        nameIdx(src.nameIdx), nIdx(src.nIdx), oIdx(src.oIdx) {
    // intentionally empty
}


/*
 * protein::CallProteinData::AminoAcid::AminoAcid
 */
protein::CallProteinData::AminoAcid::AminoAcid(unsigned int firstAtomIdx,
        unsigned int atomCnt, unsigned int cAlphaIdx, unsigned int cCarbIdx,
        unsigned int nIdx, unsigned int oIdx, unsigned int nameIdx) :
        atomCnt(atomCnt), cAlphaIdx(cAlphaIdx), cCarbIdx(cCarbIdx),
        connectivity(), firstAtomIdx(firstAtomIdx), nameIdx(nameIdx),
        nIdx(nIdx), oIdx(oIdx) {
    // intentionally empty
}


/*
 * protein::CallProteinData::AminoAcid::~AminoAcid
 */
protein::CallProteinData::AminoAcid::~AminoAcid(void) {
    // intentionally empty
}


/*
 * protein::CallProteinData::AminoAcid::SetCAlphaIndex
 */
void protein::CallProteinData::AminoAcid::SetCAlphaIndex(unsigned int idx) {
    this->cAlphaIdx = idx;
}


/*
 * protein::CallProteinData::AminoAcid::SetCCarbIndex
 */
void protein::CallProteinData::AminoAcid::SetCCarbIndex(unsigned int idx) {
    this->cCarbIdx = idx;
}


/*
 * protein::CallProteinData::AminoAcid::SetNIndex
 */
void protein::CallProteinData::AminoAcid::SetNIndex(unsigned int idx) {
    this->nIdx = idx;
}


/*
 * protein::CallProteinData::AminoAcid::SetOIndex
 */
void protein::CallProteinData::AminoAcid::SetOIndex(unsigned int idx) {
    this->oIdx = idx;
}


/*
 * protein::CallProteinData::AminoAcid::SetPosition
 */
void protein::CallProteinData::AminoAcid::SetPosition(unsigned int firstAtom,
        unsigned int atomCnt) {
    this->firstAtomIdx = firstAtom;
    this->atomCnt = atomCnt;
}


/*
 * protein::CallProteinData::AminoAcid::SetNameIndex
 */
void protein::CallProteinData::AminoAcid::SetNameIndex(unsigned int idx) {
    this->nameIdx = idx;
}


/*
 * protein::CallProteinData::AminoAcid::operator=
 */
protein::CallProteinData::AminoAcid& 
protein::CallProteinData::AminoAcid::operator=(
        const protein::CallProteinData::AminoAcid& rhs) {
    this->atomCnt = rhs.atomCnt;
    this->cAlphaIdx = rhs.cAlphaIdx;
    this->cCarbIdx = rhs.cCarbIdx;
    this->connectivity = rhs.connectivity;
    this->firstAtomIdx = rhs.firstAtomIdx;
    this->nameIdx = rhs.nameIdx;
    this->nIdx = rhs.nIdx;
    this->oIdx = rhs.oIdx;
    return *this;
}


/*
 * protein::CallProteinData::AminoAcid::operator==
 */
bool protein::CallProteinData::AminoAcid::operator==(
        const protein::CallProteinData::AminoAcid& rhs) const {
    return ((this->atomCnt == rhs.atomCnt)
        && (this->cAlphaIdx == rhs.cAlphaIdx)
        && (this->cCarbIdx == rhs.cCarbIdx)
        && (this->connectivity == rhs.connectivity) // O(n) warning!
        && (this->firstAtomIdx == rhs.firstAtomIdx)
        && (this->nameIdx == rhs.nameIdx)
        && (this->nIdx == rhs.nIdx)
        && (this->oIdx == rhs.oIdx));
}

/****************************************************************************/


/*
 * protein::CallProteinData::AtomData::AtomData
 */
protein::CallProteinData::AtomData::AtomData(unsigned int typeIdx, float charge, 
        float tempFactor, float occupancy) : charge(charge),
        occupancy(occupancy), tempFactor(tempFactor), typeIdx(typeIdx) {
    // intentionally empty
}


/*
 * protein::CallProteinData::AtomData::AtomData
 */
protein::CallProteinData::AtomData::AtomData(const AtomData& src) {
    *this = src;
}


/*
 * protein::CallProteinData::AtomData::~AtomData
 */
protein::CallProteinData::AtomData::~AtomData(void) {
    // intentionally empty
}


/*
 * protein::CallProteinData::AtomData::operator=
 */
protein::CallProteinData::AtomData& protein::CallProteinData::AtomData::operator=(
        const protein::CallProteinData::AtomData& rhs) {
    this->charge = rhs.charge;
    this->occupancy = rhs.occupancy;
    this->tempFactor = rhs.tempFactor;
    this->typeIdx = rhs.typeIdx;
    return *this;
}


/*
 * protein::CallProteinData::AtomData::operator==
 */
bool protein::CallProteinData::AtomData::operator==(
        const protein::CallProteinData::AtomData& rhs) const {
    return vislib::math::IsEqual(this->charge, rhs.charge)
        && vislib::math::IsEqual(this->occupancy, rhs.occupancy)
        && vislib::math::IsEqual(this->tempFactor, rhs.tempFactor)
        && (this->typeIdx == rhs.typeIdx);
}

/****************************************************************************/

/*
 * protein::CallProteinData::AtomType::AtomType
 */
protein::CallProteinData::AtomType::AtomType(void) : name(), rad(0.5f) {
    this->col[0] = this->col[1] = this->col[2] = 191;
}


/*
 * protein::CallProteinData::AtomType::AtomType
 */
protein::CallProteinData::AtomType::AtomType(const vislib::StringA& name,
        float rad, unsigned char colR, unsigned char colG, unsigned char colB)
        : name(name), rad(rad) {
    this->col[0] = colR;
    this->col[1] = colG;
    this->col[2] = colB;
}


/*
 * protein::CallProteinData::AtomType::AtomType
 */
protein::CallProteinData::AtomType::AtomType(const AtomType& src) {
    *this = src;
}


/*
 * protein::CallProteinData::AtomType::~AtomType
 */
protein::CallProteinData::AtomType::~AtomType(void) {
}


/*
 * protein::CallProteinData::AtomType::operator=
 */
protein::CallProteinData::AtomType& protein::CallProteinData::AtomType::operator=(
        const protein::CallProteinData::AtomType& rhs) {
    this->name = rhs.name;
    this->rad = rhs.rad;
    this->col[0] = rhs.col[0];
    this->col[1] = rhs.col[1];
    this->col[2] = rhs.col[2];
    return *this;
}


/*
 * protein::CallProteinData::AtomType::operator==
 */
bool protein::CallProteinData::AtomType::operator==(
        const protein::CallProteinData::AtomType& rhs) const {
    return vislib::math::IsEqual(this->rad, rhs.rad)
        && (this->name == rhs.name)
        && (this->col[0] == rhs.col[0])
        && (this->col[1] == rhs.col[1])
        && (this->col[2] == rhs.col[2]);
}


/****************************************************************************/

/*
 * protein::CallProteinData::Chain::Chain
 */
protein::CallProteinData::Chain::Chain(void) : aminoAcid(NULL), aminoAcidCnt(0),
        aminoAcidMemory(false), secStruct(NULL), secStructCnt(0),
        secStructMemory(false) {
}


/*
 * protein::CallProteinData::Chain::Chain
 */
protein::CallProteinData::Chain::Chain(const protein::CallProteinData::Chain& src)
        : aminoAcid(NULL), aminoAcidCnt(0), aminoAcidMemory(false),
        secStruct(NULL), secStructCnt(0), secStructMemory(false) {
    *this = src;
}


/*
 * protein::CallProteinData::Chain::~Chain
 */
protein::CallProteinData::Chain::~Chain(void) {
    if (this->aminoAcidMemory) delete[] this->aminoAcid;
    this->aminoAcid = NULL;
    this->aminoAcidCnt = 0;
    if (this->secStructMemory) delete[] this->secStruct;
    this->secStruct = NULL;
    this->secStructCnt = 0;
}


/*
 * protein::CallProteinData::Chain::AccessAminoAcid
 */
protein::CallProteinData::AminoAcid&
protein::CallProteinData::Chain::AccessAminoAcid(unsigned int idx) {
    if (!this->aminoAcidMemory) {
        throw vislib::IllegalStateException(
            "You must not access elements you do not own.",
            __FILE__, __LINE__);
    }
    if (idx >= this->aminoAcidCnt) {
        throw vislib::OutOfRangeException(idx, 0, this->aminoAcidCnt - 1,
            __FILE__, __LINE__);
    }
    return this->aminoAcid[idx];
}


/*
 * protein::CallProteinData::Chain::AccessSecondaryStructure
 */
protein::CallProteinData::SecStructure&
protein::CallProteinData::Chain::AccessSecondaryStructure(unsigned int idx) {
    if (!this->secStructMemory) {
        throw vislib::IllegalStateException(
            "You must not access elements you do not own.",
            __FILE__, __LINE__);
    }
    if (idx >= this->secStructCnt) {
        throw vislib::OutOfRangeException(idx, 0, this->secStructCnt - 1,
            __FILE__, __LINE__);
    }
    return this->secStruct[idx];
}


/*
 * protein::CallProteinData::Chain::SetAminoAcid
 */
void protein::CallProteinData::Chain::SetAminoAcid(unsigned int cnt,
        const protein::CallProteinData::AminoAcid* aminoAcids) {
    if (this->aminoAcidMemory) delete[] this->aminoAcid;
    this->aminoAcidCnt = cnt;
    this->aminoAcid 
        = const_cast<protein::CallProteinData::AminoAcid*>(aminoAcids);
    this->aminoAcidMemory = false;
}


/*
 * protein::CallProteinData::Chain::SetAminoAcidCount
 */
void protein::CallProteinData::Chain::SetAminoAcidCount(unsigned int cnt) {
    if (this->aminoAcidMemory) delete[] this->aminoAcid;
    this->aminoAcidCnt = cnt;
    this->aminoAcid = new protein::CallProteinData::AminoAcid[cnt];
    this->aminoAcidMemory = true;
}


/*
 * protein::CallProteinData::Chain::SetSecondaryStructure
 */
void protein::CallProteinData::Chain::SetSecondaryStructure(unsigned int cnt,
        const protein::CallProteinData::SecStructure* structs) {
    if (this->secStructMemory) delete[] this->secStruct;
    this->secStructCnt = cnt;
    this->secStruct 
        = const_cast<protein::CallProteinData::SecStructure*>(structs);
    this->secStructMemory = false;
}


/*
 * protein::CallProteinData::Chain::SetSecondaryStructureCount
 */
void protein::CallProteinData::Chain::SetSecondaryStructureCount(
        unsigned int cnt) {
    if (this->secStructMemory) delete[] this->secStruct;
    this->secStructCnt = cnt;
    this->secStruct = new protein::CallProteinData::SecStructure[cnt];
    this->secStructMemory = true;
}


/*
 * protein::CallProteinData::Chain::operator=
 */
protein::CallProteinData::Chain& protein::CallProteinData::Chain::operator=(
        const protein::CallProteinData::Chain& rhs) {
    if (this->aminoAcidMemory) delete[] this->aminoAcid;
    this->aminoAcidCnt = rhs.aminoAcidCnt;
    if ((this->aminoAcidMemory = rhs.aminoAcidMemory) == true) {
        this->aminoAcid 
            = new protein::CallProteinData::AminoAcid[this->aminoAcidCnt];
        for (unsigned int i = 0; i < this->aminoAcidCnt; i++) {
            this->aminoAcid[i] = rhs.aminoAcid[i];
        }
    } else {
        this->aminoAcid = rhs.aminoAcid;
    }
    if (this->secStructMemory) delete[] this->secStruct;
    this->secStructCnt = rhs.secStructCnt;
    if ((this->secStructMemory = rhs.secStructMemory) == true) {
        this->secStruct
            = new protein::CallProteinData::SecStructure[this->secStructCnt];
        for (unsigned int i = 0; i < this->secStructCnt; i++) {
            this->secStruct[i] = rhs.secStruct[i];
        }
    } else {
        this->secStruct = rhs.secStruct;
    }
    return *this;
}


/*
 * protein::CallProteinData::Chain::operator==
 */
bool protein::CallProteinData::Chain::operator==(
        const protein::CallProteinData::Chain& rhs) const {
    if ((this->aminoAcidCnt != rhs.aminoAcidCnt) 
        || (this->aminoAcidMemory != rhs.aminoAcidMemory)
        || (this->secStructCnt != rhs.secStructCnt)
        || (this->secStructMemory != rhs.secStructMemory)) return false;

    if (this->aminoAcidMemory) {
        for (unsigned int i = 0; i < this->aminoAcidCnt; i++) {
            if (!(this->aminoAcid[i] == rhs.aminoAcid[i])) return false;
        }
    } else {
        if (this->aminoAcid != rhs.aminoAcid) return false;
    }

    if (this->secStructMemory) {
        for (unsigned int i = 0; i < this->secStructCnt; i++) {
            if (!(this->secStruct[i] == rhs.secStruct[i])) return false;
        }
    } else {
        if (this->secStruct != rhs.secStruct) return false;
    }

    return true;
}


/****************************************************************************/

/*
 * protein::CallProteinData::SecStructure::SecStructure
 */
protein::CallProteinData::SecStructure::SecStructure(void) : aminoAcidCnt(0),
        atomCnt(0), firstAminoAcidIdx(0), firstAtomIdx(0), type(TYPE_COIL) {
    // intentionally empty
}


/*
 * protein::CallProteinData::SecStructure::SecStructure
 */
protein::CallProteinData::SecStructure::SecStructure(
        const protein::CallProteinData::SecStructure& src)
        : aminoAcidCnt(src.aminoAcidCnt), atomCnt(src.atomCnt),
        firstAminoAcidIdx(src.firstAminoAcidIdx),
        firstAtomIdx(src.firstAtomIdx), type(src.type) {
    // intentionally empty
}


/*
 * protein::CallProteinData::SecStructure::~SecStructure
 */
protein::CallProteinData::SecStructure::~SecStructure(void) {
    // intentionally empty
}


/*
 * protein::CallProteinData::SecStructure::SetPosition
 */
void protein::CallProteinData::SecStructure::SetPosition(
        unsigned int firstAtomIdx, unsigned int atomCnt,
        unsigned int firstAminoAcidIdx, unsigned int aminoAcidCnt) {
    this->firstAtomIdx = firstAtomIdx;
    this->atomCnt = atomCnt;
    this->firstAminoAcidIdx = firstAminoAcidIdx;
    this->aminoAcidCnt = aminoAcidCnt;
}


/*
 * protein::CallProteinData::SecStructure::SetType
 */
void protein::CallProteinData::SecStructure::SetType(
        protein::CallProteinData::SecStructure::ElementType type) {
    this->type = type;
}


/*
 * protein::CallProteinData::SecStructure::operator=
 */
protein::CallProteinData::SecStructure&
protein::CallProteinData::SecStructure::operator=(
        const protein::CallProteinData::SecStructure& rhs) {
    this->aminoAcidCnt = rhs.aminoAcidCnt;
    this->atomCnt = rhs.atomCnt;
    this->firstAminoAcidIdx = rhs.firstAminoAcidIdx;
    this->firstAtomIdx = rhs.firstAtomIdx;
    this->type = rhs.type;
    return *this;
}


/*
 * protein::CallProteinData::SecStructure::operator==
 */
bool protein::CallProteinData::SecStructure::operator==(const protein::CallProteinData::SecStructure& rhs) const {
    return ((this->aminoAcidCnt == rhs.aminoAcidCnt)
        && (this->atomCnt == rhs.atomCnt)
        && (this->firstAminoAcidIdx == rhs.firstAminoAcidIdx)
        && (this->firstAtomIdx == rhs.firstAtomIdx)
        && (this->type == rhs.type));
}


/****************************************************************************/

/*
 * protein::CallProteinData::SolventMoleculeData::SolventMoleculeData
 */
protein::CallProteinData::SolventMoleculeData::SolventMoleculeData(void)
        : atomCnt(0), name(NULL), connectivity() {
    // intentionally empty
}


/*
 * protein::CallProteinData::SolventMoleculeData::SolventMoleculeData
 */
protein::CallProteinData::SolventMoleculeData::SolventMoleculeData(
        const protein::CallProteinData::SolventMoleculeData& src)
        : atomCnt(src.atomCnt), name(src.name),
        connectivity(src.connectivity) {
    // intentionally empty
}


/*
 * protein::CallProteinData::SolventMoleculeData::SolventMoleculeData
 */
protein::CallProteinData::SolventMoleculeData::SolventMoleculeData(
        const vislib::StringA& name, unsigned int atomCnt) : atomCnt(atomCnt),
        name(name), connectivity() {
    // intentionally empty
}


/*
 * protein::CallProteinData::SolventMoleculeData::~SolventMoleculeData
 */
protein::CallProteinData::SolventMoleculeData::~SolventMoleculeData(void) {
    // intentionally empty
}


/*
 * protein::CallProteinData::SolventMoleculeData::AddConnection
 */
void protein::CallProteinData::SolventMoleculeData::AddConnection(
        const protein::CallProteinData::IndexPair& connection) {
    this->connectivity.Add(connection);
}


/*
 * protein::CallProteinData::SolventMoleculeData::AllocateConnectivityList
 */
void protein::CallProteinData::SolventMoleculeData::AllocateConnectivityList(
        unsigned int cnt) {
    this->connectivity.SetCount(cnt);
}


/*
 * protein::CallProteinData::SolventMoleculeData::ClearConnectivity
 */
void protein::CallProteinData::SolventMoleculeData::ClearConnectivity(void) {
    this->connectivity.Clear();
}


/*
 * protein::CallProteinData::SolventMoleculeData::SetAtomCount
 */
void protein::CallProteinData::SolventMoleculeData::SetAtomCount(
        unsigned int cnt) {
    this->atomCnt = cnt;
    // Perhaps the connectivity should be checked here and invalid connections
    // should be removed. But I believe that this is up the the user of this
    // class
}


/*
 * protein::CallProteinData::SolventMoleculeData::SetConnection
 */
void protein::CallProteinData::SolventMoleculeData::SetConnection(
        unsigned int idx, const protein::CallProteinData::IndexPair& connection) {
    this->connectivity[idx] = connection;
}


/*
 * protein::CallProteinData::SolventMoleculeData::SetConnections
 */
void protein::CallProteinData::SolventMoleculeData::SetConnections(
        unsigned int cnt,
        const protein::CallProteinData::IndexPair *connections) {
    this->connectivity.SetCount(cnt);
    for (unsigned int i = 0; i < cnt; i++) {
        this->connectivity[i] = connections[i];
    }
}


/*
 * protein::CallProteinData::SolventMoleculeData::SetName
 */
void protein::CallProteinData::SolventMoleculeData::SetName(
        const vislib::StringA& name) {
    this->name = name;
}


/*
 * protein::CallProteinData::SolventMoleculeData::operator=
 */
protein::CallProteinData::SolventMoleculeData&
protein::CallProteinData::SolventMoleculeData::operator=(
        const protein::CallProteinData::SolventMoleculeData& rhs) {
    this->atomCnt = rhs.atomCnt;
    this->connectivity = rhs.connectivity;
    this->name = rhs.name;
    return *this;
}


/*
 * protein::CallProteinData::SolventMoleculeData::operator==
 */
bool protein::CallProteinData::SolventMoleculeData::operator==(
        const protein::CallProteinData::SolventMoleculeData& rhs) const {
    return ((this->atomCnt == rhs.atomCnt)
        && (this->connectivity == rhs.connectivity)
        && (this->name == rhs.name));
}


/****************************************************************************/

/*
 * protein::CallProteinData::CallProteinData
 */
protein::CallProteinData::CallProteinData(void) : Call(), 
        aminoAcidNameCnt(0), aminoAcidNameMemory(false), aminoAcidNames(NULL), 
        atomTypeMemory(false), atomTypes(NULL), chainCnt(0), chains(NULL),
        chainsMemory(false), dsBondsCnt(0), dsBonds(NULL), atomTypeCnt(0),
        dsBondsMemory(false), protAtomCnt(0), protAtomData(NULL),
        protAtomPos(NULL), protDataMemory(false), protPosMemory(false),
        solAtomCnt(0), solAtomData(NULL), solAtomPos(NULL),
        solDataMemory(false), solMolCnt(NULL), solMolCntMemory(false),
        solMolTypeCnt(0), solMolTypeData(NULL), solMolTypeDataMemory(false),
        solPosMemory(false), minOccupancy(0.0f), maxOccupancy(0.0f),
        minTempFactor(0.0f), maxTempFactor(0.0f), 
        minCharge(0.0f), maxCharge(0.0f), 
        useRMS(false), currentRMSFrameID(0), currentFrameId( 0) {
    // intentionally empty
}


/*
 * protein::CallProteinData::~CallProteinData
 */
protein::CallProteinData::~CallProteinData(void) {
    this->aminoAcidNameCnt = 0;
    if (this->aminoAcidNameMemory) delete[] this->aminoAcidNames;
    this->aminoAcidNames = NULL;

    this->atomTypeCnt = 0;
    if (this->atomTypeMemory) delete[] this->atomTypes;
    this->atomTypes = NULL;

    this->chainCnt = 0;
    if (this->chainsMemory) delete[] this->chains;
    this->chains = NULL;

    this->dsBondsCnt = 0;
    if (this->dsBondsMemory) delete[] this->dsBonds;
    this->dsBonds = NULL;

    this->protAtomCnt = 0;
    if (this->protDataMemory) delete[] this->protAtomData;
    this->protAtomData = NULL;
    if (this->protPosMemory) delete[] this->protAtomPos;
    this->protAtomPos = NULL;

    this->solAtomCnt = 0;
    if (this->solDataMemory) delete[] this->solAtomData;
    this->solAtomData = NULL;
    if (this->solPosMemory) delete[] this->solAtomPos;
    this->solAtomPos = NULL;

    this->solMolTypeCnt = 0;
    if (this->solMolCntMemory) delete[] this->solMolCnt;
    this->solMolCnt = NULL;
    if (this->solMolTypeDataMemory) delete[] this->solMolTypeData;
    this->solMolTypeData = NULL;
}


/*
 * protein::CallProteinData::AccessChain
 */
protein::CallProteinData::Chain& protein::CallProteinData::AccessChain(
        unsigned int idx) {
    if ((!this->chainsMemory) || (this->chains == NULL)) {
        throw vislib::IllegalStateException(
            "Do not set values you do not own.", __FILE__, __LINE__);
    }
    if (idx >= this->chainCnt) {
        throw vislib::OutOfRangeException(idx, 0, this->chainCnt,
            __FILE__, __LINE__);
    }
    return this->chains[idx];
}


/*
 * protein::CallProteinData::AllocateAtomTypes
 */
void protein::CallProteinData::AllocateAtomTypes(unsigned int cnt) {
    this->atomTypeCnt = cnt;
    if (this->atomTypeMemory) delete[] this->atomTypes;
    this->atomTypes = new AtomType[cnt];
    this->atomTypeMemory = true;
}


/*
 * protein::CallProteinData::AllocateChains
 */
void protein::CallProteinData::AllocateChains(unsigned int cnt) {
    if (this->chainsMemory) delete[] this->chains;
    this->chainCnt = cnt;
    this->chains = new Chain[this->chainCnt];
    this->chainsMemory = true;
}


/*
 * protein::CallProteinData::AllocateProteinAtomData
 */
void protein::CallProteinData::AllocateProteinAtomData(void) {
    if (this->protDataMemory) delete[] this->protAtomData;
    this->protAtomData = new AtomData[this->protAtomCnt];
    this->protDataMemory = true;
}


/*
 * protein::CallProteinData::AllocateProteinAtomPositions
 */
void protein::CallProteinData::AllocateProteinAtomPositions(void) {
    if (this->protPosMemory) delete[] this->protAtomPos;
    this->protAtomPos = new float[3 * this->protAtomCnt];
    this->protPosMemory = true;
}


/*
 * protein::CallProteinData::AllocateSolventAtomData
 */
void protein::CallProteinData::AllocateSolventAtomData(void) {
    if (this->solDataMemory) delete[] this->solAtomData;
    this->solAtomData = new AtomData[this->solAtomCnt];
    this->solDataMemory = true;
}


/*
 * protein::CallProteinData::AllocateSolventAtomPositions
 */
void protein::CallProteinData::AllocateSolventAtomPositions(void) {
    if (this->solPosMemory) delete[] this->solAtomPos;
    this->solAtomPos = new float[3 * this->solAtomCnt];
    this->solPosMemory = true;
}


/*
 * protein::CallProteinData::CheckSolventMoleculeInformation
 */
bool protein::CallProteinData::CheckSolventMoleculeInformation(void) {
    if ((this->solMolCnt == NULL) || (this->solMolTypeData == NULL)
            || (this->solMolTypeCnt == 0)) {
        throw vislib::IllegalStateException(
            "No solvent molecule information set", __FILE__, __LINE__);
    }
    unsigned int cnt = 0;
    for (unsigned int i = 0; i < this->solMolTypeCnt; i++) {
        cnt += this->solMolCnt[i] * this->solMolTypeData[i].AtomCount();
    }
    return this->solAtomCnt == cnt;
}


/*
 * protein::CallProteinData::SetAminoAcidNameCount
 */
void protein::CallProteinData::SetAminoAcidNameCount(unsigned int cnt) {
    if (this->aminoAcidNameMemory) delete[] this->aminoAcidNames;
    this->aminoAcidNameCnt = cnt;
    this->aminoAcidNames = new vislib::StringA[cnt]; // throws bad_alloc
    this->aminoAcidNameMemory = (this->aminoAcidNames != NULL);
}


/*
 * protein::CallProteinData::SetAminoAcidName
 */
void protein::CallProteinData::SetAminoAcidName(unsigned int idx,
        const vislib::StringA& name) {
    if (!this->aminoAcidNameMemory) {
        throw vislib::IllegalStateException(
            "Amino acid name table is not owned by the interface",
            __FILE__, __LINE__);
    }
    if (idx >= this->aminoAcidNameCnt) {
        throw vislib::OutOfRangeException(idx, 0, this->aminoAcidNameCnt - 1,
            __FILE__, __LINE__);
    }
    this->aminoAcidNames[idx] = name;
}


/*
 * protein::CallProteinData::SetAminoAcidNameTable
 */
void protein::CallProteinData::SetAminoAcidNameTable(unsigned int cnt,
        const vislib::StringA *names) {
    if (this->aminoAcidNameMemory) delete[] this->aminoAcidNames;
    this->aminoAcidNameCnt = cnt;
    // DO NOT DEEP COPY
    this->aminoAcidNames = const_cast<vislib::StringA*>(names);
    this->aminoAcidNameMemory = false;
}


/*
 * protein::CallProteinData::SetAtomType
 */
void protein::CallProteinData::SetAtomType(unsigned int idx,
        const protein::CallProteinData::AtomType& type) {
    if ((!this->atomTypeMemory) || (this->atomTypes == NULL)) {
        throw vislib::IllegalStateException(
            "Do not set values you do not own.", __FILE__, __LINE__);
    }
    if (idx >= this->atomTypeCnt) {
        throw vislib::OutOfRangeException(idx, 0, this->atomTypeCnt,
            __FILE__, __LINE__);
    }
    this->atomTypes[idx] = type;
}


/*
 * protein::CallProteinData::SetAtomTypeTable
 */
void protein::CallProteinData::SetAtomTypeTable(unsigned int cnt,
        const protein::CallProteinData::AtomType *types) {
    if (this->atomTypeMemory) delete[] this->atomTypes;
    this->atomTypeCnt = cnt;
    // DO NOT DEEP COPY
    this->atomTypes = const_cast<protein::CallProteinData::AtomType*>(types);
    this->atomTypeMemory = false;
}


/*
 * protein::CallProteinData::SetDisulfidBondsPointer
 */
void protein::CallProteinData::SetDisulfidBondsPointer(unsigned int cnt,
        protein::CallProteinData::IndexPair *bonds) {
    if (this->dsBondsMemory) delete[] this->dsBonds;
    this->dsBondsCnt = cnt;
    this->dsBonds = bonds;
    this->dsBondsMemory = false;
}


/*
 * protein::CallProteinData::SetProteinAtomCount
 */
void protein::CallProteinData::SetProteinAtomCount(unsigned int cnt) {
    if (this->protDataMemory) ARY_SAFE_DELETE(this->protAtomData);
    if (this->protPosMemory) ARY_SAFE_DELETE(this->protAtomPos);
    this->protAtomCnt = cnt;
}


/*
 * protein::CallProteinData::SetProteinAtomData
 */
void protein::CallProteinData::SetProteinAtomData(unsigned int idx,
        const protein::CallProteinData::AtomData& data) {
    if ((!this->protDataMemory) || (this->protAtomData == NULL)) {
        throw vislib::IllegalStateException(
            "Do not set values you do not own.", __FILE__, __LINE__);
    }
    if (idx >= this->protAtomCnt) {
        throw vislib::OutOfRangeException(idx, 0, this->protAtomCnt,
            __FILE__, __LINE__);
    }
    this->protAtomData[idx] = data;
}


/*
 * protein::CallProteinData::SetProteinAtomDataPointer
 */
void protein::CallProteinData::SetProteinAtomDataPointer(
        protein::CallProteinData::AtomData *data) {
    if (this->protDataMemory) delete[] this->protAtomData;
    this->protDataMemory = false;
    this->protAtomData = data;
}


/*
 * protein::CallProteinData::SetProteinAtomPosition
 */
void protein::CallProteinData::SetProteinAtomPosition(unsigned int idx, float x,
        float y, float z) {
    if ((!this->protPosMemory) || (this->protAtomPos == NULL)) {
        throw vislib::IllegalStateException(
            "Do not set values you do not own.", __FILE__, __LINE__);
    }
    if (idx >= this->protAtomCnt) {
        throw vislib::OutOfRangeException(idx, 0, this->protAtomCnt,
            __FILE__, __LINE__);
    }
    this->protAtomPos[idx * 3] = x;
    this->protAtomPos[idx * 3 + 1] = y;
    this->protAtomPos[idx * 3 + 2] = z;
}


/*
 * protein::CallProteinData::SetProteinAtomPositionPointer
 */
void protein::CallProteinData::SetProteinAtomPositionPointer(float *pos) {
    if (this->protPosMemory) delete[] this->protAtomPos;
    this->protPosMemory = false;
    this->protAtomPos = pos;
}


/*
 * protein::CallProteinData::SetSolventAtomCount
 */
void protein::CallProteinData::SetSolventAtomCount(unsigned int cnt) {
    if (this->solDataMemory) ARY_SAFE_DELETE(this->solAtomData);
    if (this->solPosMemory) ARY_SAFE_DELETE(this->solAtomPos);
    this->solAtomCnt = cnt;
}


/*
 * protein::CallProteinData::SetSolventAtomData
 */
void protein::CallProteinData::SetSolventAtomData(unsigned int idx,
        const protein::CallProteinData::AtomData& data) {
    if ((!this->solDataMemory) || (this->solAtomData == NULL)) {
        throw vislib::IllegalStateException(
            "Do not set values you do not own.", __FILE__, __LINE__);
    }
    if (idx >= this->solAtomCnt) {
        throw vislib::OutOfRangeException(idx, 0, this->solAtomCnt,
            __FILE__, __LINE__);
    }
    this->solAtomData[idx] = data;
}


/*
 * protein::CallProteinData::SetSolventAtomDataPointer
 */
void protein::CallProteinData::SetSolventAtomDataPointer(
        protein::CallProteinData::AtomData *data) {
    if (this->solDataMemory) delete[] this->solAtomData;
    this->solDataMemory = false;
    this->solAtomData = data;
}


/*
 * protein::CallProteinData::SetSolventAtomPosition
 */
void protein::CallProteinData::SetSolventAtomPosition(unsigned int idx, float x,
        float y, float z) {
    if ((!this->solPosMemory) || (this->solAtomPos == NULL)) {
        throw vislib::IllegalStateException(
            "Do not set values you do not own.", __FILE__, __LINE__);
    }
    if (idx >= this->solAtomCnt) {
        throw vislib::OutOfRangeException(idx, 0, this->solAtomCnt,
            __FILE__, __LINE__);
    }
    this->solAtomPos[idx * 3] = x;
    this->solAtomPos[idx * 3 + 1] = y;
    this->solAtomPos[idx * 3 + 2] = z;
}


/*
 * protein::CallProteinData::SetSolventAtomPositionPointer
 */
void protein::CallProteinData::SetSolventAtomPositionPointer(float *pos) {
    if (this->solPosMemory) delete[] this->solAtomPos;
    this->solPosMemory = false;
    this->solAtomPos = pos;
}


/*
 * protein::CallProteinData::SetSolventMoleculeCount
 */
void protein::CallProteinData::SetSolventMoleculeCount(unsigned int idx,
        unsigned int cnt) {
    if ((!this->solMolCntMemory) || (this->solMolCnt == NULL)
            || (!this->solMolTypeDataMemory)
            || (this->solMolTypeData == NULL)) {
        throw vislib::IllegalStateException(
            "Do not set values you do not own.", __FILE__, __LINE__);
    }
    if (idx >= this->solMolTypeCnt) {
        throw vislib::OutOfRangeException(idx, 0, this->solMolTypeCnt,
            __FILE__, __LINE__);
    }

    this->solMolCnt[idx] = cnt;
}


/*
 * protein::CallProteinData::SetSolventMoleculeTypeCount
 */
void protein::CallProteinData::SetSolventMoleculeTypeCount(unsigned int cnt) {
    if (this->solMolCntMemory) delete[] this->solMolCnt;
    if (this->solMolTypeDataMemory) delete[] this->solMolTypeData;
    this->solMolTypeCnt = cnt;
    this->solMolCnt = new unsigned int[this->solMolTypeCnt];
    this->solMolCntMemory = true;
    this->solMolTypeData = new SolventMoleculeData[this->solMolTypeCnt];
    this->solMolTypeDataMemory = true;
    for (unsigned int i = 0; i < this->solMolTypeCnt; i++) {
        this->solMolCnt[i] = 0;
    }
}


/*
 * protein::CallProteinData::SetSolventMoleculeData
 */
void protein::CallProteinData::SetSolventMoleculeData(unsigned int idx,
        const protein::CallProteinData::SolventMoleculeData& data) {
    if ((!this->solMolCntMemory) || (this->solMolCnt == NULL)
            || (!this->solMolTypeDataMemory)
            || (this->solMolTypeData == NULL)) {
        throw vislib::IllegalStateException(
            "Do not set values you do not own.", __FILE__, __LINE__);
    }
    if (idx >= this->solMolTypeCnt) {
        throw vislib::OutOfRangeException(idx, 0, this->solMolTypeCnt,
            __FILE__, __LINE__);
    }

    this->solMolTypeData[idx] = data;
}
