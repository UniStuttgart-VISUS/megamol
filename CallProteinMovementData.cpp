/*
 * CallProteinMovementData.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CallProteinMovementData.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/mathfunctions.h"
#include "vislib/OutOfRangeException.h"

using namespace megamol::core;


/****************************************************************************/

/*
 * protein::CallProteinMovementData::AminoAcid::AminoAcid
 */
protein::CallProteinMovementData::AminoAcid::AminoAcid(void) : atomCnt(0), cAlphaIdx(0),
        cCarbIdx(0), connectivity(), firstAtomIdx(0), nameIdx(0), nIdx(0),
        oIdx(0) {
    // intentionally empty
}


/*
 * protein::CallProteinMovementData::AminoAcid::AminoAcid
 */
protein::CallProteinMovementData::AminoAcid::AminoAcid(
        const protein::CallProteinMovementData::AminoAcid& src) : atomCnt(src.atomCnt),
        cAlphaIdx(src.cAlphaIdx), cCarbIdx(src.cCarbIdx),
        connectivity(src.connectivity), firstAtomIdx(src.firstAtomIdx),
        nameIdx(src.nameIdx), nIdx(src.nIdx), oIdx(src.oIdx) {
    // intentionally empty
}


/*
 * protein::CallProteinMovementData::AminoAcid::AminoAcid
 */
protein::CallProteinMovementData::AminoAcid::AminoAcid(unsigned int firstAtomIdx,
        unsigned int atomCnt, unsigned int cAlphaIdx, unsigned int cCarbIdx,
        unsigned int nIdx, unsigned int oIdx, unsigned int nameIdx) :
        atomCnt(atomCnt), cAlphaIdx(cAlphaIdx), cCarbIdx(cCarbIdx),
        connectivity(), firstAtomIdx(firstAtomIdx), nameIdx(nameIdx),
        nIdx(nIdx), oIdx(oIdx) {
    // intentionally empty
}


/*
 * protein::CallProteinMovementData::AminoAcid::~AminoAcid
 */
protein::CallProteinMovementData::AminoAcid::~AminoAcid(void) {
    // intentionally empty
}


/*
 * protein::CallProteinMovementData::AminoAcid::SetCAlphaIndex
 */
void protein::CallProteinMovementData::AminoAcid::SetCAlphaIndex(unsigned int idx) {
    this->cAlphaIdx = idx;
}


/*
 * protein::CallProteinMovementData::AminoAcid::SetCCarbIndex
 */
void protein::CallProteinMovementData::AminoAcid::SetCCarbIndex(unsigned int idx) {
    this->cCarbIdx = idx;
}


/*
 * protein::CallProteinMovementData::AminoAcid::SetNIndex
 */
void protein::CallProteinMovementData::AminoAcid::SetNIndex(unsigned int idx) {
    this->nIdx = idx;
}


/*
 * protein::CallProteinMovementData::AminoAcid::SetOIndex
 */
void protein::CallProteinMovementData::AminoAcid::SetOIndex(unsigned int idx) {
    this->oIdx = idx;
}


/*
 * protein::CallProteinMovementData::AminoAcid::SetPosition
 */
void protein::CallProteinMovementData::AminoAcid::SetPosition(unsigned int firstAtom,
        unsigned int atomCnt) {
    this->firstAtomIdx = firstAtom;
    this->atomCnt = atomCnt;
}


/*
 * protein::CallProteinMovementData::AminoAcid::SetNameIndex
 */
void protein::CallProteinMovementData::AminoAcid::SetNameIndex(unsigned int idx) {
    this->nameIdx = idx;
}


/*
 * protein::CallProteinMovementData::AminoAcid::operator=
 */
protein::CallProteinMovementData::AminoAcid& 
protein::CallProteinMovementData::AminoAcid::operator=(
        const protein::CallProteinMovementData::AminoAcid& rhs) {
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
 * protein::CallProteinMovementData::AminoAcid::operator==
 */
bool protein::CallProteinMovementData::AminoAcid::operator==(
        const protein::CallProteinMovementData::AminoAcid& rhs) const {
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
 * protein::CallProteinMovementData::AtomData::AtomData
 */
protein::CallProteinMovementData::AtomData::AtomData(unsigned int typeIdx, float charge, 
        float tempFactor, float occupancy) : charge(charge),
        occupancy(occupancy), tempFactor(tempFactor), typeIdx(typeIdx) {
    // intentionally empty
}


/*
 * protein::CallProteinMovementData::AtomData::AtomData
 */
protein::CallProteinMovementData::AtomData::AtomData(const AtomData& src) {
    *this = src;
}


/*
 * protein::CallProteinMovementData::AtomData::~AtomData
 */
protein::CallProteinMovementData::AtomData::~AtomData(void) {
    // intentionally empty
}


/*
 * protein::CallProteinMovementData::AtomData::operator=
 */
protein::CallProteinMovementData::AtomData& protein::CallProteinMovementData::AtomData::operator=(
        const protein::CallProteinMovementData::AtomData& rhs) {
    this->charge = rhs.charge;
    this->occupancy = rhs.occupancy;
    this->tempFactor = rhs.tempFactor;
    this->typeIdx = rhs.typeIdx;
    return *this;
}


/*
 * protein::CallProteinMovementData::AtomData::operator==
 */
bool protein::CallProteinMovementData::AtomData::operator==(
        const protein::CallProteinMovementData::AtomData& rhs) const {
    return vislib::math::IsEqual(this->charge, rhs.charge)
        && vislib::math::IsEqual(this->occupancy, rhs.occupancy)
        && vislib::math::IsEqual(this->tempFactor, rhs.tempFactor)
        && (this->typeIdx == rhs.typeIdx);
}

/****************************************************************************/

/*
 * protein::CallProteinMovementData::AtomType::AtomType
 */
protein::CallProteinMovementData::AtomType::AtomType(void) : name(), rad(0.5f) {
    this->col[0] = this->col[1] = this->col[2] = 191;
}


/*
 * protein::CallProteinMovementData::AtomType::AtomType
 */
protein::CallProteinMovementData::AtomType::AtomType(const vislib::StringA& name,
        float rad, unsigned char colR, unsigned char colG, unsigned char colB)
        : name(name), rad(rad) {
    this->col[0] = colR;
    this->col[1] = colG;
    this->col[2] = colB;
}


/*
 * protein::CallProteinMovementData::AtomType::AtomType
 */
protein::CallProteinMovementData::AtomType::AtomType(const AtomType& src) {
    *this = src;
}


/*
 * protein::CallProteinMovementData::AtomType::~AtomType
 */
protein::CallProteinMovementData::AtomType::~AtomType(void) {
}


/*
 * protein::CallProteinMovementData::AtomType::operator=
 */
protein::CallProteinMovementData::AtomType& protein::CallProteinMovementData::AtomType::operator=(
        const protein::CallProteinMovementData::AtomType& rhs) {
    this->name = rhs.name;
    this->rad = rhs.rad;
    this->col[0] = rhs.col[0];
    this->col[1] = rhs.col[1];
    this->col[2] = rhs.col[2];
    return *this;
}


/*
 * protein::CallProteinMovementData::AtomType::operator==
 */
bool protein::CallProteinMovementData::AtomType::operator==(
        const protein::CallProteinMovementData::AtomType& rhs) const {
    return vislib::math::IsEqual(this->rad, rhs.rad)
        && (this->name == rhs.name)
        && (this->col[0] == rhs.col[0])
        && (this->col[1] == rhs.col[1])
        && (this->col[2] == rhs.col[2]);
}


/****************************************************************************/

/*
 * protein::CallProteinMovementData::Chain::Chain
 */
protein::CallProteinMovementData::Chain::Chain(void) : aminoAcid(NULL), aminoAcidCnt(0),
        aminoAcidMemory(false), secStruct(NULL), secStructCnt(0),
        secStructMemory(false) {
}


/*
 * protein::CallProteinMovementData::Chain::Chain
 */
protein::CallProteinMovementData::Chain::Chain(const protein::CallProteinMovementData::Chain& src)
        : aminoAcid(NULL), aminoAcidCnt(0), aminoAcidMemory(false),
        secStruct(NULL), secStructCnt(0), secStructMemory(false) {
    *this = src;
}


/*
 * protein::CallProteinMovementData::Chain::~Chain
 */
protein::CallProteinMovementData::Chain::~Chain(void) {
    if (this->aminoAcidMemory) delete[] this->aminoAcid;
    this->aminoAcid = NULL;
    this->aminoAcidCnt = 0;
    if (this->secStructMemory) delete[] this->secStruct;
    this->secStruct = NULL;
    this->secStructCnt = 0;
}


/*
 * protein::CallProteinMovementData::Chain::AccessAminoAcid
 */
protein::CallProteinMovementData::AminoAcid&
protein::CallProteinMovementData::Chain::AccessAminoAcid(unsigned int idx) {
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
 * protein::CallProteinMovementData::Chain::AccessSecondaryStructure
 */
protein::CallProteinMovementData::SecStructure&
protein::CallProteinMovementData::Chain::AccessSecondaryStructure(unsigned int idx) {
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
 * protein::CallProteinMovementData::Chain::SetAminoAcid
 */
void protein::CallProteinMovementData::Chain::SetAminoAcid(unsigned int cnt,
        const protein::CallProteinMovementData::AminoAcid* aminoAcids) {
    if (this->aminoAcidMemory) delete[] this->aminoAcid;
    this->aminoAcidCnt = cnt;
    this->aminoAcid 
        = const_cast<protein::CallProteinMovementData::AminoAcid*>(aminoAcids);
    this->aminoAcidMemory = false;
}


/*
 * protein::CallProteinMovementData::Chain::SetAminoAcidCount
 */
void protein::CallProteinMovementData::Chain::SetAminoAcidCount(unsigned int cnt) {
    if (this->aminoAcidMemory) delete[] this->aminoAcid;
    this->aminoAcidCnt = cnt;
    this->aminoAcid = new protein::CallProteinMovementData::AminoAcid[cnt];
    this->aminoAcidMemory = true;
}


/*
 * protein::CallProteinMovementData::Chain::SetSecondaryStructure
 */
void protein::CallProteinMovementData::Chain::SetSecondaryStructure(unsigned int cnt,
        const protein::CallProteinMovementData::SecStructure* structs) {
    if (this->secStructMemory) delete[] this->secStruct;
    this->secStructCnt = cnt;
    this->secStruct 
        = const_cast<protein::CallProteinMovementData::SecStructure*>(structs);
    this->secStructMemory = false;
}


/*
 * protein::CallProteinMovementData::Chain::SetSecondaryStructureCount
 */
void protein::CallProteinMovementData::Chain::SetSecondaryStructureCount(
        unsigned int cnt) {
    if (this->secStructMemory) delete[] this->secStruct;
    this->secStructCnt = cnt;
    this->secStruct = new protein::CallProteinMovementData::SecStructure[cnt];
    this->secStructMemory = true;
}


/*
 * protein::CallProteinMovementData::Chain::operator=
 */
protein::CallProteinMovementData::Chain& protein::CallProteinMovementData::Chain::operator=(
        const protein::CallProteinMovementData::Chain& rhs) {
    if (this->aminoAcidMemory) delete[] this->aminoAcid;
    this->aminoAcidCnt = rhs.aminoAcidCnt;
    if ((this->aminoAcidMemory = rhs.aminoAcidMemory) == true) {
        this->aminoAcid 
            = new protein::CallProteinMovementData::AminoAcid[this->aminoAcidCnt];
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
            = new protein::CallProteinMovementData::SecStructure[this->secStructCnt];
        for (unsigned int i = 0; i < this->secStructCnt; i++) {
            this->secStruct[i] = rhs.secStruct[i];
        }
    } else {
        this->secStruct = rhs.secStruct;
    }
    return *this;
}


/*
 * protein::CallProteinMovementData::Chain::operator==
 */
bool protein::CallProteinMovementData::Chain::operator==(
        const protein::CallProteinMovementData::Chain& rhs) const {
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
 * protein::CallProteinMovementData::SecStructure::SecStructure
 */
protein::CallProteinMovementData::SecStructure::SecStructure(void) : aminoAcidCnt(0),
        atomCnt(0), firstAminoAcidIdx(0), firstAtomIdx(0), type(TYPE_COIL) {
    // intentionally empty
}


/*
 * protein::CallProteinMovementData::SecStructure::SecStructure
 */
protein::CallProteinMovementData::SecStructure::SecStructure(
        const protein::CallProteinMovementData::SecStructure& src)
        : aminoAcidCnt(src.aminoAcidCnt), atomCnt(src.atomCnt),
        firstAminoAcidIdx(src.firstAminoAcidIdx),
        firstAtomIdx(src.firstAtomIdx), type(src.type) {
    // intentionally empty
}


/*
 * protein::CallProteinMovementData::SecStructure::~SecStructure
 */
protein::CallProteinMovementData::SecStructure::~SecStructure(void) {
    // intentionally empty
}


/*
 * protein::CallProteinMovementData::SecStructure::SetPosition
 */
void protein::CallProteinMovementData::SecStructure::SetPosition(
        unsigned int firstAtomIdx, unsigned int atomCnt,
        unsigned int firstAminoAcidIdx, unsigned int aminoAcidCnt) {
    this->firstAtomIdx = firstAtomIdx;
    this->atomCnt = atomCnt;
    this->firstAminoAcidIdx = firstAminoAcidIdx;
    this->aminoAcidCnt = aminoAcidCnt;
}


/*
 * protein::CallProteinMovementData::SecStructure::SetType
 */
void protein::CallProteinMovementData::SecStructure::SetType(
        protein::CallProteinMovementData::SecStructure::ElementType type) {
    this->type = type;
}


/*
 * protein::CallProteinMovementData::SecStructure::operator=
 */
protein::CallProteinMovementData::SecStructure&
protein::CallProteinMovementData::SecStructure::operator=(
        const protein::CallProteinMovementData::SecStructure& rhs) {
    this->aminoAcidCnt = rhs.aminoAcidCnt;
    this->atomCnt = rhs.atomCnt;
    this->firstAminoAcidIdx = rhs.firstAminoAcidIdx;
    this->firstAtomIdx = rhs.firstAtomIdx;
    this->type = rhs.type;
    return *this;
}


/*
 * protein::CallProteinMovementData::SecStructure::operator==
 */
bool protein::CallProteinMovementData::SecStructure::operator==(const protein::CallProteinMovementData::SecStructure& rhs) const {
    return ((this->aminoAcidCnt == rhs.aminoAcidCnt)
        && (this->atomCnt == rhs.atomCnt)
        && (this->firstAminoAcidIdx == rhs.firstAminoAcidIdx)
        && (this->firstAtomIdx == rhs.firstAtomIdx)
        && (this->type == rhs.type));
}


/****************************************************************************/

/*
 * protein::CallProteinMovementData::SolventMoleculeData::SolventMoleculeData
 */
protein::CallProteinMovementData::SolventMoleculeData::SolventMoleculeData(void)
        : atomCnt(0), name(NULL), connectivity() {
    // intentionally empty
}


/*
 * protein::CallProteinMovementData::SolventMoleculeData::SolventMoleculeData
 */
protein::CallProteinMovementData::SolventMoleculeData::SolventMoleculeData(
        const protein::CallProteinMovementData::SolventMoleculeData& src)
        : atomCnt(src.atomCnt), name(src.name),
        connectivity(src.connectivity) {
    // intentionally empty
}


/*
 * protein::CallProteinMovementData::SolventMoleculeData::SolventMoleculeData
 */
protein::CallProteinMovementData::SolventMoleculeData::SolventMoleculeData(
        const vislib::StringA& name, unsigned int atomCnt) : atomCnt(atomCnt),
        name(name), connectivity() {
    // intentionally empty
}


/*
 * protein::CallProteinMovementData::SolventMoleculeData::~SolventMoleculeData
 */
protein::CallProteinMovementData::SolventMoleculeData::~SolventMoleculeData(void) {
    // intentionally empty
}


/*
 * protein::CallProteinMovementData::SolventMoleculeData::AddConnection
 */
void protein::CallProteinMovementData::SolventMoleculeData::AddConnection(
        const protein::CallProteinMovementData::IndexPair& connection) {
    this->connectivity.Add(connection);
}


/*
 * protein::CallProteinMovementData::SolventMoleculeData::AllocateConnectivityList
 */
void protein::CallProteinMovementData::SolventMoleculeData::AllocateConnectivityList(
        unsigned int cnt) {
    this->connectivity.SetCount(cnt);
}


/*
 * protein::CallProteinMovementData::SolventMoleculeData::ClearConnectivity
 */
void protein::CallProteinMovementData::SolventMoleculeData::ClearConnectivity(void) {
    this->connectivity.Clear();
}


/*
 * protein::CallProteinMovementData::SolventMoleculeData::SetAtomCount
 */
void protein::CallProteinMovementData::SolventMoleculeData::SetAtomCount(
        unsigned int cnt) {
    this->atomCnt = cnt;
    // Perhaps the connectivity should be checked here and invalid connections
    // should be removed. But I believe that this is up the the user of this
    // class
}


/*
 * protein::CallProteinMovementData::SolventMoleculeData::SetConnection
 */
void protein::CallProteinMovementData::SolventMoleculeData::SetConnection(
        unsigned int idx, const protein::CallProteinMovementData::IndexPair& connection) {
    this->connectivity[idx] = connection;
}


/*
 * protein::CallProteinMovementData::SolventMoleculeData::SetConnections
 */
void protein::CallProteinMovementData::SolventMoleculeData::SetConnections(
        unsigned int cnt,
        const protein::CallProteinMovementData::IndexPair *connections) {
    this->connectivity.SetCount(cnt);
    for (unsigned int i = 0; i < cnt; i++) {
        this->connectivity[i] = connections[i];
    }
}


/*
 * protein::CallProteinMovementData::SolventMoleculeData::SetName
 */
void protein::CallProteinMovementData::SolventMoleculeData::SetName(
        const vislib::StringA& name) {
    this->name = name;
}


/*
 * protein::CallProteinMovementData::SolventMoleculeData::operator=
 */
protein::CallProteinMovementData::SolventMoleculeData&
protein::CallProteinMovementData::SolventMoleculeData::operator=(
        const protein::CallProteinMovementData::SolventMoleculeData& rhs) {
    this->atomCnt = rhs.atomCnt;
    this->connectivity = rhs.connectivity;
    this->name = rhs.name;
    return *this;
}


/*
 * protein::CallProteinMovementData::SolventMoleculeData::operator==
 */
bool protein::CallProteinMovementData::SolventMoleculeData::operator==(
        const protein::CallProteinMovementData::SolventMoleculeData& rhs) const {
    return ((this->atomCnt == rhs.atomCnt)
        && (this->connectivity == rhs.connectivity)
        && (this->name == rhs.name));
}


/****************************************************************************/

/*
 * protein::CallProteinMovementData::CallProteinMovementData
 */
protein::CallProteinMovementData::CallProteinMovementData(void) : Call(), 
        aminoAcidNameCnt(0), aminoAcidNameMemory(false), aminoAcidNames(NULL), 
        atomTypeMemory(false), atomTypes(NULL), chainCnt(0), chains(NULL),
        chainsMemory(false), dsBondsCnt(0), dsBonds(NULL), atomTypeCnt(0),
        dsBondsMemory(false), protAtomCnt(0), protAtomData(NULL),
        protAtomPos(NULL), protAtomMovementPos(NULL),protDataMemory(false),
        protPosMemory(false), protMovedPosMemory(false),
        solAtomCnt(0), solAtomData(NULL), solAtomPos(NULL),
        solDataMemory(false), solMolCnt(NULL), solMolCntMemory(false),
        solMolTypeCnt(0), solMolTypeData(NULL), solMolTypeDataMemory(false),
        solPosMemory(false), minOccupancy(0.0f), maxOccupancy(0.0f),
        minTempFactor(0.0f), maxTempFactor(0.0f), 
        minCharge(0.0f), maxCharge(0.0f), 
        useRMS(false), currentRMSFrameID(0), currentFrameId( 0),
        maxMovementDist( 0.0f)
{
    // intentionally empty
}


/*
 * protein::CallProteinMovementData::~CallProteinMovementData
 */
protein::CallProteinMovementData::~CallProteinMovementData(void) {
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
 * protein::CallProteinMovementData::AccessChain
 */
protein::CallProteinMovementData::Chain& protein::CallProteinMovementData::AccessChain(
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
 * protein::CallProteinMovementData::AllocateAtomTypes
 */
void protein::CallProteinMovementData::AllocateAtomTypes(unsigned int cnt) {
    this->atomTypeCnt = cnt;
    if (this->atomTypeMemory) delete[] this->atomTypes;
    this->atomTypes = new AtomType[cnt];
    this->atomTypeMemory = true;
}


/*
 * protein::CallProteinMovementData::AllocateChains
 */
void protein::CallProteinMovementData::AllocateChains(unsigned int cnt) {
    if (this->chainsMemory) delete[] this->chains;
    this->chainCnt = cnt;
    this->chains = new Chain[this->chainCnt];
    this->chainsMemory = true;
}


/*
 * protein::CallProteinMovementData::AllocateProteinAtomData
 */
void protein::CallProteinMovementData::AllocateProteinAtomData(void) {
    if (this->protDataMemory) delete[] this->protAtomData;
    this->protAtomData = new AtomData[this->protAtomCnt];
    this->protDataMemory = true;
}


/*
 * protein::CallProteinMovementData::AllocateProteinAtomPositions
 */
void protein::CallProteinMovementData::AllocateProteinAtomPositions(void) {
    if (this->protPosMemory) delete[] this->protAtomPos;
    this->protAtomPos = new float[3 * this->protAtomCnt];
    this->protPosMemory = true;
}


/*
 * protein::CallProteinMovementData::AllocateProteinMovementAtomPositions
 */
void protein::CallProteinMovementData::AllocateProteinAtomMovementPositions(void) {
    if (this->protMovedPosMemory) delete[] this->protAtomMovementPos;
    this->protAtomMovementPos = new float[3 * this->protAtomCnt];
    this->protMovedPosMemory = true;
}


/*
 * protein::CallProteinMovementData::AllocateSolventAtomData
 */
void protein::CallProteinMovementData::AllocateSolventAtomData(void) {
    if (this->solDataMemory) delete[] this->solAtomData;
    this->solAtomData = new AtomData[this->solAtomCnt];
    this->solDataMemory = true;
}


/*
 * protein::CallProteinMovementData::AllocateSolventAtomPositions
 */
void protein::CallProteinMovementData::AllocateSolventAtomPositions(void) {
    if (this->solPosMemory) delete[] this->solAtomPos;
    this->solAtomPos = new float[3 * this->solAtomCnt];
    this->solPosMemory = true;
}


/*
 * protein::CallProteinMovementData::CheckSolventMoleculeInformation
 */
bool protein::CallProteinMovementData::CheckSolventMoleculeInformation(void) {
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
 * protein::CallProteinMovementData::SetAminoAcidNameCount
 */
void protein::CallProteinMovementData::SetAminoAcidNameCount(unsigned int cnt) {
    if (this->aminoAcidNameMemory) delete[] this->aminoAcidNames;
    this->aminoAcidNameCnt = cnt;
    this->aminoAcidNames = new vislib::StringA[cnt]; // throws bad_alloc
    this->aminoAcidNameMemory = (this->aminoAcidNames != NULL);
}


/*
 * protein::CallProteinMovementData::SetAminoAcidName
 */
void protein::CallProteinMovementData::SetAminoAcidName(unsigned int idx,
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
    this->aminoAcidNames[idx];
}


/*
 * protein::CallProteinMovementData::SetAminoAcidNameTable
 */
void protein::CallProteinMovementData::SetAminoAcidNameTable(unsigned int cnt,
        const vislib::StringA *names) {
    if (this->aminoAcidNameMemory) delete[] this->aminoAcidNames;
    this->aminoAcidNameCnt = cnt;
    // DO NOT DEEP COPY
    this->aminoAcidNames = const_cast<vislib::StringA*>(names);
    this->aminoAcidNameMemory = false;
}


/*
 * protein::CallProteinMovementData::SetAtomType
 */
void protein::CallProteinMovementData::SetAtomType(unsigned int idx,
        const protein::CallProteinMovementData::AtomType& type) {
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
 * protein::CallProteinMovementData::SetAtomTypeTable
 */
void protein::CallProteinMovementData::SetAtomTypeTable(unsigned int cnt,
        const protein::CallProteinMovementData::AtomType *types) {
    if (this->atomTypeMemory) delete[] this->atomTypes;
    this->atomTypeCnt = cnt;
    // DO NOT DEEP COPY
    this->atomTypes = const_cast<protein::CallProteinMovementData::AtomType*>(types);
    this->atomTypeMemory = false;
}


/*
 * protein::CallProteinMovementData::SetDisulfidBondsPointer
 */
void protein::CallProteinMovementData::SetDisulfidBondsPointer(unsigned int cnt,
        protein::CallProteinMovementData::IndexPair *bonds) {
    if (this->dsBondsMemory) delete[] this->dsBonds;
    this->dsBondsCnt = cnt;
    this->dsBonds = bonds;
    this->dsBondsMemory = false;
}


/*
 * protein::CallProteinMovementData::SetProteinAtomCount
 */
void protein::CallProteinMovementData::SetProteinAtomCount(unsigned int cnt) {
    if (this->protDataMemory) ARY_SAFE_DELETE(this->protAtomData);
    if (this->protPosMemory) ARY_SAFE_DELETE(this->protAtomPos);
    if (this->protMovedPosMemory) ARY_SAFE_DELETE(this->protAtomMovementPos);
    this->protAtomCnt = cnt;
}


/*
 * protein::CallProteinMovementData::SetProteinAtomData
 */
void protein::CallProteinMovementData::SetProteinAtomData(unsigned int idx,
        const protein::CallProteinMovementData::AtomData& data) {
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
 * protein::CallProteinMovementData::SetProteinAtomDataPointer
 */
void protein::CallProteinMovementData::SetProteinAtomDataPointer(
        protein::CallProteinMovementData::AtomData *data) {
    if (this->protDataMemory) delete[] this->protAtomData;
    this->protDataMemory = false;
    this->protAtomData = data;
}


/*
 * protein::CallProteinMovementData::SetProteinAtomPosition
 */
void protein::CallProteinMovementData::SetProteinAtomPosition(unsigned int idx, float x,
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
 * protein::CallProteinMovementData::SetProteinAtomMovementPosition
 */
void protein::CallProteinMovementData::SetProteinAtomMovementPosition(unsigned int idx, float x,
        float y, float z) {
    if ((!this->protMovedPosMemory) || (this->protAtomMovementPos == NULL)) {
        throw vislib::IllegalStateException(
            "Do not set values you do not own.", __FILE__, __LINE__);
    }
    if (idx >= this->protAtomCnt) {
        throw vislib::OutOfRangeException(idx, 0, this->protAtomCnt,
            __FILE__, __LINE__);
    }
    this->protAtomMovementPos[idx * 3] = x;
    this->protAtomMovementPos[idx * 3 + 1] = y;
    this->protAtomMovementPos[idx * 3 + 2] = z;
}


/*
 * protein::CallProteinMovementData::SetProteinAtomPositionPointer
 */
void protein::CallProteinMovementData::SetProteinAtomPositionPointer(float *pos) {
    if (this->protPosMemory) delete[] this->protAtomPos;
    this->protPosMemory = false;
    this->protAtomPos = pos;
}


/*
 * protein::CallProteinMovementData::SetProteinAtomMovementPositionPointer
 */
void protein::CallProteinMovementData::SetProteinAtomMovementPositionPointer(float *pos) {
    if (this->protMovedPosMemory) delete[] this->protAtomMovementPos;
    this->protPosMemory = false;
    this->protAtomMovementPos = pos;
}


/*
 * protein::CallProteinMovementData::SetSolventAtomCount
 */
void protein::CallProteinMovementData::SetSolventAtomCount(unsigned int cnt) {
    if (this->solDataMemory) ARY_SAFE_DELETE(this->solAtomData);
    if (this->solPosMemory) ARY_SAFE_DELETE(this->solAtomPos);
    this->solAtomCnt = cnt;
}


/*
 * protein::CallProteinMovementData::SetSolventAtomData
 */
void protein::CallProteinMovementData::SetSolventAtomData(unsigned int idx,
        const protein::CallProteinMovementData::AtomData& data) {
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
 * protein::CallProteinMovementData::SetSolventAtomDataPointer
 */
void protein::CallProteinMovementData::SetSolventAtomDataPointer(
        protein::CallProteinMovementData::AtomData *data) {
    if (this->solDataMemory) delete[] this->solAtomData;
    this->solDataMemory = false;
    this->solAtomData = data;
}


/*
 * protein::CallProteinMovementData::SetSolventAtomPosition
 */
void protein::CallProteinMovementData::SetSolventAtomPosition(unsigned int idx, float x,
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
 * protein::CallProteinMovementData::SetSolventAtomPositionPointer
 */
void protein::CallProteinMovementData::SetSolventAtomPositionPointer(float *pos) {
    if (this->solPosMemory) delete[] this->solAtomPos;
    this->solPosMemory = false;
    this->solAtomPos = pos;
}


/*
 * protein::CallProteinMovementData::SetSolventMoleculeCount
 */
void protein::CallProteinMovementData::SetSolventMoleculeCount(unsigned int idx,
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
 * protein::CallProteinMovementData::SetSolventMoleculeTypeCount
 */
void protein::CallProteinMovementData::SetSolventMoleculeTypeCount(unsigned int cnt) {
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
 * protein::CallProteinMovementData::SetSolventMoleculeData
 */
void protein::CallProteinMovementData::SetSolventMoleculeData(unsigned int idx,
        const protein::CallProteinMovementData::SolventMoleculeData& data) {
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
