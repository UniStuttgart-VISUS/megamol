/*
 * UncertaintyDataCall.cpp
 *
 * Copyright (C) 2016 by University of Stuttgart (VISUS).
 * All rights reserved.
 *
 * This module is based on the source code of "SequenceRenderere" in protein_calls plugin (svn revision 17).
 *
 */

#include "stdafx.h"
#include "protein_calls/UncertaintyDataCall.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/OutOfRangeException.h"

using namespace megamol;
using namespace megamol::protein_uncertainty;


// ======================================================================

/*
 * UncertaintyDataCall::SecStructure::SecStructure
 */
UncertaintyDataCall::SecStructure::SecStructure(void) : aminoAcidCnt(0),
        firstAminoAcidIdx(0), type(TYPE_COIL) {
    // intentionally empty
}


/*
 * UncertaintyDataCall::SecStructure::SecStructure
 */
UncertaintyDataCall::SecStructure::SecStructure(
        const UncertaintyDataCall::SecStructure& src)
        : aminoAcidCnt(src.aminoAcidCnt), 
        firstAminoAcidIdx(src.firstAminoAcidIdx), type(src.type) {
    // intentionally empty
}


/*
 * UncertaintyDataCall::SecStructure::~SecStructure
 */
UncertaintyDataCall::SecStructure::~SecStructure(void) {
    // intentionally empty
}


/*
 * UncertaintyDataCall::SecStructure::SetPosition
 */
void UncertaintyDataCall::SecStructure::SetPosition(
        unsigned int firstAminoAcidIdx, unsigned int aminoAcidCnt) {
    this->firstAminoAcidIdx = firstAminoAcidIdx;
    this->aminoAcidCnt = aminoAcidCnt;
}


/*
 * UncertaintyDataCall::SecStructure::SetType
 */
void UncertaintyDataCall::SecStructure::SetType(
        UncertaintyDataCall::SecStructure::ElementType type) {
    this->type = type;
}


/*
 * UncertaintyDataCall::SecStructure::operator=
 */
UncertaintyDataCall::SecStructure&
UncertaintyDataCall::SecStructure::operator=(
        const UncertaintyDataCall::SecStructure& rhs) {
    this->aminoAcidCnt = rhs.aminoAcidCnt;
    this->firstAminoAcidIdx = rhs.firstAminoAcidIdx;
    this->type = rhs.type;
    return *this;
}


/*
 * UncertaintyDataCall::SecStructure::operator==
 */
bool UncertaintyDataCall::SecStructure::operator==(const UncertaintyDataCall::SecStructure& rhs) const {
    return ((this->aminoAcidCnt == rhs.aminoAcidCnt)
        && (this->firstAminoAcidIdx == rhs.firstAminoAcidIdx)
        && (this->type == rhs.type));
}

// ======================================================================

/*
 * UncertaintyDataCall::CallForGetData
 */
const unsigned int UncertaintyDataCall::CallForGetData = 0;


/*
 * UncertaintyDataCall::CallForGetExtent
 */
const unsigned int UncertaintyDataCall::CallForGetExtent = 1;


/*
 * UncertaintyDataCall::UncertaintyDataCall
 */
UncertaintyDataCall::UncertaintyDataCall(void) : AbstractGetDataCall() {

}


/*
 * UncertaintyDataCall::~UncertaintyDataCall
 */
UncertaintyDataCall::~UncertaintyDataCall(void) {
}


/*
 * Set the number of secondary structure elements.
 */
void UncertaintyDataCall::SetSecondaryStructureCount( unsigned int cnt) {
    this->secStruct.Clear();
    this->secStruct.SetCount( cnt);
}

/*
 * Set a secondary stucture element to the array.
 */
bool UncertaintyDataCall::SetSecondaryStructure( unsigned int idx, SecStructure secS) {
    if( idx < this->secStruct.Count() ) {
        this->secStruct[idx] = secS;
        return true;
    } else {
        return false;
    }
}

/*
 * Get the secondary structure.
 */
const UncertaintyDataCall::SecStructure* UncertaintyDataCall::SecondaryStructures() const {
    return this->secStruct.PeekElements();
}

/*
 * Get the number of secondary structure elements.
 */
unsigned int UncertaintyDataCall::SecondaryStructureCount() const {
    return static_cast<unsigned int>(this->secStruct.Count());
}
