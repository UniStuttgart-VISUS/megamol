/*
 * UncertaintyDataCall.cpp
 *
 * Author: Matthias Braun
 * Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 *
 * This module is based on the source code of "UncertaintyDataCall" in megamol protein_calls plugin (svn revision 17).
 *
 */


#include "stdafx.h"
#include "UncertaintyDataCall.h"


using namespace megamol;
using namespace megamol::protein_uncertainty;


/*
* UncertaintyDataCall::CallForGetData
*/
const unsigned int UncertaintyDataCall::CallForGetData = 0;


/*
* UncertaintyDataCall::UncertaintyDataCall
*/
UncertaintyDataCall::UncertaintyDataCall(void) : megamol::core::Call(),
                                                 dsspSecStructure(NULL), strideSecStructure(NULL), pdbSecStructure(NULL),
                                                 indexAminoAcidchainID(NULL), secStructUncertainty(NULL){
}


/*
* UncertaintyDataCall::~UncertaintyDataCall
*/
UncertaintyDataCall::~UncertaintyDataCall(void) {
    this->dsspSecStructure = NULL;
    this->strideSecStructure = NULL;
    this->pdbSecStructure = NULL,
    this->indexAminoAcidchainID = NULL;
    this->secStructUncertainty = NULL;
}


/*
* UncertaintyDataCall::GetMostLikelySecStructure
*/
vislib::Pair<UncertaintyDataCall::secStructure, float> UncertaintyDataCall::GetMostLikelySecStructure(unsigned int i) {
    // temp varaibles
    vislib::math::Vector<float, 4> secStruct = this->secStructUncertainty->operator[](i);
    vislib::Pair<secStructure, float> maxUncertainty(secStructure::NOTDEFINED, 0.0f);

    if (!this->secStructUncertainty)
        return maxUncertainty;
    else if (this->secStructUncertainty->Count() <= i)
        return maxUncertainty;
    else {
        maxUncertainty.Second() = secStruct[secStructure::NOTDEFINED];
        for (int x = 0; x < static_cast<int>(secStructure::NoE); x++) {
            if (maxUncertainty.Second() < secStruct[x]) {
                maxUncertainty.First() = static_cast<secStructure>(x);
                maxUncertainty.Second() = secStruct[x];
            }
        }   
        return maxUncertainty;
    }
}


/*
* UncertaintyDataCall::GetSortedSecStructureUncertainties
*/
vislib::Pair<vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::secStructure::NoE)>,
vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NoE)> > 
    UncertaintyDataCall::GetSortedSecStructureUncertainties(unsigned int i){

    // temp variables
    vislib::math::Vector<secStructure, static_cast<int>(secStructure::NoE)> sortedStructures;
    vislib::math::Vector<float, static_cast<int>(secStructure::NoE)>        sortedUncertainties;

    // initialising tmp variables
    for (int x = 0; x < static_cast<int>(secStructure::NoE); x++) {
        sortedStructures[x] = static_cast<secStructure>(x);
        sortedUncertainties[i] = this->secStructUncertainty->operator[](i)[x];
    }

    // using quicksort for sorting ...
    this->quickSortUncertainties(&sortedStructures, &sortedUncertainties, 0, static_cast<int>(secStructure::NoE));


    return vislib::Pair<vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::secStructure::NoE)>,
                        vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NoE)> >(sortedStructures, sortedUncertainties);
}


/*
* UncertaintyDataCall::quickSortUncertainties
*/
void UncertaintyDataCall::quickSortUncertainties(vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::secStructure::NoE)> *structArr,
                                                 vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NoE)> *uncerArr, int left, int right) {

    int i = left;
    int j = right;
    float tmpUncer;
    UncertaintyDataCall::secStructure tmpStruct;

    float pivot = uncerArr->operator[]((unsigned int)(left + right) / 2);

    // partition 
    while (i <= j) {
        while (uncerArr->operator[](i) < pivot)
            i++;
        while (uncerArr->operator[](j) > pivot)
            j--;
        if (i <= j) {
            // swap elements for structure and uncertainty
            tmpUncer = uncerArr->operator[](i);
            uncerArr->operator[](i) = uncerArr->operator[](j);
            uncerArr->operator[](j) = tmpUncer;

            tmpStruct = structArr->operator[](i);
            structArr->operator[](i) = structArr->operator[](j);
            structArr->operator[](j) = tmpStruct;

            i++;
            j--;
        }
    }

    // recursion
    if (left < j)
        this->quickSortUncertainties(structArr, uncerArr, left, j);
    if (i < right)
        this->quickSortUncertainties(structArr, uncerArr, i, right);

}
