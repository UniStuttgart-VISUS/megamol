/*
 * BindingSiteCall.h
 *
 * Author: Michael Krone
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#ifndef MEGAMOL_PROTEIN_CALL_BSITECALL_H_INCLUDED
#define MEGAMOL_PROTEIN_CALL_BSITECALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/String.h"
#include "vislib/math/Vector.h"
#include <glm/glm.hpp>

namespace megamol {
namespace protein_calls {

/**
 * Class for binding site calls and data interfaces.
 */

class BindingSiteCall : public megamol::core::Call {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName() {
        return "BindingSiteCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call to get binding sites.";
    }

    /** Index of the 'GetData' function */
    static const unsigned int CallForGetData;

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
        return 1;
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "dataOut";
        }
        return "";
    }

    /**
     * Get the number of binding sites.
     *
     * @return The binding site count.
     */
    inline unsigned int GetBindingSiteCount() const {
        if (!this->bindingSites)
            return 0;
        else
            return static_cast<unsigned int>(this->bindingSites->size());
    }

    /**
     * Get the residue and chain indices of a binding site.
     *
     * @param i The index of the residue.
     * @return Pointer to the array of residue indices.
     */
    inline std::vector<std::pair<char, unsigned int>>* GetBindingSite(unsigned int i) const {
        if (!this->bindingSites)
            return 0;
        else if (this->bindingSites->size() <= i)
            return 0;
        else
            //return &(*this->bindingSites)[i];
            return &(this->bindingSites->operator[](i));
    }

    /**
     * Set the binding sites pointer (residue and chain indices).
     *
     * @param bsPtr The pointer.
     */
    inline void SetBindingSite(std::vector<std::vector<std::pair<char, unsigned int>>>* bsPtr) {
        this->bindingSites = bsPtr;
    }

    /**
     * Get the residue names of a binding site.
     *
     * @param i The index of the residue.
     * @return Pointer to the array of residue names.
     */
    inline std::vector<std::string>* GetBindingSiteResNames(unsigned int i) const {
        if (!this->bindingSiteResNames)
            return 0;
        else if (this->bindingSiteResNames->size() <= i)
            return 0;
        else
            //return &(*this->bindingSites)[i];
            return &(this->bindingSiteResNames->operator[](i));
    }

    /**
     * Set the pointer to the residue names of the binding site.
     *
     * @param rnPtr The pointer.
     */
    inline void SetBindingSiteResNames(std::vector<std::vector<std::string>>* rnPtr) {
        this->bindingSiteResNames = rnPtr;
    }

    /**
     * Get the name of a binding site.
     *
     * @param i The index of the residue.
     * @return Pointer to the array of residue names.
     */
    inline std::string GetBindingSiteName(unsigned int i) const {
        if (!this->bindingSiteNames)
            return "";
        else if (this->bindingSiteNames->size() <= i)
            return "";
        else
            return (this->bindingSiteNames->operator[](i));
    }

    /**
     * Set the pointer to the names of a binding sites.
     *
     * @param nPtr The pointer.
     */
    inline void SetBindingSiteNames(std::vector<std::string>* nPtr) {
        this->bindingSiteNames = nPtr;
    }

    /**
     * Get the description of a binding site.
     *
     * @param i The index of the residue.
     * @return Pointer to the array of binding site descriptions.
     */
    inline std::string GetBindingSiteDescription(unsigned int i) const {
        if (!this->bindingSiteDescriptions)
            return "";
        else if (this->bindingSiteDescriptions->size() <= i)
            return "";
        else
            return (this->bindingSiteDescriptions->operator[](i));
    }

    /**
     * Set the pointer to the descriptions of a binding sites.
     *
     * @param nPtr The pointer.
     */
    inline void SetBindingSiteDescriptions(std::vector<std::string>* nPtr) {
        this->bindingSiteDescriptions = nPtr;
    }

    /**
     * Get the color of a binding site.
     *
     * @param i The index of the residue.
     * @return Pointer to the array of binding site descriptions.
     */
    inline glm::vec3 GetBindingSiteColor(unsigned int i) const {
        if (!this->bindingSiteColors)
            return glm::vec3(0.5f, 0.5f, 0.5f);
        else if (this->bindingSiteColors->size() <= i)
            return glm::vec3(0.5f, 0.5f, 0.5f);
        else
            return (this->bindingSiteColors->operator[](i));
    }

    /**
     * Set the pointer to the colors of a binding sites.
     *
     * @param nPtr The pointer.
     */
    inline void SetBindingSiteColors(std::vector<glm::vec3>* nPtr) {
        this->bindingSiteColors = nPtr;
    }

    /**
     * Sets the enzyme mode flag
     */
    inline void SetEnzymeMode(bool mode) {
        this->enzymeCase = mode;
    }

    /**
     * Sets whether the binding site is of a gx type
     */
    inline void SetGXTypeFlag(bool isGX) {
        this->isGxType = isGX;
    }

    inline bool isEnzymeMode() const {
        return this->enzymeCase;
    }

    inline bool isOfGxType() const {
        return this->isGxType;
    }

    BindingSiteCall();
    ~BindingSiteCall() override;

private:
    /** Pointer to binding site array */
    std::vector<std::vector<std::pair<char, unsigned int>>>* bindingSites;
    /** Pointer to binding site residue name array */
    std::vector<std::vector<std::string>>* bindingSiteResNames;
    /** The binding site name */
    std::vector<std::string>* bindingSiteNames;
    /** The binding site name */
    std::vector<std::string>* bindingSiteDescriptions;
    // color table
    std::vector<glm::vec3>* bindingSiteColors;
    /** are we in the special enzyme case? */
    bool enzymeCase;
    /** is the current enzyme type the gx type? */
    bool isGxType;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<BindingSiteCall> BindingSiteCallDescription;

} /* end namespace protein_calls */
} /* end namespace megamol */

#endif /* MEGAMOL_PROTEIN_CALL_BSITECALL_H_INCLUDED */
