/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include <vector>

namespace megamol {
namespace protein_calls {

class RamachandranDataCall : public megamol::core::Call {
public:
    enum class PointState { NONE = 0, UNSURE_ALPHA = 1, SURE_ALPHA = 2, UNSURE_BETA = 3, SURE_BETA = 4 };

    enum class ProcheckState {
        PS_NONE = 0,
        PS_UNSURE_ALPHA = 1,
        PS_MIDDLE_ALPHA = 2,
        PS_SURE_ALPHA = 3,
        PS_UNSURE_BETA = 4,
        PS_MIDDLE_BETA = 5,
        PS_SURE_BETA = 6,
        PS_UNSURE_LEFT_HANDED = 7,
        PS_MIDDLE_LEFT_HANDED = 8,
        PS_SURE_LEFT_HANDED = 9,
        PS_UNSURE_E = 10,
        PS_MIDDLE_E = 11
    };

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "RamachandranDataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call to get ramachandran plot data.";
    }

    /** Index of the 'GetData' function */
    static const unsigned int CallForGetData;

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
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

    /** Ctor. */
    RamachandranDataCall(void);

    /** Dtor. */
    virtual ~RamachandranDataCall(void);

    /**
     * Returns the currently requested time of this call
     *
     * @return The current time.
     */
    inline virtual float Time(void) const {
        return this->time;
    }

    /**
     * Sets the time stored in this call
     *
     * @param time The new time value
     */
    inline virtual void SetTime(const float time) {
        this->time = time;
    }

    /**
     * Returns the vector containing the dihedral angles.
     * The length of this vector is two times the number of residues.
     * The phi-value is at idx = 2*residueIdx.
     * The psi-value is at idx = 2*residueIdx + 1.
     * An angle smaller than -500° is an invalid angle and can be ignored.
     *
     * @return The dihedral angle vector.
     */
    inline virtual const std::vector<float>* GetAngleVector(void) const {
        return this->angleVector;
    }

    /**
     * Sets the dihedral angle vector.
     *
     * @param angleVector The vector that is set.
     */
    inline virtual void SetAngleVector(std::vector<float>& angleVector) {
        this->angleVector = &angleVector;
    }

    /**
     * Returns the vector containing the Ramachandran point states of each amino acid.
     *
     * @return The point state vector.
     */
    inline virtual const std::vector<RamachandranDataCall::PointState>* GetPointStateVector(void) const {
        return this->pointStateVector;
    }

    /**
     * Sets the point state vector.
     *
     * @param pointStateVector The new point state vector.
     */
    inline virtual void SetPointStateVector(std::vector<RamachandranDataCall::PointState>& pointStateVector) {
        this->pointStateVector = &pointStateVector;
    }

    /**
     * Returns the probability vector that stores the assignment probabilites for each amino acid.
     *
     * @return The probability vector.
     */
    inline virtual const std::vector<float>* GetProbabilityVector(void) const {
        return this->probabilityVector;
    }

    /**
     * Sets the probability vector.
     *
     * @param probabilityVector The new probability vector.
     */
    inline virtual void SetProbabilityVector(std::vector<float>& probabilityVector) {
        this->probabilityVector = &probabilityVector;
    }

    /**
     * Returns the index of the currently selected amino acid.
     *
     * @return The index of the currently selected amino acid.
     */
    inline virtual int GetSelectedAminoAcid(void) const {
        return this->selectedAminoAcid;
    }

    /**
     * Sets the index of the currently selected amino acid.
     *
     * @param selectedAminoAcid The index of the selected amino acid.
     */
    inline virtual void SetSelectedAminoAcid(const int selectedAminoAcid) {
        this->selectedAminoAcid = selectedAminoAcid;
    }

private:
    /** The requested time. */
    float time;

    /** The vector storing the dihedral angles*/
    std::vector<float>* angleVector;

    /** The vector storing the possible assignments for each residue*/
    std::vector<RamachandranDataCall::PointState>* pointStateVector;

    /** The vector storing the probability of the assignment */
    std::vector<float>* probabilityVector;

    /** Index of the selected amino acid */
    int selectedAminoAcid;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<RamachandranDataCall> RamachandranDataCallDescription;
} // namespace protein_calls
} // namespace megamol
