/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/String.h"
#include "vislib/math/Vector.h"

#include "protein_calls/UncertaintyDataCall.h"

#include <filesystem>

namespace megamol {
namespace protein {

class UncertaintyDataLoader : public megamol::core::Module {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "UncertaintyDataLoader";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Offers protein secondary structure uncertainty data.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** ctor */
    UncertaintyDataLoader();

    /** dtor */
    ~UncertaintyDataLoader() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * Call callback to get the data
     *
     * @param call The calling call
     *
     * @return True on success
     */
    bool getData(megamol::core::Call& call);

private:
    /**
     * Read the input file containing secondary structure data collected by the the python script.
     *
     * @param filename The filename of the uncertainty input data file.
     *
     *@return True on success
     */
    bool ReadInputFile(const std::filesystem::path& filename);


    /**
     * Quick sorting of the corresponding types of the secondary structure uncertainties.
     *
     * @param valueArr  The pointer to the vector keeping the uncertainty values
     * @param structArr The pointer to the vector keeping the structural indices
     * @param left      The left index of the array
     * @param right     The right index of the array
     */
    void QuickSortUncertainties(
        vislib::math::Vector<float, static_cast<int>(protein_calls::UncertaintyDataCall::secStructure::NOE)>* valueArr,
        vislib::math::Vector<protein_calls::UncertaintyDataCall::secStructure,
            static_cast<int>(protein_calls::UncertaintyDataCall::secStructure::NOE)>* structArr,
        int left, int right);

    /**
     * enumeration of available uncertainty calculation methods.
     */
    enum calculationMethod { AVERAGE = 0, EXTENDED = 1 };

    /**
     * Compute uncertainty on current secondary structure data with method AVERAGE.
     *
     *@return True on success
     */
    bool CalculateUncertaintyAverage();

    /**
     * Compute uncertainty on current secondary structure data with method EXTENDED.
     *
     *@return True on success
     */
    bool CalculateUncertaintyExtended();


    /**
     * Calculate length of continuous secondary structure assignment.
     *
     *@return True on success
     */
    bool CalculateStructureLength();


    // ------------------ variables -------------------

    /** The data callee slot */
    core::CalleeSlot dataOutSlot;

    /** The parameter slot for the uid filename */
    core::param::ParamSlot filenameSlot;

    /** The parameter slot for choosing uncertainty calculation method */
    core::param::ParamSlot methodSlot;

    /** The currently used uncertainty calculation method */
    calculationMethod currentMethod;


    /** The PDB index */
    vislib::Array<vislib::StringA> pdbIndex;

    /** The chain ID */
    vislib::Array<char> chainID;

    /** The flag giving additional information */
    vislib::Array<protein_calls::UncertaintyDataCall::addFlags> residueFlag;

    /** The amino-acid name */
    vislib::Array<vislib::StringA> aminoAcidName;

    /** The uncertainty of the assigned secondary structure types for each assignment method and for each amino-acid
     */
    vislib::Array<vislib::Array<
        vislib::math::Vector<float, static_cast<int>(protein_calls::UncertaintyDataCall::secStructure::NOE)>>>
        secStructUncertainty;
    /** The sorted assigned secondary structure types (sorted by descending uncertainty values) for each assignment
     * method and for each amino-acid */
    vislib::Array<vislib::Array<vislib::math::Vector<protein_calls::UncertaintyDataCall::secStructure,
        static_cast<int>(protein_calls::UncertaintyDataCall::secStructure::NOE)>>>
        sortedSecStructAssignment;
    /** The secondary structure assignment length for each assignment method */
    vislib::Array<vislib::Array<unsigned int>> secStructLength;
    /** The uncertainty of secondary structure assignment for each amino-acid */
    vislib::Array<float> uncertainty;

    /** The 5 STRIDE threshold values per amino-acid */
    vislib::Array<vislib::math::Vector<float, 7>> strideStructThreshold;
    /** The 4 DSSP energy values per amino-acid */
    vislib::Array<vislib::math::Vector<float, 4>> dsspStructEnergy;
    /** The 6 PROSIGN threshold values per amino-acid */
    vislib::Array<vislib::math::Vector<float, 6>> prosignStructThreshold;

    /** The pdb id */
    vislib::StringA pdbID;

    /** The pdb assignment method for helix */
    protein_calls::UncertaintyDataCall::pdbAssMethod pdbAssignmentHelix;

    /** The pdb assignment method for helix */
    protein_calls::UncertaintyDataCall::pdbAssMethod pdbAssignmentSheet;
};

} // namespace protein
} /* end namespace megamol */
