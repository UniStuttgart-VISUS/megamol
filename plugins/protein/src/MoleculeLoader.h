/**
 * MegaMol
 * Copyright (c) 2010, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

#include <chemfiles.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "protein_calls/MolecularDataCall.h"

namespace megamol::protein {
class MoleculeLoader : public core::Module {
public:
    /** Ctor. */
    MoleculeLoader();

    /** Dtor. */
    ~MoleculeLoader() override;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "MoleculeLoader";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Offers molecular data.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Call callback to get the molecular data call data
     *
     * @param c The calling call
     *
     * @return True on success
     */
    bool getMDCData(core::Call& call);

    /**
     * Call callback to get the extent of the molecular data call data
     *
     * @param c The calling call
     *
     * @return True on success
     */
    bool getMDCExtent(core::Call& call);

    /**
     * Call callback to get the modern call data
     *
     * @param c The calling call
     *
     * @return True on success
     */
    bool getMoleculeData(core::Call& call);

    /**
     * Call callback to get the extent of the modern call data
     *
     * @param c The calling call
     *
     * @return True on success
     */
    bool getMoleculeExtent(core::Call& call);

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * Loads the structure file into memory
     *
     * @param path_to_structure Path to the file containing the structure information
     * @return True on success, false otherwise
     */
    bool loadFile(std::filesystem::path const& path_to_structure);

    /**
     *  Loads the files into memory
     *
     *  @param path_to_structure Path to the file containing the structure information
     *  @param path_to_trajectory Path to the file containing the trajectory information
     *  @return True on success, false otherwise
     */
    bool loadFile(std::filesystem::path const& path_to_structure, std::filesystem::path const& path_to_trajectory);

    /**
     * Updates all file contents if necessary
     *
     * @return True if anything was updated, false otherwise
     */
    bool updateFiles();

    /**
     * Postprocesses the files targeting a molecular data call
     *
     * @param mdc The molecular data call to write stuff into
     */
    void postProcessFilesMDC(protein_calls::MolecularDataCall& mdc, bool recalc = false);

    /**
     * Postprocesses the files targeting a modern molecular call
     */
    void postProcessFilesMolecule();

private:
    /** Slot connecting this module to the requesting modules */
    core::CalleeSlot data_out_slot_;

    /** Slot for the structure file path */
    core::param::ParamSlot filename_slot_;
    /** Slot for the trajectory file path */
    core::param::ParamSlot trajectory_filename_slot_;

    /** Slot for enabling the computation of STRIDE */
    core::param::ParamSlot calc_secstruct_slot_;

    /** Slot for the selection of the radius measure */
    core::param::ParamSlot used_radius_slot_;

    /** Pointer to the structure */
    std::shared_ptr<chemfiles::Trajectory> structure_;

    /** Path to the current structure */
    std::filesystem::path path_to_current_structure_;
    /** Path to the current trajectory */
    std::filesystem::path path_to_current_trajectory_;

    /** Number of time steps in the currently saved trajectory */
    int64_t time_step_count_;

    /** The bounding box for all frames */
    vislib::math::Cuboid<float> global_bounding_box_;

    /** The bounding box for the current frame */
    vislib::math::Cuboid<float> local_bounding_box_;

    struct MDCStructuresFrame {
        /** Positions of the atoms */
        std::vector<glm::vec3> atom_positions_;
        /** B-Factor values for each atom */
        std::vector<float> b_factors_;
        /** Charges for each atom */
        std::vector<float> charges_;
        /** Occupancy values for each atom */
        std::vector<float> occupancies_;
        /** Min and max value of the B-Factor */
        std::pair<float, float> b_factor_bounds_;
        /** Min and max value of the charge */
        std::pair<float, float> charge_bounds_;
        /** Min and max value of the occupancy */
        std::pair<float, float> occupancy_bounds_;
    } mdc_structures_frame_;

    struct MDCStructures {
        /** Type indices per atom */
        std::vector<uint32_t> atom_type_indices_;
        /** Atom indices from the original pdb file */
        std::vector<int32_t> former_atom_indices_;
        /** Residue indices for each atom */
        std::vector<int32_t> atom_residue_indices_;
        /** All different atom types */
        std::vector<protein_calls::MolecularDataCall::AtomType> atom_types_;
        /** All different residues */
        std::vector<const protein_calls::MolecularDataCall::Residue*> residues_;
        /** Names for all residues */
        std::vector<vislib::StringA> residue_type_names_;
        /** indices for the solvent residues */
        std::vector<uint32_t> solvent_residue_index_;
        /** All molecule structures for the protein */
        std::vector<protein_calls::MolecularDataCall::Molecule> molecules_;
        /** All chains */
        std::vector<protein_calls::MolecularDataCall::Chain> chains_;
        /** Vector storing all atom-atom bonds */
        std::vector<glm::uvec2> connectivity_;
        /** Vector storing the visibility information for each atom */
        std::vector<int32_t> visibility_per_atom_;
        // TODO secondary structure
    } mdc_structures_;

    struct MoleculeStructuresFrame {
        // TODO
    } molecule_structures_frame_;

    struct MoleculeStructures {
        // TODO
    } molecule_structures_;

    /** The current data hash */
    size_t datahash_ = 0;
};
} // namespace megamol::protein
