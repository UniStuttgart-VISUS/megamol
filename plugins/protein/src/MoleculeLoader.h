#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "protein_calls/MolecularDataCall.h"

#include "chemfiles.hpp"

namespace megamol::protein {
class MoleculeLoader : public core::Module {
public:
    /** Ctor. */
    MoleculeLoader(void);

    /** Dtor. */
    virtual ~MoleculeLoader(void);

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "MoleculeLoader";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Offers molecular data.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void) override;

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
    virtual void release(void) override;

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
     */
    void updateFiles();

    /**
     * Postprocesses the files targeting a molecular data call
     */
    void postProcessFilesMDC();

    /**
     * Postproces the files targeting a modern molecular call
     */
    void postPorcessFilesMolecule();

private:
    /** Slot connecting this module to the requesting modules */
    core::CalleeSlot data_out_slot_;

    /** Slot for the structure file path */
    core::param::ParamSlot filename_slot_;
    /** Slot for the trajectory file path */
    core::param::ParamSlot trajectory_filename_slot_;

    /** Pointer to the structure */
    std::shared_ptr<chemfiles::Trajectory> structure_;
    /** Pointer to the trajectory */
    std::shared_ptr<chemfiles::Trajectory> trajectory_;

    /** Path to the current structure */
    std::filesystem::path path_to_current_structure_;
    /** Path to the current trajectory */
    std::filesystem::path path_to_current_trajectory_;
};
} // namespace megamol::protein
