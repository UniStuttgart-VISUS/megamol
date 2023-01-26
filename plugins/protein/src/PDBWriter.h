//
// PDBWriter.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
//  Created on: Apr 19, 2013
//      Author: scharnkn
//

#ifndef MMPROTEINPLUGIN_PDBWRITER_H_INCLUDED
#define MMPROTEINPLUGIN_PDBWRITER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/job/AbstractJob.h"
#include "mmstd/renderer/TimeControl.h"
#include "protein_calls/MolecularDataCall.h"

namespace megamol {
namespace protein {

class PDBWriter : public core::job::AbstractJob, public core::Module {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "PDBWriter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Job writing arbitrary trajectories to a series of PDB or PQR\
                files.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /**
     * Ctor
     */
    PDBWriter();

    /**
     * Dtor
     */
    ~PDBWriter() override;

    /**
     * Answers whether or not this job is still running.
     *
     * @return 'true' if this job is still running, 'false' if it has
     *         finished.
     */
    bool IsRunning(void) const override;

    /**
     * Starts the job thread.
     *
     * @return true if the job has been successfully started.
     */
    bool Start(void) override;

    /**
     * Terminates the job thread.
     *
     * @return true to acknowledge that the job will finish as soon
     *         as possible, false if termination is not possible.
     */
    bool Terminate(void) override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void) override;

    /**
     * Implementation of 'Release'.
     */
    void release(void) override;

private:
    /**
     * Callback function called when the trigger button is pressed
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool buttonCallback(core::param::ParamSlot& slot);

    /**
     * Creates the folders in the output path that do not yet exist.
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool createDirectories(vislib::StringA folder);

    /**
     * Write one frame to the respective *.pdb file.
     *
     * @param dc The data call
     * @return 'True' on success, 'false' otherwise
     */
    bool writePDB(megamol::protein_calls::MolecularDataCall* mol);

    /**
     * Write one frame to the respective *.pqr file.
     *
     * @param dc The data call
     * @return 'True' on success, 'false' otherwise
     */
    bool writePQR(megamol::protein_calls::MolecularDataCall* mol);

    core::CallerSlot dataCallerSlot;  ///> Data caller slot
    core::view::TimeControl timeCtrl; ///> The time control

    /// Parameter to trigger writing of *.pqr files instead of *.pdb
    core::param::ParamSlot writePQRSlot;

    /// Parameter to determine whether the solvent should be included
    core::param::ParamSlot includeSolventAtomsSlot;

    /// Parameter to determine whether all frames should be written into
    /// separate files
    core::param::ParamSlot writeSepFilesSlot;

    /// Parameter to determine the first frame to be written
    core::param::ParamSlot minFrameSlot;

    /// Parameter to determine the number of frames to be written
    core::param::ParamSlot nFramesSlot;

    /// Parameter to determine the stride used when writing frames
    core::param::ParamSlot strideSlot;

    /// Parameter for the filename prefix
    core::param::ParamSlot filenamePrefixSlot;

    /// Parameter for the output folder
    core::param::ParamSlot outDirSlot;

    /// Parameter for the trigger button
    core::param::ParamSlot triggerButtonSlot;

    /// Parameter for the b factor rescaling
    core::param::ParamSlot rescaleBFactorSlot;

    bool jobDone;       ///> Flag whether the job is done
    int filenameDigits; ///> Number of digits used in generated filenames

    ///  Flag whether the 'MODEL' record should be used, this allows handling
    ///  atom number > 99999
    bool useModelRecord;
};

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MMPROTEINPLUGIN_PDBWRITER_H_INCLUDED */
