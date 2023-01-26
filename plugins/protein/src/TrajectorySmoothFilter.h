//
// TrajectorySmoothFilter.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: May 8, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINPLUGIN_TRAJECTORYSMOOTHFILTER_H_INCLUDED
#define MMPROTEINPLUGIN_TRAJECTORYSMOOTHFILTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/renderer/Renderer3DModule.h"
#include "protein_calls/MolecularDataCall.h"

#include "HostArr.h"

typedef unsigned int uint;

namespace megamol {
namespace protein {

/// Filter module that computes a smoothed version of a given trajectory by
/// calculating the average.
/// Note: Does not take periodic boundary conditions into account, therefore,
/// particles that wrap around the box will be incorrect!
class TrajectorySmoothFilter : public core::Module {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TrajectorySmoothFilter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Calculates a smoothed version of a given trajectory.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    TrajectorySmoothFilter();

    /** Dtor. */
    ~TrajectorySmoothFilter() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'release'.
     */
    void release() override;

    /**
     * Call callback to get the data
     *
     * @param c The calling call
     *
     * @return True on success
     */
    bool getData(core::Call& call);

    /**
     * Call callback to get the extent of the data
     *
     * @param c The calling call
     *
     * @return True on success
     */
    bool getExtent(core::Call& call);

private:
    /**
     * Helper class to unlock frame data.
     */
    class Unlocker : public megamol::protein_calls::MolecularDataCall::Unlocker {
    public:
        /**
         * Ctor.
         *
         * @param mol The molecular data call whos 'Unlock'-method is to be
         *            called.
         */
        Unlocker(megamol::protein_calls::MolecularDataCall& mol)
                : megamol::protein_calls::MolecularDataCall::Unlocker()
                , mol(&mol) {
            // intentionally empty
        }

        /** Dtor. */
        ~Unlocker() override {
            this->Unlock();
        }

        /** Unlocks the data */
        void Unlock() override {
            this->mol->Unlock();
        }

    private:
        megamol::protein_calls::MolecularDataCall* mol;
    };

    /**
     * Update all parameters.
     *
     * @param mol   Pointer to the data call.
     */
    void updateParams(megamol::protein_calls::MolecularDataCall* mol);


    /// Caller slot to get unfiltered data
    core::CallerSlot molDataCallerSlot;

    /// Callee slot to provide filtered data
    core::CalleeSlot dataOutSlot;

    /// Parameter slot for averaging time window
    core::param::ParamSlot nAvgFramesSlot;
    uint nAvgFrames;

    /// Intermediate storage for smoothed atom positions
    HostArr<float> atomPosSmoothed;
};


} // end namespace protein
} // end namespace megamol

#endif // MMPROTEINPLUGIN_TRAJECTORYSMOOTHFILTER_H_INCLUDED
