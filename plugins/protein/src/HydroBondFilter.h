/*
 * HydroBondFilter.h
 *
 * Copyright (C) 2016 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_HYDROBONDFILTER_H_INCLUDED
#define MMPROTEINPLUGIN_HYDROBONDFILTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "protein_calls/MolecularDataCall.h"

namespace megamol::protein {

class HydroBondFilter : public core::Module {
public:
    /** Ctor */
    HydroBondFilter();

    /** Dtor */
    ~HydroBondFilter() override;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "HydroBondFilter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Filters molecular hydrogen bonds due to certain criteria.";
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
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * Call callback to get the data
     *
     * @param c The calling call
     * @return True on success
     */
    bool getData(core::Call& call);

    /**
     * Call callback to get the extents
     *
     * @param c The calling call
     * @return True on success
     */
    bool getExtent(core::Call& call);

private:
    /**
     * Struct representing a hydrogen bond
     */
    struct HBond {

        /** Index of the donor atom */
        unsigned int donorIdx;
        /** Index of the acceptor atom */
        unsigned int acceptorIdx;
        /** Index of the hydrogen atom */
        unsigned int hydrogenIdx;
        /** The angle of the connection */
        float angle;

        /** Ctor. */
        HBond() : donorIdx(0), acceptorIdx(0), hydrogenIdx(0), angle(0.0f) {}

        /**
         * Constructor for a hydrogen bond.
         *
         * @param donorIdx The index of the donor atom.
         * @param acceptorIdx The index of the acceptor atom.
         * @param hydrogenIdx The index of the hydrogen atom.
         * @param angle The angle of the connection.
         */
        HBond(unsigned int donorIdx, unsigned int acceptorIdx, unsigned int hydrogenIdx, float angle = -1.0f)
                : donorIdx(donorIdx)
                , acceptorIdx(acceptorIdx)
                , hydrogenIdx(hydrogenIdx)
                , angle(angle) {}

        /**
         * Overload of the comparison operator
         *
         * @param rhs The right hand side of the comparison
         * @return True, when this is smaller than the right hand side, false otherwise.
         */
        bool operator<(const HBond& rhs) {
            if (this->donorIdx == rhs.donorIdx)
                return this->angle < rhs.angle;
            else
                return this->donorIdx < rhs.donorIdx;
        }

        /**
         * Prints a representation of this struct to the console
         */
        void print() {
            printf("H-Bond from %u to %u over %u with angle %f\n", donorIdx, acceptorIdx, hydrogenIdx, angle);
        }
    };

    /**
     * Post process the computed hydrogen bonds by deleting unneccessary ones.
     *
     * @param mdc The call storing all relevant molecular information.
     */
    void filterHBonds(protein_calls::MolecularDataCall& mdc);

    /**
     * Fills the secondary structure vector with information about the secondary structure of every atom.
     *
     * @param mdc The call storing all relevant molecular information.
     */
    void fillSecStructVector(protein_calls::MolecularDataCall& mdc);

    /**
     * Computes whether a given hydrogen bond is valid.
     *
     * @param donorIndex The index of the donor of the hydrogen bond.
     * @param accptorIndex The index of the acceptor of the hydrogen bond.
     * @param mdc The MolecularDataCall containing the data.
     */
    bool isValidHBond(unsigned int donorIndex, unsigned int acceptorIndex, protein_calls::MolecularDataCall& mdc);

    /** caller slot */
    core::CallerSlot inDataSlot;

    /** callee slot */
    core::CalleeSlot outDataSlot;

    /** Maximal distance between donor and acceptor of a hydrogen bond */
    core::param::ParamSlot hBondDonorAcceptorDistance;

    /** Should the H-Bonds of the alpha helices be computed? */
    core::param::ParamSlot alphaHelixHBonds;

    /** Should the H-Bonds of the beta sheets be computed? */
    core::param::ParamSlot betaSheetHBonds;

    /** Should the rest of the H-Bonds be computed */
    core::param::ParamSlot otherHBonds;

    /** Should the H-Bonds be faked as bonds between two c-alphas? */
    core::param::ParamSlot cAlphaHBonds;

    /** The last known data hash of the incoming data */
    SIZE_T lastDataHash;

    /** The offset from the last known data hash */
    SIZE_T dataHashOffset;

    /** Vector containing all hydrogen bonds surviving the filtering process */
    std::vector<unsigned int> hydrogenBondsFiltered;

    /** Number of H-bonds per atom */
    std::vector<unsigned int> hBondStatistics;

    /** The secondary structure ID per atom */
    std::vector<protein_calls::MolecularDataCall::SecStructure::ElementType> secStructPerAtom;

    /** The c alpha indices per atom */
    std::vector<unsigned int> cAlphaIndicesPerAtom;
};

} // namespace megamol::protein

#endif /* MMPROTEINPLUGIN_HYDROBONDGENERATOR_H_INCLUDED */
