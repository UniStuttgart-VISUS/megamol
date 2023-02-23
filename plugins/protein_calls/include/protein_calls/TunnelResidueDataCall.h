/*
 * TunnelResidueDataCall.h
 * Copyright (C) 2006-2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmstd/data/AbstractGetData3DCall.h"
#include <vector>

namespace megamol::protein_calls {

class TunnelResidueDataCall : public core::AbstractGetData3DCall {
public:
    /** Struct representing a single tunnel */
    struct Tunnel {
    public:
        /** Ctor. */
        Tunnel() : tunnelLength(0.0f) {
            // intentionally empty
        }
        /** array storing the voronoi vertex locations of the tunnel represented */
        std::vector<float> coordinates;

        /** the numbers of stored atoms per voronoi vertex */
        std::vector<int> atomNumbers;

        /** the indices of the first atoms belonging to each voronoi vertex */
        std::vector<int> firstAtomIndices;

        /** the identifiers for all atoms stored in the call. first: identifier, second: number of occurences */
        std::vector<std::pair<int, int>> atomIdentifiers;

        /** The length of the tunnel */
        float tunnelLength;

        /** The bottleneck radius of the tunnel */
        float bottleneckRadius;
    };

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName() {
        return "TunnelResidueDataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call transporting tunnel data alongside with redidue indices beside it";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
        return AbstractGetData3DCall::FunctionCount();
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        return AbstractGetData3DCall::FunctionName(idx);
    }

    /** Ctor. */
    TunnelResidueDataCall();

    /** Dtor. */
    ~TunnelResidueDataCall() override;

    /**
     * Returns the number of tunnels stored in this call
     *
     * @return The number of tunnels.
     */
    int getTunnelNumber() const {
        return this->numTunnels;
    }

    /**
     * Sets the number of tunnels
     *
     * @param numTunnels The new number of tunnels
     */
    void setTunnelNumber(const int numTunnels) {
        this->numTunnels = numTunnels;
    }

    /**
     * Returns the tunnel descriptions
     *
     * @param return The tunnel descriptions
     */
    const Tunnel* getTunnelDescriptions() const {
        return this->tunnels;
    }

    /**
     * Sets the new tunnel description pointer
     *
     * @param tunnels The pointer to the tunnel descriptions
     */
    void setTunnelDescriptions(Tunnel* tunnels) {
        this->tunnels = tunnels;
    }

private:
    /** array containing all tunnels */
    Tunnel* tunnels;

    /** the number of tunnels stored in this call */
    int numTunnels;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<TunnelResidueDataCall> TunnelResidueDataCallDescription;

} // namespace megamol::protein_calls
