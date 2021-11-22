//
// AtomWeights.h
//
// Copyright (C) 2016 by University of Stuttgart (VISUS).
// All rights reserved.
//
#ifndef MMPROTEINPLUGIN_ATOMWEIGHTS_H_INCLUDED
#define MMPROTEINPLUGIN_ATOMWEIGHTS_H_INCLUDED

#include "mmcore/utility/log/Log.h"

#include <algorithm>
#include <string>

/**
 * Returns the weight of the chemical element with the given proton count
 * This method currently only knows the weights of the elements that are
 * common in proteins (H, C, N, O, S).
 *
 * @param protonCount The number of protons in the asked element
 * @return The weight of the element with the given symbol.
 */
static float getElementWeightByProtonCount(unsigned int protonCount) {
    switch (protonCount) {
    case 1:
        return 1.0079f; // H
    case 6:
        return 12.011f; // C
    case 7:
        return 14.007f; // N
    case 8:
        return 15.999f; // O
    case 16:
        return 32.065f; // S
    default:
        return 0.0f;
    }

    // TODO extend to whole periodic table

    megamol::core::utility::log::Log::DefaultLog.WriteMsg(
        megamol::core::utility::log::Log::LEVEL_WARN, "Unknown element detected (%u)", protonCount);
    return 0.0f;
}

/**
 * Returns the weight of the chemical element with the given element symbol.
 * This method currently only knows the weights of the elements that are
 * common in proteins (H, C, N, O, S).
 *
 * @param elementString The element symbol as string.
 * @return The weight of the element with the given symbol.
 */
static float getElementWeightBySymbolString(const vislib::StringA& elementString) {
    vislib::StringA es = elementString;
    es.ToLowerCase();
    es.TrimSpaces();

    if (es.StartsWith("h"))
        return 1.0079f;
    else if (es.StartsWith("c"))
        return 12.011f;
    else if (es.StartsWith("n"))
        return 14.007f;
    else if (es.StartsWith("o"))
        return 15.999f;
    else if (es.StartsWith("s"))
        return 32.065f;

    // TODO extend to whole periodic table
    // TODO use switch with constexpr (only possible with VS 2015)

    megamol::core::utility::log::Log::DefaultLog.WriteMsg(
        megamol::core::utility::log::Log::LEVEL_WARN, "Unknown element detected (%c)", es.PeekBuffer());
    return 0.0f;
}

#endif // MMPROTEINPLUGIN_ATOMWEIGHTS_H_INCLUDED
