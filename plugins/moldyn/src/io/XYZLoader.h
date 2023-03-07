/*
 * XYZLoader.h
 *
 * Copyright (C) 2015 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/math/Cuboid.h"
#include <vector>

namespace megamol::moldyn::io {

/**
 * Data loader of the simple xyz file format:
 * https://en.wikipedia.org/wiki/XYZ_file_format
 * http://openbabel.org/wiki/XYZ_%28format%29
 *
 * ASCII-Text file:
 *  1. line contains number of atoms
 *  2. line contains a comment (will be ignored by the loader)
 *  All following lines contain one atom each:
 *    A chemical element symbol (can be used to group)
 *    X coordinate
 *    Y coordinate
 *    Z coordinate
 */
class XYZLoader : public core::Module {
public:
    static const char* ClassName() {
        return "XYZDataSource";
    }
    static const char* Description() {
        return "Data source for the simple xyz file format";
    }
    static bool IsAvailable() {
        return true;
    }
    static float FileFormatAutoDetect(const unsigned char* data, SIZE_T dataSize);
    static const char* FilenameExtensions() {
        return ".XYZ";
    }
    static const char* FilenameSlotName() {
        return "filename";
    }
    static const char* FileTypeName() {
        return "XYZ";
    }

    XYZLoader();
    ~XYZLoader() override;

protected:
    bool create() override;
    void release() override;

private:
    bool getDataCallback(core::Call& caller);
    bool getExtentCallback(core::Call& caller);

    void clear();
    void assertData();

    core::CalleeSlot getDataSlot;

    core::param::ParamSlot filenameSlot;
    core::param::ParamSlot hasCountLineSlot;
    core::param::ParamSlot hasCommentLineSlot;
    core::param::ParamSlot hasElementSymbolSlot;
    core::param::ParamSlot groupByElementSlot;
    core::param::ParamSlot radiusSlot;

    size_t hash;
    vislib::math::Cuboid<float> bbox;
    std::vector<std::vector<float>> poss;
};

} // namespace megamol::moldyn::io
