/*
 * XYZLoader.h
 *
 * Copyright (C) 2015 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_STDMOLDYN_XYZLOADER_H_INCLUDED
#define MEGAMOL_STDMOLDYN_XYZLOADER_H_INCLUDED
#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/math/Cuboid.h"
#include <vector>

namespace megamol {
namespace moldyn {
namespace io {

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
    static const char* ClassName(void) {
        return "XYZDataSource";
    }
    static const char* Description(void) {
        return "Data source for the simple xyz file format";
    }
    static bool IsAvailable(void) {
        return true;
    }
    static float FileFormatAutoDetect(const unsigned char* data, SIZE_T dataSize);
    static const char* FilenameExtensions() {
        return ".XYZ";
    }
    static const char* FilenameSlotName(void) {
        return "filename";
    }
    static const char* FileTypeName(void) {
        return "XYZ";
    }

    XYZLoader();
    virtual ~XYZLoader();

protected:
    virtual bool create(void);
    virtual void release(void);

private:
    bool getDataCallback(core::Call& caller);
    bool getExtentCallback(core::Call& caller);

    void clear(void);
    void assertData(void);

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

} /* end namespace io */
} /* end namespace moldyn */
} /* end namespace megamol */

#endif /* MEGAMOL_STDMOLDYN_XYZLOADER_H_INCLUDED */
