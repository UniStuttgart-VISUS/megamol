/*
 * TagVolume.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_TAGVOLUME_H_INCLUDED
#define MEGAMOLCORE_TAGVOLUME_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


namespace megamol {
namespace trisoup {
namespace volumetrics {

class TagVolume {
public:
    TagVolume(unsigned int xRes, unsigned int yRes, unsigned int zRes);
    ~TagVolume(void);
    void Tag(unsigned int x, unsigned int y, unsigned int z);
    bool IsTagged(unsigned int x, unsigned int y, unsigned int z);
    void UnTag(unsigned int x, unsigned int y, unsigned int z);
    void Reset();

private:
    unsigned int xRes;
    unsigned int yRes;
    unsigned int zRes;
    unsigned int volSize;
    unsigned char* volume;
};

} /* end namespace volumetrics */
} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_TAGVOLUME_H_INCLUDED */
