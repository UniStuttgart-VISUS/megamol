/*
 * CoordSysMarker.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMTRISOUPPLG_COORDSYSMARKER_H_INCLUDED
#define MMTRISOUPPLG_COORDSYSMARKER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractTriMeshDataSource.h"
#include "vislib/String.h"


namespace megamol {
namespace trisoup_gl {


/**
 * Data source class for a simple coordinate system marker object
 */
class CoordSysMarker : public AbstractTriMeshDataSource {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "CoordSysMarker";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Data source class for a simple coordinate system marker object";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor */
    CoordSysMarker(void);

    /** Dtor */
    virtual ~CoordSysMarker(void);

protected:
    /** Ensures that the data is loaded */
    virtual void assertData(void);

private:
};

} // namespace trisoup_gl
} /* end namespace megamol */

#endif /* MMTRISOUPPLG_COORDSYSMARKER_H_INCLUDED */
