/*
 * CoordSysMarker.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMTRISOUPPLG_COORDSYSMARKER_H_INCLUDED
#define MMTRISOUPPLG_COORDSYSMARKER_H_INCLUDED
#pragma once

#include "AbstractTriMeshDataSource.h"
#include "vislib/String.h"


namespace megamol::trisoup_gl {


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
    static const char* ClassName() {
        return "CoordSysMarker";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Data source class for a simple coordinate system marker object";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    CoordSysMarker();

    /** Dtor */
    ~CoordSysMarker() override;

protected:
    /** Ensures that the data is loaded */
    void assertData() override;

private:
};

} // namespace megamol::trisoup_gl

#endif /* MMTRISOUPPLG_COORDSYSMARKER_H_INCLUDED */
