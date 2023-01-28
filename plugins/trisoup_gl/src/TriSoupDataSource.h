/*
 * TriSoupDataSource.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMTRISOUPPLG_TRISOUPDATASOURCE_H_INCLUDED
#define MMTRISOUPPLG_TRISOUPDATASOURCE_H_INCLUDED
#pragma once

#include "AbstractTriMeshLoader.h"
#include "vislib/String.h"


namespace megamol::trisoup_gl {


/**
 * Data source class for tri soup files
 */
class TriSoupDataSource : public AbstractTriMeshLoader {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TriSoupDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Data source for tri soup files";
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
    TriSoupDataSource();

    /** Dtor */
    ~TriSoupDataSource() override;

protected:
    /**
     * Loads the specified file
     *
     * @param filename The file to load
     *
     * @return True on success
     */
    bool load(const vislib::TString& filename) override;

private:
};

} // namespace megamol::trisoup_gl

#endif /* MMTRISOUPPLG_TRISOUPDATASOURCE_H_INCLUDED */
