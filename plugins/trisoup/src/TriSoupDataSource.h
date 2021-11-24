/*
 * TriSoupDataSource.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMTRISOUPPLG_TRISOUPDATASOURCE_H_INCLUDED
#define MMTRISOUPPLG_TRISOUPDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractTriMeshLoader.h"
#include "vislib/String.h"


namespace megamol {
namespace trisoup {


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
    static const char* ClassName(void) {
        return "TriSoupDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Data source for tri soup files";
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
    TriSoupDataSource(void);

    /** Dtor */
    virtual ~TriSoupDataSource(void);

protected:
    /**
     * Loads the specified file
     *
     * @param filename The file to load
     *
     * @return True on success
     */
    virtual bool load(const vislib::TString& filename);

private:
};

} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MMTRISOUPPLG_TRISOUPDATASOURCE_H_INCLUDED */
