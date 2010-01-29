/*
 * SolPathDataSource.h
 *
 * Copyright (C) 2010 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_PROTEIN_SOLPATHDATASOURCE_H_INCLUDED
#define MEGAMOL_PROTEIN_SOLPATHDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Module.h"
#include "param/ParamSlot.h"
#include "CalleeSlot.h"


namespace megamol {
namespace protein {

    /**
     * Data source for solvent path data
     */
    class SolPathDataSource : public megamol::core::Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "SolPathDataSource";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Data source for solvent path data.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** ctor */
        SolPathDataSource(void);

        /** dtor */
        virtual ~SolPathDataSource(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    private:

    };

} /* end namespace protein */
} /* end namespace megamol */

#endif /*  MEGAMOL_PROTEIN_SOLPATHDATASOURCE_H_INCLUDED */
