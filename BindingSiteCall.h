/*
 * BindingSiteCall.h
 *
 * Author: Michael Krone
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#ifndef MEGAMOL_PROTEIN_BSITECALL_H_INCLUDED
#define MEGAMOL_PROTEIN_BSITECALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Call.h"
#include "CallAutoDescription.h"
#include "vislib/Array.h"
#include "vislib/String.h"

namespace megamol {
namespace protein {

    /**
     * Class for binding site calls and data interfaces.
     */

    class BindingSiteCall : public megamol::core::Call {
    public:
        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "BindingSiteCall";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get binding sites.";
        }

        /** Index of the 'GetData' function */
        static const unsigned int CallForGetData;

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return 1;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char* FunctionName(unsigned int idx) {
            switch (idx) {
            case 0:
                return "dataOut";
            }
            return "";
        }
        
        /**
         * Get the number of binding sites.
         *
         * @return The binding site count.
         */
        inline unsigned int GetBindingSiteCount(void) const {
            if( !this->bindingSites )
                return 0;
            else
                return this->bindingSites->Count();
        }
        
        /**
         * Get the residue indices of a binding site.
         *
         * @param i The index of the residue.
         * @return Pointer to the array of residue indices.
         */
        inline vislib::Array<unsigned int> *GetBindingSite( unsigned int i) const {
            if( !this->bindingSites )
                return 0;
            else if( this->bindingSites->Count() < i )
                return 0;
            else
                //return &(*this->bindingSites)[i];
                return &(this->bindingSites->operator[](i));
        }
        
        /**
         * Get the residue names of a binding site.
         *
         * @param i The index of the residue.
         * @return Pointer to the array of residue names.
         */
        inline vislib::Array<vislib::StringA> *GetBindingSiteResNames( unsigned int i) const {
            if( !this->bindingSiteResNames )
                return 0;
            else if( this->bindingSiteResNames->Count() < i )
                return 0;
            else
                //return &(*this->bindingSites)[i];
                return &(this->bindingSiteResNames->operator[](i));
        }

        BindingSiteCall(void);
        virtual ~BindingSiteCall(void);

    private:
        /** Pointer to binding site array */
        vislib::Array<vislib::Array<unsigned int> > *bindingSites;
        /** Pointer to binding site residue name array */
        vislib::Array<vislib::Array<vislib::StringA> > *bindingSiteResNames;
    };

    /** Description class typedef */
    typedef megamol::core::CallAutoDescription<BindingSiteCall> BindingSiteCallDescription;

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MEGAMOL_PROTEIN_BSITECALL_H_INCLUDED */
