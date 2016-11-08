/*
 * UncertaintyDataCall.h
 *
 * Copyright (C) 2016 by University of Stuttgart (VISUS).
 * All rights reserved.
  *
 * This module is based on the source code of "SequenceRenderere" in protein_calls plugin (svn revision 17).
 *
 */

#ifndef PROTEIN_UNCERTAINTY_UNCERTAINTYDATACALL_H_INCLUDED
#define PROTEIN_UNCERTAINTY_UNCERTAINTYDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <vector>

#include "mmcore/Call.h"
#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"

#include "vislib/IllegalParamException.h"
#include "vislib/math/Vector.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include "vislib/macro_utils.h"

#include "protein_calls/Protein_Calls.h"


namespace megamol {
namespace protein_uncertainty {

    /**
     * Base class of rendering graph calls and of data interfaces for 
     * molecular data (e.g. protein-solvent-systems).
     *
     * Note that all data has to be sorted!
     * There are no IDs anymore, always use the index values into the right 
     * tables.
     *
     * All data has to be stored in the corresponding data source object. This
     * interface object will only pass pointers to the renderer objects. A
     * compatible data source should therefore use the public nested class of
     */

	class PROTEIN_CALLS_API UncertaintyDataCall : public megamol::core::AbstractGetDataCall {
		
    public:

        /**
         * Nested class holding all information about one segment of a proteins
         * secondary structure.
         */
		class PROTEIN_CALLS_API SecStructure {
        public:

            /** possible types of secondary structure elements */
            enum ElementType {
                TYPE_COIL  = 0,
                TYPE_SHEET = 1,
                TYPE_HELIX = 2,
                TYPE_TURN  = 3
            };

            /**
             * Default ctor initialising all elements to zero (or equivalent
             * values).
             */
            SecStructure(void);

            /**
             * Copy ctor performing a deep copy.
             *
             * @param src The object to clone from.
             */
            SecStructure(const SecStructure& src);

            /** Dtor. */
            ~SecStructure(void);

            /**
             * Returns the size of the element in (partial) amino acids.
             *
             * @return The size of the element in (partial) amino acids.
             */
            inline unsigned int AminoAcidCount(void) const {
                return this->aminoAcidCnt;
            }

            /**
             * Returns the index of the amino acid in which this element
             * starts.
             *
             * @return The index of the amino acid in which this element
             *         starts.
             */
            inline unsigned int FirstAminoAcidIndex(void) const {
                return this->firstAminoAcidIdx;
            }

            /**
             * Sets the position of the secondary structure element by the
             * indices of the amino acid where this element starts and the 
             * size of the element in (partial) amino acids.
             *
             * @param firstAminoAcidIdx The index of the amino acid where this
             *                          element starts.
             * @param aminoAcidCnt The size of the element in (partial) amino
             *                     acids.
             */
            void SetPosition( unsigned int firstAminoAcidIdx, 
                unsigned int aminoAcidCnt);

            /**
             * Sets the type of the element.
             *
             * @param type The new type for this element.
             */
            void SetType(ElementType type);

            /**
             * Returns the type of this element.
             *
             * @return The type of this element.
             */
            inline ElementType Type(void) const {
                return this->type;
            }

            /**
             * Assignment operator.
             *
             * @param rhs The right hand side to clone from.
             *
             * @return The reference to 'this'.
             */
            SecStructure& operator=(const SecStructure& rhs);

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'true' if 'this' and 'rhs' are equal, 'false' if not.
             */
            bool operator==(const SecStructure& rhs) const;

            /**
             * Test for inequality.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'true' if 'this' and 'rhs' are not equal, 'false' if
             *         they are equal.
             */
            inline bool operator!=(const SecStructure& rhs) const {
                return !(*this == rhs);
            }

        private:

            /** The size of the element in (partial) amino acids */
            unsigned int aminoAcidCnt;

            /** The index of the amino acid in which this element starts */
            unsigned int firstAminoAcidIdx;

            /** The type of this element */
            ElementType type;

        };


        /** Index of the 'GetData' function */
        static const unsigned int CallForGetData;

        /** Index of the 'GetExtent' function */
        static const unsigned int CallForGetExtent;

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "UncertaintyDataCall";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get uncertainty data for secondary structure";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return 2;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            switch( idx) {
                case 0:
                    return "GetData";
                case 1:
                    return "GetExtend";
            }
            return "";
        }

        /** Ctor. */
        UncertaintyDataCall(void);

        /** Dtor. */
        virtual ~UncertaintyDataCall(void);

        // -------------------- get and set routines --------------------

               /**
         * Set the number of secondary structure elements.
         *
         * @param secStructCnt The secondary structure element count.
         */
        void SetSecondaryStructureCount( unsigned int cnt);

        /**
         * Set a secondary stucture element to the array.
         *
         * @param idx   The index of the element.
         * @param secS  The secondary structure element.
         *
         * @return 'true' if successful, 'false' otherwise.
         */
        bool SetSecondaryStructure( unsigned int idx, SecStructure secS);

        /**
         * Get the secondary structure.
         *
         * @return The secondary structure array.
         */
        const SecStructure* SecondaryStructures() const;

        /**
         * Get the number of secondary structure elements.
         *
         * @return The secondary structure element count.
         */
        unsigned int SecondaryStructureCount() const;


        inline UncertaintyDataCall& operator=(const UncertaintyDataCall& s) {
            AbstractGetDataCall::operator=(s);
            this->secStruct = s.secStruct; // TODO: besser zeiger und anzahl ?!
            return *this;
        }

    private:
        // -------------------- variables --------------------

        /** The array of secondary structures */
        VISLIB_MSVC_SUPPRESS_WARNING(4251)
        vislib::Array<SecStructure> secStruct;

    };

    /** Description class typedef */
    typedef megamol::core::factories::CallAutoDescription<UncertaintyDataCall> UncertaintyDataCallDescription;
    


} /* end namespace protein_uncertainty */
} /* end namespace megamol */

#endif /* PROTEIN_UNCERTAINTY_UNCERTAINTYDATACALL_H_INCLUDED */
