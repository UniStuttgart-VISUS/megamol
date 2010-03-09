/*
 * MultiParticleDataCall.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MULTIPARTICLEDATACALL_H_INCLUDED
#define MEGAMOLCORE_MULTIPARTICLEDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractGetData3DCall.h"
#include "CallAutoDescription.h"
#include "vislib/assert.h"
#include "vislib/Array.h"


namespace megamol {
namespace core {
namespace moldyn {


    /**
     * Call for multi-stream particle data.
     */
    class MEGAMOLCORE_API MultiParticleDataCall : public AbstractGetData3DCall {
    public:

        /**
         * Class holding all data of a single particle type
         *
         * TODO: This class currenty can only hold data for spheres and should
         *       be extended to be able to handle data for arbitrary glyphs.
         *       This also applies to interpolation of data.
         */
        class MEGAMOLCORE_API Particles {
        public:

            /** possible values for the vertex data */
            enum VertexDataType {
                VERTDATA_NONE, //< indicates that this object is void
                VERTDATA_FLOAT_XYZ, //< use global radius
                VERTDATA_FLOAT_XYZR
            };

            /** possible values for the colour data */
            enum ColourDataType {
                COLDATA_NONE, //< use global colour
                COLDATA_UINT8_RGB,
                COLDATA_UINT8_RGBA,
                COLDATA_FLOAT_RGB,
                COLDATA_FLOAT_RGBA,
                COLDATA_FLOAT_I //< single float value to be mapped by a transfer function
            };

            /**
             * Ctor
             */
            Particles(void);

            /**
             * Copy ctor
             *
             * @param src The object to clone from
             */
            Particles(const Particles& src);

            /**
             * Dtor
             */
            ~Particles(void);

            /**
             * Answer the colour data type
             *
             * @return The colour data type
             */
            inline ColourDataType GetColourDataType(void) const {
                return this->colDataType;
            }

            /**
             * Answer the colour data pointer
             *
             * @return The colour data pointer
             */
            inline const void * GetColourData(void) const {
                return this->colPtr;
            }

            /**
             * Answer the colour data stride
             *
             * @return The colour data stride
             */
            inline unsigned int GetColourDataStride(void) const {
                return this->colStride;
            }

            /**
             * Answer the number of stored objects
             *
             * @return The number of stored objects
             */
            inline UINT64 GetCount(void) const {
                return this->count;
            }

            /**
             * Answer the global colour
             *
             * @return The global colour as a pointer to four unsigned bytes
             *         storing the RGBA colour components
             */
            inline const unsigned char * GetGlobalColour(void) const {
                return this->col;
            }

            /**
             * Answer the global radius
             *
             * @return The global radius
             */
            inline float GetGlobalRadius(void) const {
                return this->radius;
            }

            /**
             * Answer the maximum colour index value to be mapped
             *
             * @return The maximum colour index value to be mapped
             */
            inline float GetMaxColourIndexValue(void) const {
                return this->maxColI;
            }

            /**
             * Answer the minimum colour index value to be mapped
             *
             * @return The minimum colour index value to be mapped
             */
            inline float GetMinColourIndexValue(void) const {
                return this->minColI;
            }

            /**
             * Answer the vertex data type
             *
             * @return The vertex data type
             */
            inline VertexDataType GetVertexDataType(void) const {
                return this->vertDataType;
            }

            /**
             * Answer the vertex data pointer
             *
             * @return The vertex data pointer
             */
            inline const void * GetVertexData(void) const {
                return this->vertPtr;
            }

            /**
             * Answer the vertex data stride
             *
             * @return The vertex data stride
             */
            inline unsigned int GetVertexDataStride(void) const {
                return this->vertStride;
            }

            /**
             * Sets the colour data
             *
             * @param t The type of the colour data
             * @param p The pointer to the colour data (must not be NULL if t
             *          is not 'COLDATA_NONE'
             * @param s The stride of the colour data
             */
            void SetColourData(ColourDataType t, const void *p,
                    unsigned int s = 0) {
                ASSERT((p != NULL) || (t == COLDATA_NONE));
                this->colDataType = t;
                this->colPtr = p;
                this->colStride = s;
            }

            /**
             * Sets the colour map index values
             *
             * @param minVal The minimum colour index value to be mapped
             * @param maxVal The maximum colour index value to be mapped
             */
            void SetColourMapIndexValues(float minVal, float maxVal) {
                this->maxColI = maxVal;
                this->minColI = minVal;
            }

            /**
             * Sets the number of objects stored and resets all data pointers!
             *
             * @param cnt The number of stored objects
             */
            void SetCount(UINT64 cnt) {
                this->colDataType = COLDATA_NONE;
                this->colPtr = NULL; // DO NOT DELETE
                this->vertDataType = VERTDATA_NONE;
                this->vertPtr = NULL; // DO NOT DELETE

                this->count = cnt;
            }

            /**
             * Sets the global colour data
             *
             * @param r The red colour component
             * @param g The green colour component
             * @param b The blue colour component
             * @param a The opacity alpha
             */
            void SetGlobalColour(unsigned int r, unsigned int g,
                    unsigned int b, unsigned int a = 255) {
                this->col[0] = r;
                this->col[1] = g;
                this->col[2] = b;
                this->col[3] = a;
            }

            /**
             * Sets the global radius
             *
             * @param r The global radius
             */
            void SetGlobalRadius(float r) {
                this->radius = r;
            }

            /**
             * Sets the vertex data
             *
             * @param t The type of the vertex data
             * @param p The pointer to the vertex data (must not be NULL if t
             *          is not 'VERTDATA_NONE'
             * @param s The stride of the vertex data
             */
            void SetVertexData(VertexDataType t, const void *p,
                    unsigned int s = 0) {
                ASSERT((p != NULL) || (t == VERTDATA_NONE));
                this->vertDataType = t;
                this->vertPtr = p;
                this->vertStride = s;
            }

            /**
             * Assignment operator
             *
             * @param rhs The right hand side operand
             *
             * @return A reference to 'this'
             */
            Particles& operator=(const Particles& rhs);

            /**
             * Test for equality
             *
             * @param rhs The right hand side operand
             *
             * @return 'true' if 'this' and 'rhs' are equal.
             */
            bool operator==(const Particles& rhs) const;

        private:

            /** The global colour */
            unsigned char col[4];

            /** The colour data type */
            ColourDataType colDataType;

            /** The colour data pointer */
            const void *colPtr;

            /** The colour data stride */
            unsigned int colStride;

            /** The number of objects stored */
            UINT64 count;

            /** The maximum colour index value to be mapped */
            float maxColI;

            /** The minimum colour index value to be mapped */
            float minColI;

            /** The global radius */
            float radius;

            /** The vertex data type */
            VertexDataType vertDataType;

            /** The vertex data pointer */
            const void *vertPtr;

            /** The vertex data stride */
            unsigned int vertStride;

        };

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "MultiParticleDataCall";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get multi-stream particle sphere data";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return AbstractGetData3DCall::FunctionCount();
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            return AbstractGetData3DCall::FunctionName(idx);
        }

        /** Ctor. */
        MultiParticleDataCall(void);

        /** Dtor. */
        virtual ~MultiParticleDataCall(void);

        /**
         * Accesses the particles of list item 'idx'
         *
         * @param idx The zero-based index of the particle list to return
         *
         * @return The requested particle list
         */
        Particles& AccessParticles(unsigned int idx) {
            return this->lists[idx];
        }

        /**
         * Accesses the particles of list item 'idx'
         *
         * @param idx The zero-based index of the particle list to return
         *
         * @return The requested particle list
         */
        const Particles& AccessParticles(unsigned int idx) const {
            return this->lists[idx];
        }

        /**
         * Answer the number of particle lists
         *
         * @return The number of particle lists
         */
        inline unsigned int GetParticleListCount(void) const {
            return static_cast<unsigned int>(this->lists.Count());
        }

        /**
         * Sets the number of particle lists. All list items are in undefined
         * states afterward.
         *
         * @param cnt The new number of particle lists
         */
        void SetParticleListCount(unsigned int cnt) {
            this->lists.SetCount(cnt);
        }

        /**
         * Assignment operator.
         * Makes a deep copy of all members. While for data these are only
         * pointers, the pointer to the unlocker object is also copied.
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this
         */
        MultiParticleDataCall& operator=(const MultiParticleDataCall& rhs);

    private:

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /** Array of lists of particles */
        vislib::Array<Particles> lists;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

    };

    /** Description class typedef */
    typedef CallAutoDescription<MultiParticleDataCall>
        MultiParticleDataCallDescription;

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MULTIPARTICLEDATACALL_H_INCLUDED */
