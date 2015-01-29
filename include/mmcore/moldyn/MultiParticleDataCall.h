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

#include "mmcore/moldyn/AbstractParticleDataCall.h"
#include "mmcore/CallAutoDescription.h"
#include "vislib/assert.h"
#include "vislib/Map.h"
#include "vislib/Array.h"

namespace megamol {
namespace core {
namespace moldyn {


    /**
     * Class holding all data of a single particle type
     *
     * TODO: This class currently can only hold data for spheres and should
     *       be extended to be able to handle data for arbitrary glyphs.
     *       This also applies to interpolation of data.
     */
    class MEGAMOLCORE_API SimpleSphericalParticles {
    public:

        /** possible values for the vertex data */
        enum VertexDataType {
            VERTDATA_NONE, //< indicates that this object is void
            VERTDATA_FLOAT_XYZ, //< use global radius
            VERTDATA_FLOAT_XYZR,
            VERTDATA_SHORT_XYZ //< quantized positions and global radius
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
        SimpleSphericalParticles(void);

        /**
         * Copy ctor
         *
         * @param src The object to clone from
         */
        SimpleSphericalParticles(const SimpleSphericalParticles& src);

        /**
         * Dtor
         */
        ~SimpleSphericalParticles(void);

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
         * Answer the global particle type
         *
         * @return the global type
         */
        inline unsigned int GetGlobalType(void) const {
            return this->particleType;
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
        //    ASSERT((p != NULL) || (t == COLDATA_NONE));
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
         * Sets the global particle type
         *
         * @param t The global type
         */
        void SetGlobalType(unsigned int t) {
            this->particleType = t;
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
            ASSERT(this->disabledNullChecks || (p != NULL) || (t == VERTDATA_NONE));
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
        SimpleSphericalParticles& operator=(const SimpleSphericalParticles& rhs);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return 'true' if 'this' and 'rhs' are equal.
         */
        bool operator==(const SimpleSphericalParticles& rhs) const;

		/**
         * Disable NULL-checks in case we have an OpenGL-VAO
         * @param disable flag to disable/enable the checks
         */
		void disableNullChecksForVAOs(bool disable = true)
		{
			disabledNullChecks = disable;
		}
		
		/**
		* Defines wether we transport VAOs instead of real data
		* @param vao flag to disable/enable the checks
		*/
		void SetIsVAO(bool vao)
		{
			this->isVAO = vao;
		}

		/**
		* Disable NULL-checks in case we have an OpenGL-VAO
		* @param disable flag to disable/enable the checks
		*/
		bool IsVAO()
		{
			return this->isVAO;
		}

		/**
		* If we handle clusters this could be useful
		*/
		struct ClusterInfos
		{
			/** a map with clusterid to particleids relation*/
			vislib::Map<int, vislib::Array<int>> data;
			/** the map in plain data for upload to gpu */
			unsigned int *plainData;
			/** size of the plain data*/
			size_t sizeofPlainData;
			/** number of clusters*/
			unsigned int numClusters;
			ClusterInfos() : data(), plainData(0), sizeofPlainData(0), numClusters(0) {};
		};
		
		/**
		* Sets the local ClusterInfos-struct
		*/
		void SetClusterInfos(ClusterInfos *infos)
		{
			this->clusterInfos = infos;
		}

		/**
		* gets the local ClusterInfos-struct
		*/
		ClusterInfos *GetClusterInfos()
		{
			return this->clusterInfos;
		}

		/**
		* Sets the VertexArrayObject, VertexBuffer and ColorBuffer used
		*/
		void SetVAOs(unsigned int vao, unsigned int vb, unsigned int cb)
		{
			this->glVAO = vao;
			this->glVB = vb;
			this->glCB = cb;
		}

		/**
		* Gets the VertexArrayObject, VertexBuffer and ColorBuffer used
		*/
		void GetVAOs(unsigned int &vao, unsigned int &vb, unsigned int &cb)
		{
			vao = this->glVAO;
			vb = this->glVB;
			cb = this->glCB;
		}

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

        /** The global type of particles in the list */
        unsigned int particleType;

        /** The vertex data type */
        VertexDataType vertDataType;

        /** The vertex data pointer */
        const void *vertPtr;

        /** The vertex data stride */
        unsigned int vertStride;
		
		/** disable NULL-checks if used with OpenGL-VAO */
		bool disabledNullChecks;

		/** do we use a VertexArrayObject? */
		bool isVAO;

		/** Vertex Array Object to transport */
		unsigned int glVAO;
		/** Vertex Buffer to transport */
		unsigned int glVB;
		/** Color Buffer to transport */
		unsigned int glCB;

		/** local Cluster Infos*/
		ClusterInfos *clusterInfos;
    };


    MEGAMOLCORE_APIEXT template class MEGAMOLCORE_API AbstractParticleDataCall<SimpleSphericalParticles>;


    /**
     * Call for multi-stream particle data.
     */
    class MEGAMOLCORE_API MultiParticleDataCall
        : public AbstractParticleDataCall<SimpleSphericalParticles> {
    public:

        /** typedef for legacy name */
        typedef SimpleSphericalParticles Particles;

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "MultiParticleDataCall";
        }

        /** Ctor. */
        MultiParticleDataCall(void);

        /** Dtor. */
        virtual ~MultiParticleDataCall(void);

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

    };


    /** Description class typedef */
    typedef CallAutoDescription<MultiParticleDataCall>
        MultiParticleDataCallDescription;


} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MULTIPARTICLEDATACALL_H_INCLUDED */
