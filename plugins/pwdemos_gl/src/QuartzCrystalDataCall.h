/*
 * QuartzCrystalDataCall.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmstd/data/AbstractGetData3DCall.h"
#include "vislib/Array.h"
#include "vislib/math/Vector.h"


namespace megamol::demos_gl {

/**
 * Call transporting BDMD data frames
 */
class CrystalDataCall : public megamol::core::AbstractGetData3DCall {
public:
    /**
     * Definition of a single crystal
     */
    class Crystal {
    public:
        /**
         * Ctor
         */
        Crystal();

        /**
         * Dtor
         */
        ~Crystal();

        /**
         * Adds a face to the crystal
         *
         * @param vec The normal vector/position vector of the face
         */
        void AddFace(const vislib::math::Vector<float, 3>& vec);

        /**
         * Adds a face to the crystal
         *
         * @param x The x component of the normal vector/position vector of the face
         * @param y The y component of the normal vector/position vector of the face
         * @param z The z component of the normal vector/position vector of the face
         */
        inline void AddFace(float x, float y, float z) {
            this->AddFace(vislib::math::Vector<float, 3>(x, y, z));
        }

        /**
         * Makes sure the mesh data is available
         */
        void AssertMesh() const;

        /**
         * Calculates the mesh data for the crystal
         *
         * @param setBoundingRad Flag whether or not to set the bounding
         *                       radius to the value calculated from the
         *                       mesh.
         * @param force If set to 'true' the mesh will be calculated even
         *              if it is already present
         */
        void CalculateMesh(bool setBoundingRad, bool force = false);

        /**
         * Clears the crystal data
         */
        void Clear();

        /**
         * Clears the mesh data
         */
        void ClearMesh();

        /**
         * Answer the base radius
         *
         * @return The base radius
         */
        inline float GetBaseRadius() const {
            return this->baseRad;
        }

        /**
         * Answer the bounding radius
         *
         * @return The bounding radius
         */
        inline float GetBoundingRadius() const {
            return this->boundRad;
        }

        /**
         * Answers the mesh per-face triangle-fan vertex count
         *
         * @return The mesh per-face triangle-fan vertex count
         */
        inline const unsigned int* GetMeshTriangleCounts() const {
            return this->triangleCnt;
        }

        /**
         * Answers the mesh per-face triangle-fan data
         *
         * @return The mesh per-face triangle-fan data
         */
        inline unsigned int** const GetMeshTriangles() const {
            return this->triangles;
        }

        /**
         * Answers the mesh vertex data
         *
         * @return The mesh vertex data
         */
        inline const float* GetMeshVertexData() const {
            return this->vertices;
        }

        /**
         * Answer a face from the crystal
         *
         * @param idx The zero-based index of the face
         *
         * @return The normal vector/position vector of the idx'th face of the crystal
         */
        inline const vislib::math::Vector<float, 3>& GetFace(unsigned int idx) const {
            return this->faces[static_cast<SIZE_T>(idx)];
        }

        /**
         * Answer the number of faces
         *
         * @return The number of faces
         */
        inline unsigned int GetFaceCount() const {
            return static_cast<unsigned int>(this->faces.Count());
        }

        /**
         * Answer if the crystal is empty
         *
         * @return False if at least one face is stored
         */
        inline bool IsEmpty() const {
            return this->faces.IsEmpty();
        }

        /**
         * Sets the base radius
         *
         * @param rad The new base radius
         */
        inline void SetBaseRadius(float rad) {
            this->baseRad = rad;
            //this->ClearMesh(); // mesh does not depend on this variable anymore!
        }

        /**
         * Sets the bounding radius
         *
         * @param rad The new bounding radius
         */
        inline void SetBoundingRadius(float rad) {
            this->boundRad = rad;
            // mesh does not depend on this variable
        }

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return True if this equals rhs
         */
        bool operator==(const Crystal& rhs) const;

    private:
        /** The base radius */
        float baseRad;

        /** The bounding sphere radius */
        float boundRad;

        /** The normal vectors, position vectors for the faces of the crystal */
        vislib::Array<vislib::math::Vector<float, 3>> faces;

        /** The mesh vertices */
        float* vertices;

        /** The number of triangles per face in the mesh */
        unsigned int* triangleCnt;

        /** The mesh triangles per face */
        unsigned int** triangles;
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
    static const char* ClassName() {
        return "QuartzCrystalDataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call transporting quartz crystal definitions";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
        return AbstractGetData3DCall::FunctionCount();
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        return AbstractGetData3DCall::FunctionName(idx);
    }

    /** Ctor */
    CrystalDataCall();

    /** Dtor. */
    ~CrystalDataCall() override;

    /**
     * Gets the number of crystals
     *
     * @return The number of crystals
     */
    inline unsigned int GetCount() const {
        return this->count;
    }

    /**
     * Gets the array of crystals
     *
     * @return Pointer to the array of crystals
     */
    inline const Crystal* GetCrystals() const {
        return this->crystals;
    }

    /**
     * Sets the data
     *
     * @param cnt The number of crystals
     * @param crystals The array of crystals
     */
    inline void SetCrystals(unsigned int cnt, const Crystal* crystals) {
        this->count = cnt;
        this->crystals = crystals;
    }

private:
    /** Number of crystals */
    unsigned int count;

    /** Crystals */
    const Crystal* crystals;
};

} // namespace megamol::demos_gl
