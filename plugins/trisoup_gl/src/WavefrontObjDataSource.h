/*
 * WavefrontObjDataSource.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMTRISOUPPLG_WAVEFRONTOBJDATASOURCE_H_INCLUDED
#define MMTRISOUPPLG_WAVEFRONTOBJDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractTriMeshLoader.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include "vislib/StringTokeniser.h"


namespace megamol::trisoup_gl {


/**
 * Data source class for wavefront OBJ files
 */
class WavefrontObjDataSource : public AbstractTriMeshLoader {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "WavefrontObjDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Data source for wavefront OBJ files";
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
    WavefrontObjDataSource();

    /** Dtor */
    ~WavefrontObjDataSource() override;

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
    /** Internat utility struct to store a single triangle */
    typedef struct _tri_t {

        /** The indices of the vertices */
        unsigned int v1, v2, v3;

        /** The indices of the normals */
        unsigned int n1, n2, n3;

        /** The indices of the texture coordinates */
        unsigned int t1, t2, t3;

        /** validity flags for normal and texture coodrinate information */
        bool n, t;

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return True if this and rhs are equal
         */
        bool operator==(const struct _tri_t& rhs) const {
            return (this->v1 == rhs.v1) && (this->v2 == rhs.v2) && (this->v3 == rhs.v3) && (this->n1 == rhs.n1) &&
                   (this->n2 == rhs.n2) && (this->n3 == rhs.n3) && (this->t1 == rhs.t1) && (this->t2 == rhs.t2) &&
                   (this->t3 == rhs.t3) && (this->n == rhs.n) && (this->t == rhs.t);
        }

    } Tri;

    /**
     * Loads a material library file
     *
     * @param filename The file name of the material library file to load
     * @param names Array to store the material names
     */
    void loadMaterialLibrary(const vislib::TString& filename, vislib::Array<vislib::StringA>& names);

    /**
     * Creates a mesh from the loaded triangles
     *
     * @param mesh The object to store the new mesh
     * @param tris The incoming triangles
     * @param vu unsigned-int-Array of the same size as 'v' free to use
     * @param v The vertices array
     * @param n The normal vectors array
     * @param t The texture coordinates array
     */
    void makeMesh(megamol::geocalls_gl::CallTriMeshDataGL::Mesh& mesh,
        const vislib::Array<WavefrontObjDataSource::Tri>& tris, unsigned int* vu,
        const vislib::Array<vislib::math::Vector<float, 3>>& v, const vislib::Array<vislib::math::Vector<float, 3>>& n,
        const vislib::Array<vislib::math::Vector<float, 2>>& t);


    /** vertex store for lines */
    vislib::Array<vislib::Array<float>> lineVerts;
};

} // namespace megamol::trisoup_gl

#endif /* MMTRISOUPPLG_WAVEFRONTOBJDATASOURCE_H_INCLUDED */
