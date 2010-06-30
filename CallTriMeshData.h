/*
 * CallTriMeshData.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMTRISOUPPLG_CALLTRIMESHDATA_H_INCLUDED
#define MMTRISOUPPLG_CALLTRIMESHDATA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractGetData3DCall.h"
#include "CallAutoDescription.h"
#include "vislib/assert.h"
#include "vislib/Array.h"
#include "vislib/String.h"


namespace megamol {
namespace trisoup {

    /**
     * Call transporting tri soup mesh data
     */
    class CallTriMeshData : public core::AbstractGetData3DCall {
    public:

        /**
         * Subclass storing material information
         */
        class Material {
        public:

            /** Possible values for the illumination modell */
            enum IlluminationModel {
                ILLUM_NO_LIGHT = 0, //: no lighting
                ILLUM_DIFF = 1,     //: diffuse lighting only
                ILLUM_DIFF_SPEC = 2 //: both diffuse lighting and specular highlights
            };

            /** Ctor */
            Material(void);

            /** Dtor */
            ~Material(void);

            /**
             * Gets specular component of the Phong shading model ranges between 0 and 128
             *
             * @return specular component
             */
            inline float GetNs(void) const {
                return this->Ns;
            }

            /**
             * Gets Unknown
             *
             * @return Unknown
             */
            inline float GetNi(void) const {
                return this->Ni;
            }

            /**
             * Gets alpha transparency (?)
             *
             * @return alpha transparency (?)
             */
            inline float GetD(void) const {
                return this->d;
            }

            /**
             * Gets alpha transparency (?)
             *
             * @return alpha transparency (?)
             */
            inline float GetTr(void) const {
                return this->Tr;
            }

            /**
             * Gets Unknown
             *
             * @return Unknown
             */
            inline const float* GetTf(void) const {
                return this->Tf;
            }

            /**
             * Gets illumination model
             *
             * @return illumination model
             */
            inline IlluminationModel GetIllum(void) const {
                return this->illum;
            }

            /**
             * Gets ambient colour
             *
             * @return ambient colour
             */
            inline const float* GetKa(void) const {
                return this->Ka;
            }

            /**
             * Gets diffuse colour
             *
             * @return diffuse colour
             */
            inline const float* GetKd(void) const {
                return this->Kd;
            }

            /**
             * Gets specular colour
             *
             * @return specular colour
             */
            inline const float* GetKs(void) const {
                return this->Ks;
            }

            /**
             * Gets emissive colour
             *
             * @return emissive colour
             */
            inline const float* GetKe(void) const {
                return this->Ke;
            }

            /**
             * Gets colour texture map file
             *
             * @return colour texture map file
             */
            inline const vislib::TString& GetMapFileName(void) const {
                return this->mapFileName;
            }

            /**
             * Gets bump/normal texture map file
             *
             * @return bump/normal texture map file
             */
            inline const vislib::TString& GetBumpMapFileName(void) const {
                return this->bumpMapFileName;
            }

            /**
             * Gets OpenGL texture object ID for colour texture
             *
             * @return OpenGL texture object ID for colour texture
             */
            unsigned int  GetMapID(void) const;

            /**
             * Gets OpenGL texture object ID for bump/normal texture
             *
             * @return OpenGL texture object ID for bump/normal texture
             */
            unsigned int  GetBumpMapID(void) const;

            /**
             * Sets specular component of the Phong shading model ranges between 0 and 128
             *
             * @param Ns specular component
             */
            inline void SetNs(float Ns) {
                this->Ns = Ns;
            }

            /**
             * Sets 
             *
             * @param Ni 
             */
            inline void SetNi(float Ni) {
                this->Ni = Ni;
            }

            /**
             * Sets 
             *
             * @param d 
             */
            inline void SetD(float d) {
                this->d = d;
            }

            /**
             * Sets 
             *
             * @param Tr 
             */
            inline void SetTr(float Tr) {
                this->Tr = Tr;
            }

            /**
             * Sets 
             *
             * @param Tf0 First component of 
             * @param Tf1 Second component of 
             * @param Tf2 Third component of 
             */
            inline void SetTf(float Tf0, float Tf1, float Tf2) {
                this->Tf[0] = Tf0;
                this->Tf[1] = Tf1;
                this->Tf[2] = Tf2;
            }

            /**
             * Sets 
             *
             * @param illum 
             */
            inline void SetIllum(IlluminationModel illum) {
                this->illum = illum;
            }

            /**
             * Sets 
             *
             * @param Ka0 First component of 
             * @param Ka1 Second component of 
             * @param Ka2 Third component of 
             */
            inline void SetKa(float Ka0, float Ka1, float Ka2) {
                this->Ka[0] = Ka0;
                this->Ka[1] = Ka1;
                this->Ka[2] = Ka2;
            }

            /**
             * Sets 
             *
             * @param Kd0 First component of 
             * @param Kd1 Second component of 
             * @param Kd2 Third component of 
             */
            inline void SetKd(float Kd0, float Kd1, float Kd2) {
                this->Kd[0] = Kd0;
                this->Kd[1] = Kd1;
                this->Kd[2] = Kd2;
            }

            /**
             * Sets 
             *
             * @param Ks0 First component of 
             * @param Ks1 Second component of 
             * @param Ks2 Third component of 
             */
            inline void SetKs(float Ks0, float Ks1, float Ks2) {
                this->Ks[0] = Ks0;
                this->Ks[1] = Ks1;
                this->Ks[2] = Ks2;
            }

            /**
             * Sets 
             *
             * @param Ke0 First component of 
             * @param Ke1 Second component of 
             * @param Ke2 Third component of 
             */
            inline void SetKe(float Ke0, float Ke1, float Ke2) {
                this->Ke[0] = Ke0;
                this->Ke[1] = Ke1;
                this->Ke[2] = Ke2;
            }

            /**
             * Sets the file name to load the colour texture from
             *
             * @param filename The file name
             */
            void SetMapFileName(const vislib::TString& filename);

            /**
             * Sets the file name to load the bump/normal texture from
             *
             * @param filename The file name
             */
            void SetBumpMapFileName(const vislib::TString& filename);

            /**
             * Test for equality
             *
             * @param rhs The right hand side operand
             *
             * @return True if this and rhs are equal
             */
            bool operator==(const Material& rhs) const;

        private:

            /**
             * Tries to load a texture from a file
             *
             * @param filename Specifies the file name of the texture file to
             *                 be loaded. Will be cleared on error.
             *
             * @return The OpenGL texture object id on success or 0 on failure
             */
            static unsigned int loadTexture(vislib::TString &filename);

            /** Forbidden copy ctor */
            Material(const Material& src);

            /** specular component of the Phong shading model ranges between 0 and 128 */
            float Ns;

            /** Unknown */
            float Ni;

            /** alpha transparency (?) */
            float d;

            /** alpha transparency (?) */
            float Tr;

            /** Unknown */
            float Tf[3];

            /** The illumination model */
            IlluminationModel illum;

            /** ambient colour */
            float Ka[3];

            /** diffuse colour */
            float Kd[3];

            /** specular colour */
            float Ks[3];

            /** emissive colour */
            float Ke[3];

            /** colour texture map file */
            mutable vislib::TString mapFileName;

            /** bump/normal texture map file */
            mutable vislib::TString bumpMapFileName;

            /** OpenGL texture object ID for colour texture */
            mutable unsigned int  mapID;

            /** OpenGL texture object ID for bump/normal texture */
            mutable unsigned int  bumpMapID;

        };

        /**
         * Subclass storing the pointers to the mesh data
         */
        class Mesh {
        public:

            /** Ctor */
            Mesh(void);

            /** Dtor */
            ~Mesh(void);

            /**
             * Gets Triangle count (ignored if t is NULL)
             *
             * @return Triangle count (ignored if t is NULL)
             */
            inline unsigned int GetTriCount(void) const {
                return this->triCnt;
            }

            /**
             * Gets Triangle vertex indices (3 times tc) or NULL
             *
             * @return Triangle vertex indices (3 times tc) or NULL
             */
            inline const unsigned int * GetTriIndexPointer(void) const {
                return this->tri;
            }

            /**
             * Gets Vertex count
             *
             * @return Vertex count
             */
            inline unsigned int GetVertexCount(void) const {
                return this->vrtCnt;
            }

            /**
             * Gets Vertices (3 times vc)
             *
             * @return Vertices (3 times vc)
             */
            inline const float * GetVertexPointer(void) const {
                return this->vrt;
            }

            /**
             * Gets Normals (3 times vc)
             *
             * @return Normals (3 times vc)
             */
            inline const float * GetNormalPointer(void) const {
                return this->nrm;
            }

            /**
             * Gets Colors (3 times vc)
             *
             * @return Colors (3 times vc)
             */
            inline const unsigned char * GetColourPointer(void) const {
                return this->col;
            }

            /**
             * Gets Texture coordinates (2 times vc)
             *
             * @return Texture coordinates (2 times vc)
             */
            inline const float * GetTextureCoordinatePointer(void) const {
                return this->tex;
            }

            /**
             * Gets The material
             *
             * @return The material
             */
            inline const Material * GetMaterial(void) const {
                return this->mat;
            }

            /**
             * Sets the triangle data
             *
             * @param cnt The number of triangles
             * @param indices Pointer to 3 times cnt unsigned ints holding the
             *                indices of the vertices used by the triangles
             * @param takeOwnership If true the object will take ownership of
             *                      all the memory of the pointers provided
             *                      and will free the memory on its
             *                      destruction. Otherwise the caller must
             *                      ensure the memory stays valid as long as
             *                      it is used.
             */
            void SetTriangleData(unsigned int cnt, unsigned int *indices, bool takeOwnership);

            /**
             * Sets the vertex data
             *
             * @param cnt The number of vertices
             * @param vertices Pointer to 3 times cnt floats holding the vertices
             * @param normals Pointer to 3 times cnt floats holding the normal vectors
             * @param colours Pointer to 3 times cnt unsigned bytes holding the colours
             * @param textureCoordinates Pointer to 2 times cnt float holding the texture coordinates
             * @param takeOwnership If true the object will take ownership of
             *                      all the memory of the pointers provided
             *                      and will free the memory on its
             *                      destruction. Otherwise the caller must
             *                      ensure the memory stays valid as long as
             *                      it is used.
             */
            void SetVertexData(unsigned int cnt, float *vertices, float *normals, unsigned char *colours, float *textureCoordinates, bool takeOwnership);

            /**
             * Sets the material
             * The ownership of the 'mat' object is not changed. The caller
             * must ensure that the object stays valid as long as it is used
             *
             * @param mat Pointer to the material object to be referenced
             */
            inline void SetMaterial(const Material * mat) {
                this->mat = mat;
            }

            /**
             * Test for equality
             *
             * @param rhs The right hand side operand
             *
             * @return True if this and rhs are equal
             */
            bool operator==(const Mesh& rhs) const;

        protected:
        private:

            /** Forbidden copy ctor */
            Mesh(const Mesh& src);

            /** Triangle count (ignored if t is NULL) */
            unsigned int triCnt;

            /** Triangle vertex indices (3 times tc) or NULL */
            unsigned int * tri;

            /** Flag indicating the if the triangle data memory is owned by this object and will be freed on its destruction */
            bool triMemOwned;

            /** Vertex count */
            unsigned int vrtCnt;

            /** Vertices (3 times vc) */
            float * vrt;

            /** Normals (3 times vc) */
            float * nrm;

            /** Colors (3 times vc) */
            unsigned char * col;

            /** Texture coordinates (2 times vc) */
            float * tex;

            /** Flag indicating the if the triangle data memory is owned by this object and will be freed on its destruction */
            bool vrtMemOwned;

            /** The material */
            const Material * mat;

        };

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "CallTriMeshData";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call transporting tri soup mesh data";
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

        /** Ctor */
        CallTriMeshData(void);

        /** Dtor */
        virtual ~CallTriMeshData(void);

        /**
         * Answer the number of objects
         *
         * @return The number of objects
         */
        inline unsigned int Count(void) const {
            return this->objCnt;
        }

        /**
         * Returns a pointer to the array of objects
         *
         * @return A pointer to the array of objects
         */
        inline const Mesh *Objects(void) const {
            return this->objs;
        }

        /**
         * Sets the object array pointer.
         * The call does not take ownership of the memory of the objects. The
         * caller must ensure the pointer and memory stays valid as long as
         * they are used.
         *
         * @param cnt The number of objects
         * @param objs A pointer to the array of objects
         */
        inline void SetObjects(unsigned int cnt, const Mesh *objs) {
            this->objCnt = cnt;
            this->objs = objs;
        }

    private:

        /** The number of objects */
        unsigned int objCnt;

        /** Pointer to the array of objects */
        const Mesh *objs;

    };

    /** Description class typedef */
    typedef core::CallAutoDescription<CallTriMeshData> CallTriMeshDataDescription;

} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MMTRISOUPPLG_CALLTRIMESHDATA_H_INCLUDED */
