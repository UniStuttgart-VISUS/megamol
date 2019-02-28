/*
 * CallTriMeshData.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010-2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GEOMETRY_CALLS_CALLTRIMESHDATA_H_INCLUDED
#define MEGAMOL_GEOMETRY_CALLS_CALLTRIMESHDATA_H_INCLUDED
#pragma once

#include "geometry_calls/geometry_calls.h"
#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/assert.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include "vislib/macro_utils.h"
#include <climits>

#define MAX_PARAMETER_NUMBER 100

namespace megamol {
namespace geocalls {

    /**
     * Call transporting tri soup mesh data
     */
    class GEOMETRY_CALLS_API CallTriMeshData : public core::AbstractGetData3DCall {
    public:

        /**
         * Subclass storing material information
         */
        class GEOMETRY_CALLS_API Material {
        public:

            /** Possible values for the illumination modell */
            enum IlluminationModel {
                ILLUM_NO_LIGHT = 0, //: no lighting
                ILLUM_DIFF = 1,     //: diffuse lighting only
                ILLUM_DIFF_SPEC = 2 //: both diffuse lighting and specular highlights
            };

            /** Ctor */
            Material(void);

            /**
             * Copy ctor
             *
             * @param src The object to clone from
             */
            Material(const Material& src);

            /** Dtor */
            ~Material(void);

            /**
             * Dyes the current color by multiplying 'r', 'g', and 'b' to all
             * colour values
             *
             * @param r The red colour component
             * @param g The green colour component
             * @param b The blue colour component
             */
            void Dye(float r, float g, float b);

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
            unsigned int GetMapID(void) const;

            /**
             * Gets OpenGL texture object ID for bump/normal texture
             *
             * @return OpenGL texture object ID for bump/normal texture
             */
            unsigned int GetBumpMapID(void) const;

            /**
             * Resets this colour material to default
             */
            void MakeDefault(void);

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
             * Sets illumination model
             *
             * @param illum illumination model
             */
            inline void SetIllum(IlluminationModel illum) {
                this->illum = illum;
            }

            /**
             * Sets ambient colour
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
             * Sets diffuse colour
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
             * Sets specular colour
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
             * Sets emissive colour
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
             * Assignment operator
             *
             * @param rhs The right hand side operand
             *
             * @return Reference to this
             */
            Material& operator=(const Material& rhs);

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
            VISLIB_MSVC_SUPPRESS_WARNING(4251)
            mutable vislib::TString mapFileName;

            /** bump/normal texture map file */
            VISLIB_MSVC_SUPPRESS_WARNING(4251)
            mutable vislib::TString bumpMapFileName;

            /** OpenGL texture object ID for colour texture */
            mutable unsigned int  mapID;

            /** OpenGL texture object ID for bump/normal texture */
            mutable unsigned int  bumpMapID;

        };

        /**
         * Subclass storing the pointers to the mesh data
         */
        class GEOMETRY_CALLS_API Mesh {
        public:

            /** Possible data types */
            enum DataType {
                DT_NONE,
                DT_BYTE, // UINT8
                DT_UINT16,
                DT_UINT32,
                DT_INT16,
                DT_INT32,
                DT_FLOAT,
                DT_DOUBLE
            };

            /** Ctor */
            Mesh(void);

            /**
             * Copy ctor
             *
             * @param src The object to clone from
             */
            Mesh(const Mesh& src);

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
             * Answer the data type for triangle vertex index data
             *
             * @return The data type for triangle vertex index data
             */
            inline DataType GetTriDataType(void) const {
                return this->triDT;
            }

            /**
             * Answer if triangle vertex indices data has been set to non-null
             *
             * @return True if triangle vertex indices data is present
             */
            inline bool HasTriIndexPointer(void) const {
                return this->tri.dataByte != NULL;
            }

            /**
             * Gets Triangle vertex indices (3 times tc) or NULL
             *
             * @return Triangle vertex indices (3 times tc) or NULL
             */
            inline const unsigned char * GetTriIndexPointerByte(void) const {
                ASSERT(this->triDT == DT_BYTE);
                return this->tri.dataByte;
            }

            /**
             * Gets Triangle vertex indices (3 times tc) or NULL
             *
             * @return Triangle vertex indices (3 times tc) or NULL
             */
            inline const unsigned short * GetTriIndexPointerUInt16(void) const {
                ASSERT(this->triDT == DT_UINT16);
                return this->tri.dataUInt16;
            }

            /**
             * Gets Triangle vertex indices (3 times tc) or NULL
             *
             * @return Triangle vertex indices (3 times tc) or NULL
             */
            inline const unsigned int * GetTriIndexPointerUInt32(void) const {
                ASSERT(this->triDT == DT_UINT32);
                return this->tri.dataUInt32;
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
             * Answer the data type for vertices
             *
             * @return The data type for vertices
             */
            inline DataType GetVertexDataType(void) const {
                return this->vrtDT;
            }

            /**
             * Gets Vertices (3 times vc)
             *
             * @return Vertices (3 times vc)
             */
            inline const double * GetVertexPointerDouble(void) const {
                ASSERT(this->vrtDT == DT_DOUBLE);
                return this->vrt.dataDouble;
            }

            /**
             * Gets Vertices (3 times vc)
             *
             * @return Vertices (3 times vc)
             */
            inline const float * GetVertexPointerFloat(void) const {
                ASSERT(this->vrtDT == DT_FLOAT);
                return this->vrt.dataFloat;
            }

            /**
             * Answer the data type for normals
             *
             * @return The data type for normals
             */
            inline DataType GetNormalDataType(void) const {
                return this->nrmDT;
            }

            /**
             * Answer if normal data has been set to non-null
             *
             * @return True if normal data is present
             */
            inline bool HasNormalPointer(void) const {
                return this->nrm.dataDouble != NULL;
            }

            /**
             * Gets Normals (3 times vc)
             *
             * @return Normals (3 times vc)
             */
            inline const double * GetNormalPointerDouble(void) const {
                ASSERT(this->nrmDT == DT_DOUBLE);
                return this->nrm.dataDouble;
            }

            /**
             * Gets Normals (3 times vc)
             *
             * @return Normals (3 times vc)
             */
            inline const float * GetNormalPointerFloat(void) const {
                ASSERT(this->nrmDT == DT_FLOAT);
                return this->nrm.dataFloat;
            }

            /**
             * Answer if colours data has been set to non-null
             *
             * @return True if colours data is present
             */
            inline bool HasColourPointer(void) const {
                return this->col.dataDouble != NULL;
            }

            /**
             * Answer the data type for colours
             *
             * @return The data type for colours
             */
            inline DataType GetColourDataType(void) const {
                return this->colDT;
            }

            /**
             * Gets Colors (3 times vc)
             *
             * @return Colors (3 times vc)
             */
            inline const unsigned char * GetColourPointerByte(void) const {
                ASSERT(this->colDT == DT_BYTE);
                return this->col.dataByte;
            }

            /**
             * Gets Colors (3 times vc)
             *
             * @return Colors (3 times vc)
             */
            inline const double * GetColourPointerDouble(void) const {
                ASSERT(this->colDT == DT_DOUBLE);
                return this->col.dataDouble;
            }

            /**
             * Gets Colors (3 times vc)
             *
             * @return Colors (3 times vc)
             */
            inline const float * GetColourPointerFloat(void) const {
                ASSERT(this->colDT == DT_FLOAT);
                return this->col.dataFloat;
            }

            /**
             * Answer if texture coordinates data has been set to non-null
             *
             * @return True if texture coordinates data is present
             */
            inline bool HasTextureCoordinatePointer(void) const {
                return this->tex.dataDouble != NULL;
            }

            /**
             * Answer the data type for texture coordinates
             *
             * @return The data type for texture coordinates
             */
            inline DataType GetTextureCoordinateDataType(void) const {
                return this->texDT;
            }

            /**
             * Gets Texture coordinates (2 times vc)
             *
             * @return Texture coordinates (2 times vc)
             */
            inline const double * GetTextureCoordinatePointerDouble(void) const {
                ASSERT(this->texDT == DT_DOUBLE);
                return this->tex.dataDouble;
            }

            /**
             * Gets Texture coordinates (2 times vc)
             *
             * @return Texture coordinates (2 times vc)
             */
            inline const float * GetTextureCoordinatePointerFloat(void) const {
                ASSERT(this->texDT == DT_FLOAT);
                return this->tex.dataFloat;
            }

            /**
             * Adds a new vertex attribute.
             *
             * @return The index of the newly added attribute, or UINT_MAX in the case of failure
             */
            inline unsigned int AddVertexAttribPointer(uint8_t * ptr) {
                if (this->vattCount == MAX_PARAMETER_NUMBER) return UINT_MAX;
                this->vattDTypes[this->vattCount] = DT_BYTE;
                this->vattVector = this->allocateAdditionalEntry(this->vattVector, this->vattCount);
                this->vattVector[this->vattCount].dataByte = ptr;
                this->vattCount = this->vattCount + 1;
                return this->vattCount - 1;
            }

            /**
             * Adds a new vertex attribute.
             *
             * @return The index of the newly added attribute, or UINT_MAX in the case of failure
             */
            inline unsigned int AddVertexAttribPointer(double * ptr) {
                if (this->vattCount == MAX_PARAMETER_NUMBER) return UINT_MAX;
                this->vattDTypes[this->vattCount] = DT_DOUBLE;
                this->vattVector = this->allocateAdditionalEntry(this->vattVector, this->vattCount);
                this->vattVector[this->vattCount].dataDouble = ptr;
                this->vattCount = this->vattCount + 1;
                return this->vattCount - 1;
            }

            /**
             * Adds a new vertex attribute.
             *
             * @return The index of the newly added attribute, or UINT_MAX in the case of failure
             */
            inline unsigned int AddVertexAttribPointer(float * ptr) {
                if (this->vattCount == MAX_PARAMETER_NUMBER) return UINT_MAX;
                this->vattDTypes[this->vattCount] = DT_FLOAT;
                this->vattVector = this->allocateAdditionalEntry(this->vattVector, this->vattCount);
                this->vattVector[this->vattCount].dataFloat = ptr;
                this->vattCount = this->vattCount + 1;
                return this->vattCount - 1;
            }

            /**
             * Adds a new vertex attribute.
             *
             * @return The index of the newly added attribute, or UINT_MAX in the case of failure
             */
            inline unsigned int AddVertexAttribPointer(int16_t * ptr) {
                if (this->vattCount == MAX_PARAMETER_NUMBER) return UINT_MAX;
                this->vattDTypes[this->vattCount] = DT_INT16;
                this->vattVector = this->allocateAdditionalEntry(this->vattVector, this->vattCount);
                this->vattVector[this->vattCount].dataInt16 = ptr;
                this->vattCount = this->vattCount + 1;
                return this->vattCount - 1;
            }

            /**
             * Adds a new vertex attribute.
             *
             * @return The index of the newly added attribute, or UINT_MAX in the case of failure
             */
            inline unsigned int AddVertexAttribPointer(int32_t * ptr) {
                if (this->vattCount >= MAX_PARAMETER_NUMBER) return UINT_MAX;
                this->vattDTypes[this->vattCount] = DT_INT32;
                this->vattVector = this->allocateAdditionalEntry(this->vattVector, this->vattCount);
                this->vattVector[this->vattCount].dataInt32 = ptr;
                this->vattCount = this->vattCount + 1;
                return this->vattCount - 1;
            }

            /**
             * Adds a new vertex attribute.
             *
             * @return The index of the newly added attribute, or UINT_MAX in the case of failure
             */
            inline unsigned int AddVertexAttribPointer(uint16_t * ptr) {
                if (this->vattCount >= MAX_PARAMETER_NUMBER) return UINT_MAX;
                this->vattDTypes[this->vattCount] = DT_UINT16;
                this->vattVector = this->allocateAdditionalEntry(this->vattVector, this->vattCount);
                this->vattVector[this->vattCount].dataUInt16 = ptr;
                this->vattCount = this->vattCount + 1;
                return this->vattCount - 1;
            }

            /**
             * Adds a new vertex attribute.
             *
             * @return The index of the newly added attribute, or UINT_MAX in the case of failure
             */
            inline unsigned int AddVertexAttribPointer(unsigned int * ptr) {
                if (this->vattCount >= MAX_PARAMETER_NUMBER) return UINT_MAX;
                this->vattDTypes[this->vattCount] = DT_UINT32;
                this->vattVector = this->allocateAdditionalEntry(this->vattVector, this->vattCount);
                this->vattVector[this->vattCount].dataUInt32 = ptr;
                this->vattCount = this->vattCount + 1;
                return this->vattCount - 1;
            }

            /**
            * Answer the data type for vertex attrib
            *
            * @return The data type for vertex attrib
            */
            inline DataType GetVertexAttribDataType(unsigned int attribID) const {
                if (attribID >= this->vattCount) {
                    return DT_NONE;
                }
                return this->vattDTypes[attribID];
            }

            /**
             * Answer the number of vertex attributes.
             *
             * @return The number of vertex attributes.
             */
            inline unsigned int GetVertexAttribCount(void) const {
                return this->vattCount;
            }

            /**
            * Answer if vertex attrib data has been set to non-null
            *
            * @return True if vertex attrib data is present
            */
            inline bool HasVertexAttribPointer(void) const {
                bool hasPointer = (this->vattCount > 0);
                bool hasData = false;
                if (hasPointer) {
                    hasData = hasData || (this->vattVector[0].dataByte != nullptr);
                    hasData = hasData || (this->vattVector[0].dataDouble != nullptr);
                    hasData = hasData || (this->vattVector[0].dataFloat != nullptr);
                    hasData = hasData || (this->vattVector[0].dataInt16 != nullptr);
                    hasData = hasData || (this->vattVector[0].dataInt32 != nullptr);
                    hasData = hasData || (this->vattVector[0].dataUInt16 != nullptr);
                    hasData = hasData || (this->vattVector[0].dataUInt32 != nullptr);
                }
                return hasPointer && hasData;
            }

            /**
             * Gets vertex attrib
             *
             * @return vertex attrib
             */
            inline const uint8_t * GetVertexAttribPointerByte(unsigned int attribID) const {
                if (attribID >= this->vattCount) {
                    return nullptr;
                }
                ASSERT(this->vattDTypes[attribID] == DT_BYTE);
                return this->vattVector[attribID].dataByte;
            }

            /**
             * Gets vertex attrib
             *
             * @return vertex attrib
             */
            inline const double * GetVertexAttribPointerDouble(unsigned int attribID) const {
                if (attribID >= this->vattCount) {
                    return nullptr;
                }
                ASSERT(this->vattDTypes[attribID] == DT_DOUBLE);
                return this->vattVector[attribID].dataDouble;
            }

            /**
             * Gets vertex attrib
             *
             * @return vertex attrib
             */
            inline const float * GetVertexAttribPointerFloat(unsigned int attribID) const {
                if (attribID >= this->vattCount) {
                    return nullptr;
                }
                ASSERT(this->vattDTypes[attribID] == DT_FLOAT);
                return this->vattVector[attribID].dataFloat;
            }

            /**
             * Gets vertex attrib
             *
             * @return vertex attrib
             */
            inline const int16_t * GetVertexAttribPointerInt16(unsigned int attribID) const {
                if (attribID >= this->vattCount) {
                    return nullptr;
                }
                ASSERT(this->vattDTypes[attribID] == DT_INT16);
                return this->vattVector[attribID].dataInt16;
            }

            /**
             * Gets vertex attrib
             *
             * @return vertex attrib
             */
            inline const int * GetVertexAttribPointerInt32(unsigned int attribID) const {
                if (attribID >= this->vattCount) {
                    return nullptr;
                }
                ASSERT(this->vattDTypes[attribID] == DT_INT32);
                return this->vattVector[attribID].dataInt32;
            }

            /**
             * Gets vertex attrib
             *
             * @return vertex attrib
             */
            inline const uint16_t * GetVertexAttribPointerUInt16(unsigned int attribID) const {
                if (attribID >= this->vattCount) {
                    return nullptr;
                }
                ASSERT(this->vattDTypes[attribID] == DT_UINT16);
                return this->vattVector[attribID].dataUInt16;
            }

            /**
             * Gets vertex attrib
             *
             * @return vertex attrib
             */
            inline const unsigned int * GetVertexAttribPointerUInt32(unsigned int attribID) const {
                if (attribID >= this->vattCount) {
                    return nullptr;
                }
                ASSERT(this->vattDTypes[attribID] == DT_UINT32);
                return this->vattVector[attribID].dataUInt32;
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
             *                indices of the vertices used by the triangles; Must not be NULL
             * @param takeOwnership If true the object will take ownership of
             *                      all the memory of the pointers provided
             *                      and will free the memory on its
             *                      destruction. Otherwise the caller must
             *                      ensure the memory stays valid as long as
             *                      it is used.
             */
            template<class Tp>
            inline void SetTriangleData(unsigned int cnt, Tp indices, bool takeOwnership) {
                this->clearTriData();
                this->triCnt = cnt;
                if (cnt > 0) {
                    ASSERT(indices != NULL);
                    this->setTriData(indices);
                    this->triMemOwned = takeOwnership;
                }
            }

            /**
            * Sets the vertex data
            *
            * @param cnt The number of vertices
            * @param vertices Pointer to 3 times cnt floats holding the vertices; Must not be NULL
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
            template<class Tp1, class Tp2, class Tp3, class Tp4>
            inline void SetVertexData(unsigned int cnt,
                Tp1 vertices, Tp2 normals, Tp3 colours, Tp4 textureCoordinates,
                bool takeOwnership) {
                this->clearVrtData();
                this->vrtCnt = cnt;
                if (cnt > 0) {
                    ASSERT(vertices != NULL);
                    this->setVrtData(vertices);
                    this->setNrmData(normals);
                    this->setColData(colours);
                    this->setTexData(textureCoordinates);
                    this->vrtMemOwned = takeOwnership;
                }
            }

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
             * Sets the vertex attrib data pointer
             * for a specific attribute.
             * If the attribute is not present, nothing is changed...
             *
             * @param v The new pointer value
             */
            inline void SetVertexAttribData(uint8_t *v, unsigned int attribID) {
                if (attribID >= this->vattCount) return;
                this->vattDTypes[attribID] = DT_BYTE;
                this->vattVector[attribID].dataByte = v;
            }

            /**
             * Sets the vertex attrib data pointer
             * for a specific attribute.
             * If the attribute is not present, nothing is changed...
             *
             * @param v The new pointer value
             */
            inline void SetVertexAttribData(double *v, unsigned int attribID) {
                if (attribID >= this->vattCount) return;
                this->vattDTypes[attribID] = DT_DOUBLE;
                this->vattVector[attribID].dataDouble = v;
            }

            /**
             * Sets the vertex attrib data pointer
             * for a specific attribute.
             * If the attribute is not present, nothing is changed...
             *
             * @param v The new pointer value
             */
            inline void SetVertexAttribData(float *v, unsigned int attribID) {
                if (attribID >= this->vattCount) return;
                this->vattDTypes[attribID] = DT_FLOAT;
                this->vattVector[attribID].dataFloat = v;
            }

            /**
             * Sets the vertex attrib data pointer
             * for a specific attribute.
             * If the attribute is not present, nothing is changed...
             *
             * @param v The new pointer value
             */
            inline void SetVertexAttribData(int16_t *v, unsigned int attribID) {
                if (attribID >= this->vattCount) return;
                this->vattDTypes[attribID] = DT_INT16;
                this->vattVector[attribID].dataInt16 = v;
            }

            /**
             * Sets the vertex attrib data pointer
             * for a specific attribute.
             * If the attribute is not present, nothing is changed...
             *
             * @param v The new pointer value
             */
            inline void SetVertexAttribData(int *v, unsigned int attribID) {
                if (attribID >= this->vattCount) return;
                this->vattDTypes[attribID] = DT_INT32;
                this->vattVector[attribID].dataInt32 = v;
            }

            /**
             * Sets the vertex attrib data pointer
             * for a specific attribute.
             * If the attribute is not present, nothing is changed...
             *
             * @param v The new pointer value
             */
            inline void SetVertexAttribData(uint16_t *v, unsigned int attribID) {
                if (attribID >= this->vattCount) return;
                this->vattDTypes[attribID] = DT_UINT16;
                this->vattVector[attribID].dataUInt16 = v;
            }

            /**
             * Sets the vertex attrib data pointer
             * for a specific attribute.
             * If the attribute is not present, nothing is changed...
             *
             * @param v The new pointer value
             */
            inline void SetVertexAttribData(unsigned int *v, unsigned int attribID) {
                if (attribID >= this->vattCount) return;
                this->vattDTypes[attribID] = DT_UINT32;
                this->vattVector[attribID].dataUInt32 = v;
            }

            /**
             * Assignment operator
             *
             * @param rhs The right hand side operand
             *
             * @return Reference to this
             */
            Mesh& operator=(const Mesh& rhs);

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

            /**
             * Clears all triangle index data
             */
            void clearTriData(void);

            /**
             * Clears all vertex data
             */
            void clearVrtData(void);

            /**
             * Sets the triangle data pointer
             *
             * @param v The new pointer value
             */
            inline void setTriData(unsigned char *v) {
                this->triDT = DT_BYTE;
                this->tri.dataByte = v;
            }

            /**
             * Sets the triangle data pointer
             *
             * @param v The new pointer value
             */
            inline void setTriData(unsigned short *v) {
                this->triDT = DT_UINT16;
                this->tri.dataUInt16 = v;
            }

            /**
             * Sets the triangle data pointer
             *
             * @param v The new pointer value
             */
            inline void setTriData(unsigned int *v) {
                this->triDT = DT_UINT32;
                this->tri.dataUInt32 = v;
            }

            /**
             * Sets the triangle data pointer
             *
             * @param v The new pointer value
             */
            template<class Tp>
            inline void setTriData(Tp v) {
                ASSERT(v == NULL);
                this->triDT = DT_NONE;
                this->tri.dataUInt32 = NULL;
            }

            /**
             * Sets the vertex data pointer
             *
             * @param v The new pointer value
             */
            inline void setVrtData(float *v) {
                this->vrtDT = DT_FLOAT;
                this->vrt.dataFloat = v;
            }

            /**
             * Sets the vertex data pointer
             *
             * @param v The new pointer value
             */
            inline void setVrtData(double *v) {
                this->vrtDT = DT_DOUBLE;
                this->vrt.dataDouble = v;
            }

            /**
             * Sets the vertex data pointer
             *
             * @param v The new pointer value
             */
            template<class Tp>
            inline void setVrtData(Tp v) {
                ASSERT(v == NULL);
                this->vrtDT = DT_NONE;
                this->vrt.dataDouble = NULL;
            }

            /**
             * Sets the normal data pointer
             *
             * @param v The new pointer value
             */
            inline void setNrmData(float *v) {
                this->nrmDT = (v == NULL) ? DT_NONE : DT_FLOAT;
                this->nrm.dataFloat = v;
            }

            /**
             * Sets the normal data pointer
             *
             * @param v The new pointer value
             */
            inline void setNrmData(double *v) {
                this->nrmDT = (v == NULL) ? DT_NONE : DT_DOUBLE;
                this->nrm.dataDouble = v;
            }

            /**
             * Sets the normal data pointer
             *
             * @param v The new pointer value
             */
            template<class Tp>
            inline void setNrmData(Tp v) {
                ASSERT(v == NULL);
                this->nrmDT = DT_NONE;
                this->nrm.dataDouble = NULL;
            }

            /**
             * Sets the colour data pointer
             *
             * @param v The new pointer value
             */
            inline void setColData(unsigned char *v) {
                this->colDT = (v == NULL) ? DT_NONE : DT_BYTE;
                this->col.dataByte = v;
            }

            /**
             * Sets the colour data pointer
             *
             * @param v The new pointer value
             */
            inline void setColData(float *v) {
                this->colDT = (v == NULL) ? DT_NONE : DT_FLOAT;
                this->col.dataFloat = v;
            }

            /**
             * Sets the colour data pointer
             *
             * @param v The new pointer value
             */
            inline void setColData(double *v) {
                this->colDT = (v == NULL) ? DT_NONE : DT_DOUBLE;
                this->col.dataDouble = v;
            }

            /**
             * Sets the colour data pointer
             *
             * @param v The new pointer value
             */
            template<class Tp>
            inline void setColData(Tp v) {
                ASSERT(v == NULL);
                this->colDT = DT_NONE;
                this->col.dataDouble = NULL;
            }

            /**
             * Sets the texture coordinate data pointer
             *
             * @param v The new pointer value
             */
            inline void setTexData(float *v) {
                this->texDT = (v == NULL) ? DT_NONE : DT_FLOAT;
                this->tex.dataFloat = v;
            }

            /**
             * Sets the texture coordinate data pointer
             *
             * @param v The new pointer value
             */
            inline void setTexData(double *v) {
                this->texDT = (v == NULL) ? DT_NONE : DT_DOUBLE;
                this->tex.dataDouble = v;
            }

            /**
             * Sets the texture coordinate data pointer
             *
             * @param v The new pointer value
             */
            template<class Tp>
            inline void setTexData(Tp v) {
                ASSERT(v == NULL);
                this->texDT = DT_NONE;
                this->tex.dataDouble = NULL;
            }

            /**
            * Sets the triangle data pointer
            *
            * @param v The new pointer value
            */
            template<class Tp>
            inline void setVertexAttribData(Tp v, unsigned int attribID) {
                ASSERT(v == NULL);
                if (attribID >= this->vattCount) return;
                this->vattDTypes[attribID] = DT_NONE;
                this->vattVector[attribID].dataUInt32 = NULL;
            }

            /**
             * Reallocates an array containing an additional entry
             *
             * @param arrayPtr The array pointer.
             * @param oldSize The number of elements of the input array. 
             * @return New array pointer for an array one element bigger.
             *         Ideally, this pointer is assigned to the old one.
             */
            template<class Tp>
            inline Tp * allocateAdditionalEntry(Tp * arrayPtr, unsigned int oldSize) {
                Tp * result = new Tp[oldSize + 1];
                std::memcpy(result, arrayPtr, oldSize * sizeof(Tp));
                delete[] arrayPtr;
                arrayPtr = nullptr;
                return result;
            }

            /** Triangle count (ignored if t is NULL) */
            unsigned int triCnt;

            /** Data type for triangle index data */
            DataType triDT;

            /** Triangle vertex indices (3 times tc) or NULL */
            union _tri_t {
                unsigned char * dataByte;
                unsigned short * dataUInt16;
                unsigned int * dataUInt32;
            } tri;

            /**
             * Flag indicating the if the triangle index data memory is owned
             * by this object and will be freed on its destruction
             */
            bool triMemOwned;

            /** Vertex count */
            unsigned int vrtCnt;

            /** The vertex data type */
            DataType vrtDT;

            /** Vertices (3 times vc) */
            union _vrt_t {
                float * dataFloat;
                double * dataDouble;
            } vrt;

            /** The normal data type */
            DataType nrmDT;

            /** Normals (3 times vc) */
            union _nrm_t {
                float * dataFloat;
                double * dataDouble;
            } nrm;

            /** The colour data type */
            DataType colDT;

            /** Colors (3 times vc) */
            union _col_t {
                unsigned char * dataByte;
                float * dataFloat;
                double * dataDouble;
            } col;

            /** The texture coordinates data type */
            DataType texDT;

            /** Texture coordinates (2 times vc) */
            union _tex_t {
                float * dataFloat;
                double * dataDouble;
            } tex;

            /** The vertex attrib data types */            
            DataType vattDTypes[MAX_PARAMETER_NUMBER];

            /** Vertex attrib data possibilities */
            union _vatt_t {
                unsigned int * dataUInt32;
                int * dataInt32;
                uint16_t * dataUInt16;
                int16_t * dataInt16;
                uint8_t * dataByte;
                float * dataFloat;
                double * dataDouble;
            };

            /** Vector with pointers to all attributes */
            _vatt_t * vattVector;

            /** Count of currently present vertex attributes */
            unsigned int vattCount;

            /**
             * Flag indicating the if the vertex data memory is owned by this
             * object and will be freed on its destruction
             */
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
    typedef megamol::core::factories::CallAutoDescription<CallTriMeshData> CallTriMeshDataDescription;

} /* end namespace geocalls */
} /* end namespace megamol */

#endif /* MEGAMOL_GEOMETRY_CALLS_CALLTRIMESHDATA_H_INCLUDED */
