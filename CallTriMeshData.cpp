/*
 * CallTriMeshData.cpp
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "CallTriMeshData.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include "vislib/BitmapCodecCollection.h"
#include "vislib/BitmapImage.h"
#include "vislib/Exception.h"
#include "vislib/Log.h"
#include "vislib/UnsupportedOperationException.h"

using namespace megamol::trisoup;


/*
 * CallTriMeshData::Material::Material
 */
CallTriMeshData::Material::Material(void) : Ns(0.0f), Ni(0.0f), d(0.0f),
        Tr(0.0f), illum(ILLUM_DIFF_SPEC), mapFileName(), bumpMapFileName(),
        mapID(0), bumpMapID(0) {
    this->Tf[0] = this->Tf[1] = this->Tf[2] = 0.0f;
    this->Ka[0] = this->Ka[1] = this->Ka[2] = 0.2f;
    this->Kd[0] = this->Kd[1] = this->Kd[2] = 0.8f;
    this->Ks[0] = this->Ks[1] = this->Ks[2] = 0.0f;
    this->Ke[0] = this->Ke[1] = this->Ke[2] = 0.0f;
}


/*
 * CallTriMeshData::Material::~Material
 */
CallTriMeshData::Material::~Material(void) {
    if (this->mapID != 0) {
        ::glDeleteTextures(1, &this->mapID);
        this->mapID = 0;
    }
    if (this->bumpMapID != 0) {
        ::glDeleteTextures(1, &this->bumpMapID);
        this->bumpMapID = 0;
    }
}


/*
 * CallTriMeshData::Material::GetMapID
 */
unsigned int CallTriMeshData::Material::GetMapID(void) const {
    if ((this->mapID == 0) && !this->mapFileName.IsEmpty()) {
        this->mapID = this->loadTexture(this->mapFileName);
    }
    return this->mapID;
}


/*
 * CallTriMeshData::Material::GetBumpMapID
 */
unsigned int CallTriMeshData::Material::GetBumpMapID(void) const {
    if ((this->bumpMapID == 0) && !this->bumpMapFileName.IsEmpty()) {
        this->bumpMapID = this->loadTexture(this->bumpMapFileName);
    }
    return this->bumpMapID;
}


/*
 * CallTriMeshData::Material::SetMapFileName
 */
void CallTriMeshData::Material::SetMapFileName(const vislib::TString& filename) {
    if (this->mapFileName.Equals(filename)) return;
    this->mapFileName = filename;
    if (this->mapID != 0) {
        ::glDeleteTextures(1, &this->mapID);
        this->mapID = 0;
    }
}


/*
 * CallTriMeshData::Material::SetBumpMapFileName
 */
void CallTriMeshData::Material::SetBumpMapFileName(const vislib::TString& filename) {
    if (this->bumpMapFileName.Equals(filename)) return;
    this->bumpMapFileName = filename;
    if (this->bumpMapID != 0) {
        ::glDeleteTextures(1, &this->bumpMapID);
        this->bumpMapID = 0;
    }
}


/*
 * CallTriMeshData::Material::operator==
 */
bool CallTriMeshData::Material::operator==(const CallTriMeshData::Material& rhs) const {
    // epsilon comparisons not required, since we want to check for *exact equality*
    return this->bumpMapFileName.Equals(rhs.bumpMapFileName)
        // Not checking texture ids!
        && (this->d == rhs.d)
        && (this->illum == rhs.illum)
        && (::memcmp(this->Ka, rhs.Ka, sizeof(float) * 3) == 0)
        && (::memcmp(this->Kd, rhs.Kd, sizeof(float) * 3) == 0)
        && (::memcmp(this->Ke, rhs.Ke, sizeof(float) * 3) == 0)
        && (::memcmp(this->Ks, rhs.Ks, sizeof(float) * 3) == 0)
        && this->mapFileName.Equals(rhs.mapFileName)
        // Not checking texture ids!
        && (this->Ni == rhs.Ni)
        && (this->Ns == rhs.Ns)
        && (::memcmp(this->Tf, rhs.Tf, sizeof(float) * 3) == 0)
        && (this->Tr == rhs.Tr);
}


/*
 * CallTriMeshData::Material::loadTexture
 */
unsigned int CallTriMeshData::Material::loadTexture(vislib::TString &filename) {
    vislib::graphics::BitmapImage img;
    try {
        if (!vislib::graphics::BitmapCodecCollection::DefaultCollection().LoadBitmapImage(img, filename)) {
            throw vislib::Exception("No suitable codec found", __FILE__, __LINE__);
        }
    } catch(vislib::Exception ex) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to load texture \"%s\": %s (%s, %d)\n", vislib::StringA(filename).PeekBuffer(),
            ex.GetMsgA(), ex.GetFile(), ex.GetLine());
        filename.Clear();
        return 0;
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to load texture \"%s\": unexpected exception\n", vislib::StringA(filename).PeekBuffer());
        filename.Clear();
        return 0;
    }

    bool hasAlpha = false;
    for (unsigned int i = 0; i < img.GetChannelCount(); i++) {
        if (img.GetChannelLabel(i) == vislib::graphics::BitmapImage::CHANNEL_ALPHA) {
            hasAlpha = true;
            break;
        }
    }
    img.Convert(hasAlpha ? vislib::graphics::BitmapImage::TemplateByteRGBA
        : vislib::graphics::BitmapImage::TemplateByteRGB);
    img.FlipVertical();

    GLuint tex;
    ::glGenTextures(1, &tex);
    ::glBindTexture(GL_TEXTURE_2D, tex);
    ::glPixelStorei(GL_PACK_ALIGNMENT, 1);
    ::glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    ::glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    ::gluBuild2DMipmaps(GL_TEXTURE_2D, hasAlpha ? 4 : 3,
        img.Width(), img.Height(),
        hasAlpha ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE,
        img.PeekData());

    return tex;
}


/*
 * CallTriMeshData::Material::Material
 */
CallTriMeshData::Material::Material(const CallTriMeshData::Material& src) {
    throw vislib::UnsupportedOperationException("CallTriMeshData::Material Copy-Ctor", __FILE__, __LINE__);
}


/*
 * CallTriMeshData::Mesh::Mesh
 */
CallTriMeshData::Mesh::Mesh(void) : triCnt(0), tri(NULL), triMemOwned(false),
        vrtCnt(0), vrt(NULL), nrm(NULL), col(NULL), tex(NULL), vrtMemOwned(false), mat(NULL) {
    // intentionally empty
}


/*
 * CallTriMeshData::Mesh::~Mesh
 */
CallTriMeshData::Mesh::~Mesh(void) {
    this->triCnt = 0;
    if (this->triMemOwned) {
        SAFE_DELETE(this->tri);
    }
    this->tri = NULL;
    this->vrtCnt = 0;
    if (this->vrtMemOwned) {
        SAFE_DELETE(this->vrt);
        SAFE_DELETE(this->nrm);
        SAFE_DELETE(this->col);
        SAFE_DELETE(this->tex);
    }
    this->vrt = NULL;
    this->nrm = NULL;
    this->col = NULL;
    this->tex = NULL;
    this->mat = NULL; // DO NOT DELETE
}


/*
 * CallTriMeshData::Mesh::SetTriangleData
 */
void CallTriMeshData::Mesh::SetTriangleData(unsigned int cnt, unsigned int *indices, bool takeOwnership) {
    if (this->triMemOwned) {
        SAFE_DELETE(this->tri);
    }
    this->triCnt = cnt;
    this->tri = indices;
    this->triMemOwned = takeOwnership;
}


/*
 * CallTriMeshData::Mesh::SetVertexData
 */
void CallTriMeshData::Mesh::SetVertexData(unsigned int cnt, float *vertices, float *normals, unsigned char *colours, float *textureCoordinates, bool takeOwnership) {
    if (this->vrtMemOwned) {
        SAFE_DELETE(this->vrt);
        SAFE_DELETE(this->nrm);
        SAFE_DELETE(this->col);
        SAFE_DELETE(this->tex);
    }
    this->vrtCnt = cnt;
    this->vrt = vertices;
    this->nrm = normals;
    this->col = colours;
    this->tex = textureCoordinates;
    this->vrtMemOwned = takeOwnership;
}


/*
 * CallTriMeshData::Mesh::operator==
 */
bool CallTriMeshData::Mesh::operator==(const CallTriMeshData::Mesh& rhs) const {
    return (this->col == rhs.col)
        && (this->mat == rhs.mat)
        && (this->nrm == rhs.nrm)
        && (this->tex == rhs.tex)
        && (this->tri == rhs.tri)
        && (this->triCnt == rhs.triCnt)
        && (this->triMemOwned == rhs.triMemOwned)
        && (this->vrt == rhs.vrt)
        && (this->vrtCnt == rhs.vrtCnt)
        && (this->vrtMemOwned == rhs.vrtMemOwned);
}


/*
 * CallTriMeshData::Mesh::Mesh
 */
CallTriMeshData::Mesh::Mesh(const CallTriMeshData::Mesh& src) {
    throw vislib::UnsupportedOperationException("CallTriMeshData::Mesh Copy-Ctor", __FILE__, __LINE__);
}


/*
 * CallTriMeshData::CallTriMeshData
 */
CallTriMeshData::CallTriMeshData(void) : core::AbstractGetData3DCall(), objCnt(0), objs(NULL) {
}


/*
 * CallTriMeshData::~CallTriMeshData
 */
CallTriMeshData::~CallTriMeshData(void) {
    this->objCnt = 0;
    this->objs = NULL; // DO NOT DELETE
}
