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

/****************************************************************************/


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
 * CallTriMeshData::Material::Dye
 */
void CallTriMeshData::Material::Dye(float r, float g, float b) {
    this->Ka[0] *= r; this->Ka[1] *= g; this->Ka[2] *= b;
    this->Kd[0] *= r; this->Kd[1] *= g; this->Kd[2] *= b;
    this->Ks[0] *= r; this->Ks[1] *= g; this->Ks[2] *= b;
    this->Ke[0] *= r; this->Ke[1] *= b; this->Ke[2] *= b;
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
 * CallTriMeshData::Material::MakeDefault
 */
void CallTriMeshData::Material::MakeDefault(void) {
    this->Ns = 0.0f;
    this->Ni = 0.0f;
    this->d = 0.0f;
    this->Tr = 0.0f;
    this->illum = ILLUM_DIFF_SPEC;
    this->mapFileName.Clear();
    this->bumpMapFileName.Clear();
    if (this->mapID != 0) {
        ::glDeleteTextures(1, &this->mapID);
        this->mapID = 0;
    }
    if (this->bumpMapID != 0) {
        ::glDeleteTextures(1, &this->bumpMapID);
        this->bumpMapID = 0;
    }
    this->Tf[0] = this->Tf[1] = this->Tf[2] = 0.0f;
    this->Ka[0] = this->Ka[1] = this->Ka[2] = 0.2f;
    this->Kd[0] = this->Kd[1] = this->Kd[2] = 0.8f;
    this->Ks[0] = this->Ks[1] = this->Ks[2] = 0.0f;
    this->Ke[0] = this->Ke[1] = this->Ke[2] = 0.0f;
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

/****************************************************************************/


/*
 * CallTriMeshData::Mesh::Mesh
 */
CallTriMeshData::Mesh::Mesh(void) : triCnt(0), triDT(DT_NONE), /*tri(NULL), */triMemOwned(false),
        vrtCnt(0), vrtDT(DT_NONE), /*vrt(NULL), */nrmDT(DT_NONE), /*nrm(NULL), */
        colDT(DT_NONE), /*col(NULL), */texDT(DT_NONE), /*tex(NULL), */vrtMemOwned(false), mat(NULL) {
    this->tri.dataByte = NULL;
    this->vrt.dataFloat = NULL;
    this->nrm.dataFloat = NULL;
    this->col.dataByte = NULL;
    this->tex.dataFloat = NULL;
}


/*
 * CallTriMeshData::Mesh::~Mesh
 */
CallTriMeshData::Mesh::~Mesh(void) {
    this->clearTriData();
    this->clearVrtData();
    this->mat = NULL; // DO NOT DELETE

}


/*
 * CallTriMeshData::Mesh::operator==
 */
bool CallTriMeshData::Mesh::operator==(const CallTriMeshData::Mesh& rhs) const {
    return (this->col.dataByte == rhs.col.dataByte)
        && (this->colDT == rhs.colDT)
        && (this->mat == rhs.mat)
        && (this->nrm.dataFloat == rhs.nrm.dataFloat)
        && (this->nrmDT == rhs.nrmDT)
        && (this->tex.dataFloat == rhs.tex.dataFloat)
        && (this->texDT == rhs.texDT)
        && (this->tri.dataByte == rhs.tri.dataByte)
        && (this->triDT == rhs.triDT)
        && (this->triCnt == rhs.triCnt)
        && (this->triMemOwned == rhs.triMemOwned)
        && (this->vrt.dataFloat == rhs.vrt.dataFloat)
        && (this->vrtDT == rhs.vrtDT)
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
 * CallTriMeshData::Mesh::clearTriData
 */
void CallTriMeshData::Mesh::clearTriData(void) {
    this->triCnt = 0;
    if (this->triMemOwned) {
        if (this->tri.dataByte != NULL) {
            switch (this->triDT) {
                case DT_BYTE: delete[] this->tri.dataByte; break;
                case DT_UINT16: delete[] this->tri.dataUInt16; break;
                case DT_UINT32: delete[] this->tri.dataUInt32; break;
                default: ASSERT(false);
            }
        }
    }
    this->triDT = DT_NONE;
    this->tri.dataByte = NULL;
}


/*
 * CallTriMeshData::Mesh::clearVrtData
 */
void CallTriMeshData::Mesh::clearVrtData(void) {
    this->vrtCnt = 0;
    if (this->vrtMemOwned) {
        if (this->vrt.dataFloat != NULL) {
            switch (this->vrtDT) {
                case DT_FLOAT: delete[] this->vrt.dataFloat; break;
                case DT_DOUBLE: delete[] this->vrt.dataDouble; break;
                default: ASSERT(false);
            }
        }
        if (this->nrm.dataFloat != NULL) {
            switch (this->nrmDT) {
                case DT_FLOAT: delete[] this->nrm.dataFloat; break;
                case DT_DOUBLE: delete[] this->nrm.dataDouble; break;
                default: ASSERT(false);
            }
        }
        if (this->col.dataByte != NULL) {
            switch (this->colDT) {
                case DT_BYTE: delete[] this->col.dataByte; break;
                case DT_FLOAT: delete[] this->col.dataFloat; break;
                case DT_DOUBLE: delete[] this->col.dataDouble; break;
                default: ASSERT(false);
            }
        }
        if (this->tex.dataFloat != NULL) {
            switch (this->texDT) {
                case DT_FLOAT: delete[] this->tex.dataFloat; break;
                case DT_DOUBLE: delete[] this->tex.dataDouble; break;
                default: ASSERT(false);
            }
        }
    }
    this->vrtDT = DT_NONE;
    this->vrt.dataFloat = NULL;
    this->nrmDT = DT_NONE;
    this->nrm.dataFloat = NULL;
    this->colDT = DT_NONE;
    this->col.dataByte = NULL;
    this->texDT = DT_NONE;
    this->tex.dataFloat = NULL;
}

/****************************************************************************/


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
