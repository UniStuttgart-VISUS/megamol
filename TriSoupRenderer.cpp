/*
 * TriSoupRenderer.cpp
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2008-2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "TriSoupRenderer.h"
#include "CallTriMeshData.h"
#include "param/BoolParam.h"
#include "param/EnumParam.h"
#include "view/CallRender3D.h"
#include <GL/gl.h>


#include "param/ButtonParam.h"
#include "param/StringParam.h"
#include "param/FilePathParam.h"
#include "vislib/KeyCode.h"
#include "vislib/Log.h"
#include "vislib/mathfunctions.h"
#include "vislib/MemmappedFile.h"
#include "vislib/ShallowPoint.h"
#include "vislib/Vector.h"

using namespace megamol;
using namespace megamol::trisoup;



using namespace megamol::core;


/*
 * TriSoupRenderer::TriSoupRenderer
 */
TriSoupRenderer::TriSoupRenderer(void) : Renderer3DModule(),
        getDataSlot("getData", "The slot to fetch the tri-mesh data"),
        showVertices("showVertices", "Flag whether to show the verices of the object"),
        lighting("lighting", "Flag whether or not use lighting for the surface"),
        cullface("cullface", "Flag whether or not use back-face culling for the surface"),
        surStyle("style", "The rendering style for the surface") {

    this->getDataSlot.SetCompatibleCall<CallTriMeshDataDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->showVertices.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->showVertices);

    this->lighting.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->lighting);

    this->cullface.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->cullface);

    param::EnumParam *ep = new param::EnumParam(0);
    ep->SetTypePair(0, "Filled");
    ep->SetTypePair(1, "Wireframe");
    this->surStyle << ep;
    this->MakeSlotAvailable(&this->surStyle);
}


/*
 * TriSoupRenderer::~TriSoupRenderer
 */
TriSoupRenderer::~TriSoupRenderer(void) {
    this->Release();
}


/*
 * TriSoupRenderer::create
 */
bool TriSoupRenderer::create(void) {
    // intentionally empty
    return true;
}


/*
 * TriSoupRenderer::GetCapabilities
 */
bool TriSoupRenderer::GetCapabilities(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(view::CallRender3D::CAP_RENDER | view::CallRender3D::CAP_LIGHTING);

    return true;
}


/*
 * TriSoupRenderer::GetExtents
 */
bool TriSoupRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;
    CallTriMeshData *ctmd = this->getDataSlot.CallAs<CallTriMeshData>();
    if (ctmd == NULL) return false;
    if (!(*ctmd)(1)) return false;

    cr->SetTimeFramesCount(1);
    cr->AccessBoundingBoxes().Clear();
    cr->AccessBoundingBoxes() = ctmd->AccessBoundingBoxes();
    float scale = ctmd->AccessBoundingBoxes().ClipBox().LongestEdge();
    if (scale > 0.0f) scale = 2.0f / scale;
    cr->AccessBoundingBoxes().MakeScaledWorld(scale);

    return true;
}


/*
 * TriSoupRenderer::release
 */
void TriSoupRenderer::release(void) {
    // intentionally empty
}


/*
 * TriSoupRenderer::Render
 */
bool TriSoupRenderer::Render(Call& call) {
    CallTriMeshData *ctmd = this->getDataSlot.CallAs<CallTriMeshData>();
    if (ctmd == NULL) return false;

    if (!(*ctmd)(1)) return false;
    float scale = ctmd->AccessBoundingBoxes().ClipBox().LongestEdge();
    if (scale > 0.0f) scale = 2.0f / scale;
    ::glScalef(scale, scale, scale);

    if (!(*ctmd)(0)) return false;

    bool normals = false;
    bool colors = false;
    bool textures = false;

    ::glEnable(GL_DEPTH_TEST);
    if (this->lighting.Param<param::BoolParam>()->Value()) {
        ::glEnable(GL_LIGHTING);
    } else {
        ::glDisable(GL_LIGHTING);
    }
    ::glDisable(GL_BLEND);
    ::glEnableClientState(GL_VERTEX_ARRAY);
    ::glDisableClientState(GL_NORMAL_ARRAY);
    ::glDisableClientState(GL_COLOR_ARRAY);
    ::glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    if (this->cullface.Param<param::BoolParam>()->Value()) {
        ::glEnable(GL_CULL_FACE);
    } else {
        ::glDisable(GL_CULL_FACE);
    }
    ::glEnable(GL_COLOR_MATERIAL);
    ::glEnable(GL_TEXTURE_2D);
    ::glBindTexture(GL_TEXTURE_2D, 0);
    ::glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    ::glEnable(GL_NORMALIZE);

    if (this->surStyle.Param<param::EnumParam>()->Value() == 1) {
        ::glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }

    ::glColor3f(1.0f, 1.0f, 1.0f);
    for (unsigned int i = 0; i < ctmd->Count(); i++) {
        const CallTriMeshData::Mesh& obj = ctmd->Objects()[i];

        ::glVertexPointer(3, GL_FLOAT, 0, obj.GetVertexPointer());
        if (obj.GetNormalPointer() != NULL) {
            if (!normals) {
                ::glEnableClientState(GL_NORMAL_ARRAY);
                normals = true;
            }
            ::glNormalPointer(GL_FLOAT, 0, obj.GetNormalPointer());
        } else if (normals) {
            ::glDisableClientState(GL_NORMAL_ARRAY);
            normals = false;
        }
        if (obj.GetColourPointer() != NULL) {
            if (!colors) {
                ::glEnableClientState(GL_COLOR_ARRAY);
                colors = true;
            }
            ::glColorPointer(3, GL_UNSIGNED_BYTE, 0, obj.GetColourPointer());
        } else if (colors) {
            ::glDisableClientState(GL_COLOR_ARRAY);
            colors = false;
        }
        if (obj.GetTextureCoordinatePointer() != NULL) {
            if (!textures) {
                ::glEnableClientState(GL_TEXTURE_COORD_ARRAY);
                textures = true;
            }
            ::glTexCoordPointer(2, GL_FLOAT, 0, obj.GetTextureCoordinatePointer());
        } else if (textures) {
            ::glDisableClientState(GL_TEXTURE_COORD_ARRAY);
            textures = false;
        }

        if (obj.GetMaterial() != NULL) {
            const CallTriMeshData::Material &mat = *obj.GetMaterial();

            ::glDisable(GL_COLOR_MATERIAL);
            GLfloat mat_ambient[4] = { mat.GetKa()[0], mat.GetKa()[1], mat.GetKa()[2], 1.0f };
            GLfloat mat_diffuse[4] = { mat.GetKd()[0], mat.GetKd()[1], mat.GetKd()[2], 1.0f };
            GLfloat mat_specular[4] = { mat.GetKs()[0], mat.GetKs()[1], mat.GetKs()[2], 1.0f };
            GLfloat mat_emission[4] = { mat.GetKe()[0], mat.GetKe()[1], mat.GetKe()[2], 1.0f };
            GLfloat mat_shininess[1] = { mat.GetNs() };
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, mat_emission);
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);

            GLuint mapid = mat.GetMapID();
            if (mapid > 0) {
                //::glActiveTexture(GL_TEXTURE0);
                ::glEnable(GL_COLOR_MATERIAL);
                ::glBindTexture(GL_TEXTURE_2D, mapid);
                ::glEnable(GL_TEXTURE_2D);
                ::glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                ::glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            }
        } else {
            GLfloat mat_ambient[4] = { 0.2f, 0.2f, 0.2f, 1.0f };
            GLfloat mat_diffuse[4] = { 0.8f, 0.8f, 0.8f, 1.0f };
            GLfloat mat_specular[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
            GLfloat mat_emission[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
            GLfloat mat_shininess[1] = { 0.0f };
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, mat_emission);
            ::glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
            ::glBindTexture(GL_TEXTURE_2D, 0);
            ::glEnable(GL_COLOR_MATERIAL);

        }

        if (obj.GetTriIndexPointer() != NULL) {
            ::glDrawElements(GL_TRIANGLES, obj.GetTriCount() * 3, GL_UNSIGNED_INT, obj.GetTriIndexPointer());
			//::glDrawElements(GL_TRIANGLES, obj.GetTriCount(), GL_UNSIGNED_INT, obj.GetTriIndexPointer());
        } else {
            ::glDrawArrays(GL_TRIANGLES, 0, obj.GetVertexCount());
        }
    }
    if (normals) ::glDisableClientState(GL_NORMAL_ARRAY);
    if (colors) ::glDisableClientState(GL_COLOR_ARRAY);
    if (textures) ::glDisableClientState(GL_TEXTURE_COORD_ARRAY);

    {
        GLfloat mat_ambient[4] = { 0.2f, 0.2f, 0.2f, 1.0f };
        GLfloat mat_diffuse[4] = { 0.8f, 0.8f, 0.8f, 1.0f };
        GLfloat mat_specular[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
        GLfloat mat_emission[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
        GLfloat mat_shininess[1] = { 0.0f };
        ::glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
        ::glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
        ::glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
        ::glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, mat_emission);
        ::glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
        ::glBindTexture(GL_TEXTURE_2D, 0);
    }

    ::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if (this->showVertices.Param<param::BoolParam>()->Value()) {
        ::glEnable(GL_POINT_SIZE);
        ::glPointSize(3.0f);
        ::glDisable(GL_LIGHTING);

        ::glColor3f(1.0f, 0.0f, 0.0f);
        for (unsigned int i = 0; i < ctmd->Count(); i++) {
            ::glVertexPointer(3, GL_FLOAT, 0, ctmd->Objects()[i].GetVertexPointer());
            ::glDrawArrays(GL_POINTS, 0, ctmd->Objects()[i].GetVertexCount());
        }
    }

    ::glEnable(GL_CULL_FACE);
    ::glDisableClientState(GL_VERTEX_ARRAY);
    ::glDisable(GL_POINT_SIZE);
    ::glEnable(GL_BLEND);

    return true;
}
