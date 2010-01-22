/*
 * TriSoupRenderer.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "TriSoupRenderer.h"
#include "param/BoolParam.h"
#include "param/ButtonParam.h"
#include "view/CallRender3D.h"
#include "param/EnumParam.h"
#include "param/StringParam.h"
#include "vislib/KeyCode.h"
#include "vislib/Log.h"
#include "vislib/mathfunctions.h"
#include "vislib/MemmappedFile.h"
#include "vislib/ShallowPoint.h"
#include "vislib/Vector.h"
#include <GL/gl.h>

using namespace megamol::core;


/*
 * misc::TriSoupRenderer::TriSoupRenderer
 */
misc::TriSoupRenderer::TriSoupRenderer(void) : Renderer3DModule(),
        filename("filename", "The path to the trisoup file to load."), cnt(0),
        clusters(NULL), showVertices("showVertices", "Flag whether to show the verices of the object"),
        lighting("lighting", "Flag whether or not use lighting for the surface"),
        surStyle("style", "The rendering style for the surface"),
        testButton("test", "A test button for button testing"), bbox() {

    this->filename.SetParameter(new param::StringParam(""));
    this->MakeSlotAvailable(&this->filename);

    this->showVertices.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->showVertices);

    this->lighting.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->lighting);

    param::EnumParam *ep = new param::EnumParam(0);
    ep->SetTypePair(0, "Filled");
    ep->SetTypePair(1, "Wireframe");
    this->surStyle << ep;
    this->MakeSlotAvailable(&this->surStyle);

    this->testButton << new param::ButtonParam(vislib::sys::KeyCode('t'));
    this->testButton.SetUpdateCallback(this, &TriSoupRenderer::bullchit);
    this->MakeSlotAvailable(&this->testButton);

    this->bbox.Set(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);

}


/*
 * misc::TriSoupRenderer::~TriSoupRenderer
 */
misc::TriSoupRenderer::~TriSoupRenderer(void) {
    this->Release();
    this->cnt = 0;
    ARY_SAFE_DELETE(this->clusters);
}


/*
 * misc::TriSoupRenderer::create
 */
bool misc::TriSoupRenderer::create(void) {
    this->tryLoadFile();
    this->filename.ResetDirty();
    return true;
}


/*
 * misc::TriSoupRenderer::GetCapabilities
 */
bool misc::TriSoupRenderer::GetCapabilities(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(
        view::CallRender3D::CAP_RENDER
        | view::CallRender3D::CAP_LIGHTING
        );

    return true;
}


/*
 * misc::TriSoupRenderer::GetExtents
 */
bool misc::TriSoupRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetTimeFramesCount(1);
    cr->AccessBoundingBoxes().Clear();
    cr->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    cr->AccessBoundingBoxes().SetWorldSpaceBBox(this->bbox);

    return true;
}


/*
 * misc::TriSoupRenderer::release
 */
void misc::TriSoupRenderer::release(void) {
    this->cnt = 0;
    ARY_SAFE_DELETE(this->clusters);
}


/*
 * misc::TriSoupRenderer::Render
 */
bool misc::TriSoupRenderer::Render(Call& call) {

    if (this->filename.IsDirty()) {
        // load the data.
        this->tryLoadFile();
        this->filename.ResetDirty();
    }
    bool normals = false;
    bool colors = false;

    glEnable(GL_DEPTH_TEST);
    if (this->lighting.Param<param::BoolParam>()->Value()) {
        glEnable(GL_LIGHTING);
    } else {
        glDisable(GL_LIGHTING);
    }
    glDisable(GL_BLEND);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT); // Grrrrr. Arglarglargl
    glEnable(GL_COLOR_MATERIAL);

    if (this->surStyle.Param<param::EnumParam>()->Value() == 1) {
        glPolygonMode(GL_BACK, GL_LINE);
    }

    glColor3f(1.0f, 1.0f, 1.0f);

    for (unsigned int i = 0; i < this->cnt; i++) {
        glVertexPointer(3, GL_FLOAT, 0, this->clusters[i].v);

        if (this->clusters[i].n != NULL) {
            if (!normals) {
                glEnableClientState(GL_NORMAL_ARRAY);
                normals = true;
            }
            glNormalPointer(GL_FLOAT, 0, this->clusters[i].n);
        } else {
            if (normals) {
                glDisableClientState(GL_NORMAL_ARRAY);
                normals = false;
            }
        }

        if (this->clusters[i].c != NULL) {
            if (!colors) {
                glEnableClientState(GL_COLOR_ARRAY);
                colors = true;
            }
            glColorPointer(3, GL_UNSIGNED_BYTE, 0, this->clusters[i].c);
        } else {
            if (colors) {
                glDisableClientState(GL_COLOR_ARRAY);
                colors = false;
                glColor3f(1.0f, 1.0f, 1.0f);
            }
        }

        glDrawElements(GL_TRIANGLES, this->clusters[i].tc * 3,
            GL_UNSIGNED_INT, this->clusters[i].t);
    }
    if (normals) {
        glDisableClientState(GL_NORMAL_ARRAY);
    }

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if (this->showVertices.Param<param::BoolParam>()->Value()) {
        glEnable(GL_POINT_SIZE);
        glPointSize(3.0f);
        glDisable(GL_LIGHTING);

        glColor3f(1.0f, 0.0f, 0.0f);
        for (unsigned int i = 0; i < this->cnt; i++) {
            glVertexPointer(3, GL_FLOAT, 0, this->clusters[i].v);
            glDrawArrays(GL_POINTS, 0, this->clusters[i].vc);
        }
    }

    glEnable(GL_CULL_FACE);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_POINT_SIZE);
    glEnable(GL_BLEND);

    return true;
}


/*
 * misc::TriSoupRenderer::tryLoadFile
 */
void misc::TriSoupRenderer::tryLoadFile(void) {
    using vislib::sys::MemmappedFile;
    using vislib::sys::File;
    using vislib::sys::Log;

#define FILE_READ(A, B) if ((B) != file.Read((A), (B))) {\
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,\
            "Unable to load file \"%s\": data corruption",\
            vislib::StringA(fn).PeekBuffer());\
        this->cnt = 0;\
        ARY_SAFE_DELETE(this->clusters);\
        return;\
    }

    File::FileSize r;
    MemmappedFile file;
    const char theHeader[] = "Triangle Soup File 100\0\xFF";
    char rb[100];
    unsigned int ui;
    float minX, minY, minZ, maxX, maxY, maxZ;
    float xo, yo, zo, scale;

    this->cnt = 0;
    ARY_SAFE_DELETE(this->clusters);

    const vislib::TString& fn
        = this->filename.Param<param::StringParam>()->Value();
    if (fn.IsEmpty()) {
        // no file to load
        return;
    }

    if (!file.Open(fn, File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to open file \"%s\"", vislib::StringA(fn).PeekBuffer());
        return;
    }

    r = file.Read(rb, sizeof(theHeader) - 1);
    if (memcmp(rb, theHeader, sizeof(theHeader) - 1) != 0) {
        file.Close();
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to load file \"%s\": Wrong format header",
            vislib::StringA(fn).PeekBuffer());
        return;
    }

    FILE_READ(&ui, sizeof(unsigned int));

    FILE_READ(&minX, sizeof(float));
    FILE_READ(&minY, sizeof(float));
    FILE_READ(&minZ, sizeof(float));
    FILE_READ(&maxX, sizeof(float));
    FILE_READ(&maxY, sizeof(float));
    FILE_READ(&maxZ, sizeof(float));

    xo = (minX + maxX) * -0.5f;
    yo = (minY + maxY) * -0.5f;
    zo = (minZ + maxZ) * -0.5f;

    scale = vislib::math::Abs(maxX - minX);
    scale = vislib::math::Max(scale, vislib::math::Abs(maxY - minY));
    scale = vislib::math::Max(scale, vislib::math::Abs(maxZ - minZ));
    if (scale > 0.0f) {
        scale = 2.0f / scale;
    }

    this->bbox.Set(
        vislib::math::Abs(maxX - minX) * scale * -0.5f,
        vislib::math::Abs(maxY - minY) * scale * -0.5f,
        vislib::math::Abs(maxZ - minZ) * scale * -0.5f,
        vislib::math::Abs(maxX - minX) * scale * 0.5f,
        vislib::math::Abs(maxY - minY) * scale * 0.5f,
        vislib::math::Abs(maxZ - minZ) * scale * 0.5f);

    this->cnt = ui;
    this->clusters = new Cluster[this->cnt];

    for (unsigned int i = 0; i < this->cnt; i++) {
        FILE_READ(&this->clusters[i].id, sizeof(unsigned int));
        FILE_READ(&this->clusters[i].vc, sizeof(unsigned int));
        this->clusters[i].v = new float[3 * this->clusters[i].vc];
        FILE_READ(this->clusters[i].v,
            sizeof(float) * 3 * this->clusters[i].vc);
        for (unsigned int j = 0; j < this->clusters[i].vc; j++) {
            this->clusters[i].v[j * 3] += xo;
            this->clusters[i].v[j * 3 + 1] += yo;
            this->clusters[i].v[j * 3 + 2] += zo;
            this->clusters[i].v[j * 3] *= scale;
            this->clusters[i].v[j * 3 + 1] *= scale;
            this->clusters[i].v[j * 3 + 2] *= scale;
        }
        FILE_READ(&this->clusters[i].tc, sizeof(unsigned int));
        this->clusters[i].t = new unsigned int[3 * this->clusters[i].tc];
        FILE_READ(this->clusters[i].t,
            sizeof(unsigned int) * 3 * this->clusters[i].tc);

        // Calculate the vertex normals
        unsigned int *nc = new unsigned int[this->clusters[i].vc];
        ::memset(nc, 0, this->clusters[i].vc * sizeof(unsigned int));
        this->clusters[i].n = new float[3 * this->clusters[i].vc];
        for (unsigned int j = 0; j < 3 * this->clusters[i].vc; j++) {
            this->clusters[i].n[j] = 0.0f;
        }
        for (unsigned int j = 0; j < this->clusters[i].tc; j++) {
            unsigned int v1 = this->clusters[i].t[j * 3] * 3;
            unsigned int v2 = this->clusters[i].t[j * 3 + 1] * 3;
            unsigned int v3 = this->clusters[i].t[j * 3 + 2] * 3;
            vislib::math::ShallowPoint<float, 3> p1(&this->clusters[i].v[v1]);
            vislib::math::ShallowPoint<float, 3> p2(&this->clusters[i].v[v2]);
            vislib::math::ShallowPoint<float, 3> p3(&this->clusters[i].v[v3]);
            vislib::math::Vector<float, 3> e1 = p2 - p1;
            vislib::math::Vector<float, 3> e2 = p1 - p3;
            vislib::math::Vector<float, 3> n = e1.Cross(e2);
            n.Normalise();

            this->clusters[i].n[v1 + 0] += n.X();
            this->clusters[i].n[v1 + 1] += n.Y();
            this->clusters[i].n[v1 + 2] += n.Z();
            this->clusters[i].n[v2 + 0] += n.X();
            this->clusters[i].n[v2 + 1] += n.Y();
            this->clusters[i].n[v2 + 2] += n.Z();
            this->clusters[i].n[v3 + 0] += n.X();
            this->clusters[i].n[v3 + 1] += n.Y();
            this->clusters[i].n[v3 + 2] += n.Z();
            nc[v1 / 3]++;
            nc[v2 / 3]++;
            nc[v3 / 3]++;

        }
        for (unsigned int j = 0; j < 3 * this->clusters[i].vc; j++) {
            this->clusters[i].n[j] /= float(nc[j / 3]);
        }
        delete[] nc;

        // Calculate fancy colors
        this->clusters[i].c = new unsigned char[this->clusters[i].vc * 3];
        for (unsigned int j = 0; j < 3 * this->clusters[i].vc; j += 3) {
            float a = float(j) / float(3 * (this->clusters[i].vc - 1));
            if (a < 0.0f) a = 0.0f; else if (a > 1.0f) a = 1.0f;
            this->clusters[i].c[j + 2] = int(255.f * a);

            //this->clusters[i].c[j + 2] = (j / 3) % 256;

            this->clusters[i].c[j + 1] = 0;
            this->clusters[i].c[j + 0] = 255 - this->clusters[i].c[j + 2];
        }

    }

    file.Close();
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
        "File \"%s\" loaded successfully\n",
        vislib::StringA(fn).PeekBuffer());
}


/*
 * misc::TriSoupRenderer::bullchit
 */
bool misc::TriSoupRenderer::bullchit(param::ParamSlot& slot) {
    vislib::sys::Log::DefaultLog.WriteMsg(2000,
        "Hurglhurglhurgl!\n");
    return true;
}
