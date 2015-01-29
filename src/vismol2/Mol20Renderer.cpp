/*
 * Mol20Renderer.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "Mol20Renderer.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/vismol2/Mol20DataCall.h"

using namespace megamol::core;


/*
 * vismol2::Mol20Renderer::Mol20Renderer
 */
vismol2::Mol20Renderer::Mol20Renderer(void) : Renderer3DModule(),
        getDataSlot("getdata", "Connects to the data source") {

    this->getDataSlot.SetCompatibleCall<vismol2::Mol20DataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);
}


/*
 * vismol2::Mol20Renderer::~Mol20Renderer
 */
vismol2::Mol20Renderer::~Mol20Renderer(void) {
    this->Release(); // implicitly calls 'release'
}


/*
 * vismol2::Mol20Renderer::create
 */
bool vismol2::Mol20Renderer::create(void) {
    // intentionally empty (ATM)
    return true;
}


/*
 * vismol2::Mol20Renderer::release
 */
void vismol2::Mol20Renderer::release(void) {
    // intentionally empty (ATM)
}


/*
 * vismol2::Mol20Renderer::GetCapabilities
 */
bool vismol2::Mol20Renderer::GetCapabilities(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(
        view::CallRender3D::CAP_RENDER
        | view::CallRender3D::CAP_LIGHTING
        | view::CallRender3D::CAP_ANIMATION
        );

    return true;
}


/*
 * vismol2::Mol20Renderer::GetExtents
 */
bool vismol2::Mol20Renderer::GetExtents(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    Mol20DataCall *dc = this->getDataSlot.CallAs<Mol20DataCall>();
    if (dc != NULL) {
        if ((*dc)(1)) {
            cr->SetTimeFramesCount(dc->Time());
            cr->DataSpaceBoundingBox() = dc->BoundingBox();
            cr->ObjectSpaceBoundingBox() = dc->BoundingBox();
        } else {
            dc = NULL; // set default data as sfx
        }
    }

    if (dc == NULL) {
        cr->SetTimeFramesCount(1);
        cr->DataSpaceBoundingBox().Set(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        cr->ObjectSpaceBoundingBox() = cr->DataSpaceBoundingBox();
        return false;
    }

    return true;
}


/*
 * vismol2::Mol20Renderer::Render
 */
bool vismol2::Mol20Renderer::Render(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    Mol20DataCall *dc = this->getDataSlot.CallAs<Mol20DataCall>();
    if (dc == NULL) return false;
    dc->SetTime(static_cast<unsigned int>(cr->Time()));

    if (!(*dc)(0)) return false;

    // get data
    vismol2::cluster_t& cluster = dc->Frame()->data.start;
    float alpha = cr->Time() - static_cast<float>(dc->Time());
    if (alpha < 0.0f) alpha = 0.0f;
    else if (alpha > 1.0f) alpha = 1.0f;
    float beta = 1.0f - alpha;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE); // simple additive blending
    glEnable(GL_POINT_SMOOTH);
    glPointSize(5.0f);
    glColor3ub(192, 16, 8);
    glBegin(GL_POINTS);

    this->render(cluster, 0.0f, 0.0f, 0.0f, alpha, beta);

    glEnd();
    glDisable(GL_POINT_SMOOTH);
    glDisable(GL_BLEND);

    return true;
}


/*
 * vismol2::Mol20Renderer::render
 */
void vismol2::Mol20Renderer::render(vismol2::cluster_t& cluster,
        float ox, float oy, float oz, float alpha, float beta) {
    float x, y, z;
    unsigned char a;

    /*
        cluster.points[i].interpolType stores significant information!
        for clusters:
            254 : root for clustered/liquid molecules
            255 : root for unclustered/vapor molecules
        for points:
            0   : linear interpolation
            1   : jumping (cyclic boundary condition)
            2   : linear, disappearing (joining a cluster)
            3   : linear, appearing (leaving a cluster)
            4   : jumping, disappearing
            5   : jumping, appearing
            6   : linear, inside a cluster
            7   : jumping, inside a cluster
    */

    if (cluster.clusters != NULL) {

        for (unsigned int i = 0; i < static_cast<unsigned int>(cluster.size); i++) {
            this->render(cluster.clusters[i],
                ox + (cluster.points[i].x * beta + cluster.points[i].tpx * alpha) * cluster.scale,
                oy + (cluster.points[i].y * beta + cluster.points[i].tpy * alpha) * cluster.scale,
                oz + (cluster.points[i].z * beta + cluster.points[i].tpz * alpha) * cluster.scale,
                alpha, beta);
        }

    } else {

        for (unsigned int i = 0; i < static_cast<unsigned int>(cluster.size); i++) {
            switch (cluster.points[i].interpolType) {
                case 0: case 2: case 3: case 6: // linear
                    x = (cluster.points[i].x * beta + cluster.points[i].tpx * alpha);
                    y = (cluster.points[i].y * beta + cluster.points[i].tpy * alpha);
                    z = (cluster.points[i].z * beta + cluster.points[i].tpz * alpha);
                    a = 255;
                    break;
                case 1: case 4: case 5: case 7: // jumping
                    if (alpha < 0.5f) {
                        x = cluster.points[i].x;
                        y = cluster.points[i].y;
                        z = cluster.points[i].z;
                        a = static_cast<unsigned char>(static_cast<int>(beta * 510.0f) - 255);
                    } else {
                        x = cluster.points[i].tpx;
                        y = cluster.points[i].tpy;
                        z = cluster.points[i].tpz;
                        a = static_cast<unsigned char>(static_cast<int>(alpha * 510.0f) - 255);
                    }
                    break;
                default:
                    x = y = z = 0.0f;
                    a = 0;
                    break;
            }
            glColor4ub( // ultra-ugly, but okey for now
                static_cast<unsigned char>(static_cast<float>(cluster.points[i].r) * beta
                    + static_cast<float>(cluster.points[i].tcr) * alpha),
                static_cast<unsigned char>(static_cast<float>(cluster.points[i].g) * beta
                    + static_cast<float>(cluster.points[i].tcg) * alpha),
                static_cast<unsigned char>(static_cast<float>(cluster.points[i].b) * beta
                    + static_cast<float>(cluster.points[i].tcb) * alpha),
                a);
            glVertex3f(ox + x * cluster.scale, oy + y * cluster.scale, oz + z * cluster.scale);
        }

    }

}
