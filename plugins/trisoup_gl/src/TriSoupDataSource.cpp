/*
 * TriSoupDataSource.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "TriSoupDataSource.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"
#include "vislib/assert.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/Vector.h"
#include "vislib/sys/MemmappedFile.h"

using namespace megamol;
using namespace megamol::trisoup_gl;


/*
 * TriSoupDataSource::TriSoupDataSource
 */
TriSoupDataSource::TriSoupDataSource(void) : AbstractTriMeshLoader() {
    // intentionally empty
}


/*
 * TriSoupDataSource::~TriSoupDataSource
 */
TriSoupDataSource::~TriSoupDataSource(void) {
    this->Release();
}


/*
 * TriSoupDataSource::load
 */
bool TriSoupDataSource::load(const vislib::TString& filename) {
    using megamol::core::utility::log::Log;
    using vislib::sys::File;
    using vislib::sys::MemmappedFile;

#define FILE_READ(A, B)                                                                     \
    if ((B) != file.Read((A), (B))) {                                                       \
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load file: data corruption"); \
        return false;                                                                       \
    }

    File::FileSize r;
    MemmappedFile file;
    const char theHeader[] = "Triangle Soup File 100\0\xFF";
    char rb[100];
    unsigned int ui;
    float minX, minY, minZ, maxX, maxY, maxZ;
    float xo, yo, zo, scale;
    ASSERT(this->objs.IsEmpty());

    if (filename.IsEmpty()) {
        // no file to load
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "No file to load (filename empty)");
        return true;
    }

    if (!file.Open(filename, File::READ_ONLY, File::SHARE_READ, File::OPEN_ONLY)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to open file");
        return false;
    }

    r = file.Read(rb, sizeof(theHeader) - 1);
    if (memcmp(rb, theHeader, sizeof(theHeader) - 1) != 0) {
        file.Close();
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load file: Wrong format header");
        return false;
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

    this->bbox.Set(vislib::math::Abs(maxX - minX) * scale * -0.5f, vislib::math::Abs(maxY - minY) * scale * -0.5f,
        vislib::math::Abs(maxZ - minZ) * scale * -0.5f, vislib::math::Abs(maxX - minX) * scale * 0.5f,
        vislib::math::Abs(maxY - minY) * scale * 0.5f, vislib::math::Abs(maxZ - minZ) * scale * 0.5f);

    this->objs.SetCount(ui);

    for (unsigned int i = 0; i < this->objs.Count(); i++) {

        FILE_READ(&ui, sizeof(unsigned int)); // object id currently not used

        unsigned int vrtCnt;
        FILE_READ(&vrtCnt, sizeof(unsigned int));

        float* v = new float[3 * vrtCnt];
        float* n = new float[3 * vrtCnt];
        unsigned char* c = new unsigned char[3 * vrtCnt];
        this->objs[i].SetVertexData(
            vrtCnt, v, n, c, NULL, true); // Do not delete v, n, or c since the memory is now owned by objs[i]

        FILE_READ(v, sizeof(float) * 3 * vrtCnt);

        for (unsigned int j = 0; j < vrtCnt; j++) {
            v[j * 3] += xo;
            v[j * 3 + 1] += yo;
            v[j * 3 + 2] += zo;
            v[j * 3] *= scale;
            v[j * 3 + 1] *= scale;
            v[j * 3 + 2] *= scale;
        }

        unsigned int triCnt;
        FILE_READ(&triCnt, sizeof(unsigned int));

        unsigned int* t = new unsigned int[3 * triCnt];
        this->objs[i].SetTriangleData(triCnt, t, true); // Do not delet t since the memory is now owned by objs[i]

        FILE_READ(t, sizeof(unsigned int) * 3 * triCnt);
        // fix triangle orientations (hard)
        for (unsigned int j = 0; j < triCnt; j++) {
            unsigned int tmp = t[j * 3 + 1];
            t[j * 3 + 1] = t[j * 3 + 2];
            t[j * 3 + 2] = tmp;
        }

        // Calculate the vertex normals
        unsigned int* nc = new unsigned int[vrtCnt];
        ::memset(nc, 0, vrtCnt * sizeof(unsigned int));
        ::memset(n, 0, vrtCnt * 3 * sizeof(float));

        for (unsigned int j = 0; j < triCnt; j++) {
            unsigned int v1 = t[j * 3] * 3;
            unsigned int v2 = t[j * 3 + 1] * 3;
            unsigned int v3 = t[j * 3 + 2] * 3;
            vislib::math::ShallowPoint<float, 3> p1(&v[v1]);
            vislib::math::ShallowPoint<float, 3> p2(&v[v2]);
            vislib::math::ShallowPoint<float, 3> p3(&v[v3]);
            vislib::math::Vector<float, 3> e1 = p2 - p1;
            vislib::math::Vector<float, 3> e2 = p1 - p3;
            vislib::math::Vector<float, 3> nrml = e2.Cross(e1);
            nrml.Normalise();

            n[v1 + 0] += nrml.X();
            n[v1 + 1] += nrml.Y();
            n[v1 + 2] += nrml.Z();
            n[v2 + 0] += nrml.X();
            n[v2 + 1] += nrml.Y();
            n[v2 + 2] += nrml.Z();
            n[v3 + 0] += nrml.X();
            n[v3 + 1] += nrml.Y();
            n[v3 + 2] += nrml.Z();
            nc[v1 / 3]++;
            nc[v2 / 3]++;
            nc[v3 / 3]++;
        }

        for (unsigned int j = 0; j < 3 * vrtCnt; j++) {
            n[j] /= float(nc[j / 3]);
        }
        delete[] nc;

        // Calculate fancy colors
        this->objs[i].SetMaterial(NULL);
        for (unsigned int j = 0; j < 3 * vrtCnt; j += 3) {
            float a = float(j) / float(3 * (vrtCnt - 1));
            if (a < 0.0f)
                a = 0.0f;
            else if (a > 1.0f)
                a = 1.0f;
            c[j + 2] = int(255.f * a);
            c[j + 1] = 0;
            c[j + 0] = 255 - c[j + 2];
        }
    }

    file.Close();

    return true;
}
