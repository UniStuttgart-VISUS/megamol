/*
 * MegaMolLogo.cpp
 *
 * Copyright (C) 2008 - 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/view/special/MegaMolLogo.h"
#include "MegaMolLogoData.inl"
#include "vislib/assert.h"
#include "vislib/Array.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/Pair.h"
#include "vislib/math/Vector.h"


/*
 * megamol::core::view::special::MegaMolLogo::MegaMolLogo
 */
megamol::core::view::special::MegaMolLogo::MegaMolLogo(void)
        : vertices(NULL), colours(NULL), triIdxs(NULL), triIdxCnt(0),
        lineIdxs(NULL), lineIdxCnt(0), maxX(0.0f) {
    // intentionally empty
}


/*
 * megamol::core::view::special::MegaMolLogo::~MegaMolLogo
 */
megamol::core::view::special::MegaMolLogo::~MegaMolLogo(void) {
    this->Release();
    ASSERT(this->vertices == NULL);
    ASSERT(this->colours == NULL);
    ASSERT(this->triIdxs == NULL);
    ASSERT(this->lineIdxs == NULL);
}


/*
 * megamol::core::view::special::MegaMolLogo::Create
 */
void megamol::core::view::special::MegaMolLogo::Create(void) {
    ASSERT(this->vertices == NULL);
    ASSERT(this->colours == NULL);
    ASSERT(this->triIdxs == NULL);
    ASSERT(this->lineIdxs == NULL);
    ASSERT((MegaMolLogoElements % 3) == 0);
    // MegaMolLogoElements (count)
    // MegaMolLogoVertices
    // MegaMolLogoColors
    vislib::Array<vislib::math::Vector<float, 2> > pos;
    vislib::Array<vislib::math::Vector<unsigned char, 3> > col;
    this->triIdxs = new unsigned int[MegaMolLogoElements];
    this->triIdxCnt = MegaMolLogoElements;
    pos.AssertCapacity(MegaMolLogoElements);
    col.AssertCapacity(MegaMolLogoElements);
    this->maxX = 0.0f;

    for (unsigned int i = 0; i < MegaMolLogoElements; i++) {
        vislib::math::Vector<float, 2> p(MegaMolLogoVertices[i * 3], -MegaMolLogoVertices[i * 3 + 2]);
        INT_PTR idx = pos.IndexOf(p);
        if (idx == vislib::Array<vislib::math::Vector<float, 2> >::INVALID_POS) {
            idx = pos.Count();
            pos.Append(p);
            col.Append(vislib::math::Vector<unsigned char, 3>(MegaMolLogoColors + i * 3));
        }
        this->triIdxs[i] = static_cast<unsigned int>(idx);
    }

    ASSERT(pos.Count() == col.Count());
    this->vertices = new float[pos.Count() * 2];
    this->colours = new unsigned char[col.Count() * 3];
    for (unsigned int i = 0; i < pos.Count(); i++) {
        this->vertices[i * 2] = pos[i].X();
        this->vertices[i * 2 + 1] = pos[i].Y();
        this->colours[i * 3] = col[i].X();
        this->colours[i * 3 + 1] = col[i].Y();
        this->colours[i * 3 + 2] = col[i].Z();
        if (this->maxX < this->vertices[i * 2]) {
            this->maxX = this->vertices[i * 2];
        }
    }

    vislib::Array<vislib::Pair<unsigned int, unsigned int> > lines;
    lines.AssertCapacity(MegaMolLogoElements);
    for (unsigned int i = 0; i < MegaMolLogoElements; i += 3) {
        for (unsigned int j = 0; j < 3; j++) {
            unsigned int i1 = this->triIdxs[i + j];
            unsigned int i2 = this->triIdxs[i + ((j + 1) % 3)];
            vislib::Pair<unsigned int, unsigned int> line(vislib::math::Min(i1, i2), vislib::math::Max(i1, i2));
            if (lines.Contains(line)) {
                lines.RemoveAll(line);
            } else {
                lines.Add(line);
            }
        }
    }

    this->lineIdxCnt = static_cast<unsigned int>(lines.Count() * 2);
    this->lineIdxs = new unsigned int[this->lineIdxCnt];
    for (unsigned int i = 0; i < lines.Count(); i++) {
        this->lineIdxs[i * 2] = lines[i].First();
        this->lineIdxs[i * 2 + 1] = lines[i].Second();
    }
}


/*
 * megamol::core::view::special::MegaMolLogo::Draw
 */
void megamol::core::view::special::MegaMolLogo::Draw(void) {
    ::glEnable(GL_BLEND);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    ::glEnable(GL_LINE_SMOOTH);
    ::glLineWidth(1.1f);

    ::glEnableClientState(GL_VERTEX_ARRAY);
    ::glEnableClientState(GL_COLOR_ARRAY);

    ::glVertexPointer(2, GL_FLOAT, 0, this->vertices);
    ::glColorPointer(3, GL_UNSIGNED_BYTE, 0, this->colours);

    ::glDrawElements(GL_TRIANGLES, this->triIdxCnt, GL_UNSIGNED_INT, this->triIdxs);
    ::glDrawElements(GL_LINES, this->lineIdxCnt, GL_UNSIGNED_INT, this->lineIdxs);

    ::glDisableClientState(GL_VERTEX_ARRAY);
    ::glDisableClientState(GL_COLOR_ARRAY);
}


/*
 * megamol::core::view::special::MegaMolLogo::Release
 */
void megamol::core::view::special::MegaMolLogo::Release(void) {
    ARY_SAFE_DELETE(this->vertices);
    ARY_SAFE_DELETE(this->colours);
    ARY_SAFE_DELETE(this->triIdxs);
    this->triIdxCnt = 0;
    ARY_SAFE_DELETE(this->lineIdxs);
    this->lineIdxCnt = 0;
}
