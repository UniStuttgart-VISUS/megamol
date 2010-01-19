/*
 * MegaMolLogo.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MegaMolLogo.h"
#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */
#include "special/MegaMolLogoData.inl"
#include <GL/gl.h>


/*
 * megamol::core::special::MegaMolLogo::MegaMolLogo
 */
megamol::core::special::MegaMolLogo::MegaMolLogo(void)
        : vertices(MegaMolLogoVertices), colours(MegaMolLogoColors),
        count(MegaMolLogoElements) {
    // intentionally empty
}


/*
 * megamol::core::special::MegaMolLogo::~MegaMolLogo
 */
megamol::core::special::MegaMolLogo::~MegaMolLogo(void) {
    this->vertices = NULL; // DO NOT DELETE
    this->colours = NULL; // DO NOT DELETE
    this->count = 0;
}


/*
 * megamol::core::special::MegaMolLogo::Create
 */
void megamol::core::special::MegaMolLogo::Create(void) {
}


/*
 * megamol::core::special::MegaMolLogo::Draw
 */
void megamol::core::special::MegaMolLogo::Draw(void) {
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glVertexPointer(3, GL_FLOAT, 0, this->vertices);
    glColorPointer(3, GL_UNSIGNED_BYTE, 0, this->colours);
    glDrawArrays(GL_TRIANGLES, 0, this->count);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}


/*
 * megamol::core::special::MegaMolLogo::Release
 */
void megamol::core::special::MegaMolLogo::Release(void) {
}
