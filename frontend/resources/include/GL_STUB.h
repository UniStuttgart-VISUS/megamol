/*
 * GL_STUB.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#ifndef GL_STUB
#ifdef MEGAMOL_USE_OPENGL
#define GL_STUB(value)
#else // MEGAMOL_USE_OPENGL
#define GL_STUB(value) \
    { return value; }
#endif // MEGAMOL_USE_OPENGL
#endif // GL_STUB
