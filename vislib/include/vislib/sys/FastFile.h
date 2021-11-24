/*
 * FastFile.h
 *
 * Copyright (C) 2015 by TU Dresden. Alle Rechte vorbehalten.
 */
#ifndef VISLIB_FASTFILE_H_INCLUDED
#define VISLIB_FASTFILE_H_INCLUDED
#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32
#include "mmcore/utility/sys/MemmappedFile.h"
#else
#include "vislib/sys/BufferedFile.h"
#endif

namespace vislib {
namespace sys {

/**
 * The FastFile typedef selects the best file implementation for the
 * operating system.
 *
 * On Windows the MemmappedFile is 1-2x faster than the BufferedFile.
 *
 * On Linux the BufferedFile is ~20x faster than the MemmappedFile.
 */
#ifdef _WIN32
typedef MemmappedFile FastFile;
#else
typedef BufferedFile FastFile;
#endif

} // namespace sys
} // namespace vislib

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_FASTFILE_H_INCLUDED */
