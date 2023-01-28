/*
 * CriticalSection.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 by Christoph Mueller (christoph.mueller@vis.uni-stuttgart.de). Alle Rechte vorbehalten.
 * Copyright 2019 MegaMol Dev Team
 */

#ifndef VISLIB_CRITICALSECTION_H_INCLUDED
#define VISLIB_CRITICALSECTION_H_INCLUDED
#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "Lockable.h"
#include "Mutex.h"

namespace vislib::sys {

using CriticalSection = Mutex;

/** name typedef for Lockable with this SyncObject */
typedef Lockable<CriticalSection> CriticalSectionLockable;


} // namespace vislib::sys

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CRITICALSECTION_H_INCLUDED */
