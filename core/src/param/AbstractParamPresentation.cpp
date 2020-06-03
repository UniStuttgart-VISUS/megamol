/*
 * AbstractParamPresentation.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/param/AbstractParamPresentation.h"

using namespace megamol::core::param;

AbstractParamPresentation::AbstractParamPresentation(void) : visible(true), read_only(false), presentation(AbstractParamPresentation::Presentations::RawValue)  {
}