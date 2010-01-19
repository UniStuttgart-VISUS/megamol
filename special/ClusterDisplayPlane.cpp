/*
 * ClusterDisplayPlane.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ClusterDisplayPlane.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include "vislib/StringTokeniser.h"

using namespace megamol::core;


/*
 * special::ClusterDisplayPlane::planes
 */
vislib::PtrArray<special::ClusterDisplayPlane>
special::ClusterDisplayPlane::planes;


/*
 * special::ClusterDisplayPlane::Plane
 */
const special::ClusterDisplayPlane *
special::ClusterDisplayPlane::Plane(unsigned int id,
        const CoreInstance& inst) {
    SIZE_T l = planes.Count();
    for (SIZE_T i = 0; i < l; i++) {
        if (planes[i]->Id() == id) {
            return planes[i];
        }
    }

    ClusterDisplayPlane *plane = new ClusterDisplayPlane();
    if (!plane->loadConfiguration(id, inst)) {
        delete plane;
        return NULL;
    }

    planes.Append(plane);

    return plane;
}


/*
 * special::ClusterDisplayPlane::ClusterDisplayPlane
 */
special::ClusterDisplayPlane::ClusterDisplayPlane(
        special::ClusterDisplayPlane::PlaneType type, float width,
        float height) : id(0), planeType(type), width(width), height(height) {
    // intentionally empty
}


/*
 * special::ClusterDisplayPlane::ClusterDisplayPlane
 */
special::ClusterDisplayPlane::ClusterDisplayPlane(
        const special::ClusterDisplayPlane& src) : id(src.id),
        planeType(src.planeType), width(src.width), height(src.height) {
    // intentionally empty
}

/*
 * special::ClusterDisplayPlane::~ClusterDisplayPlane
 */
special::ClusterDisplayPlane::~ClusterDisplayPlane(void) {
    // intentionally empty
}


/*
 * special::ClusterDisplayPlane::operator=
 */
special::ClusterDisplayPlane& special::ClusterDisplayPlane::operator=(
        const special::ClusterDisplayPlane& rhs) {
    this->id = rhs.id;
    this->planeType = rhs.planeType;
    this->width = rhs.width;
    this->height = rhs.height;
    return *this;
}


/*
 * special::ClusterDisplayPlane::ClusterDisplayPlane
 */
special::ClusterDisplayPlane::ClusterDisplayPlane(void) : id(0),
        planeType(TYPE_VOID), width(1.0f), height(1.0f) {
    // intentionally empty
}


/*
 * special::ClusterDisplayPlane::loadConfiguration
 */
bool special::ClusterDisplayPlane::loadConfiguration(unsigned int id,
        const CoreInstance& inst) {
    vislib::StringA str;
    this->id = id;
    this->planeType = TYPE_VOID;
    if (id == 0) return true;

    str.Format("viewplane-%u", id);
    if (inst.Configuration().IsConfigValueSet(str)) {
        vislib::Array<vislib::StringW> items
            = vislib::StringTokeniserW::Split(
            inst.Configuration().ConfigValue(str), L";", true);
        if (items.Count() == 3) {
            items[0].TrimSpaces();
            items[1].TrimSpaces();
            items[2].TrimSpaces();

            try {
                if (items[0].Equals(L"mono", false)) {
                    this->planeType = TYPE_MONO;
                } else if (items[0].Equals(L"stereoleft", false)
                        || items[0].Equals(L"stereo left", false)
                        || items[0].Equals(L"stereo-left", false)
                        || items[0].Equals(L"left", false)) {
                    this->planeType = TYPE_STEREO_LEFT;
                } else if (items[0].Equals(L"stereoright", false)
                        || items[0].Equals(L"stereo right", false)
                        || items[0].Equals(L"stereo-right", false)
                        || items[0].Equals(L"right", false)) {
                    this->planeType = TYPE_STEREO_RIGHT;
                } else if (items[0].Equals(L"dome", false)) {
                    this->planeType = TYPE_DOME;
                } else {
                    this->planeType = TYPE_VOID;
                }

                this->width = static_cast<float>(
                    vislib::CharTraitsW::ParseDouble(items[1]));
                this->height = static_cast<float>(
                    vislib::CharTraitsW::ParseDouble(items[2]));

                return true;
            } catch(...) {
            }
        }
    }

    return false;
}
