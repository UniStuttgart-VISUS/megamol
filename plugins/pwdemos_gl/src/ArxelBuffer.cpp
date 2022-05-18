/*
 * ArxelBuffer.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "ArxelBuffer.h"
#include "stdafx.h"
#include "vislib/memutils.h"

#include <climits>

namespace megamol {
namespace demos_gl {


/*
 * ArxelBuffer::Initialize
 */
void ArxelBuffer::Initialize(ArxelBuffer& buffer, int& state, const InitValues& ctxt) {
    buffer.width = ctxt.width;
    buffer.height = ctxt.height;
    if (buffer.data != NULL)
        delete[] buffer.data;
    buffer.data = ((buffer.width == 0) || (buffer.height == 0)) ? NULL : new ArxelType[buffer.width * buffer.height];
    state = 0;
}


/*
 * ArxelBuffer::ArxelBuffer
 */
ArxelBuffer::ArxelBuffer(void) : width(0), height(0), data(NULL), borderXVal(2), borderYVal(3) {}


/*
 * ArxelBuffer::~ArxelBuffer
 */
ArxelBuffer::~ArxelBuffer(void) {
    this->width = 0;
    this->height = 0;
    ARY_SAFE_DELETE(this->data);
}


/*
 * helper struct for housekeeping and sorting in ArxelBuffer::Fill
 */
struct tableEdge {
    int end;
    float x;
    float invSlope;

    bool operator==(const tableEdge& rhs) const {
        return this->end == rhs.end && this->x == rhs.x && this->invSlope == rhs.invSlope;
    }

    static int AETComparator(const tableEdge& one, const tableEdge& other) {
        return vislib::math::Compare(one.x, other.x);
    }
};


/*
 * ArxelBuffer::Fill
 */
UINT64 ArxelBuffer::Fill(const vislib::Array<vislib::math::Point<int, 2>>& polygon, const ArxelType& val, bool dryRun) {
    vislib::math::Point<int, 2> p1;
    vislib::math::Point<int, 2> p2;
    vislib::math::Point<int, 2> p;
    int bottom = INT_MAX, top = -INT_MAX, y, x;
    UINT64 pixelCount = 0;
    //int sign = 1;

    for (SIZE_T i = 0; i < polygon.Count(); i++) {
        if ((p1 = polygon[i]).Y() < bottom) {
            bottom = p1.Y();
        }
        if (p1.Y() > top) {
            top = p1.Y();
        }
    }

    //for (SIZE_T i = 0; i < polygon.Count(); i++) {
    //    p1 = polygon[i];
    //    p2 = polygon[(i + 1) % polygon.Count()];
    //    if (p2.Y() < p1.Y()) {
    //        p = p2;
    //    } else {
    //        p = p1;
    //    }
    //    //if (p.Y() == bottom) {
    //    //    if (Get(p.X() + 1, p.Y()) != 0) {
    //    //        if (Get(p.X() + 1, p.Y() - 1) != 0) {
    //    //            sign = -1;
    //    //        }
    //    //        break;
    //    //    } else if (Get(p.X() - 1, p.Y()) != 0) {
    //    //        if (Get(p.X() - 1, p.Y() - 1) != 0) {
    //    //            sign = -1;
    //    //        }
    //    //        break;
    //    //    }
    //    //}
    //}

    vislib::Array<vislib::Array<tableEdge>> edgeTable;
    for (y = bottom; y <= top; y++) {
        vislib::Array<tableEdge> edgeLine;
        edgeLine.SetCapacityIncrement(4);
        for (SIZE_T i = 0; i < polygon.Count(); i++) {
            p = polygon[i];
            p2 = polygon[(i + 1) % polygon.Count()];
            if (p2.Y() < p.Y()) {
                p1 = p2;
                p2 = p;
            } else {
                p1 = p;
            }
            // all go up; also, horizontal edges are useless
            if (p1.Y() == y && p2.Y() != y) {
                tableEdge e;
                e.x = static_cast<float>(p1.X());
                e.end = p2.Y();
                e.invSlope = static_cast<float>(p2.X() - p1.X()) / static_cast<float>(p2.Y() - p1.Y());
                edgeLine.Add(e);
            }
        }
        edgeTable.Add(edgeLine);
    }

    vislib::Array<tableEdge> activeEdges;
    y = 0;
    //while (edgeTable.Count() > 0 || activeEdges.Count() > 0) {
    while (y < edgeTable.Count()) {
        for (x = 0; x < activeEdges.Count(); x++) {
            if (activeEdges[x].end == y + bottom) {
                activeEdges.RemoveAt(x);
                x = -1;
            }
        }
        //if (y < edgeTable.Count()) {
        for (x = 0; x < edgeTable[y].Count(); x++) {
            activeEdges.Add(edgeTable[y][x]);
        }
        activeEdges.Sort(tableEdge::AETComparator);
        for (x = 0; x < activeEdges.Count(); x += 2) {
            // this is stupid, but I'm lazy
            //pixelCount += this->Line(static_cast<int>(activeEdges[x].x), y + bottom,
            //    static_cast<int>(activeEdges[x + 1].x), y + bottom, val, dryRun);
            pixelCount += static_cast<int>(activeEdges[x + 1].x) - static_cast<int>(activeEdges[x].x) + 1;
            if (!dryRun) {
                for (int i = static_cast<int>(activeEdges[x].x); i <= activeEdges[x + 1].x; i++) {
                    Set(i, y + bottom, val);
                }
            }
        }

        y++;
        for (x = 0; x < activeEdges.Count(); x++) {
            activeEdges[x].x += activeEdges[x].invSlope;
        }
    }

    return pixelCount /* * sign*/;
}

} // namespace demos_gl
} /* end namespace megamol */
