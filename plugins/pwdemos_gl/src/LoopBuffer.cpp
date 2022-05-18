/*
 * LoopBuffer.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "LoopBuffer.h"
#include "stdafx.h"

namespace megamol {
namespace demos_gl {

#define DEFAULT_LOOP_VDATA_CAPACITY 10
#define DEFAULT_LOOP_VDATA_INCREMENT 20


/*
 * LoopBuffer::Loop::Loop
 */
LoopBuffer::Loop::Loop(void)
        : area(0)
        , bbox(0, 0, 0, 0)
        , edgeVals(static_cast<SIZE_T>(DEFAULT_LOOP_VDATA_CAPACITY), static_cast<SIZE_T>(DEFAULT_LOOP_VDATA_INCREMENT))
        , enclosingLoop(NULL)
        , vertices(static_cast<SIZE_T>(DEFAULT_LOOP_VDATA_CAPACITY), static_cast<SIZE_T>(DEFAULT_LOOP_VDATA_INCREMENT))
        , whiteArxels(0) {
    // intentionally empty
}


/*
 * LoopBuffer::Loop::Loop
 */
LoopBuffer::Loop::Loop(const LoopBuffer::Loop& src)
        : area(src.area)
        , bbox(src.bbox)
        , edgeVals(src.edgeVals)
        , enclosingLoop(src.enclosingLoop)
        , vertices(src.vertices) {
    // intentionally empty
}


/*
 * LoopBuffer::Loop::~Loop
 */
LoopBuffer::Loop::~Loop(void) {
    this->enclosingLoop = NULL; // DO NOT DELETE
}


/*
 * LoopBuffer::Loop::AddVertex
 */
void LoopBuffer::Loop::AddVertex(const vislib::math::Point<int, 2>& vertex, const ArxelBuffer::ArxelType& edge) {
    if (this->vertices.IsEmpty()) {
        this->bbox.Set(vertex.X(), vertex.Y(), vertex.X(), vertex.Y());
    } else {
        if (this->vertices.Last() == vertex)
            return; // do not add edges of length zero
        this->bbox.GrowToPoint(vertex);
    }
    this->vertices.Add(vertex);
    this->edgeVals.Add(edge);
}


/*
 * LoopBuffer::Loop::ClearVertices
 */
void LoopBuffer::Loop::ClearVertices(void) {
    this->vertices.Clear();
    this->edgeVals.Clear();
    this->bbox.SetNull();
}


/*
 * LoopBuffer::Loop::Contains
 */
bool LoopBuffer::Loop::Contains(const vislib::math::Point<int, 2>& point) const {
    unsigned int leftEdgeCnt = 0; // number of edges left of p

    int lix = point.X();
    int liy = point.Y();

    if (!this->bbox.Contains(point))
        return false;

    for (SIZE_T k = 0; k < this->Length(); k++) {
        const vislib::math::Point<int, 2>& p = this->vertices[k];
        const vislib::math::Point<int, 2>& pn = this->vertices[(k + 1) % this->vertices.Count()];
        int yMin = vislib::math::Min(p.Y(), pn.Y());
        int yMax = vislib::math::Max(p.Y(), pn.Y()); // no part of edge

        if ((yMin > liy) || (yMax <= liy))
            continue;

        int x = static_cast<int>(static_cast<float>(p.X()) + static_cast<float>(pn.X() - p.X()) *
                                                                 static_cast<float>(liy - p.Y()) /
                                                                 static_cast<float>(pn.Y() - p.Y()));

        if (x < lix) {
            leftEdgeCnt++;
        }
    }

    return ((leftEdgeCnt % 2) == 1);
}


/*
 * LoopBuffer::Loop::operator==
 */
bool LoopBuffer::Loop::operator==(const LoopBuffer::Loop& rhs) const {
    return (this->area == rhs.area) && (this->bbox == rhs.bbox) && (this->edgeVals == rhs.edgeVals) &&
           (this->enclosingLoop == rhs.enclosingLoop) && (this->vertices == rhs.vertices);
}


/*
 * LoopBuffer::Loop::operator=
 */
LoopBuffer::Loop& LoopBuffer::Loop::operator=(const LoopBuffer::Loop& rhs) {
    this->area = rhs.area;
    this->bbox = rhs.bbox;
    this->edgeVals = rhs.edgeVals;
    this->enclosingLoop = rhs.enclosingLoop;
    this->vertices = rhs.vertices;
    return *this;
}


/*
 * LoopBuffer::LoopBuffer
 */
LoopBuffer::LoopBuffer(void) : loops() {
    // intentionally empty
}


/*
 * LoopBuffer::~LoopBuffer
 */
LoopBuffer::~LoopBuffer(void) {
    // intentionally empty
}


/*
 * LoopBuffer::Clear
 */
void LoopBuffer::Clear(void) {
    this->loops.Clear();
}


/*
 * LoopBuffer::NewLoop
 */
LoopBuffer::Loop& LoopBuffer::NewLoop(void) {
    this->loops.Add(Loop());
    return this->loops.Last();
}


/*
 * LoopBuffer::NewLoopComplete
 */
void LoopBuffer::NewLoopComplete(void) {
    bool emptyLoop = (this->loops.Last().Length() <= 2); // degenerated loop

    if (this->loops.Last().Length() == 3) { // test for degenerated triangles
        const vislib::math::Point<int, 2> p1 = this->loops.Last().Vertex(0);
        const vislib::math::Point<int, 2> p2 = this->loops.Last().Vertex(1);
        const vislib::math::Point<int, 2> p3 = this->loops.Last().Vertex(2);
        const vislib::math::Vector<int, 2> v1 = p1 - p2;
        const vislib::math::Vector<int, 2> v2 = p1 - p3;
        // test for degenerated triangles
        if (v1.IsNull() || v2.IsNull() || v1.IsParallel(v2)) {
            emptyLoop = true;
        }
    }

    if (emptyLoop) {
        this->loops.RemoveLast();
    }
}

} // namespace demos_gl
} /* end namespace megamol */
