/*
 * BlockVolumeMesh.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "BlockVolumeMesh.h"
#include "vislib/assert.h"
#include "vislib/Log.h"

using namespace megamol;
using namespace megamol::trisoup;


/*
 * BlockVolumeMesh::BlockVolumeMesh
 */
BlockVolumeMesh::BlockVolumeMesh(void) : AbstractTriMeshDataSource() {
}


/*
 * BlockVolumeMesh::~BlockVolumeMesh
 */
BlockVolumeMesh::~BlockVolumeMesh(void) {
    this->Release();
    ASSERT(this->objs.IsEmpty());
    ASSERT(this->mats.IsEmpty());
}


/*
 * BlockVolumeMesh::assertData
 */
void BlockVolumeMesh::assertData(void) {

    // TODO: Implement

}
