/*
* OSPRayStreamLines.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayStreamLines.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/EnumParam.h"
#include "vislib/sys/Log.h"
#include "mmcore/Call.h"
#include "tbb/tbb.h"


using namespace megamol::ospray;


OSPRayStreamLines::OSPRayStreamLines(void) :
    AbstractOSPRayStructure(),
    /* Streamline integration parameters */
    nStreamlinesSlot("nStreamlines", "Set the number of streamlines"),
    streamlineMaxStepsSlot("nSteps", "Set the number of steps for streamline integration"),
    streamlineStepSlot("step", "Set stepsize for the streamline integration"),
    seedClipZSlot("seedClipZ", "Starting z value for the clipping plane"),
    seedIsoSlot("seedIso", "Iso value for the seed point"),
    streamtubesThicknessSlot("tubesScl", "The scale factor for the streamtubes thickness"),
    minColSlot("minCol", "Minimum color value"),
    maxColSlot("maxCol", "Maximum color value"),
    streamlineEpsilon("epsilon", "Minimum value of a vector"),
    samplingDirection("samplingDirection", "direction of streamline sampling"),
    streamlineRadius("radius", "Radius of the streamlines"),

    getDataSlot("getdata", "Connects to the data source") {

    core::param::EnumParam *sd = new core::param::EnumParam(directionEnum::FORWARD);
    sd->SetTypePair(directionEnum::FORWARD, "Forward");
    sd->SetTypePair(directionEnum::BIDIRECTIONAL, "Bidirectional");
    sd->SetTypePair(directionEnum::BACKWARD, "Backward");
    this->samplingDirection << sd;
    this->MakeSlotAvailable(&this->samplingDirection);

    // Set the number of streamlines
    this->nStreamlinesSlot.SetParameter(new core::param::IntParam(10, 0));
    this->MakeSlotAvailable(&this->nStreamlinesSlot);

    // Set the number of steps for streamline integration
    this->streamlineMaxStepsSlot.SetParameter(new core::param::IntParam(10));
    this->MakeSlotAvailable(&this->streamlineMaxStepsSlot);

    // Set the step size for streamline integration
    this->streamlineStepSlot.SetParameter(new core::param::FloatParam(1.0f));
    this->MakeSlotAvailable(&this->streamlineStepSlot);

    // Set the step size for streamline integration
    this->seedClipZSlot.SetParameter(new core::param::FloatParam(0.5f, 0.0f));
    this->MakeSlotAvailable(&this->seedClipZSlot);

    // Set the step size for streamline integration
    this->seedIsoSlot.SetParameter(new core::param::FloatParam(0.5f));
    this->MakeSlotAvailable(&this->seedIsoSlot);

    this->streamlineEpsilon.SetParameter(new core::param::FloatParam(0.1f, 0.0f));
    this->MakeSlotAvailable(&this->streamlineEpsilon);

    this->streamlineRadius.SetParameter(new core::param::FloatParam(1.0f));
    this->MakeSlotAvailable(&this->streamlineRadius);

    this->getDataSlot.SetCompatibleCall<megamol::protein_calls::VTIDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

}


bool OSPRayStreamLines::readData(megamol::core::Call &call) {

    // read Data, calculate  shape parameters, fill data vectors

    CallOSPRayStructure *os = dynamic_cast<CallOSPRayStructure*>(&call);
    megamol::protein_calls::VTIDataCall *cd = this->getDataSlot.CallAs<megamol::protein_calls::VTIDataCall>();

    this->structureContainer.dataChanged = false;
    if (cd == NULL) return false;
    if (os->getTime() > cd->FrameCount()) {
        cd->SetFrameID(cd->FrameCount() - 1, true); // isTimeForced flag set to true
    } else {
        cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    }
    if (this->datahash != cd->DataHash() || this->time != os->getTime() || this->InterfaceIsDirty()) {
        this->datahash = cd->DataHash();
        this->time = os->getTime();
        this->structureContainer.dataChanged = true;
    } else {
        return true;
    }


    if (!(*cd)(megamol::protein_calls::VTIDataCall::CallForGetExtent)) return false;
    if (!(*cd)(megamol::protein_calls::VTIDataCall::CallForGetData)) return false;

    this->gridSize = cd->GetGridsize();
    this->gridOrg = cd->GetOrigin();
    this->gridSpacing = cd->GetSpacing();

    //this->field = (float*)cd->GetPointDataByIdx(0, 0);
    this->field.SetData((float*)cd->GetPointDataByIdx(1, 0), gridSize[0], gridSize[1], gridSize[2],
        gridSpacing[0], gridSpacing[1], gridSpacing[2],
        gridOrg[0], gridOrg[1], gridOrg[2]);


    // implement streamline calculation
    this->initStreamLines((directionEnum)samplingDirection.Param<core::param::EnumParam>()->Value());

    float zHeight = (gridSize.GetZ() - 1) * gridSpacing.GetZ();
    this->genSeedPoints(cd, zHeight*this->seedClipZSlot.Param<core::param::FloatParam>()->Value(), this->seedIsoSlot.Param<core::param::FloatParam>()->Value()); // Isovalues


    //this->vertexData.assign(this->seedPoints.begin(), this->seedPoints.end());

    for (unsigned int i = 0; i < nStreamlines; i++) {
        this->integrateRK4(this->seedPoints[3*i+0], this->seedPoints[3 * i + 1], this->seedPoints[3 * i + 2]);
    }


    // Write stuff into the structureContainer

    this->structureContainer.type = structureTypeEnum::GEOMETRY;
    //this->structureContainer.geometryType = geometryTypeEnum::STREAMLINES;
    this->structureContainer.geometryType = geometryTypeEnum::SPHERES;
    this->structureContainer.partCount = this->vertexData.size()/3;
    this->structureContainer.vertexData = std::make_shared<std::vector<float>>(std::move(this->vertexData));
    this->structureContainer.vertexLength = 3;
    this->structureContainer.colorLength = 0;
    this->structureContainer.globalRadius = streamlineRadius.Param<core::param::FloatParam>()->Value();


    // material container
    CallOSPRayMaterial *cm = this->getMaterialSlot.CallAs<CallOSPRayMaterial>();
    if (cm != NULL) {
        auto gmp = cm->getMaterialParameter();
        if (gmp->isValid) {
            this->structureContainer.materialContainer = cm->getMaterialParameter();
        }
    } else {
        this->structureContainer.materialContainer = NULL;
    }

    return true;
}


OSPRayStreamLines::~OSPRayStreamLines() {
    //
}

bool OSPRayStreamLines::create() {
    return true;
}

void OSPRayStreamLines::release() {

}

/*
ospray::OSPRaySphereGeometry::InterfaceIsDirty()
*/
bool OSPRayStreamLines::InterfaceIsDirty() {
    CallOSPRayMaterial *cm = this->getMaterialSlot.CallAs<CallOSPRayMaterial>();
    cm->getMaterialParameter();
    if (
        cm->InterfaceIsDirty() ||
        this->nStreamlinesSlot.IsDirty() ||
        this->streamlineMaxStepsSlot.IsDirty() ||
        this->streamlineStepSlot.IsDirty() ||
        this->seedClipZSlot.IsDirty() ||
        this->seedIsoSlot.IsDirty() ||
        this->streamtubesThicknessSlot.IsDirty() ||
        this->minColSlot.IsDirty() ||
        this->maxColSlot.IsDirty() ||
        this->samplingDirection.IsDirty() ||
        this->streamlineEpsilon.IsDirty() || 
        this->streamlineRadius.IsDirty()
        ) {
        this->nStreamlinesSlot.ResetDirty();
        this->streamlineMaxStepsSlot.ResetDirty();
        this->streamlineStepSlot.ResetDirty();
        this->seedClipZSlot.ResetDirty();
        this->seedIsoSlot.ResetDirty();
        this->streamtubesThicknessSlot.ResetDirty();
        this->minColSlot.ResetDirty();
        this->maxColSlot.ResetDirty();
        this->samplingDirection.ResetDirty();
        this->streamlineEpsilon.ResetDirty();
        this->streamlineRadius.ResetDirty();

        return true;
    } else {
        return false;
    }
}



bool OSPRayStreamLines::getExtends(megamol::core::Call &call) {
    CallOSPRayStructure *os = dynamic_cast<CallOSPRayStructure*>(&call);
    megamol::protein_calls::VTIDataCall *cd = this->getDataSlot.CallAs<megamol::protein_calls::VTIDataCall>();

    if (cd == NULL) return false;
    if (os->getTime() > cd->FrameCount()) {
        cd->SetFrameID(cd->FrameCount() - 1, true);  // isTimeForced flag set to true
    } else {
        cd->SetCalltime(os->getTime());
        cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    }

    if (!(*cd)(megamol::protein_calls::VTIDataCall::CallForGetExtent)) return false;

    this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes>(cd->AccessBoundingBoxes());
    this->extendContainer.timeFramesCount = cd->FrameCount();
    this->extendContainer.isValid = true;

    return true;
}


void OSPRayStreamLines::genSeedPoints(megamol::protein_calls::VTIDataCall *cd, float zClip, float isoval) {

    float posZ = gridOrg.GetZ() + zClip; // Start above the lower boundary

    float xMax = gridOrg.GetX() + gridSpacing.GetX()*(gridSize.GetX() - 1);
    float yMax = gridOrg.GetY() + gridSpacing.GetY()*(gridSize.GetY() - 1);
    float xMin = gridOrg.GetX();
    float yMin = gridOrg.GetY();

    // Initialize random seed
    //srand(static_cast<unsigned int>(::time(NULL)));
    srand(10);
    this->seedPoints.clear();

    while (this->seedPoints.size() / 3 < this->nStreamlines) {
        std::vector<float> pos(3);
        pos[0] = (gridOrg.GetX() + (float(rand() % 10000) / 10000.0f)*(xMax - xMin));
        pos[1] = (gridOrg.GetY() + (float(rand() % 10000) / 10000.0f)*(yMax - yMin));
        pos[2] = (posZ + (float(rand() % 10000) / 10000.0f)*(10));

        float sample = this->sampleFieldAtPosTrilin(pos.data(), (float*)cd->GetPointDataByIdx(0,0));
        //Vec3f vec = field.GetAtTrilin(Vec3f(pos[0], pos[1], pos[2]));
        //float sample = vec.Length();

        // Sample density value
        if ((sample - isoval) > 0.00) {
            this->seedPoints.push_back(pos[0]);
            this->seedPoints.push_back(pos[1]);
            this->seedPoints.push_back(pos[2]);
        }
    }
}

/*
Vec3f OSPRayStreamLines::sampleFieldAtPosTrilin_vec3f(float* pos) {
    float cx, cy, cz;
    cx = (pos[0] - gridOrg[0]) / gridSpacing[0];
    cy = (pos[1] - gridOrg[1]) / gridSpacing[1];
    cz = (pos[2] - gridOrg[2]) / gridSpacing[2];

    vislib::math::Vector<unsigned int,3> cellId;
    cellId[0] = static_cast<unsigned int>(cx);
    cellId[1] = static_cast<unsigned int>(cy);
    cellId[2] = static_cast<unsigned int>(cz);

    cx -= cellId[0]; // alpha
    cy -= cellId[1]; // beta
    cz -= cellId[2]; // gamma

    // Get neighbour vecs
    Vec3f  n[8];
    n[0] = this->GetAt(cellId.X() + 0, cellId.Y() + 0, cellId.Z() + 0);
    n[1] = this->GetAt(cellId.X() + 1, cellId.Y() + 0, cellId.Z() + 0);
    n[2] = this->GetAt(cellId.X() + 0, cellId.Y() + 1, cellId.Z() + 0);
    n[3] = this->GetAt(cellId.X() + 1, cellId.Y() + 1, cellId.Z() + 0);
    n[4] = this->GetAt(cellId.X() + 0, cellId.Y() + 0, cellId.Z() + 1);
    n[5] = this->GetAt(cellId.X() + 1, cellId.Y() + 0, cellId.Z() + 1);
    n[6] = this->GetAt(cellId.X() + 0, cellId.Y() + 1, cellId.Z() + 1);
    n[7] = this->GetAt(cellId.X() + 1, cellId.Y() + 1, cellId.Z() + 1);


    // Use trilinear interpolation to sample the volume
    return protein_calls::Interpol::Trilin(n[0], n[1], n[2], n[3],
        n[4], n[5], n[6], n[7], cx, cy, cz);
}
*/


float OSPRayStreamLines::sampleFieldAtPosTrilin(float* pos, float* field) {
    std::vector<int> c(3);
    std::vector<float> f(3);

    // Get id of the cell containing the given position and interpolation
    // coefficients
    f[0] = (pos[0] - gridOrg[0]) / gridSpacing[0];
    f[1] = (pos[1] - gridOrg[1]) / gridSpacing[1];
    f[2] = (pos[2] - gridOrg[2]) / gridSpacing[2];
    c[0] = (int)(f[0]);
    c[1] = (int)(f[1]);
    c[2] = (int)(f[2]);
    f[0] = f[0] - (float)c[0]; // alpha
    f[1] = f[1] - (float)c[1]; // beta
    f[2] = f[2] - (float)c[2]; // gamma

    c[0] = vislib::math::Clamp(c[0], int(0), gridSize[0] - 2);
    c[1] = vislib::math::Clamp(c[1], int(0), gridSize[1] - 2);
    c[2] = vislib::math::Clamp(c[2], int(0), gridSize[2] - 2);

    // Get values at corners of current cell
    float s[8];
    s[0] = field[gridSize[0] * (gridSize[1] * (c[2] + 0) + (c[1] + 0)) + c[0] + 0];
    s[1] = field[gridSize[0] * (gridSize[1] * (c[2] + 0) + (c[1] + 0)) + c[0] + 1];
    s[2] = field[gridSize[0] * (gridSize[1] * (c[2] + 0) + (c[1] + 1)) + c[0] + 0];
    s[3] = field[gridSize[0] * (gridSize[1] * (c[2] + 0) + (c[1] + 1)) + c[0] + 1];
    s[4] = field[gridSize[0] * (gridSize[1] * (c[2] + 1) + (c[1] + 0)) + c[0] + 0];
    s[5] = field[gridSize[0] * (gridSize[1] * (c[2] + 1) + (c[1] + 0)) + c[0] + 1];
    s[6] = field[gridSize[0] * (gridSize[1] * (c[2] + 1) + (c[1] + 1)) + c[0] + 0];
    s[7] = field[gridSize[0] * (gridSize[1] * (c[2] + 1) + (c[1] + 1)) + c[0] + 1];


    // Use trilinear interpolation to sample the volume
    return protein_calls::Interpol::Trilin(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], f[0], f[1], f[2]);
}


/*
* OSPRayStreamLines::InitStreamLines
*/
void OSPRayStreamLines::initStreamLines(directionEnum dir) {
    if ((dir == directionEnum::FORWARD) || (dir == directionEnum::BACKWARD)) {
        this->nSegments = this->streamlineMaxStepsSlot.Param<core::param::IntParam>()->Value();
    } else if (dir == directionEnum::BIDIRECTIONAL) {
        this->nSegments = this->streamlineMaxStepsSlot.Param<core::param::IntParam>()->Value() * 2;
    }

    this->nStreamlines = this->nStreamlinesSlot.Param<core::param::IntParam>()->Value();
    this->dir = dir;

}


/*
 * OSPRayStreamLines::IntegrateRK4
 */
void OSPRayStreamLines::integrateRK4(float startX, float startY, float startZ) {

    float eps = streamlineEpsilon.Param<core::param::FloatParam>()->Value();


    float step = streamlineStepSlot.Param<core::param::FloatParam>()->Value() /10;

    bool vanishing = false, gridLeft = false;
    unsigned int l0 = 0, l1 = 0;
    Vec3f v0, v1, v2, v3, x0, x1, x2, x3, color;

    // 1. Forward
    if ((this->dir == directionEnum::FORWARD) || (dir == directionEnum::BIDIRECTIONAL)) {

        x0.Set(startX, startY, startZ);

        // Test whether the grid has been left
        if (!isValidGridPos(x0.PeekComponents())) {
            gridLeft = true;
        }

        //unsigned int id;
        //while (!(vanishing || gridLeft || l0 > this->nSegments)) {
        for (unsigned int id = 0; id <= this->nSegments; id++) {
            v0.Set(0.0f, 0.0f, 0.0f);
            v1.Set(0.0f, 0.0f, 0.0f);
            v2.Set(0.0f, 0.0f, 0.0f);
            v3.Set(0.0f, 0.0f, 0.0f);

            // Find new position using fourth order Runge-Kutta method
            if (isValidGridPos(x0.PeekComponents())) {
                //v0 = sampleFieldAtPosTrilin_vec3f(x0.PeekComponents());
                v0 = this->field.GetAtTrilin(x0);
                //v0 = Vec3f(0.0f, 0.0f, 1.0f);
                //if (v0.Norm() <= eps) vanishing = true;
                v0.Normalise();
                v0 *= step;
            } else {
                this->vertexData.push_back(x0.X());
                this->vertexData.push_back(x0.Y());
                this->vertexData.push_back(x0.Z());

                this->indexData.push_back(id);
                break;
            }

            x1 = x0 + 0.5f*v0;
            if (isValidGridPos(x1.PeekComponents())) {
                //v1 = sampleFieldAtPosTrilin_vec3f(x1.PeekComponents());
                v1 = this->field.GetAtTrilin(x1);
                v1.Normalise();
                v1 *= step;
            } else {
                this->vertexData.push_back(x0.X());
                this->vertexData.push_back(x0.Y());
                this->vertexData.push_back(x0.Z());

                this->indexData.push_back(id);
                break;
            }

            x2 = x0 + 0.5f*v1;
            if (isValidGridPos(x2.PeekComponents())) {
                //v2 = sampleFieldAtPosTrilin_vec3f(x2.PeekComponents());
                v2 = this->field.GetAtTrilin(x2);
                v2.Normalise();
                v2 *= step;
            } else {
                this->vertexData.push_back(x0.X());
                this->vertexData.push_back(x0.Y());
                this->vertexData.push_back(x0.Z());

                this->indexData.push_back(id);
                break;
            }

            x3 = x0 + v2;
            if (isValidGridPos(x3.PeekComponents())) {
                //v3 = sampleFieldAtPosTrilin_vec3f(x3.PeekComponents());
                v3 = this->field.GetAtTrilin(x3);
                v3.Normalise();
                v3 *= step;
            } else {
                this->vertexData.push_back(x0.X());
                this->vertexData.push_back(x0.Y());
                this->vertexData.push_back(x0.Z());

                this->indexData.push_back(id);
                break;
            }


            x0 += (1.0f / 6.0f)*(v0 + 2.0f*v1 + 2.0f*v2 + v3);

            // Add position and tangent to streamline
            this->vertexData.push_back(x0.X());
            this->vertexData.push_back(x0.Y());
            this->vertexData.push_back(x0.Z());

            this->indexData.push_back(id);
            ++id;

            this->tangentData.push_back(v0.X());
            this->tangentData.push_back(v0.Y());
            this->tangentData.push_back(v0.Z());

            this->texCoordData.push_back((x0.X() - gridOrg.GetX()) / ((gridSize.GetX() - 1)*gridSpacing.GetX()));
            this->texCoordData.push_back((x0.Y() - gridOrg.GetY()) / ((gridSize.GetY() - 1)*gridSpacing.GetY()));
            this->texCoordData.push_back((x0.Z() - gridOrg.GetZ()) / ((gridSize.GetZ() - 1)*gridSpacing.GetZ()));

            // Test whether the grid has been left
            if (!isValidGridPos(x0.PeekComponents())) {
                gridLeft = true;
            }

            l0++;
        }
    }

    // 2. Backward
    if ((dir == directionEnum::BACKWARD) || (dir == directionEnum::BIDIRECTIONAL)) {

        x0.Set(startX, startY, startZ);
        vanishing = false;
        gridLeft = false;

        // Test whether the grid has been left
        if (!isValidGridPos(x0.PeekComponents())) {
            gridLeft = true;
        }

        //unsigned int id = 0;
        //while (!(vanishing || gridLeft || l1 >= this->nSegments)) {
        for (unsigned int id = 0; id <= this->nSegments; id++) {
            v0.Set(0.0f, 0.0f, 0.0f);
            v1.Set(0.0f, 0.0f, 0.0f);
            v2.Set(0.0f, 0.0f, 0.0f);
            v3.Set(0.0f, 0.0f, 0.0f);

            // Find new position using fourth order Runge-Kutta method
            if (isValidGridPos(x0.PeekComponents())) {
                //v0 = sampleFieldAtPosTrilin_vec3f(x0.PeekComponents());
                v0 = this->field.GetAtTrilin(x0);
                //if (v0.Norm() <= eps) vanishing = true;
                v0.Normalise();
                v0 *= step;
            } else {
                this->vertexData.push_back(x0.X());
                this->vertexData.push_back(x0.Y());
                this->vertexData.push_back(x0.Z());

                this->indexData.push_back(id);
                break;
            }

            x1 = x0 - 0.5f*v0;
            if (isValidGridPos(x1.PeekComponents())) {
                //v1 = sampleFieldAtPosTrilin_vec3f(x1.PeekComponents());
                v1 = this->field.GetAtTrilin(x1);
                v1.Normalise();
                v1 *= step;
            } else {
                this->vertexData.push_back(x0.X());
                this->vertexData.push_back(x0.Y());
                this->vertexData.push_back(x0.Z());

                this->indexData.push_back(id);
                break;
            }

            x2 = x0 - 0.5f*v1;
            if (isValidGridPos(x2.PeekComponents())) {
                //v2 = sampleFieldAtPosTrilin_vec3f(x2.PeekComponents());
                v2 = this->field.GetAtTrilin(x2);
                v2.Normalise();
                v2 *= step;
            } else {
                this->vertexData.push_back(x0.X());
                this->vertexData.push_back(x0.Y());
                this->vertexData.push_back(x0.Z());

                this->indexData.push_back(id);
                break;
            }

            x3 = x0 - v2;
            if (isValidGridPos(x3.PeekComponents())) {
                //v3 = sampleFieldAtPosTrilin_vec3f(x3.PeekComponents());
                v3 = this->field.GetAtTrilin(x3);
                v3.Normalise();
                v3 *= step;
            } else {
                this->vertexData.push_back(x0.X());
                this->vertexData.push_back(x0.Y());
                this->vertexData.push_back(x0.Z());

                this->indexData.push_back(id);
                break;
            }


            x0 -= (1.0f / 6.0f)*(v0 + 2.0f*v1 + 2.0f*v2 + v3);

            // Add position and tangent to streamline

            this->vertexData.push_back(x0.X());
            this->vertexData.push_back(x0.Y());
            this->vertexData.push_back(x0.Z());

            this->indexData.push_back(id);
            ++id;

            this->tangentData.push_back(v0.X());
            this->tangentData.push_back(v0.Y());
            this->tangentData.push_back(v0.Z());

            this->texCoordData.push_back((x0.X() - gridOrg.GetX()) / ((gridSize.GetX() - 1)*gridSpacing.GetX()));
            this->texCoordData.push_back((x0.Y() - gridOrg.GetY()) / ((gridSize.GetY() - 1)*gridSpacing.GetY()));
            this->texCoordData.push_back((x0.Z() - gridOrg.GetZ()) / ((gridSize.GetZ() - 1)*gridSpacing.GetZ()));

            // Test whether the grid has been left
            if (!isValidGridPos(x0.PeekComponents())) {
                gridLeft = true;
            }

            l1++;
        }
    }
}


/* OSPRayStreamLines::IsValidGridpos */
bool OSPRayStreamLines::isValidGridPos(float* pos) {
    if (pos[0] < gridOrg.GetX()) return false;
    if (pos[1] < gridOrg.GetY()) return false;
    if (pos[2] < gridOrg.GetZ()) return false;
    if (pos[0] >= (gridOrg.GetX() + (gridSize.GetX() - 1)*gridSpacing.GetX())) return false;
    if (pos[1] >= (gridOrg.GetY() + (gridSize.GetY() - 1)*gridSpacing.GetY())) return false;
    if (pos[2] >= (gridOrg.GetZ() + (gridSize.GetZ() - 1)*gridSpacing.GetZ())) return false;
    return true;
}




/* OSPRayStreamLines::GetAt */
/*
Vec3f OSPRayStreamLines::GetAt(unsigned int posX, unsigned int posY, unsigned int posZ) {

    ASSERT(field != NULL);

    //printf("POS %u %u %u\n", posX, posY, posZ);

    ASSERT(posX < gridSize[0]);
    ASSERT(posY < gridSize[1]);
    ASSERT(posZ < gridSize[2]);

    ASSERT(posX >= 0);
    ASSERT(posY >= 0);
    ASSERT(posZ >= 0);

    auto index_x = (gridSize[0] * (gridSize[1] * posZ + posY) + posX) + 0;
    auto index_y = (gridSize[0] * (gridSize[1] * posZ + posY) + posX) + 1;
    auto index_z = (gridSize[0] * (gridSize[1] * posZ + posY) + posX) + 2;

    return Vec3f(field[index_x], field[index_y], field[index_z]);
}

*/