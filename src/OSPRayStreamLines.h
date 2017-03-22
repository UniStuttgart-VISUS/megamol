/*
* OSPRayStreamLines.h
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/
#pragma once

#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "AbstractOSPRayStructure.h"
#include "protein_calls/VTIDataCall.h"
#include "VecField3f.h"

namespace megamol {
namespace ospray {

enum directionEnum {
    FORWARD,
    BACKWARD,
    BIDIRECTIONAL
};

class OSPRayStreamLines : public AbstractOSPRayStructure {

public:

    /**
    * Answer the name of this module.
    *
    * @return The name of this module.
    */
    static const char *ClassName(void) {
        return "OSPRayStreamLines";
    }

    /**
    * Answer a human readable description of this module.
    *
    * @return A human readable description of this module.
    */
    static const char *Description(void) {
        return "Creator for OSPRay stream lines.";
    }

    /**
    * Answers whether this module is available on the current system.
    *
    * @return 'true' if the module is available, 'false' otherwise.
    */
    static bool IsAvailable(void) {
        return true;
    }

    /** Dtor. */
    virtual ~OSPRayStreamLines(void);

    /** Ctor. */
    OSPRayStreamLines(void);

protected:

    virtual bool create();
    virtual void release();

    virtual bool readData(core::Call &call);
    virtual bool getExtends(core::Call &call);

    bool InterfaceIsDirty();

    /** The call for data */
    core::CallerSlot getDataSlot;


private:

    core::param::ParamSlot nStreamlinesSlot;
    core::param::ParamSlot streamlineMaxStepsSlot;
    core::param::ParamSlot streamlineStepSlot;
    core::param::ParamSlot seedClipZSlot;
    core::param::ParamSlot seedIsoSlot;
    core::param::ParamSlot streamtubesThicknessSlot;
    core::param::ParamSlot minColSlot;
    core::param::ParamSlot maxColSlot;

    core::param::ParamSlot samplingDirection;
    core::param::ParamSlot streamlineRadius;


    std::vector<float> seedPoints;
    unsigned int nStreamlines;

    void genSeedPoints(megamol::protein_calls::VTIDataCall *cd, float zClip, float isoval);

    float sampleFieldAtPosTrilin(float* pos, float* field);
    void initStreamLines(directionEnum dir);
    void integrateRK4(float startX, float startY, float startZ, unsigned int line);
    bool isValidGridPos(float* pos);

    int nSegments;

    //float* field;
    VecField3f field;
    Vec3f gridOrg;
    Vec3f gridSpacing;
    vislib::math::Vector<int,3> gridSize;


    std::vector<float> vertexData;
    std::vector<unsigned int> indexData;
    std::vector<float> tangentData;
    std::vector<float> texCoordData;

    directionEnum dir;
};


template<class T, typename... Args>
void Parallel(const Args&... args) {
    T t(args...);
    int n = t.size();
    parallel_for(blocked_range<int>(1, n), t);
}




} // namespace ospray
} // namespace megamol