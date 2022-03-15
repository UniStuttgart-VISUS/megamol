/*
 * MoleculeBallifier.h
 *
 * Copyright (C) 2012 by TU Dresden
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_MOLECULEBALLIFIER_H_INCLUDED
#define MMPROTEINPLUGIN_MOLECULEBALLIFIER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "protein_calls/ProteinColor.h"
#include "vislib/RawStorage.h"

namespace megamol {
namespace protein {

class MoleculeBallifier : public megamol::core::Module {
public:
    static const char* ClassName(void) {
        return "MoleculeBallifier";
    }
    static const char* Description(void) {
        return "MoleculeBallifier";
    }
    static bool IsAvailable(void) {
        return true;
    }
    MoleculeBallifier(void);
    virtual ~MoleculeBallifier(void);

protected:
    virtual bool create(void);
    virtual void release(void);

private:
    bool getData(core::Call& c);
    bool getExt(core::Call& c);

    core::CalleeSlot outDataSlot;
    core::CallerSlot inDataSlot;
    megamol::core::param::ParamSlot colorTableFileParam_;
    megamol::core::param::ParamSlot coloringModeParam0_;
    megamol::core::param::ParamSlot coloringModeParam1_;
    megamol::core::param::ParamSlot cmWeightParam_;
    megamol::core::param::ParamSlot minGradColorParam_;
    megamol::core::param::ParamSlot midGradColorParam_;
    megamol::core::param::ParamSlot maxGradColorParam_;
    megamol::core::param::ParamSlot specialColorParam_;

    protein_calls::ProteinColor::ColoringMode curColoringMode0_;
    protein_calls::ProteinColor::ColoringMode curColoringMode1_;

    std::vector<glm::vec3> colorArray_;
    std::vector<glm::vec3> colorLookupTable_;
    std::vector<glm::vec3> fileLookupTable_;
    std::vector<glm::vec3> rainbowColors_;

    SIZE_T inHash, outHash;
    vislib::RawStorage data;
    int frameOld;
};

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MMPROTEINPLUGIN_MOLECULEBALLIFIER_H_INCLUDED */
