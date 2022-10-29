#ifndef MEGAMOL_MOLDYN_VOXELGENERATOR_H_INCLUDED
#define MEGAMOL_MOLDYN_VOXELGENERATOR_H_INCLUDED

#pragma once
#include "mmcore/Module.h"

namespace megamol {
namespace moldyn_gl {
namespace rendering {

class VoxelGenerator : public core::Module { // TODO is code::Module correct?

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "VoxelGenerator";
    }

    /**
    * Answer a human readable description of this module.
    *
    * @return A human readable description of this module.
    */
    static const char* Description(void) {
        return "Generate Voxel texture for particle data"; //TODO
    }

    /**
    * Answers whether this module is available on the current system.
    *
    * @return 'true' if the module is available, 'false' otherwise.
    */
    static inline bool IsAvailable(void) {
        return true;   //TODO
    }

    /**
    * Initialises a new instance.
    */
    VoxelGenerator(void);

    /**
    * Finalises an instance.
    */
    virtual ~VoxelGenerator(void);
};

} // namespace rendering
} // namespace moldyn_gl
} // namespace megamol

#endif /* MEGAMOL_MOLDYN_VOXELGENERATOR_H_INCLUDED */
