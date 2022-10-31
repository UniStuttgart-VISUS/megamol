#ifndef MEGAMOL_MOLDYN_VOXELGENERATOR_H_INCLUDED
#define MEGAMOL_MOLDYN_VOXELGENERATOR_H_INCLUDED

#pragma once


#include "geometry_calls/MultiParticleDataCall.h"
#include "geometry_calls/VolumetricDataCall.h"

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
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
        return "Generate voxel texture for particle data"; //TODO
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

protected:

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

     /**
     * Generates Voxels.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool onGenerateVoxels(core::Call& call);

private:
    ///**
    // * TODO: Document
    // *
    // * @param t           ...
    // * @param outScaling  ...
    // *
    // * @return Pointer to MultiParticleDataCall ...
    // */
    //MultiParticleDataCall* getData(unsigned int t, float& out_scaling);

    /** The slot that requests the data. */
    core::CalleeSlot generate_voxels_slot_; //TODO VolumetricDataCall

    core::CallerSlot get_data_slot_; // MultiParticleDataCall 

    /**
     * TODO: Document
     *
     * @param t           ...
     * @param outScaling  ...
     *
     * @return Pointer to MultiParticleDataCall ...
     */
    geocalls::MultiParticleDataCall* getData(unsigned int t, float& out_scaling); // get particle data
};

} // namespace rendering
} // namespace moldyn_gl
} // namespace megamol

#endif /* MEGAMOL_MOLDYN_VOXELGENERATOR_H_INCLUDED */
