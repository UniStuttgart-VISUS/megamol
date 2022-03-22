#pragma once

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "chemfiles.hpp"

namespace megamol::protein {
class MoleculeLoader : public core::Module {
public:
    /** Ctor. */
    MoleculeLoader(void);

    /** Dtor. */
    virtual ~MoleculeLoader(void);

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "MoleculeLoader";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Offers molecular data.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Call callback to get the data
     *
     * @param c The calling call
     *
     * @return True on success
     */
    bool getData(core::Call& call);

    /**
     * Call callback to get the extent of the data
     *
     * @param c The calling call
     *
     * @return True on success
     */
    bool getExtent(core::Call& call);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

private:
    core::param::ParamSlot filenameSlot_;
    core::param::ParamSlot trajectoryFilenameSlot_;

    std::shared_ptr<chemfiles::Trajectory> base_;
    std::shared_ptr<chemfiles::Trajectory> trajectory_;
};
} // namespace megamol::protein
