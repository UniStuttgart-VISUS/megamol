/*
 * FlagCall.h
 *
 * Author: Guido Reina
 * Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#ifndef MEGAMOL_FLAGCALL_H_INCLUDED
#define MEGAMOL_FLAGCALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "FlagStorage.h"
#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/Array.h"

namespace megamol {
namespace core {

/**
 * Call for passing flag data (FlagStorage) between modules that tries to be thread-safe
 * and conflict-free by behaving like a unique pointer. A unique pointer cannot be used since
 * it interferes with the leftCall = rightCall paradigm.
 */

class MEGAMOLCORE_API FlagCall : public megamol::core::Call {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "FlagCall"; }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) { return "Call to get index-synced flag data"; }

    /** Index of the 'GetData' function */
    static const unsigned int CallMapFlags;

    static const unsigned int CallUnmapFlags;

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) { return 2; }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "mapFlags";
        case 1:
            return "unmapFlags";
        default:
            return "out_of_bounds_error";
        }
        return "";
    }

    /**
     * Extract the flags from the call, effectively removing ownership of the data from the call. When you are done
     * using the data in your callback, you need to return the data to the call!
     */
    inline std::shared_ptr<FlagStorage::FlagVectorType> GetFlags(void) {
        std::shared_ptr<FlagStorage::FlagVectorType> ret;
        ret.swap(this->flags);
        return ret;
    }

    /**
     * Return the version of the flags contained. Note that uninitialized flags carry
     * version number 0.
     */
    inline FlagStorage::FlagVersionType GetVersion() const { return this->version; }

    /**
     * Returns the pointer to the flags to the call, changing the version, thus indicating that you
     * changed the flags. This by design means you do not have ownership of the data anymore!
     */
    inline void SetFlags(std::shared_ptr<FlagStorage::FlagVectorType>& f, FlagStorage::FlagVersionType version) {
        this->flags = f;
        this->version = version;
        f.reset();
    }

    /**
     * Returns the pointer to the flags to the call, indicating that their content was not changed. This by design means
     * you do not have ownership of the data anymore!
     */
    inline void SetFlags(std::shared_ptr<FlagStorage::FlagVectorType>& f) {
        this->flags = f;
        f.reset();
    }

    /**
     * Makes sure there are enough flags for count items. The storage is initialized
     * with FlagStorage::ENABLED by default.
     */
    inline void validateFlagsCount(const uint32_t count, FlagStorage::FlagItemType init = FlagStorage::ENABLED) {
        auto f = this->flags.get();
        if (f && f->size() != count) {
            f->resize(count, init);
            ++version;
        }
    }


    FlagCall(void);
    virtual ~FlagCall(void);

private:
    std::shared_ptr<FlagStorage::FlagVectorType> flags;
    FlagStorage::FlagVersionType version;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<FlagCall> FlagCallDescription;

} // namespace core
} /* end namespace megamol */

#endif /* MEGAMOL_FLAGCALL_H_INCLUDED */
