#pragma once

#include <unordered_map>
#include <vector>
#include "mmcore/AbstractGetData3DCall.h"

#include "thermodyn.h"

namespace megamol {
namespace thermodyn {

class thermodyn_API PathLineDataCall : public megamol::core::AbstractGetData3DCall {
public:
    using pathline_t = std::vector<float>;
    using pathline_store_t = std::unordered_map<uint64_t, pathline_t>;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName(void) { return "PathLineDataCall"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description(void) { return "Transports pathlines."; }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) { return AbstractGetData3DCall::FunctionCount(); }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) { return AbstractGetData3DCall::FunctionName(idx); }

    std::vector<int> const& GetEntrySize() const { return entrySizes_; }

    std::vector<bool> const& HasColors() const { return colsPresent_; }

    std::vector<bool> const& HasDirections() const { return dirsPresent_; }

    std::vector<pathline_store_t> const* GetPathStore() const { return pathStore_; }

    void SetEntrySizes(std::vector<int> const& entrySizes) { entrySizes_ = entrySizes; }

    void SetColorFlags(std::vector<bool> const& colsPresent) { colsPresent_ = colsPresent; }

    void SetDirFlags(std::vector<bool> const& dirsPresent) { dirsPresent_ = dirsPresent; }

    void SetPathStore(std::vector<pathline_store_t> const* pathStore) { pathStore_ = pathStore; }

private:
    std::vector<int> entrySizes_;

    std::vector<bool> dirsPresent_;

    std::vector<bool> colsPresent_;

    std::vector<pathline_store_t> const* pathStore_ = nullptr;
}; // end class PathLineDataCall

/** Call Descriptor.  */
typedef core::factories::CallAutoDescription<PathLineDataCall> PathLineDataCallDescription;

} // end namespace thermodyn
} // end namespace megamol
