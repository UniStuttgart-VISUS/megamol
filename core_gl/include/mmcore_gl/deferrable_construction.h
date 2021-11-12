#pragma once

namespace megamol {
namespace core {

/**
 * Interface for classes that need to defer construction.
 */
class deferrable_construction {
public:
    /** dtor */
    virtual ~deferrable_construction() = default;

protected:
    /**
     * This method should hold all initializations that need to be deferred from constructor.
     *
     * @return true if containing initializations were successful. If false is returned, the object needs to be
     * deconstructed as the object state is not consistent.
     */
    virtual bool create() = 0;

    /**
     * This method should hold all de-initializations that need to be done before destruction of this object.
     */
    virtual void release() = 0;

}; // end class deferrable_construction

} // end namespace core
} // end namespace megamol
