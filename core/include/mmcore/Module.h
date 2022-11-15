/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <string>
#include <vector>

#include "FrontendResource.h"
#include "FrontendResourcesMap.h"
#include "mmcore/AbstractNamedObjectContainer.h"


namespace megamol::core {

/** forward declaration */
class AbstractSlot;
namespace factories {
class ModuleDescription;
}

/**
 * Base class of all graph modules
 */
class Module : public AbstractNamedObjectContainer {
public:
    virtual std::vector<std::string> requested_lifetime_resources() {
        return {"GlobalValueStore"};
    }

    friend class ::megamol::core::factories::ModuleDescription;

    /** Shared ptr type alias */
    typedef ::std::shared_ptr<Module> ptr_type;

    /** Shared ptr type alias */
    typedef ::std::shared_ptr<const Module> const_ptr_type;

    /**
     * Utility function to dynamically cast to a shared_ptr of this type
     *
     * @param p The shared pointer to cast from
     *
     * @return A shared pointer of this type
     */
    template<class T>
    inline static ptr_type dynamic_pointer_cast(std::shared_ptr<T> p) {
        return std::dynamic_pointer_cast<Module, T>(p);
    }

    /**
     * Utility function to dynamically cast to a shared_ptr of this type
     *
     * @param p The shared pointer to cast from
     *
     * @return A shared pointer of this type
     */
    template<class T>
    inline static const_ptr_type dynamic_pointer_cast(std::shared_ptr<const T> p) {
        return std::dynamic_pointer_cast<const Module, const T>(p);
    }

    /**
     * Answer whether or not this module supports being used in a
     * quickstart. Overwrite if you don't want your module to be used in
     * quickstarts.
     *
     * This default implementation returns 'true'
     *
     * @return Whether or not this module supports being used in a
     *         quickstart.
     */
    static bool SupportQuickstart() {
        return true;
    }

    /**
     * Ctor.
     *
     * Be aware of the fact that most of your initialisation code should
     * be placed in 'create' since the ctor cannot fail and important
     * members (such as 'instance') are set after the ctor returns.
     */
    Module();

    /** Dtor. */
    ~Module() override;

    /**
     * Tries to create this module. Do not overwrite this method!
     * Overwrite 'create'!
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool Create(std::vector<megamol::frontend::FrontendResource> resources = {});

    /**
     * Finds the slot with the given name.
     *
     * @param name The name of the slot to find (case sensitive!)
     *
     * @return The found slot of this module, or 'NULL' if there is no
     *         slot with this name.
     */
    AbstractSlot* FindSlot(const vislib::StringA& name);

    template<class S>
    std::vector<S*> GetSlots();

    /**
     * Gets the name of the current instance as specified on the command
     * line, i.e. the name one level below the root namespace.
     * Caution: This can only work after the module is properly
     * inserted into the module graph and its parent is known,
     * calling it on create, e.g. will not yield satisfactory results.
     *
     * @return the according name
     */
    vislib::StringA GetDemiRootName() const;

    /**
     * Releases the module and all resources. Do not overwrite this method!
     * Overwrite 'release'!
     */
    void Release(std::vector<megamol::frontend::FrontendResource> resources = {});

    /**
     * Clears the cleanup mark for this and all dependent objects.
     */
    void ClearCleanupMark() override;

    /**
     * Performs the cleanup operation by removing and deleteing of all
     * marked objects.
     */
    void PerformCleanup() override;

    inline void SetClassName(const char* name) {
        this->className = name;
    }

    inline const char* ClassName() const {
        return this->className;
    }

    bool isCreated() const {
        return this->created;
    }

    bool AnyParameterDirty() const;

    void ResetAllDirtyFlags();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create() = 0;

    /**
     * Implementation of 'Release'.
     */
    virtual void release() = 0;

    /**
     * Makes the given slot for this module available.
     *
     * @param slot Slot to be made available.
     */
    void MakeSlotAvailable(AbstractSlot* slot);
    void SetSlotUnavailable(AbstractSlot* slot);

private:
    /** Sets the name of the module */
    void setModuleName(const vislib::StringA& name);

    /** Flag whether this module is created or not */
    bool created;

    const char* className;

    /* Allow the container to access the internal create flag */
    friend class ::megamol::core::AbstractNamedObjectContainer;

protected:
    // usage: auto const& resource = frontend_resources.get<ResourceType>()
    megamol::frontend_resources::FrontendResourcesMap frontend_resources;
};

template<class S>
std::vector<S*> Module::GetSlots() {
    child_list_type::iterator iter, end;
    iter = this->ChildList_Begin();
    end = this->ChildList_End();
    std::vector<S*> res;
    for (; iter != end; ++iter) {
        S* slot = dynamic_cast<S*>(iter->get());
        if (slot == NULL)
            continue;
        res.push_back(slot);
    }
    return res;
}


} // namespace megamol::core
