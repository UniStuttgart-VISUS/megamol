/*
 * AbstractNamedObjectContainer.cpp
 *
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/AbstractNamedObjectContainer.h"
#include <cstring>
#include "vislib/assert.h"
#include "vislib/sys/Log.h"
#include "vislib/String.h"
#include <algorithm>
#include "mmcore/Module.h"

using namespace megamol::core;


/*
 * AbstractNamedObjectContainer::~AbstractNamedObjectContainer
 */
AbstractNamedObjectContainer::~AbstractNamedObjectContainer(void) {
    if (this->children.size() > 0) {
        vislib::StringA msg;
        vislib::StringA name = "::";
        name.Append(this->name);
        AbstractNamedObject::ptr_type ano = this->parent.lock();
        while (ano) {
            name.Prepend(ano->Name());
            name.Prepend("::");
            ano = ano->Parent();
        }
        msg.Format("Possible memory problem detected: NamedObjectContainer (%s) with children destructed", name.PeekBuffer());
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN, msg.PeekBuffer());
        this->children.clear();
    }
    // The child list should already be empty at this time
}


/*
 * AbstractNamedObjectContainer::AbstractNamedObjectContainer
 */
AbstractNamedObjectContainer::AbstractNamedObjectContainer(void)
        : AbstractNamedObject(), children() {
    // intentionally empty
}


/*
 * AbstractNamedObjectContainer::addChild
 */
void AbstractNamedObjectContainer::addChild(AbstractNamedObject::ptr_type child) {
    if (!child) return;
    ASSERT(!child->Parent());
    this->children.push_back(child);
    Module* mod = dynamic_cast<Module *>(this);
    if (mod) {
        // for modules, calling "shared_from_this" is illegal if they have not been created!
        if (mod->created) {
            child->setParent(this->shared_from_this());
        }
    } else {
        child->setParent(this->shared_from_this());
    }
}


/*
 * AbstractNamedObjectContainer::removeChild
 */
void AbstractNamedObjectContainer::removeChild(AbstractNamedObject::ptr_type child) {
    if (!child) return;
    //ASSERT(child->Parent().get() == this);
    this->children.remove(child);
    child->setParent(AbstractNamedObject::ptr_type(nullptr));
}


/*
 * AbstractNamedObjectContainer::findChild
 */
AbstractNamedObject::ptr_type AbstractNamedObjectContainer::findChild(
        const vislib::StringA& name) {
    child_list_type::iterator end = this->children.end();
    child_list_type::iterator found = std::find_if(
        this->children.begin(), 
        end,
        [&](AbstractNamedObject::ptr_type c) {
            return c->Name().Equals(name);
        });
    return (found != end) ? *found : AbstractNamedObject::ptr_type(nullptr);
}


/*
 * AbstractNamedObjectContainer::FindNamedObject
 */
AbstractNamedObject::ptr_type AbstractNamedObjectContainer::FindNamedObject(const char *name, bool forceRooted) {
    AbstractNamedObject::ptr_type f;
    ptr_type c = std::dynamic_pointer_cast<AbstractNamedObjectContainer>(this->shared_from_this());
    const char *next = nullptr;
    vislib::StringA n;

    // skip global namespace operator if presentd
    if (::strncmp(name, "::", 2) == 0) {
        forceRooted = true;
        name += 2;
    }

    // if forced to search from the root, search from the root, otherwise we search from this object (see init of c)
    if (forceRooted) {
        c = std::dynamic_pointer_cast<AbstractNamedObjectContainer>(this->RootModule());
        if (!c) return nullptr;
    }

    // while we still have a name to search:
    while (*name != 0) {
        // search for a direct child
        if (c) {
            f = c->findChild(name);
            if (f) break;
        }

        // search for a child with the next name segment
        next = ::strstr(name, "::");
        if (next != nullptr) {
            n = vislib::StringA(name, static_cast<int>(next - name));
            name = next + 2;
        } else {
            // this is the last name segment, thus no seperator was found
            n = name;
            name += n.Length();
        }

        if (!c) {
            // not found
            return ptr_type(nullptr);
        }
        // search for a child with a fitting name
        f = c->findChild(n);
        // try casting to a container for the next loop.
        c = std::dynamic_pointer_cast<AbstractNamedObjectContainer>(f);
    }

    // return the found child
    return f;
}


/*
 * AbstractNamedObjectContainer::SetAllCleanupMarks
 */
void AbstractNamedObjectContainer::SetAllCleanupMarks(void) {
    AbstractNamedObject::SetAllCleanupMarks();
    for (AbstractNamedObject::ptr_type i : this->children) i->SetAllCleanupMarks();
}


/*
 * AbstractNamedObjectContainer::PerformCleanup
 */
void AbstractNamedObjectContainer::PerformCleanup(void) {
    AbstractNamedObject::PerformCleanup();
    // inform all children that we perform a cleanup
    for (AbstractNamedObject::ptr_type i : this->children) {
        i->PerformCleanup();
    }

    child_list_type remoov; // list of children to be removed
    for (AbstractNamedObject::ptr_type i : this->children) {
        if (i->CleanupMark()) {
            remoov.push_back(i);
        }
    }

    // actually remove
    for (AbstractNamedObject::ptr_type i : remoov) {
#if defined(DEBUG) || defined(_DEBUG)
        // Debug-check: no children to be removed should have children of their own left!
        ptr_type c = std::dynamic_pointer_cast<AbstractNamedObjectContainer>(i);
        if (c) {
            ASSERT(c->children.empty());
        }
#endif /* defined(DEBUG) || defined(_DEBUG) */

        this->removeChild(i);
    }
    remoov.clear(); // these should be the very last shared_ptr instances!
}


/*
 * AbstractNamedObjectContainer::DisconnectCalls
 */
void AbstractNamedObjectContainer::DisconnectCalls(void) {
    // propagate 'DisconnectCalls' to all children
    for (AbstractNamedObject::ptr_type i : this->children) {
        i->DisconnectCalls();
    }
}


/*
 * AbstractNamedObjectContainer::IsParamRelevant
 */
bool AbstractNamedObjectContainer::IsParamRelevant(
        vislib::SingleLinkedList<const AbstractNamedObject*>& searched,
        const vislib::SmartPtr<param::AbstractParam>& param) const {
    // ensure each container is only asked ones
    // We need this, because this call might propagate through the graph and not only through the tree
    if (searched.Contains(this)) {
        return false;
    } else {
        searched.Add(this);
    }

    // ask all children if the given parameter is relevant for them
    for (AbstractNamedObject::ptr_type i : this->children) {
        if (i->IsParamRelevant(searched, param)) {
            return true;
        }
    }

    // none is interested
    return false;
}


/*
 * AbstractNamedObjectContainer::fixParentBackreferences
 */
void AbstractNamedObjectContainer::fixParentBackreferences(void) {
    // required for lazy initialization of module slots made available in module::ctor
    for (auto i : this->children) {
        if (i->parent.expired()) {
            i->setParent(this->shared_from_this());
        }
    }
}