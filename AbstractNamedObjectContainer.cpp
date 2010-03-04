/*
 * AbstractNamedObjectContainer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "AbstractNamedObjectContainer.h"
#include <cstring>
#include "vislib/assert.h"
#include "vislib/Log.h"
#include "vislib/String.h"

using namespace megamol::core;


/*
 * AbstractNamedObjectContainer::~AbstractNamedObjectContainer
 */
AbstractNamedObjectContainer::~AbstractNamedObjectContainer(void) {
    if (this->children.Count() > 0) {
        vislib::StringA msg;
        vislib::StringA name = "::";
        name.Append(this->name);
        AbstractNamedObject *ano = this->parent;
        while (ano != NULL) {
            name.Prepend(ano->Name());
            name.Prepend("::");
            ano = ano->Parent();
        }
        msg.Format("Possible memory problem detected: NamedObjectContainer (%s) with children destructed",
            name.PeekBuffer());
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN, msg.PeekBuffer());
        this->children.Clear();
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
void AbstractNamedObjectContainer::addChild(AbstractNamedObject *child) {
    if (child == NULL) return;
    ASSERT(child->Parent() == NULL);
    this->children.Add(child);
    child->setParent(this);
}


/*
 * AbstractNamedObjectContainer::removeChild
 */
void AbstractNamedObjectContainer::removeChild(AbstractNamedObject *child) {
    if (child == NULL) return;
    ASSERT(child->Parent() == this);
    this->children.Remove(child);
    child->setParent(NULL);
}


/*
 * AbstractNamedObjectContainer::getChildIterator
 */
AbstractNamedObjectContainer::ChildList::Iterator
AbstractNamedObjectContainer::getChildIterator(void) {
    return this->children.GetIterator();
}


/*
 * AbstractNamedObjectContainer::getConstChildIterator
 */
vislib::ConstIterator<AbstractNamedObjectContainer::ChildList::Iterator>
AbstractNamedObjectContainer::getConstChildIterator(void) const {
    return this->children.GetConstIterator();
}


/*
 * AbstractNamedObjectContainer::findChild
 */
AbstractNamedObject *AbstractNamedObjectContainer::findChild(
        const vislib::StringA& name) {
    ChildList::Iterator iter = this->children.GetIterator();
    while (iter.HasNext()) {
        AbstractNamedObject *child = iter.Next();
        if (child->Name().Equals(name)) {
            return child;
        }
    }
    return NULL;
}


/*
 * AbstractNamedObjectContainer::FindNamedObject
 */
AbstractNamedObject *AbstractNamedObjectContainer::FindNamedObject(const char *name, bool forceRooted) {
    AbstractNamedObject *f = NULL;
    AbstractNamedObjectContainer *c = this;
    const char *next = NULL;
    vislib::StringA n;

    if (::strncmp(name, "::", 2) == 0) {
        forceRooted = true;
        name += 2;
    }
    if (forceRooted) {
        c = dynamic_cast<AbstractNamedObjectContainer*>(this->RootModule());
        if (c == NULL) {
            return NULL;
        }
    }

    while (*name != 0) {
        next = ::strstr(name, "::");
        if (next != NULL) {
            n = vislib::StringA(name, static_cast<int>(next - name));
            name = next + 2;
        } else {
            n = name;
            name += n.Length();
        }
        if (c == NULL) {
            return NULL;
        }
        f = c->findChild(n);
        c = dynamic_cast<AbstractNamedObjectContainer*>(f);
    }

    return f;
}


/*
 * AbstractNamedObjectContainer::SetAllCleanupMarks
 */
void AbstractNamedObjectContainer::SetAllCleanupMarks(void) {
    AbstractNamedObject::SetAllCleanupMarks();
    ChildList::Iterator iter = this->children.GetIterator();
    while (iter.HasNext()) {
        iter.Next()->SetAllCleanupMarks();
    }
}


/*
 * AbstractNamedObjectContainer::PerformCleanup
 */
void AbstractNamedObjectContainer::PerformCleanup(void) {
    AbstractNamedObject::PerformCleanup();

    AbstractNamedObject *ano;
    ChildList::Iterator iter = this->children.GetIterator();
    while (iter.HasNext()) {
        iter.Next()->PerformCleanup();
    }

    ChildList remoov;
    iter = this->getChildIterator();
    while (iter.HasNext()) {
        ano = iter.Next();
        if (ano->CleanupMark()) {
            remoov.Add(ano);
        }
    }

    iter = remoov.GetIterator();
    while (iter.HasNext()) {
        ano = iter.Next();

#if defined(DEBUG) || defined(_DEBUG)
        // no children should have any further children
        AbstractNamedObjectContainer *anoc = dynamic_cast<AbstractNamedObjectContainer*>(ano);
        if (anoc != NULL) {
            ASSERT(!anoc->GetChildIterator().HasNext());
        }
#endif /* defined(DEBUG) || defined(_DEBUG) */

        this->removeChild(ano);
        delete ano; // <= woho! Finally!
    }

}


/*
 * AbstractNamedObjectContainer::DisconnectCalls
 */
void AbstractNamedObjectContainer::DisconnectCalls(void) {
    ChildList::Iterator iter = this->children.GetIterator();
    while (iter.HasNext()) {
        iter.Next()->DisconnectCalls();
    }
}


/*
 * AbstractNamedObjectContainer::IsParamRelevant
 */
bool AbstractNamedObjectContainer::IsParamRelevant(
        vislib::SingleLinkedList<const AbstractNamedObject*>& searched,
        const vislib::SmartPtr<param::AbstractParam>& param) const {
    if (searched.Contains(this)) {
        return false;
    } else {
        searched.Add(this);
    }

    vislib::ConstIterator<ChildList::Iterator> iter
        = this->children.GetConstIterator();
    while (iter.HasNext()) {
        if (iter.Next()->IsParamRelevant(searched, param)) {
            return true;
        }
    }

    return false;
}
