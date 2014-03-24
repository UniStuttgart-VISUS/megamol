/*
 * ShaderSource.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/ShaderSource.h"
#include <cstdarg>
#include "the/already_exists_exception.h"
#include "the/memory.h"
#include "the/no_such_element_exception.h"
#include "vislib/CharTraits.h"
#include "the/argument_exception.h"
#include "vislib/String.h"
#include "vislib/sysfunctions.h"


/*
 * vislib::graphics::gl::ShaderSource::StringSnippet::StringSnippet
 */
vislib::graphics::gl::ShaderSource::StringSnippet::StringSnippet(void) 
        : txt() {
}


/* 
 * vislib::graphics::gl::ShaderSource::StringSnippet::StringSnippet
 */
vislib::graphics::gl::ShaderSource::StringSnippet::StringSnippet(
        const vislib::StringA& str) : txt(str) {
}


/*
 * vislib::graphics::gl::ShaderSource::StringSnippet::StringSnippet
 */
vislib::graphics::gl::ShaderSource::StringSnippet::StringSnippet(
        const char *str) : txt(str) {
}


/*
 * vislib::graphics::gl::ShaderSource::StringSnippet::~StringSnippet
 */
vislib::graphics::gl::ShaderSource::StringSnippet::~StringSnippet(void) {
}


/*
 * vislib::graphics::gl::ShaderSource::StringSnippet::LoadFile
 */
bool vislib::graphics::gl::ShaderSource::StringSnippet::LoadFile(
        const char* filename) {
    return vislib::sys::ReadTextFile(this->txt, filename);
}


/*
 * vislib::graphics::gl::ShaderSource::StringSnippet::PeekCode
 */
const char *vislib::graphics::gl::ShaderSource::StringSnippet::PeekCode(void) const {
    return this->txt.PeekBuffer();
}


/*
 * vislib::graphics::gl::ShaderSource::StringSnippet::Set
 */
void vislib::graphics::gl::ShaderSource::StringSnippet::Set(
        const vislib::StringA& str) {
    this->txt = str;
}


/*
 * vislib::graphics::gl::ShaderSource::StringSnippet::Set
 */
void vislib::graphics::gl::ShaderSource::StringSnippet::Set(const char *str) {
    this->txt = str;
}


/*
 * vislib::graphics::gl::ShaderSource::StringSnippet::String
 */
const vislib::StringA& 
vislib::graphics::gl::ShaderSource::StringSnippet::String(void) const {
    return this->txt;
}

/*****************************************************************************/

/*
 * vislib::graphics::gl::ShaderSource::VersionSnippet::VersionSnippet
 */
vislib::graphics::gl::ShaderSource::VersionSnippet::VersionSnippet(
        int version) : verStr() {
    this->Set(version);
}


/*
 * vislib::graphics::gl::ShaderSource::VersionSnippet::~VersionSnippet
 */
vislib::graphics::gl::ShaderSource::VersionSnippet::~VersionSnippet(void) {
}


/*
 * vislib::graphics::gl::ShaderSource::VersionSnippet::PeekCode
 */
const char *vislib::graphics::gl::ShaderSource::VersionSnippet::PeekCode(void) const {
    return this->verStr;
}


/*
 * vislib::graphics::gl::ShaderSource::VersionSnippet::
 */
void vislib::graphics::gl::ShaderSource::VersionSnippet::Set(int version) {
    this->verStr.Format("#version %d\n", version);
}


/*
 * vislib::graphics::gl::ShaderSource::VersionSnippet::Version
 */
int vislib::graphics::gl::ShaderSource::VersionSnippet::Version(void) const {
    return vislib::CharTraitsA::ParseInt(
        this->verStr.PeekBuffer() + 9);
}

/*****************************************************************************/

/*
 * vislib::graphics::gl::ShaderSource::ShaderSource
 */
vislib::graphics::gl::ShaderSource::ShaderSource(void) 
    : snippets(), names(), code(NULL) {
    // intentionally empty
}


/*
 * vislib::graphics::gl::ShaderSource::ShaderSource
 */
vislib::graphics::gl::ShaderSource::ShaderSource(
        const vislib::graphics::gl::ShaderSource& src) 
        : snippets(), names(), code(NULL) {
    *this = src;
}


/*
 * vislib::graphics::gl::ShaderSource::~ShaderSource
 */
vislib::graphics::gl::ShaderSource::~ShaderSource(void) {
    the::safe_array_delete(this->code);
}


/*
 * vislib::graphics::gl::ShaderSource::Append
 */
const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
vislib::graphics::gl::ShaderSource::Append(
        const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
        code) {
    if (code.IsNull()) {
        throw the::argument_exception("code", __FILE__, __LINE__);
    }
    this->snippets.Append(code);
    the::safe_array_delete(this->code);
    return this->snippets.Last();
}


/*
 * vislib::graphics::gl::ShaderSource::Append
 */
const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
vislib::graphics::gl::ShaderSource::Append(
        const vislib::StringA& name,
        const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
        code) {
    if (code.IsNull()) {
        throw the::argument_exception("code", __FILE__, __LINE__);
    }
    if (this->names.Contains(name)) {
        throw the::already_exists_exception(name, __FILE__, __LINE__);
    }
    this->names[name] = this->snippets.Count();
    this->snippets.Append(code);
    the::safe_array_delete(this->code);
    return this->snippets.Last();
}


/*
 * vislib::graphics::gl::ShaderSource::Clear
 */
void vislib::graphics::gl::ShaderSource::Clear(void) {
    this->snippets.Clear();
    this->names.Clear();
    the::safe_array_delete(this->code);
}


/*
 * vislib::graphics::gl::ShaderSource::Code
 */
const char ** vislib::graphics::gl::ShaderSource::Code(void) const {
    if (this->code == NULL) {
        size_t cnt = this->snippets.Count();
        this->code = new const char*[cnt];
        for (size_t i = 0; i < cnt; i++) {
            this->code[i] = const_cast<Snippet*>(this->snippets[i].operator->()
                )->PeekCode();
        }
    }
    return this->code;
}


/*
 * vislib::graphics::gl::ShaderSource::Count
 */
size_t vislib::graphics::gl::ShaderSource::Count(void) const {
    return this->snippets.Count();
}


/*
 * vislib::graphics::gl::ShaderSource::Insert
 */
const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
vislib::graphics::gl::ShaderSource::Insert(const size_t idx, 
        const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
        code) {
    if (code.IsNull()) {
        throw the::argument_exception("code", __FILE__, __LINE__);
    }
    Map<StringA, size_t>::Iterator i = this->names.GetIterator();
    while (i.HasNext()) {
        Map<StringA, size_t>::ElementPair& ep = i.Next();
        if (ep.Value() >= idx) ep.Value()++;
    }
    this->snippets.Insert(idx, code);
    the::safe_array_delete(this->code);  
    return this->snippets[idx];
}


/*
 * vislib::graphics::gl::ShaderSource::Insert
 */
const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
vislib::graphics::gl::ShaderSource::Insert(const size_t idx, 
        const vislib::StringA& name,
        const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
        code) {
    if (code.IsNull()) {
        throw the::argument_exception("code", __FILE__, __LINE__);
    }
    if (this->names.Contains(name)) {
        throw the::already_exists_exception(name, __FILE__, __LINE__);
    }
    Map<StringA, size_t>::Iterator i = this->names.GetIterator();
    while (i.HasNext()) {
        Map<StringA, size_t>::ElementPair& ep = i.Next();
        if (ep.Value() >= idx) ep.Value()++;
    }
    this->names[name] = idx;
    this->snippets.Insert(idx, code);
    the::safe_array_delete(this->code);  
    return this->snippets[idx];
}


/*
 * vislib::graphics::gl::ShaderSource::NameIndex
 */
size_t vislib::graphics::gl::ShaderSource::NameIndex(const vislib::StringA& name) {
    if (!this->names.Contains(name)) {
        throw the::no_such_element_exception(name, __FILE__, __LINE__);
    }
    return this->names[name];
}


/*
 * vislib::graphics::gl::ShaderSource::Prepend
 */
const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
vislib::graphics::gl::ShaderSource::Prepend(
        const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
        code) {
    if (code.IsNull()) {
        throw the::argument_exception("code", __FILE__, __LINE__);
    }
    Map<StringA, size_t>::Iterator i = this->names.GetIterator();
    while (i.HasNext()) {
        i.Next().Value()++;
    }
    this->snippets.Prepend(code);
    the::safe_array_delete(this->code);
    return this->snippets.First();
}


/*
 * vislib::graphics::gl::ShaderSource::Prepend
 */
const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
vislib::graphics::gl::ShaderSource::Prepend( const vislib::StringA& name,
        const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
        code) {
    if (code.IsNull()) {
        throw the::argument_exception("code", __FILE__, __LINE__);
    }
    if (this->names.Contains(name)) {
        throw the::already_exists_exception(name, __FILE__, __LINE__);
    }
    Map<StringA, size_t>::Iterator i = this->names.GetIterator();
    while (i.HasNext()) {
        i.Next().Value()++;
    }
    this->names[name] = 0;
    this->snippets.Prepend(code);
    the::safe_array_delete(this->code);
    return this->snippets.First();
}


/*
 * vislib::graphics::gl::ShaderSource::Remove
 */
void vislib::graphics::gl::ShaderSource::Remove(size_t idx) {
    SingleLinkedList<StringA> keys = this->names.FindKeys(idx);
    Map<StringA, size_t>::Iterator i = this->names.GetIterator();
    while (i.HasNext()) {
        Map<StringA, size_t>::ElementPair& ep = i.Next();
        if (ep.Value() >= idx) ep.Value()--;
    }
    SingleLinkedList<StringA>::Iterator i2 = keys.GetIterator();
    while (i2.HasNext()) {
        this->names.Remove(i2.Next());
    }
    this->snippets.Erase(idx);
    the::safe_array_delete(this->code);
}


/*
 * vislib::graphics::gl::ShaderSource::RemoveName
 */
void vislib::graphics::gl::ShaderSource::RemoveName(vislib::StringA& name) {
    this->names.Remove(name);
}


/*
 * vislib::graphics::gl::ShaderSource::Replace
 */
vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>
vislib::graphics::gl::ShaderSource::Replace(size_t idx,
        const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>&
        code) {
    vislib::SmartPtr<Snippet> retval = this->snippets[idx];
    this->snippets[idx] = code;
    the::safe_array_delete(this->code);
    return retval;
}


/*
 * vislib::graphics::gl::ShaderSource::Set
 */
const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
vislib::graphics::gl::ShaderSource::Set(
        const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
        code) {
    if (code.IsNull()) {
        throw the::argument_exception("code", __FILE__, __LINE__);
    }
    this->snippets = Array<SmartPtr<Snippet> >(1, code);
    the::safe_array_delete(this->code);
    return this->snippets.First();
}


/*
 * vislib::graphics::gl::ShaderSource::SetName
 */
void vislib::graphics::gl::ShaderSource::SetName(const vislib::StringA& name,
        size_t idx) {
    if ((idx < 0) || (idx >= this->snippets.Count())) {
        throw the::index_out_of_range_exception(int(idx), 0, int(this->snippets.Count()) - 1,
            __FILE__, __LINE__);
    }
    if (this->names.Contains(name)) {
        throw the::already_exists_exception(name, __FILE__, __LINE__);
    }
    this->names[name] = idx;
}


/*
 * vislib::graphics::gl::ShaderSource::WholeCode
 */
vislib::StringA vislib::graphics::gl::ShaderSource::WholeCode(void) const {
    vislib::StringA retval;
    size_t cnt = this->snippets.Count();
    for (size_t i = 0; i < cnt; i++) {
        retval.Append(const_cast<Snippet*>(this->snippets[i].operator->()
            )->PeekCode());
    }
    return retval;
}


/*
 * vislib::graphics::gl::ShaderSource::operator[]
 */
const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
vislib::graphics::gl::ShaderSource::operator[](const size_t idx) const {
    return this->snippets[idx];
}


/*
 * vislib::graphics::gl::ShaderSource::operator=
 */
vislib::graphics::gl::ShaderSource&
vislib::graphics::gl::ShaderSource::operator=(
        const vislib::graphics::gl::ShaderSource& rhs) {
    if (&rhs != this) {
        this->snippets.Clear();
        this->snippets.SetCount(rhs.Count());
        for (unsigned int i = 0; i < rhs.Count(); i++) {
            this->snippets[i] = rhs.snippets[i];
        }

        this->names.Clear();
        ConstIterator<Map<StringA, size_t>::Iterator> ci = rhs.names.GetConstIterator();
        while (ci.HasNext()) {
            const Map<StringA, size_t>::ElementPair& ep = ci.Next();
            this->names[ep.Key()] = ep.Value();
        }

        the::safe_array_delete(this->code);
    }
    return *this;
}
