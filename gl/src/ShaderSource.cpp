/*
 * ShaderSource.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/ShaderSource.h"
#include <cstdarg>
#include "vislib/memutils.h"
#include "vislib/CharTraits.h"
#include "vislib/IllegalParamException.h"
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
const char *vislib::graphics::gl::ShaderSource::StringSnippet::PeekCode(void) {
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
const char *vislib::graphics::gl::ShaderSource::VersionSnippet::PeekCode(void) {
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
int vislib::graphics::gl::ShaderSource::VersionSnippet::Version(void) {
    return vislib::CharTraitsA::ParseInt(
        this->verStr.PeekBuffer() + 9);
}

/*****************************************************************************/

/*
 * vislib::graphics::gl::ShaderSource::ShaderSource
 */
vislib::graphics::gl::ShaderSource::ShaderSource(void) 
    : snippets(), code(NULL) {
}


/*
 * vislib::graphics::gl::ShaderSource::~ShaderSource
 */
vislib::graphics::gl::ShaderSource::~ShaderSource(void) {
    ARY_SAFE_DELETE(this->code);
}


/*
 * vislib::graphics::gl::ShaderSource::Append
 */
const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
vislib::graphics::gl::ShaderSource::Append(
        const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
        code) {
    if (code.IsNull()) {
        throw IllegalParamException("code", __FILE__, __LINE__);
    }
    this->snippets.Append(code);
    ARY_SAFE_DELETE(this->code);
    return this->snippets.Last();
}


/*
 * vislib::graphics::gl::ShaderSource::Clear
 */
void vislib::graphics::gl::ShaderSource::Clear(void) {
    this->snippets.Clear();
    ARY_SAFE_DELETE(this->code);  
}


/*
 * vislib::graphics::gl::ShaderSource::Code
 */
const char ** vislib::graphics::gl::ShaderSource::Code(void) const {
    if (this->code == NULL) {
        SIZE_T cnt = this->snippets.Count();
        this->code = new const char*[cnt];
        for (SIZE_T i = 0; i < cnt; i++) {
            this->code[i] = const_cast<Snippet*>(this->snippets[i].operator->()
                )->PeekCode();
        }
    }
    return this->code;
}


/*
 * vislib::graphics::gl::ShaderSource::Count
 */
SIZE_T vislib::graphics::gl::ShaderSource::Count(void) const {
    return this->snippets.Count();
}


/*
 * vislib::graphics::gl::ShaderSource::Insert
 */
const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
vislib::graphics::gl::ShaderSource::Insert(const SIZE_T idx, 
        const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
        code) {
    if (code.IsNull()) {
        throw IllegalParamException("code", __FILE__, __LINE__);
    }
    this->snippets.Insert(idx, code);
    ARY_SAFE_DELETE(this->code);  
    return this->snippets[idx];
}


/*
 * vislib::graphics::gl::ShaderSource::Prepend
 */
const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
vislib::graphics::gl::ShaderSource::Prepend(
        const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
        code) {
    if (code.IsNull()) {
        throw IllegalParamException("code", __FILE__, __LINE__);
    }
    this->snippets.Prepend(code);
    ARY_SAFE_DELETE(this->code);
    return this->snippets.First();
}


/*
 * vislib::graphics::gl::ShaderSource::Remove
 */
void vislib::graphics::gl::ShaderSource::Remove(SIZE_T idx) {
    this->snippets.Erase(idx);
    ARY_SAFE_DELETE(this->code);
}


/*
 * vislib::graphics::gl::ShaderSource::Set
 */
const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
vislib::graphics::gl::ShaderSource::Set(
        const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
        code) {
    if (code.IsNull()) {
        throw IllegalParamException("code", __FILE__, __LINE__);
    }
    this->snippets = Array<SmartPtr<Snippet> >(1, code);
    ARY_SAFE_DELETE(this->code);
    return this->snippets.First();
}


/*
 * vislib::graphics::gl::ShaderSource::WholeCode
 */
vislib::StringA vislib::graphics::gl::ShaderSource::WholeCode(void) const {
    vislib::StringA retval;
    SIZE_T cnt = this->snippets.Count();
    for (SIZE_T i = 0; i < cnt; i++) {
        retval.Append(const_cast<Snippet*>(this->snippets[i].operator->()
            )->PeekCode());
    }
    return retval;
}


/*
 * vislib::graphics::gl::ShaderSource::operator[]
 */
const vislib::SmartPtr<vislib::graphics::gl::ShaderSource::Snippet>& 
vislib::graphics::gl::ShaderSource::operator[](const SIZE_T idx) const {
    return this->snippets[idx];
}
