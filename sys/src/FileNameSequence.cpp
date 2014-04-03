/*
 * FileNameSequence.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/FileNameSequence.h"
#include "the/assert.h"
#include "vislib/File.h"
#include "the/argument_exception.h"
#include "the/index_out_of_range_exception.h"
#include "vislib/Path.h"
#include "the/string.h"
#include "the/text/string_builder.h"


/****************************************************************************/

/*
 * vislib::sys::FileNameSequence::FileNameElement::FileNameElement
 */
vislib::sys::FileNameSequence::FileNameElement::FileNameElement(void)
        : buffer(NULL), unicodeBuffer(false) {
    // Intentionally empty
}


/*
 * vislib::sys::FileNameSequence::FileNameElement::~FileNameElement
 */
vislib::sys::FileNameSequence::FileNameElement::~FileNameElement(void) {
    if (this->unicodeBuffer) {
        wchar_t *wb = reinterpret_cast<wchar_t*>(this->buffer);
        delete wb;
    } else {
        delete this->buffer;
    }
    this->buffer = NULL;
    this->unicodeBuffer = false;
}


/*
 * vislib::sys::FileNameSequence::FileNameElement::TextA
 */
const char * vislib::sys::FileNameSequence::FileNameElement::TextA(void) const {
    if (this->buffer != NULL) {
        if (this->unicodeBuffer) {
            wchar_t *wb = reinterpret_cast<wchar_t*>(this->buffer);
            delete wb;
        } else {
            delete this->buffer;
        }
    }
    this->buffer = this->makeTextA();
    this->unicodeBuffer = false;
    return this->buffer;
}


/*
 * vislib::sys::FileNameSequence::FileNameElement::TextW
 */
const wchar_t * vislib::sys::FileNameSequence::FileNameElement::TextW(void) const {
    if (this->buffer != NULL) {
        if (this->unicodeBuffer) {
            wchar_t *wb = reinterpret_cast<wchar_t*>(this->buffer);
            delete wb;
        } else {
            delete this->buffer;
        }
    }
    this->buffer = reinterpret_cast<char*>(this->makeTextW());
    this->unicodeBuffer = false;
    return reinterpret_cast<wchar_t*>(this->buffer);
}


/****************************************************************************/

/*
 * vislib::sys::FileNameSequence::FileNameCountElement::FileNameCountElement
 */
vislib::sys::FileNameSequence::FileNameCountElement::FileNameCountElement(
        void) : FileNameElement(), digits(1), maxVal(0), minVal(0), step(1),
        value(0) {
    // Intentionally empty
}


/*
 * vislib::sys::FileNameSequence::FileNameCountElement::FileNameCountElement
 */
vislib::sys::FileNameSequence::FileNameCountElement::FileNameCountElement(
        unsigned int digits, unsigned int minVal, unsigned int maxVal,
        unsigned int step) : FileNameElement(), digits(digits),
        maxVal(minVal), minVal(maxVal), step(step), value(minVal) {
    // Intentionally empty
}


/*
 * vislib::sys::FileNameSequence::FileNameCountElement::~FileNameCountElement
 */
vislib::sys::FileNameSequence::FileNameCountElement::~FileNameCountElement(
        void) {
    // Intentionally empty
}


/*
 * vislib::sys::FileNameSequence::FileNameCountElement::Count
 */
unsigned int vislib::sys::FileNameSequence::FileNameCountElement::Count(void)
        const {
    unsigned int diff = this->maxVal + 1 - this->minVal;
    return diff / this->step + ((diff % this->step) > 0 ? 1 : 0);
}


/*
 * vislib::sys::FileNameSequence::FileNameCountElement::CounterIndex
 */
unsigned int
vislib::sys::FileNameSequence::FileNameCountElement::CounterIndex(void)
        const {
    THE_ASSERT(this->value >= this->minVal);
    unsigned int p = this->value - this->minVal;
    THE_ASSERT((p % this->step) == 0);
    return p / this->step;
}


/*
 * vislib::sys::FileNameSequence::FileNameCountElement::IncreaseCounter
 */
bool
vislib::sys::FileNameSequence::FileNameCountElement::IncreaseCounter(void) {
    unsigned int nv = this->value + this->step;
    if (nv > this->maxVal) {
        return false;
    } else {
        this->value = nv;
        return true;
    }
}


/*
 * vislib::sys::FileNameSequence::FileNameCountElement::SetCounterIndex
 */
void vislib::sys::FileNameSequence::FileNameCountElement::SetCounterIndex(
        unsigned int idx) {
    if (idx >= this->Count()) {
        throw the::index_out_of_range_exception(idx, 0, this->Count(), __FILE__, __LINE__);
    }
    this->value = this->minVal + this->step * idx;
    THE_ASSERT(this->value <= this->maxVal);
}


/*
 * vislib::sys::FileNameSequence::FileNameCountElement::SetDigits
 */
void vislib::sys::FileNameSequence::FileNameCountElement::SetDigits(
        unsigned int digits) {
    this->digits = digits;
}


/*
 * vislib::sys::FileNameSequence::FileNameCountElement::SetRange
 */
void vislib::sys::FileNameSequence::FileNameCountElement::SetRange(
        unsigned int minVal, unsigned int maxVal, unsigned int step) {
    if (maxVal < minVal) {
        throw the::argument_exception("maxVal must not be less minVal",
            __FILE__, __LINE__);
    }
    this->minVal = minVal;
    this->maxVal = maxVal;
    this->SetStepSize(step); // sfx: will fix 'value'!
}


/*
 * vislib::sys::FileNameSequence::FileNameCountElement::SetStepSize
 */
void vislib::sys::FileNameSequence::FileNameCountElement::SetStepSize(
        unsigned int step) {
    unsigned int v = this->value;
    this->step = step;
    if (v < this->minVal) {
        this->value = this->minVal;
        return;
    }
    v = (this->value - this->minVal) / this->step;
    if (((this->value - this->minVal) % this->step) * 2 >= this->step) v++;
    if ((this->minVal + v * this->step) > this->maxVal) {
        v = this->Count();
    }
    this->value = this->minVal + v * this->step;
}


/*
 * vislib::sys::FileNameSequence::FileNameCountElement::makeTextA
 */
char * vislib::sys::FileNameSequence::FileNameCountElement::makeTextA(void)
        const {
    the::astring s1, s2;
    the::text::astring_builder::format_to(s1, "%%.%uu", this->digits); // paranoia way to do this
    the::text::astring_builder::format_to(s2, s1.c_str(), this->value);
    unsigned int len = static_cast<unsigned int>(s2.size());
    char *buf = new char[len + 1];
    buf[len] = 0;
    memcpy(buf, s2.c_str(), len * sizeof(char));
    return buf;
}


/*
 * vislib::sys::FileNameSequence::FileNameCountElement::makeTextW
 */
wchar_t * vislib::sys::FileNameSequence::FileNameCountElement::makeTextW(void)
        const {
    the::wstring s1, s2;
    the::text::wstring_builder::format_to(s1, L"%%.%uu", this->digits); // paranoia way to do this
    the::text::wstring_builder::format_to(s2, s1.c_str(), this->value);
    unsigned int len = static_cast<unsigned int>(s2.size());
    wchar_t *buf = new wchar_t[len + 1];
    buf[len] = 0;
    memcpy(buf, s2.c_str(), len * sizeof(wchar_t));
    return buf;
}


/****************************************************************************/

/*
 * vislib::sys::FileNameSequence::FileNameSequence
 */
vislib::sys::FileNameSequence::FileNameSequence(void) : elements(),
        reversedPriority(false) {
    // Intentionally empty
}


/*
 * vislib::sys::FileNameSequence::~FileNameSequence
 */
vislib::sys::FileNameSequence::~FileNameSequence(void) {
    // Intentionally empty
}


/*
 * vislib::sys::FileNameSequence::Autodetect
 */
void vislib::sys::FileNameSequence::Autodetect(
        const the::astring& firstFileName) {
    this->autodetect(firstFileName);
}


/*
 * vislib::sys::FileNameSequence::Autodetect
 */
void vislib::sys::FileNameSequence::Autodetect(
        const the::wstring& firstFileName) {
    this->autodetect(firstFileName);
}


/*
 * vislib::sys::FileNameSequence::Count
 */
unsigned int vislib::sys::FileNameSequence::Count(void) {
    unsigned int cnt = 1;
    for (unsigned int i = 0; i < this->elements.Count(); i++) {
        if (this->elements[i] == NULL) continue;
        cnt *= this->elements[i]->Count();
    }
    return cnt;
}


/*
 * vislib::sys::FileNameSequence::FileNameA
 */
the::astring vislib::sys::FileNameSequence::FileNameA(unsigned int idx) {
    THE_ASSERT(this->IsValid() == true);
    this->setIndex(idx);
    the::astring rv;
    for (unsigned int i = 0; i < this->elements.Count(); i++) {
        rv.append(this->elements[i]->TextA());
    }
    return rv;
}


/*
 * vislib::sys::FileNameSequence::FileNameW
 */
the::wstring vislib::sys::FileNameSequence::FileNameW(unsigned int idx) {
    THE_ASSERT(this->IsValid() == true);
    this->setIndex(idx);
    the::wstring rv;
    for (unsigned int i = 0; i < this->elements.Count(); i++) {
        rv.append(this->elements[i]->TextW());
    }
    return rv;
}


/*
 * vislib::sys::FileNameSequence::IsValid
 */
bool vislib::sys::FileNameSequence::IsValid(void) const {
    if (this->elements.Count() <= 0) return false;
    for (unsigned int i = 0; i < this->elements.Count(); i++) {
        if (this->elements[i] == NULL) return false;
        if (this->elements[i]->Count() == 0) return false;
    }
    return true;
}


/*
 * vislib::sys::FileNameSequence::autodetect
 */
template<class T>
void vislib::sys::FileNameSequence::autodetect(const T& filename) {
    this->elements.Clear();
    if (!File::IsFile(filename.c_str())) return;

    // separate path from name
    auto i = filename.find(static_cast<typename T::value_type>(Path::SEPARATOR_A));
    unsigned int len, start = 0;
    const typename T::value_type* buffer;
    if (i == T::npos) {
        // it seams there is no path
        len = static_cast<unsigned int>(filename.size());
        buffer = filename.c_str();
    } else {
        // path as first string element
        i++;
        this->elements.Add(new FileNameStringElement<T>(
            filename.substr(0, i).c_str()));
        len = static_cast<unsigned int>(filename.size() - i);
        buffer = filename.c_str() + i;
    }

    // now search for numbers
    bool dig, seqdig = false;
    FileNameCountElement *cntElement = NULL;
    for (unsigned int pos = 0; pos < len; pos++) {
        dig = the::text::char_utility::is_digit(buffer[pos]);
        if (dig != seqdig) {
            if (seqdig) {
                // number element
                unsigned int val = static_cast<unsigned int>(
                    the::text::string_utility::parse_int(
                        T(buffer + start, pos - start)));
                this->elements.Add(cntElement = new FileNameCountElement(
                    pos - start, val, val));
                // range will be adjusted later on
            } else {
                // string element
                this->elements.Add(new FileNameStringElement<T>(
                    T(buffer + start, pos - start)));
            }
            start = pos;
            seqdig = dig;
        }
    }
    if (start != len) {
        if (seqdig) {
            // number element
            unsigned int val = static_cast<unsigned int>(
                the::text::string_utility::parse_int(T(buffer + start, len - start)));
            this->elements.Add(cntElement = new FileNameCountElement(
                len - start, val, val));
            // range will be adjusted later on
        } else {
            // string element
            this->elements.Add(new FileNameStringElement<T>(
                T(buffer + start, len - start)));
        }
    }

    // search for the maximum values of the LAST counter only (ATM)
    if (cntElement != NULL) {
        unsigned int idx = 0;
        do {
            cntElement->SetRange(cntElement->MinValue(),
                cntElement->MaxValue() + 1);
            idx++;
        } while(File::IsFile(this->FileNameW(idx).c_str()));
        cntElement->SetRange(cntElement->MinValue(),
            cntElement->MaxValue() - 1);
    }
}


/*
 * vislib::sys::FileNameSequence::setIndex
 */
void vislib::sys::FileNameSequence::setIndex(unsigned int idx) {
    unsigned int ec = static_cast<unsigned int>(this->elements.Count());
    unsigned int i;
    for (unsigned int j = 0; j < ec; j++) {
        i = ((reversedPriority) ? (j) : (ec - (1 + j)));
        unsigned int c = this->elements[i]->Count();
        if (c <= 1) continue;
        FileNameCountElement *fnc 
            = this->elements[i].DynamicCast<FileNameCountElement>();
        if (fnc == NULL) continue;
        fnc->SetCounterIndex(idx % c);
        idx /= c;
    }
}
