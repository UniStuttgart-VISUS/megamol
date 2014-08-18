/*
 * FileNameSequence.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/FileNameSequence.h"
#include "vislib/assert.h"
#include "vislib/CharTraits.h"
#include "vislib/File.h"
#include "vislib/IllegalParamException.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/Path.h"
#include "vislib/String.h"


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
    ASSERT(this->value >= this->minVal);
    unsigned int p = this->value - this->minVal;
    ASSERT((p % this->step) == 0);
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
        throw OutOfRangeException(idx, 0, this->Count(), __FILE__, __LINE__);
    }
    this->value = this->minVal + this->step * idx;
    ASSERT(this->value <= this->maxVal);
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
        throw IllegalParamException("maxVal must not be less minVal",
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
    vislib::StringA s1, s2;
    s1.Format("%%.%uu", this->digits); // paranoia way to do this
    s2.Format(s1, this->value);
    unsigned int len = s2.Length();
    char *buf = new char[len + 1];
    buf[len] = 0;
    memcpy(buf, s2.PeekBuffer(), len * sizeof(char));
    return buf;
}


/*
 * vislib::sys::FileNameSequence::FileNameCountElement::makeTextW
 */
wchar_t * vislib::sys::FileNameSequence::FileNameCountElement::makeTextW(void)
        const {
    vislib::StringW s1, s2;
    s1.Format(L"%%.%uu", this->digits); // paranoia way to do this
    s2.Format(s1, this->value);
    unsigned int len = s2.Length();
    wchar_t *buf = new wchar_t[len + 1];
    buf[len] = 0;
    memcpy(buf, s2.PeekBuffer(), len * sizeof(wchar_t));
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
        const vislib::StringA& firstFileName) {
    this->autodetect(firstFileName);
}


/*
 * vislib::sys::FileNameSequence::Autodetect
 */
void vislib::sys::FileNameSequence::Autodetect(
        const vislib::StringW& firstFileName) {
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
vislib::StringA vislib::sys::FileNameSequence::FileNameA(unsigned int idx) {
    ASSERT(this->IsValid() == true);
    this->setIndex(idx);
    vislib::StringA rv;
    for (unsigned int i = 0; i < this->elements.Count(); i++) {
        rv.Append(this->elements[i]->TextA());
    }
    return rv;
}


/*
 * vislib::sys::FileNameSequence::FileNameW
 */
vislib::StringW vislib::sys::FileNameSequence::FileNameW(unsigned int idx) {
    ASSERT(this->IsValid() == true);
    this->setIndex(idx);
    vislib::StringW rv;
    for (unsigned int i = 0; i < this->elements.Count(); i++) {
        rv.Append(this->elements[i]->TextW());
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
void vislib::sys::FileNameSequence::autodetect(const vislib::String<T>& filename) {
    this->elements.Clear();
    if (!File::IsFile(filename)) return;

    // separate path from name
    typename String<T>::Size i = filename.FindLast(
        static_cast<typename T::Char>(Path::SEPARATOR_A));
    unsigned int len, start = 0;
    const typename String<T>::Char * buffer;
    if (i == String<T>::INVALID_POS) {
        // it seams there is no path
        len = filename.Length();
        buffer = filename.PeekBuffer();
    } else {
        // path as first string element
        i++;
        this->elements.Add(new FileNameStringElement<T>(
            filename.Substring(0, i)));
        len = filename.Length() - i;
        buffer = filename.PeekBuffer() + i;
    }

    // now search for numbers
    bool dig, seqdig = false;
    FileNameCountElement *cntElement = NULL;
    for (unsigned int pos = 0; pos < len; pos++) {
        dig = T::IsDigit(buffer[pos]);
        if (dig != seqdig) {
            if (seqdig) {
                // number element
                unsigned int val = static_cast<unsigned int>(
                    T::ParseInt(String<T>(buffer + start, pos - start)));
                this->elements.Add(cntElement = new FileNameCountElement(
                    pos - start, val, val));
                // range will be adjusted later on
            } else {
                // string element
                this->elements.Add(new FileNameStringElement<T>(
                    String<T>(buffer + start, pos - start)));
            }
            start = pos;
            seqdig = dig;
        }
    }
    if (start != len) {
        if (seqdig) {
            // number element
            unsigned int val = static_cast<unsigned int>(
                T::ParseInt(String<T>(buffer + start, len - start)));
            this->elements.Add(cntElement = new FileNameCountElement(
                len - start, val, val));
            // range will be adjusted later on
        } else {
            // string element
            this->elements.Add(new FileNameStringElement<T>(
                String<T>(buffer + start, len - start)));
        }
    }

    // search for the maximum values of the LAST counter only (ATM)
    if (cntElement != NULL) {
        unsigned int idx = 0;
        do {
            cntElement->SetRange(cntElement->MinValue(),
                cntElement->MaxValue() + 1);
            idx++;
        } while(File::IsFile(this->FileNameW(idx)));
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
