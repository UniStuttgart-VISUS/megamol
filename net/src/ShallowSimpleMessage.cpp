/*
 * ShallowSimpleMessage.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/ShallowSimpleMessage.h"


/*
 * vislib::net::ShallowSimpleMessage::ShallowSimpleMessage
 */
vislib::net::ShallowSimpleMessage::ShallowSimpleMessage(
        void *storage, const SIZE_T cntStorage) 
        : Super(), cntStorage(cntStorage), storage(storage) {
    VLSTACKTRACE("ShallowSimpleMessage::ShallowSimpleMessage", 
        __FILE__, __LINE__);
    ASSERT(storage != NULL);
    ASSERT((cntStorage == 0) 
        || (cntStorage >= sizeof(SimpleMessageHeaderData)));

    this->GetHeader().SetData(static_cast<SimpleMessageHeaderData*>(storage));
    // Note: Cannot do that in initialiser!
    if (this->cntStorage == 0) {
        this->cntStorage = this->GetMessageSize();
    }
}


/*
 * vislib::net::ShallowSimpleMessage::~ShallowSimpleMessage
 */
vislib::net::ShallowSimpleMessage::~ShallowSimpleMessage(void) {
    VLSTACKTRACE("ShallowSimpleMessage::~ShallowSimpleMessage",
        __FILE__, __LINE__);
}


/*
 * vislib::net::ShallowSimpleMessage::SetStorage
 */
void vislib::net::ShallowSimpleMessage::SetStorage(void *storage, 
        const SIZE_T cntStorage) {
    VLSTACKTRACE("ShallowSimpleMessage::SetStorage", __FILE__, __LINE__);
    ASSERT(storage != NULL);
    ASSERT((cntStorage == 0) 
        || (cntStorage >= sizeof(SimpleMessageHeaderData)));
    
    // Note: Order of assignments is important!
    this->storage = storage;
    this->GetHeader().SetData(static_cast<SimpleMessageHeaderData*>(storage));
    this->cntStorage = (cntStorage > 0) ? cntStorage : this->GetMessageSize();
}


/*
 * vislib::net::ShallowSimpleMessage::assertStorage
 */
bool vislib::net::ShallowSimpleMessage::assertStorage(void *& outStorage, const SIZE_T size) {
    // intentionally empty atm
    return false;
}
