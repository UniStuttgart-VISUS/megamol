/*
 * BufferMTPConnection.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "vislib/assert.h"
#include "vislib/memutils.h"
#include "vislib/sys/AutoLock.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/sys/Event.h"


namespace megamol {
namespace demos_gl {


/**
 * Buffer connection for multi-threaded pipeline processors
 * Template parameter T is the buffer class (required is default ctor)
 */
template<class T>
class BufferMTPConnection {
public:
    /** Possible states for buffers */
    enum State { STATE_EMPTY, STATE_PRODUCING, STATE_FILLED, STATE_CONSUMING };

    /**
     * Ctor
     *
     * @param bufSize The number of buffers pooled in the connection
     */
    BufferMTPConnection(unsigned int bufSize = 4);

    /** Dtor */
    ~BufferMTPConnection(void);

    /**
     * Closes the connection aborting the communication. All buffers will
     * be reset to empty state before the method returns. The method waits
     * for buffers in producing or consuming state to be returned.
     */
    void AbortClose(void);

    /**
     * Marks the buffer 'buf' as now empty after being consumed. The
     * caller will no longer use 'buf'. The caller is responsible to clear
     * the buffers data.
     *
     * @param buf The buffer just consumed, now again available as empty
     *              buffer; must not be NULL
     */
    void BufferConsumed(T* buf);

    /**
     * Marks the buffer 'buf' as now filled. The caller will no longer use
     * 'buf'.
     *
     * @param buf The buffer just filled; must not be NULL
     */
    void BufferFilled(T* buf);

    /**
     * Informs the connection that there will be no new data and that the
     * connection should be closes as soon as the last filled buffer was
     * consumed. The method returns immediately.
     */
    void EndOfDataClose(void);

    /**
     * Gets an empty buffer and switches it's state from empty to
     * producing. If no empty buffer is currenty available the method
     * will block until one becomes available or will return NULL. The
     * block will be released when the connection is closed. The caller
     * is responible for cleaning up the state of the buffer as data from
     * a previouse use of the buffer might still be present.
     *
     * @param wait If set to true the method will block/wait when no empty
     *             buffer is available. If set to false the method will
     *             immediatly return with NULL.
     *
     * @return A previously empty buffer, now in producing state, or NULL
     *         if there is no empty buffer or the connection was closed
     */
    T* GetEmptyBuffer(bool wait);

    /**
     * Gets a filled buffer and switches it's state from filled to
     * consuming. If no filled buffer is currently available the method
     * will block until one becomes available or will return NULL. The
     * block will be released when the connection is closed.
     *
     * @param wait If set to true the method will block/wait when no
     *             filled buffer is avilable. If set to false the method
     *             will immediately return with NULL.
     *
     * @return A previously filled buffer, now in consuming state, or NULL
     *         if there is no filled buffer or the connection was closed
     */
    T* GetFilledBuffer(bool wait);

    /**
     * Answer whether or not the end-of-data flag is set
     *
     * @return True if there will no buffers be filled anymore
     */
    inline bool IsEndOfData(void) const {
        return this->endOfData;
    }

    /**
     * Answer whether or not the connection is open
     *
     * @return True if the connection is open
     */
    inline bool IsOpen(void) const {
        return this->isOpen;
    }

    /**
     * Opens the connection
     */
    void Open(void);

private:
    /** type for buffers */
    typedef struct buffer_t {
    public:
        /** The real buffer */
        T buf;

        /** The state */
        State state;

        /** The next buffer in line */
        struct buffer_t* next;

    } Buffer;

    /** Event when a buffer switched to full or empty */
    vislib::sys::Event bufferStateCompleted;

    /** Buffers currently being consumed */
    Buffer* buffersConsuming;

    /** The thread lock for the buffer pointers */
    vislib::sys::CriticalSection bufferLock;

    /** Buffers currently being produced */
    Buffer* buffersProducing;

    /** The empty buffers */
    Buffer* emptyBuffers;

    /** There will no more buffers be filled */
    bool endOfData;

    /** The full buffers */
    Buffer* fullBuffers;

    /** flag whether or not this connection is open */
    bool isOpen;

    /** event for waiting for an empty buffer */
    vislib::sys::Event newEmptyBuffer;

    /** event for waiting for a full buffer */
    vislib::sys::Event newFullBuffer;
};


/*
 * BufferMTPConnection<T>::BufferMTPConnection
 */
template<class T>
BufferMTPConnection<T>::BufferMTPConnection(unsigned int bufSize)
        : bufferStateCompleted(false)
        , buffersConsuming(NULL)
        , bufferLock()
        , buffersProducing(NULL)
        , emptyBuffers(NULL)
        , endOfData(false)
        , fullBuffers(NULL)
        , isOpen(false)
        , newEmptyBuffer(false)
        , newFullBuffer(false) {

    ASSERT(bufSize > 0);

    for (unsigned int i = 0; i < bufSize; i++) {
        Buffer* b = new Buffer();
        b->state = STATE_EMPTY;
        b->next = this->emptyBuffers;
        this->emptyBuffers = b;
    }
}


/*
 * BufferMTPConnection<T>::~BufferMTPConnection
 */
template<class T>
BufferMTPConnection<T>::~BufferMTPConnection(void) {

    this->AbortClose();

    ASSERT(this->buffersConsuming == NULL);
    ASSERT(this->buffersProducing == NULL);
    ASSERT(this->fullBuffers == NULL);
    ASSERT(this->isOpen == false);

    while (this->emptyBuffers != NULL) {
        Buffer* b = this->emptyBuffers;
        this->emptyBuffers = b->next;
        delete b;
    }
}


/*
 * BufferMTPConnection<T>::AbortClose
 */
template<class T>
void BufferMTPConnection<T>::AbortClose(void) {
    this->bufferLock.Lock();
    this->isOpen = false; // don't give any buffers to callers

    // wait for buffers currently in use to return
    while ((this->buffersConsuming != NULL) || (this->buffersProducing != NULL)) {
        this->bufferLock.Unlock();
        this->bufferStateCompleted.Wait();
        this->bufferLock.Lock();
    }
    ASSERT(this->buffersConsuming == NULL);
    ASSERT(this->buffersProducing == NULL);

    // make all buffers empty
    if (this->emptyBuffers == NULL) {
        this->emptyBuffers = this->fullBuffers;
    } else {
        Buffer* leb = this->emptyBuffers;
        while (leb->next != NULL)
            leb = leb->next;
        leb->next = this->fullBuffers;
    }
    this->fullBuffers = NULL;

    // marking all buffers as empty
    Buffer* eb = this->emptyBuffers;
    ASSERT(eb != NULL);
    while (eb != NULL) {
        eb->state = STATE_EMPTY;
        eb = eb->next;
    }

    this->bufferLock.Unlock();
}


/*
 * BufferMTPConnection<T>::BufferConsumed
 */
template<class T>
void BufferMTPConnection<T>::BufferConsumed(T* buf) {
    vislib::sys::AutoLock(this->bufferLock);

    ASSERT(buf != NULL);

    // find matching buffer object and remove from list of consuming buffers
    ASSERT(this->buffersConsuming != NULL);
    Buffer* bb = this->buffersConsuming;
    if (buf == &bb->buf) {
        this->buffersConsuming = this->buffersConsuming->next;
    } else {
        Buffer* bp = bb;
        bb = bb->next;
        ASSERT(bb != NULL);
        while (buf != &bb->buf) {
            bp = bb;
            bb = bb->next;
            ASSERT(bb != NULL);
        }
        bp->next = bb->next;
    }
    bb->next = NULL;
    ASSERT(bb->state == STATE_CONSUMING);

    // add bb to end of empty list
    bb->state = STATE_EMPTY;
    if (this->emptyBuffers == NULL) {
        this->emptyBuffers = bb;
    } else {
        Buffer* i = this->emptyBuffers;
        while (i->next != NULL)
            i = i->next;
        i->next = bb;
    }

    // fire events
    this->newEmptyBuffer.Set();
    this->bufferStateCompleted.Set();
}


/*
 * BufferMTPConnection<T>::BufferFilled
 */
template<class T>
void BufferMTPConnection<T>::BufferFilled(T* buf) {
    vislib::sys::AutoLock(this->bufferLock);

    ASSERT(buf != NULL);

    // find matching buffer object and remove from list of producing buffers
    ASSERT(this->buffersProducing != NULL);
    Buffer* bb = this->buffersProducing;
    if (buf == &bb->buf) {
        this->buffersProducing = this->buffersProducing->next;
    } else {
        Buffer* bp = bb;
        bb = bb->next;
        ASSERT(bb != NULL);
        while (buf != &bb->buf) {
            bp = bb;
            bb = bb->next;
            ASSERT(bb != NULL);
        }
        bp->next = bb->next;
    }
    bb->next = NULL;
    ASSERT(bb->state == STATE_PRODUCING);

    // add bb to end of full list
    bb->state = STATE_FILLED;
    if (this->fullBuffers == NULL) {
        this->fullBuffers = bb;
    } else {
        Buffer* i = this->fullBuffers;
        while (i->next != NULL)
            i = i->next;
        i->next = bb;
    }

    // fire events
    this->newFullBuffer.Set();
    this->bufferStateCompleted.Set();
}


/*
 * BufferMTPConnection<T>::EndOfDataClose
 */
template<class T>
void BufferMTPConnection<T>::EndOfDataClose(void) {
    vislib::sys::AutoLock(this->bufferLock);

    this->endOfData = true;
    this->isOpen = ((this->fullBuffers != NULL) || (this->buffersProducing != NULL));

    this->newEmptyBuffer.Set();
    this->newFullBuffer.Set();
}


/*
 * BufferMTPConnection<T>::GetEmptyBuffer
 */
template<class T>
T* BufferMTPConnection<T>::GetEmptyBuffer(bool wait) {
    Buffer* rv = NULL;

    this->bufferLock.Lock();

    if (this->isOpen && !this->endOfData) {

        if (this->emptyBuffers == NULL) {
            if (wait) {
                while (this->emptyBuffers == NULL) {
                    this->bufferLock.Unlock();

                    this->newEmptyBuffer.Wait();
                    if (!this->isOpen || this->endOfData)
                        return NULL;

                    this->bufferLock.Lock();
                }
            } else {
                this->bufferLock.Unlock();
                return NULL;
            }
        }

        rv = this->emptyBuffers;
        ASSERT(rv->state == STATE_EMPTY);
        this->emptyBuffers = rv->next;
        rv->state = STATE_PRODUCING;
        rv->next = this->buffersProducing;
        this->buffersProducing = rv;
    }

    this->bufferLock.Unlock();

    return (rv == NULL) ? NULL : &rv->buf;
}


/*
 * BufferMTPConnection<T>::GetFilledBuffer
 */
template<class T>
T* BufferMTPConnection<T>::GetFilledBuffer(bool wait) {
    Buffer* rv = NULL;

    this->bufferLock.Lock();

    if (this->isOpen) {

        if (this->fullBuffers == NULL) {

            if (this->endOfData && (this->buffersProducing == NULL)) {
                this->isOpen = false;
                this->bufferLock.Unlock();
                return NULL;
            }

            if (!wait)
                return NULL;

            while (this->fullBuffers == NULL) {
                this->bufferLock.Unlock();

                this->newFullBuffer.Wait();
                if (!this->isOpen)
                    return NULL;

                this->bufferLock.Lock();
            }
        }

        rv = this->fullBuffers;
        ASSERT(rv->state == STATE_FILLED);
        this->fullBuffers = rv->next;
        rv->state = STATE_CONSUMING;
        rv->next = this->buffersConsuming;
        this->buffersConsuming = rv;

        if ((this->fullBuffers == NULL) && (this->buffersProducing == NULL) && this->endOfData) {
            this->isOpen = false;
        }
    }

    this->bufferLock.Unlock();

    return (rv == NULL) ? NULL : &rv->buf;
}


/*
 * BufferMTPConnection<T>::Open
 */
template<class T>
void BufferMTPConnection<T>::Open(void) {
    vislib::sys::AutoLock(this->bufferLock);

    ASSERT(this->isOpen == false);
    ASSERT(this->buffersConsuming == NULL);
    ASSERT(this->buffersProducing == NULL);
    ASSERT(this->fullBuffers == NULL);
    ASSERT(this->emptyBuffers != NULL);

    this->isOpen = true;
    this->endOfData = false;

    this->newEmptyBuffer.Set();
    this->newFullBuffer.Set();
}


} // namespace demos_gl
} /* end namespace megamol */
