/*
 * SimpleMessageHeader.cs
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;


namespace Vislib.Interop {

    /// <summary>
    /// This class wraps a <see cref="SimpleMessageHeaderData"/> structure and
    /// provides special accessors and conversion methods.
    /// </summary>
    public class SimpleMessageHeader {

        /// <summary>
        /// First reserved message ID. All user-defined message IDs should be 
        /// less than this value.
        /// </summary>
        /// <remarks>
        /// This value is copied from SimpleMessageHeaderData.h of the C++ 
        /// version.
        /// </remarks>
        public static UInt32 VLSNP1_FIRST_RESERVED_MESSAGE_ID {
            get {
                return (UInt32.MaxValue - 1024);
            }
        }

        /// <summary>
        /// Answer the size of the header block in bytes.
        /// </summary>
        public static int Size {
            get {
                return Marshal.SizeOf(typeof(SimpleMessageHeaderData));
            }
        }

        /// <summary>
        /// Create a new instance with message ID 0 and zero-sized body.
        /// </summary>
        public SimpleMessageHeader() : this(0, 0) {}

        /// <summary>
        /// Create a new instance from the given header data.
        /// </summary>
        /// <param name="data">The header data to wrap.</param>
        public SimpleMessageHeader(SimpleMessageHeaderData data) {
            this.data = data;
        }

        /// <summary>
        /// Create a new instance from a serialised 
        /// <see cref="SimpleMessageHeaderData"/> structure.
        /// </summary>
        /// <param name="data">Byte representation of the header data
        /// structure.</param>
        public SimpleMessageHeader(byte[] data) {
            this.UpdateData(data);
        }

        /// <summary>
        /// Create a new instance using the given message ID and body size.
        /// </summary>
        /// <param name="messageId">The message ID to be set.</param>
        /// <param name="bodySize">The body size in bytes to be set.</param>
        public SimpleMessageHeader(UInt32 messageId, UInt32 bodySize) {
            this.MessageID = messageId;
            this.BodySize = bodySize;
        }

        /// <summary>
        /// Gets or sets the message ID.
        /// </summary>
        public UInt32 MessageID {
            get {
                return this.data.MessageID;
            }
            set {
                this.data.MessageID = value;
            }
        }

        /// <summary>
        /// Gets or sets the body size in bytes.
        /// </summary>
        public UInt32 BodySize {
            get {
                return this.data.BodySize;
            }
            set {
                this.data.BodySize = value;
            }
        }

        /// <summary>
        /// Gets or sets the underlying <see cref="SimpleMessageHeaderData"/>.
        /// </summary>
        public SimpleMessageHeaderData Data {
            get {
                return this.data;
            }
            set {
                this.data = value;
            }
        }

        /// <summary>
        /// Update the header data from the given byte array.
        /// </summary>
        /// <param name="data">Byte representation of the header data
        /// structure.</param>
        public void UpdateData(byte[] data) {
            this.data = StructureConverter.BytesToStructure<
                SimpleMessageHeaderData>(data);
        }

        /// <summary>
        /// The underlying <see cref="SimpleMessageHeaderData"/>.
        /// </summary>
        private SimpleMessageHeaderData data;
    }

}
