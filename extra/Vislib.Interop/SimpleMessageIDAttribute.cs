/*
 * SimpleMessageIDAttribute.cs
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

using System;
using System.Collections.Generic;
using System.Text;

namespace Vislib.Interop {

    /// <summary>
    /// This attribute allows for automatic packing of structures into
    /// VISlib simple messages. The header will draw the message ID from the
    /// attribute if available.
    /// </summary>
    [AttributeUsage(AttributeTargets.Struct | AttributeTargets.Class)]
    public class SimpleMessageIDAttribute : Attribute {

        /// <summary>
        /// Creates a new attribute with the given message ID.
        /// </summary>
        /// <param name="id">The message ID.</param>
        public SimpleMessageIDAttribute(UInt32 id) {
            this.ID = id;
        }

        /// <summary>
        /// Gets the message ID.
        /// </summary>
        public UInt32 ID {
            get;
            private set;
        }
    }
}
