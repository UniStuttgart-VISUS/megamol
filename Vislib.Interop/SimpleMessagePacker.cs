/*
 * SimpleMessagePacker.cs
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;
using System.Linq;


namespace Vislib.Interop {

    /// <summary>
    /// This utility class composes the byte representation of VISlib simple
    /// messages.
    /// </summary>
    public static class SimpleMessagePacker {

        /// <summary>
        /// Pack the given message header and body into a single message array.
        /// </summary>
        /// <remarks>
        /// The method will update the body size in the header to match the body
        /// size specified in the parameters. The message ID, however, will 
        /// remain unchanged.
        /// </remarks>
        /// <param name="header">The message header.</param>
        /// <param name="body">The message body.</param>
        /// <param name="offset">The offset into <paramref name="body"/> to copy
        /// the body data from.</param>
        /// <param name="length">The number of bytes to copy from 
        /// <paramref name="body"/>.</param>
        /// <returns>An array holding the whole message.</returns>
        /// <exception cref="ArgumentException">In case 
        /// <paramref name="offset"/> is negative.</exception>
        /// <exception cref="ArgumentException">In case
        /// <paramref name="length"/> is less than one.</exception>
        /// <exception cref="ArgumentException">In case <paramref name="body"/>
        /// is not enough to copy <paramref name="length"/> bytes starting at
        /// offset <paramref name="offset"/>.</exception>
        public static byte[] Pack(SimpleMessageHeader header, byte[] body, 
                int offset, int length) {
            if (offset < 0) {
                throw new ArgumentException("The offset must not be negative.",
                    "offset");
            }
            if (length < 1) {
                throw new ArgumentException("The body data must not be empty.",
                    "length");
            }
            if (body.Length < offset + length) {
                throw new ArgumentException("The message body is too short.", 
                    "body");
            }

            header.BodySize = (UInt32) length;
            byte[] retval = new byte[SimpleMessageHeader.Size 
                + header.BodySize];
            
            StructureConverter.StructureToBytes(retval, header.Data);
            Buffer.BlockCopy(body, offset, retval, SimpleMessageHeader.Size, 
                length);

            return retval;
        }

        /// <summary>
        /// Pack the given message header and body into a single message array.
        /// </summary>
        /// <remarks>
        /// The method will update the body size in the header to match the body
        /// size specified in the parameters. The message ID, however, will 
        /// remain unchanged.
        /// </remarks>
        /// <param name="header">The message header.</param>
        /// <param name="body">The message body.</param>
        /// <param name="offset">The offset into <paramref name="body"/> to copy
        /// the body data from.</param>
        /// <returns>An array holding the whole message.</returns>
        public static byte[] Pack(SimpleMessageHeader header, byte[] body, 
                int offset) {
            return SimpleMessagePacker.Pack(header, body, offset, 
                body.Length - offset);
        }

        /// <summary>
        /// Pack the given message header and body into a single message array.
        /// </summary>
        /// <remarks>
        /// The method will update the body size in the header to match the body
        /// size specified in the parameters. The message ID, however, will 
        /// remain unchanged.
        /// </remarks>
        /// <param name="header">The message header.</param>
        /// <param name="body">The message body.</param>
        /// <returns>An array holding the whole message.</returns>
        public static byte[] Pack(SimpleMessageHeader header, byte[] body) {
            return SimpleMessagePacker.Pack(header, body, 0);
        }

        /// <summary>
        /// Pack the given message header and body into a single message array.
        /// </summary>
        /// <remarks>
        /// The method will update the body size in the header to match the body
        /// size specified in the parameters. The message ID, however, will 
        /// remain unchanged.
        /// </remarks>
        /// <typeparam name="T">The type of structure that is passed in as 
        /// message body.</typeparam>
        /// <param name="header">The message header.</param>
        /// <param name="body">The message body.</param>
        /// <returns>An array holding the whole message.</returns>
        public static byte[] Pack<T>(SimpleMessageHeader header, T body) 
                where T : struct {
            header.BodySize = (UInt32) Marshal.SizeOf(body.GetType());
            byte[] retval = new byte[SimpleMessageHeader.Size 
                + header.BodySize];

            StructureConverter.StructureToBytes(retval, header.Data);
            StructureConverter.StructureToBytes(retval,
                SimpleMessageHeader.Size, body);

            return retval;
        }

        /// <summary>
        /// Pack the given message body into a message array with a message
        /// header prepended.
        /// </summary>
        /// <remarks>
        /// The message body must have a 
        /// <seealso cref="SimpleMessageIDAttribute"/> for this method to
        /// succeed.
        /// </remarks>
        /// <typeparam name="T">The type of structure that is passed in as 
        /// message body.</typeparam>
        /// <param name="body">The message body.</param>
        /// <returns>An array holding the whole message.</returns>
        public static byte[] Pack<T>(T body)
                where T : struct {
            SimpleMessageIDAttribute att = null;

            try {
                att = (from a in body.GetType().GetCustomAttributes(true)
                       where a is SimpleMessageIDAttribute
                       select a as SimpleMessageIDAttribute).Single();
            } catch {
            }
            
            if (att == null) {
                throw new ArgumentException("The message body must have a "
                    + "SimpleMessageIDAttribute.", "body");
            }

            return SimpleMessagePacker.Pack(new SimpleMessageHeader(att.ID, 
                (UInt32) Marshal.SizeOf(body.GetType())), body);
        }

        /// <summary>
        /// Unpacks the message header.
        /// </summary>
        /// <remarks>
        /// To just convert the begin of the message into a 
        /// <see cref="SimpleMessageHeader"/>, call the appropriate ctor. This 
        /// will skip any sanity check performed by this method.
        /// </remarks>
        /// <param name="msg">The complete message containing header and body.
        /// </param>
        /// <param name="bodyOffset">Receives the offset of the message body in
        /// bytes.</param>
        /// <returns>The message header contained in <paramref name="msg"/>.
        /// </returns>
        public static SimpleMessageHeader Unpack(byte[] msg, 
                out int bodyOffset) {
            bodyOffset = SimpleMessageHeader.Size;
            if (msg.Length < bodyOffset) {
                throw new ArgumentException("The message is too short to "
                    + "contain even the message header.", "msg");
            }

            SimpleMessageHeader retval = new SimpleMessageHeader(msg);
            if (msg.Length < retval.BodySize + bodyOffset) {
                throw new ArgumentException("The message does not contain "
                    + "enough body bytes according to the header.", "msg");
            }

            return retval;
        }
    }
}
