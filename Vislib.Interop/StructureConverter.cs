/*
 * StructureConverter.cs
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

using System;
using System.Runtime.InteropServices;


namespace Vislib.Interop {

    /// <summary>
    /// This class provides structure to byte array and vice versa conversion.
    /// </summary>
    public static class StructureConverter {

        /// <summary>
        /// Convert a byte array to a structure of the given type.
        /// </summary>
        /// <typeparam name="T">The type of structure that is contained in
        /// <paramref name="bytes"/>.</typeparam>
        /// <param name="bytes">An array holding the bytes of the 
        /// structure.</param>
        /// <param name="offset">The offset into <paramref name="bytes"/>
        /// at which the structure starts.</param>
        /// <returns>The structure contained in <paramref name="bytes"/>.
        /// </returns>
        public static T BytesToStructure<T>(byte[] bytes, int offset)  
                where T : struct {
            T retval = default(T);
            int len = Marshal.SizeOf(typeof(T));
            IntPtr ptr = Marshal.AllocHGlobal(len);

            Marshal.Copy(bytes, offset, ptr, len);
            retval = (T)Marshal.PtrToStructure(ptr, typeof(T));
            Marshal.FreeHGlobal(ptr);

            return retval;
        }

        /// <summary>
        /// Convert a byte array to a structure of the given type.
        /// </summary>
        /// <typeparam name="T">The type of structure that is contained in
        /// <paramref name="bytes"/>.</typeparam>
        /// <param name="bytes">An array holding the bytes of the 
        /// structure.</param>
        /// <returns>The structure contained in <paramref name="bytes"/>.
        /// </returns>
        public static T BytesToStructure<T>(byte[] bytes) where T : struct {
            return StructureConverter.BytesToStructure<T>(bytes, 0);
        }

        /// <summary>
        /// Convert a structure to a byte array.
        /// </summary>
        /// <typeparam name="T">The type of structure to be converted into a
        /// byte array.</typeparam>
        /// <param name="dst">The array that will receive the structure. The 
        /// array must have at least <c>Marshal.SizeOf(obj) + offset</c> 
        /// elements.
        /// </param>
        /// <param name="offset">The offset into <paramref name="dst"/> where
        /// the structure should be copied to.</param>
        /// <param name="obj">The structure to be converted.</param>
        public static void StructureToBytes<T>(byte[] dst, int offset,
                T obj) where T : struct {
            int len = Marshal.SizeOf(obj);
            IntPtr ptr = Marshal.AllocHGlobal(len);

            Marshal.StructureToPtr(obj, ptr, true);
            Marshal.Copy(ptr, dst, offset, len);
            Marshal.FreeHGlobal(ptr);
        }

        /// <summary>
        /// Convert a structure to a byte array.
        /// </summary>
        /// <typeparam name="T">The type of structure to be converted into a
        /// byte array.</typeparam>
        /// <param name="dst">The array that will receive the structure. The 
        /// array must have at least <c>Marshal.SizeOf(obj)</c> elements.
        /// </param>
        /// <param name="obj">The structure to be converted.</param>
        public static void StructureToBytes<T>(byte[] dst, T obj) 
                where T: struct {
            StructureConverter.StructureToBytes(dst, 0, obj);
        }
    }
}
