using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MegaMolConf.Communication {

    /// <summary>
    /// Utility class to convert parameter type code
    /// </summary>
    public static class ParameterTypeUtility {

        /// <summary>
        /// Answer the parameter type code from a string type name representation (first six characters)
        /// </summary>
        /// <param name="str">a string type name representation (first six characters)</param>
        /// <returns>The parameter type code or zero if invalid/unknown</returns>
        public static ulong MakeParameterTypeCode(string str) {
            return MakeParameterTypeCode(Encoding.ASCII.GetBytes(str));
        }

        /// <summary>
        /// Answer the parameter type code from a byte representation (first six byte)
        /// </summary>
        /// <param name="dat">a byte representation (first six byte)</param>
        /// <returns>The parameter type code or zero if invalid/unknown</returns>
        public static ulong MakeParameterTypeCode(byte[] dat) {
            if (dat.Length < 6) return 0; // error!
            byte[] c = new byte[8] { dat[0], dat[1], dat[2], dat[3], dat[4], dat[5], 0, 0 };
            return BitConverter.ToUInt64(c, 0);
        }

        /// <summary>
        /// Answer the parameter type for a given type code
        /// </summary>
        /// <param name="typeCode">The parameter type code</param>
        /// <returns>The parameter type</returns>
        public static ParameterType TypeFromCode(ParameterTypeCode typeCode) {
            switch(typeCode) {
                case ParameterTypeCode.MMBOOL: return ParameterType.BoolParam;
                case ParameterTypeCode.MMBUTN: return ParameterType.ButtonParam;
                case ParameterTypeCode.MMENUM: return ParameterType.EnumParam;
                case ParameterTypeCode.MMFILA: return ParameterType.FilePathParam;
                case ParameterTypeCode.MMFILW: return ParameterType.FilePathParam;
                case ParameterTypeCode.MMFLOT: return ParameterType.FloatParam;
                case ParameterTypeCode.MMINTR: return ParameterType.IntParam;
                case ParameterTypeCode.MMSTRA: return ParameterType.StringParam;
                case ParameterTypeCode.MMSTRW: return ParameterType.StringParam;
                case ParameterTypeCode.MMTRRY: return ParameterType.TernaryParam;
                case ParameterTypeCode.MMVC2F: return ParameterType.Vector2fParam;
                case ParameterTypeCode.MMVC3F: return ParameterType.Vector3fParam;
                case ParameterTypeCode.MMVC4F: return ParameterType.Vector4fParam;
                case ParameterTypeCode.MMFENU: return ParameterType.FlexEnumParam;
            }
            return ParameterType.Unknown;
        }

    }

}
