using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MegaMolConf.Communication {

    /// <summary>
    /// Type description of MegaMol parameters
    /// </summary>
    public class ParameterTypeDescription {

        private Dictionary<string, object> extra = new Dictionary<string, object>();

        /// <summary>
        /// The parameter type
        /// </summary>
        public ParameterType Type { get; private set; }

        /// <summary>
        /// The parameter type code
        /// </summary>
        public ParameterTypeCode TypeCode { get; private set; }

        /// <summary>
        /// Gets the extra settings values
        /// </summary>
        public IReadOnlyDictionary<string, object> ExtraSettings {
            get { return extra; }
        }

        public bool SetFromHexStringDescription(string hex) {
            byte[] theBytes;

            if (hex.Length % 2 == 0) {
                theBytes = new byte[hex.Length / 2];
                for (int x = 0; x < hex.Length; x += 2) {
                    theBytes[x / 2] = Convert.ToByte(hex.Substring(x, 2), 16);
                }
                SetFromBinDescription(theBytes);
                return true;
            }
            return false;
        }

        public void SetFromBinDescription(byte[] dat) {
            if (dat.Length < 6) throw new ArgumentException("dat insufficient");

            TypeCode = (ParameterTypeCode)ParameterTypeUtility.MakeParameterTypeCode(dat);
            Type = ParameterTypeUtility.TypeFromCode(TypeCode);
            extra.Clear();

            if (dat.Length <= 6) return; // no further data available

            switch (Type) {
                case ParameterType.ButtonParam: {
                        if (dat.Length < 6 + 2) return;
                        extra["KeyCode"] = BitConverter.ToUInt16(dat, 6);
                    } break;
                case ParameterType.EnumParam: {
                        UInt32 cnt = BitConverter.ToUInt32(dat, 6);
                        int p = 10;
                        for (UInt32 i = 0; i < cnt; ++i) {
                            Int32 val = BitConverter.ToInt32(dat, p);
                            p += 4;
                            int start = p;
                            while (dat[p] != 0) p++; // seek end
                            string name = Encoding.UTF8.GetString(dat, start, p - start);
                            p++;

                            extra["v" + i.ToString()] = val;
                            extra["n" + i.ToString()] = name;
                        }
                    } break;
                case ParameterType.FlexEnumParam: {
                        UInt32 cnt = BitConverter.ToUInt32(dat, 6);
                        int p = 10;
                        for (UInt32 i = 0; i < cnt; ++i) {
                            Int32 val = BitConverter.ToInt32(dat, p);
                            p += 4;
                            int start = p;
                            while (dat[p] != 0)
                                p++; // seek end
                            string name = Encoding.UTF8.GetString(dat, start, p - start);
                            p++;

                            extra["v" + i.ToString()] = val;
                            extra["n" + i.ToString()] = name;
                        }
                    }
                    break;
                case ParameterType.FloatParam: {
                        if (dat.Length < 6 + 4 + 4) return;
                        extra["min"] = BitConverter.ToSingle(dat, 6);
                        extra["max"] = BitConverter.ToSingle(dat, 6 + 4);
                    }
                    break;
                case ParameterType.IntParam: {
                        if (dat.Length < 6 + 4 + 4) return;
                        extra["min"] = BitConverter.ToInt32(dat, 6);
                        extra["max"] = BitConverter.ToInt32(dat, 6 + 4);
                    }
                    break;
                case ParameterType.Vector2fParam: {
                        if (dat.Length < 6 + 4 * 4) return;
                        extra["minX"] = BitConverter.ToSingle(dat, 6);
                        extra["minY"] = BitConverter.ToSingle(dat, 6 + 4);
                        extra["maxX"] = BitConverter.ToSingle(dat, 6 + 8);
                        extra["maxY"] = BitConverter.ToSingle(dat, 6 + 12);
                    }
                    break;
                case ParameterType.Vector3fParam: {
                        if (dat.Length < 6 + 6 * 4) return;
                        extra["minX"] = BitConverter.ToSingle(dat, 6);
                        extra["minY"] = BitConverter.ToSingle(dat, 6 + 4);
                        extra["minZ"] = BitConverter.ToSingle(dat, 6 + 8);
                        extra["maxX"] = BitConverter.ToSingle(dat, 6 + 12);
                        extra["maxY"] = BitConverter.ToSingle(dat, 6 + 16);
                        extra["maxZ"] = BitConverter.ToSingle(dat, 6 + 20);
                    }
                    break;
                case ParameterType.Vector4fParam: {
                        if (dat.Length < 6 + 8 * 4) return;
                        extra["minX"] = BitConverter.ToSingle(dat, 6);
                        extra["minY"] = BitConverter.ToSingle(dat, 6 + 4);
                        extra["minZ"] = BitConverter.ToSingle(dat, 6 + 8);
                        extra["minW"] = BitConverter.ToSingle(dat, 6 + 12);
                        extra["maxX"] = BitConverter.ToSingle(dat, 6 + 16);
                        extra["maxY"] = BitConverter.ToSingle(dat, 6 + 20);
                        extra["maxZ"] = BitConverter.ToSingle(dat, 6 + 24);
                        extra["maxW"] = BitConverter.ToSingle(dat, 6 + 28);
                    }
                    break;
            }
        }

    }

}
