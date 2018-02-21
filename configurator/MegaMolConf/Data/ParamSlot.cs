using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace MegaMolConf.Data {
    [DebuggerDisplay("ParamSlot {Name}")]
    public sealed class ParamSlot {
        public string Name { get; set; }
        public string Description { get; set; }
        public ParamTypeBase Type { get; set; }

        /// <summary>
        /// Creates a type object based on a type info description
        /// </summary>
        /// <param name="typeInfo">The type info description</param>
        /// <returns>The type object</returns>
        public Data.ParamTypeBase TypeFromTypeInfo(byte[] typeInfo) {
            string typeName = string.Empty;
            if (typeInfo.Length >= 6) {
                typeName = Encoding.ASCII.GetString(typeInfo, 0, 6);
            }

            Data.ParamTypeBase ptb = null;

            if (typeName == "MMBUTN") {
                ptb = new Data.ParamType.Button();
            } else if (typeName == "MMBOOL") {
                ptb = new Data.ParamType.Bool();
            } else if (typeName == "MMENUM") {
                ptb = new Data.ParamType.Enum();
                int pos = 6;
                int cnt = (int)BitConverter.ToUInt32(typeInfo, pos);
                ((Data.ParamType.Enum)ptb).ValueNames = new string[cnt];
                ((Data.ParamType.Enum)ptb).Values = new int[cnt];
                pos += 4;
                for (int i = 0; i < cnt; i++) {
                    ((Data.ParamType.Enum)ptb).Values[i] = (int)BitConverter.ToUInt32(typeInfo, pos);
                    pos += 4;
                    int ePos = pos;
                    while ((ePos < typeInfo.Length) && (typeInfo[ePos] != 0)) ePos++;
                    ((Data.ParamType.Enum)ptb).ValueNames[i] = Encoding.ASCII.GetString(typeInfo, pos, ePos - pos);
                    pos = ePos + 1;
                }
            } else if (typeName == "MMFENU") {
                ptb = new Data.ParamType.FlexEnum();
                int pos = 6;
                int cnt = (int)BitConverter.ToUInt32(typeInfo, pos);
                ((Data.ParamType.FlexEnum)ptb).Values = new string[cnt];
                pos += 4;
                for (int i = 0; i < cnt; i++) {
                    int ePos = pos;
                    while ((ePos < typeInfo.Length) && (typeInfo[ePos] != 0))
                        ePos++;
                    ((Data.ParamType.FlexEnum)ptb).Values[i] = Encoding.ASCII.GetString(typeInfo, pos, ePos - pos);
                    pos = ePos + 1;
                }
            } else if (typeName == "MMFLOT") {
                ptb = new Data.ParamType.Float();
                ((Data.ParamType.Float)ptb).MinValue = BitConverter.ToSingle(typeInfo, 6);
                ((Data.ParamType.Float)ptb).MaxValue = BitConverter.ToSingle(typeInfo, 10);
            } else if (typeName == "MMINTR") {
                ptb = new Data.ParamType.Int();
                ((Data.ParamType.Int)ptb).MinValue = BitConverter.ToInt32(typeInfo, 6);
                ((Data.ParamType.Int)ptb).MaxValue = BitConverter.ToInt32(typeInfo, 10);
            } else if (typeName == "MMFILW" || typeName == "MMFILA") {
                ptb = new Data.ParamType.FilePath();
            } else if (typeName == "MMTF1W" || typeName == "MMTF1A") {
                ptb = new Data.ParamType.TransferFunc1D();
            } else {
                ptb = new Data.ParamType.String();
            }

            if (ptb != null) {
                ptb.TypeName = typeName;
            }
            return ptb;
        }
    }
}
