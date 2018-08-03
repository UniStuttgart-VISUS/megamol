using System;
using System.Linq;

namespace MegaMolConf.Data.ParamType {

    public sealed class Enum : ParamTypeValueBase {
        public int DefaultValue { get; set; }
        public int[] Values { get; set; }
        public string[] ValueNames { get; set; }
        public override void ParseDefaultValue(string v) {
            DefaultValue = ParseValue(v);
        }
        public int ParseValue(string v) {
            System.Diagnostics.Debug.Assert(Values.Length == ValueNames.Length);

            for (int i = 0; i < ValueNames.Length; i++) {
                if (ValueNames[i].Equals(v)) {
                    return Values[i];
                }
            }

            int iv = int.Parse(v);
            if (!Values.Contains(iv)) {
                throw new Exception();
            }

            return iv;
        }
        public override string DefaultValueString() {
            return DefaultValue.ToString();
        }
        public override bool ValuesEqual(string a, string b) {
            return ParseValue(a).Equals(ParseValue(b));
        }
    }

}
