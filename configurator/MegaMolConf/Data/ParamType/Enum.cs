using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MegaMolConf.Data.ParamType {

    public sealed class Enum : ParamTypeValueBase {
        public int DefaultValue { get; set; }
        public int[] Values { get; set; }
        public string[] ValueNames { get; set; }
        public override void ParseDefaultValue(string v) {
            this.DefaultValue = this.ParseValue(v);
        }
        public int ParseValue(string v) {
            System.Diagnostics.Debug.Assert(this.Values.Length == this.ValueNames.Length);

            for (int i = 0; i < this.ValueNames.Length; i++) {
                if (this.ValueNames[i].Equals(v)) {
                    return this.Values[i];
                }
            }

            int iv = int.Parse(v);
            if (!this.Values.Contains(iv)) {
                throw new Exception();
            }

            return iv;
        }
        public override string DefaultValueString() {
            return this.DefaultValue.ToString();
        }
        public override bool ValuesEqual(string a, string b) {
            return ParseValue(a).Equals(ParseValue(b));
        }
    }

}
