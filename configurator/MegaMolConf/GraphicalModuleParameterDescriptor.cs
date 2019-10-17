using System;
using System.ComponentModel;

namespace MegaMolConf {

    internal class GraphicalModuleParameterDescriptor : PropertyDescriptor {

        private Data.ParamSlot p;
        private bool useInCmdLine;

        public GraphicalModuleParameterDescriptor(Data.ParamSlot p, bool useInCmdLine)
            : base(p.Name, new Attribute[] { }) {
            this.p = p;
            this.useInCmdLine = useInCmdLine;
        }

        public Data.ParamSlot Parameter {
            get { return p; }
        }

        public override bool CanResetValue(object component) {
            return ((GraphicalModule)component).ParameterValues[p] != ((Data.ParamTypeValueBase)p.Type).DefaultValueString();
        }

        public override Type ComponentType {
            get { return typeof(GraphicalModule); }
        }

        public override object GetValue(object component) {
            if (p.Type is Data.ParamType.Enum) {
                int v = ((Data.ParamType.Enum)p.Type).ParseValue(((GraphicalModule)component).ParameterValues[p]);
                for (int j = 0; j < ((Data.ParamType.Enum)p.Type).Values.Length; j++) {
                    if (((Data.ParamType.Enum)p.Type).Values[j] == v) {
                        return new EnumParamEditor.Element(v, ((Data.ParamType.Enum)p.Type).ValueNames[j]);
                    }
                }
                throw new Exception("Internal Error 72");
            } else if (p.Type is Data.ParamType.FlexEnum && ((Data.ParamType.FlexEnum)p.Type).Values.Length > 0) {
                return new FlexEnumParamEditor.Element(((GraphicalModule)component).ParameterValues[p]);
                throw new Exception("Internal Error 73");
            }
            return ((GraphicalModule)component).ParameterValues[p];
        }

        public override bool IsReadOnly {
            get { return false; }
        }

        public override Type PropertyType {
            get {
                if (p.Type is Data.ParamType.Bool) return typeof(bool);
                if (p.Type is Data.ParamType.Int) return typeof(int);
                if (p.Type is Data.ParamType.Float) return typeof(float);
                if (p.Type is Data.ParamType.Enum) return typeof(EnumParamEditor.Element); // TODO: Change!
                if (p.Type is Data.ParamType.FlexEnum && ((Data.ParamType.FlexEnum)p.Type).Values.Length > 0) return typeof(FlexEnumParamEditor.Element); // TODO: Change!
                if (p.Type is Data.ParamType.FilePath) return typeof(string); // TODO: is this really the sensible solution?
                if (p.Type is Data.ParamType.TransferFunction) return typeof(string); // TODO: is this really the sensible solution?
                return typeof(string); // all else is string (because buttons are handled by another class)
            }
        }

        public override void ResetValue(object component) {
            ((GraphicalModule)component).ParameterValues[p] = ((Data.ParamTypeValueBase)p.Type).DefaultValueString();
        }

        //[System.Diagnostics.DebuggerStepThrough]
        public override void SetValue(object component, object value) {
            if (p.Type is Data.ParamType.Int) {
                int i = int.Parse(value.ToString());
                if ((i < ((Data.ParamType.Int)p.Type).MinValue) || (i > ((Data.ParamType.Int)p.Type).MaxValue)) {
                    throw new Exception(String.Format("Value must be in Range [{0} ... {1}]",
                        ((Data.ParamType.Int)p.Type).MinValue, ((Data.ParamType.Int)p.Type).MaxValue));
                }
            }
            if (p.Type is Data.ParamType.Float) {
                float f = float.Parse(value.ToString());
                if ((f < ((Data.ParamType.Float)p.Type).MinValue) || (f > ((Data.ParamType.Float)p.Type).MaxValue)) {
                    throw new Exception(String.Format("Value must be in Range [{0} ... {1}]",
                        ((Data.ParamType.Float)p.Type).MinValue, ((Data.ParamType.Float)p.Type).MaxValue));
                }
                value = f.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
            ((GraphicalModule)component).ParameterValues[p] = value.ToString();
        }

        public override bool ShouldSerializeValue(object component) {
            return CanResetValue(component);
        }

        public override string DisplayName {
            get {
                string name = p.Name;
                if (useInCmdLine) name += "*";
                return name;
            }
        }

        public override string Description {
            get {
                return p.Description;
            }
        }

        public override string Category {
            get {
                return "Parameters";
            }
        }

        public bool UseInCmdLine {
            get { return useInCmdLine; }
            set { useInCmdLine = value; }
        }

        public override object GetEditor(Type editorBaseType)
        {
            if (p.Type is Data.ParamType.Enum) {
                return new EnumParamEditor((Data.ParamType.Enum)p.Type);
            }
            else if (p.Type is Data.ParamType.FlexEnum && ((Data.ParamType.FlexEnum)p.Type).Values.Length > 0) {
                return new FlexEnumParamEditor((Data.ParamType.FlexEnum)p.Type);
            }
            else if (p.Type is Data.ParamType.FilePath) {
                return new System.Windows.Forms.Design.FileNameEditor();
            }
            else if (p.Type is Data.ParamType.Color) {
                return new ColorEditor();
            }
            return new System.ComponentModel.Design.MultilineStringEditor();
        }

    }

}
