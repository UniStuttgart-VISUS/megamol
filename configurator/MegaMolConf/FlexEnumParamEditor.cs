using System;
using System.ComponentModel;
using System.Drawing.Design;
using System.Globalization;
using System.Windows.Forms;
using System.Windows.Forms.Design;

namespace MegaMolConf {

    internal class FlexEnumParamEditor : UITypeEditor {

        internal class ElementTypeConverter : TypeConverter {
            public override bool CanConvertTo(ITypeDescriptorContext context, Type destinationType) {
                return typeof(string) == destinationType;
            }
            public override object ConvertTo(ITypeDescriptorContext context, CultureInfo culture, object value, Type destinationType) {
                if (typeof(string) == destinationType) {
                    return ((Element)value).DisplayName;
                }
                return string.Empty;
            }
        }

        [TypeConverter(typeof(ElementTypeConverter))]
        internal class Element {
            public string Value { get; set; }
            public Element(string n) {
                Value = n;
            }
            public string DisplayName {
                get { return Value; }
            }
            public override string ToString() {
                return Value;
            }
        };

        private Data.ParamType.FlexEnum e;

        private IWindowsFormsEditorService _editorService;

        public FlexEnumParamEditor(Data.ParamType.FlexEnum e) {
            this.e = e;
        }

        public override UITypeEditorEditStyle GetEditStyle(ITypeDescriptorContext context) {
            // drop down mode (we'll host a listbox in the drop down)
            return UITypeEditorEditStyle.DropDown;
        }

        public override object EditValue(ITypeDescriptorContext context, IServiceProvider provider, object value) {
            _editorService = (IWindowsFormsEditorService)provider.GetService(typeof(IWindowsFormsEditorService));

            // use a list box
            ListBox lb = new ListBox();
            lb.SelectionMode = SelectionMode.One;
            lb.SelectedValueChanged += OnListBoxSelectedValueChanged;
            
            lb.DisplayMember = "DisplayName";

            for (int i = 0; i < this.e.Values.Length; i++)
            {
                Element e = new Element(this.e.Values[i]);
                int index = lb.Items.Add(e);
                if (e.Value == ((Element)value).Value)
                {
                    lb.SelectedIndex = i;
                }
            }

            // show this model stuff
            _editorService.DropDownControl(lb);
            if (lb.SelectedItem == null) // no selection, return the passed-in value as is
                return value;

            return lb.SelectedItem;
        }

        private void OnListBoxSelectedValueChanged(object sender, EventArgs e) {
            // close the drop down as soon as something is clicked
            _editorService.CloseDropDown();
        }

    }

}
