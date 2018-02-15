# MegaMol Development Manual

<!-- TOC -->

## Contents

- [Contents](#contents)
    - [Adding new Parameter Type](#add-param)

<!-- /TOC -->

<a name="add-param"></a>

## Adding new Parameter Type

### mmconsole
This depends on whether the new parameter should behave differently from existing parameters (UI-wise). That is, whether values need to be validated against something (e.g. existence of a file) or should trigger specific behaviour on update.

1.    Choose a unique type identifier. It should start with `MM` and be at most 6 characters long (including the discriminator suffix `A` or `W` if it behaves differently when compling with WString support!)

1.
    For new behaviour:
    * Add a new Param class in `console/src/gl/ATParamBar.[h|cpp]`. It needs to be derived vom `ValueParam` and implement at least `Get` and `Set`.
    *  Instance the parameter in `gl::ATParamBar::addParamVariable`.

    If you can use existing behaviour:
    * Instance the correct, already existing parameter class in `console/src/gl/ATParamBar.[h|cpp]`. StringParam is the default and fallback, so if that is enough, you can skip this step. 

### Core

1. Add a new subclass of `core::param::AbstractParam` in `core/src/param/` and `core/include/mmcore/param`.

1.
    Integrate the new parameter into the state file generator for the configurator: `core/src/jon/PluginsStateFileGeneratorJob.cpp`.
    * Add code for dispatching in `PluginsStateFileGeneratorJob::WriteParamInfo`. Beware that for subclasses you need to make sure that the new code needs to go before the code of the superclass, otherwise the cast will succeed for the superclass first and shortcut through the if statements.
    * Add the corresponding `WriteParamInfo` override.

###  Configurator

1. In `configurator\MegaMolConf\Connection\ParameterTypeCode.cs`, add an enum identifier and the value for serialization. Note that the value is basically the memory dump of the identifier (beware of endianness, read the number backwards.). For string-type parameters, remember you have to add `.*A` and `.*W`.
1. Add the enum entries to `configurator\MegaMolConf\Connection\ParameterType.cs`.
1. Extend the switch in `configurator\MegaMolConf\Connection\ParameterTypeUtility.cs` You can use the same code path for `.*A` and `.*W` here.
1. Define a new `Data.ParamType` in `configurator\MegaMolConf\Data\ParamType\`. This will allow you to have a new behaviour for this type. We will define a custom editor for the property grid in a moment. This also needs no distinction for `.*A` and `.*W`.
1. `ParamSlot.cs`: Create a new branch in `TypeFromTypeInfo` to instantiate the new ParamType.
1. Add the ParamType to the state by extending `configurator\MegaMolConf\data\ParamTypeBase.cs`.
1. If you want to add a specific editor to better manipulate a parameter, add a class implementing `UITypeEditor`. This class. To make the PropertyGrid use this editor, extend `GraphicalModuleDescriptor::GetEditor` to return an instance of this new class.

extend `GraphicalModuleDescriptor`??? no that is for something else.