
function OnFinish(selProj, selObj)
{
	try
	{
        // Set additional symbols:
        var now = new Date();
        var today = now.getDate() + '.' + (now.getMonth() + 1) + '.' + now.getFullYear();
        wizard.AddSymbol('CURRENT_DATE', today);

        var additionalLibraries = '';
        var linAdditionalLibraries = '';
        if (wizard.FindSymbol('VISLIB_GL')) {
            additionalLibraries += ' opengl32.lib';
            linAdditionalLibraries += ' -lGL -lglut';
        }
        if (wizard.FindSymbol('VISLIB_NET')) {
            additionalLibraries += ' Ws2_32.lib Iphlpapi.lib';
        }
        if (wizard.FindSymbol('VISLIB_SYS')) {
            additionalLibraries += ' shlwapi.lib advapi32.lib';
        }
        wizard.AddSymbol('ADDITIONAL_LIBRARIES', additionalLibraries);
        wizard.AddSymbol('LIN_ADDITIONAL_LIBRARIES', linAdditionalLibraries);

//        if (!wizard.FindSymbol('PROPS_FILENAME') 
//                || (wizard.FindSymbol('PROPS_FILENAME') == '')) {
//            wizard.AddSymbol('PROPS_FILENAME', 'ExtLibs.vsprops');
//        }

		var strProjectPath = wizard.FindSymbol('PROJECT_PATH');
		var strProjectName = wizard.FindSymbol('PROJECT_NAME');

		selProj = CreateCustomProject(strProjectName, strProjectPath);
		//AddConfig(selProj, strProjectName);
		AddFilters(selProj);

		var InfFile = CreateCustomInfFile();
		AddFilesToCustomProj(selProj, strProjectName, strProjectPath, InfFile);
		PchSettings(selProj);
		InfFile.Delete();
		
		AddConfig(selProj, strProjectName);
		
		if (false && wizard.FindSymbol('X64_SUPPORT')) {
		    selProj.Object.AddPlatform('x64');
		}

		selProj.Object.Save();
	}
	catch(e)
	{
		if (e.description.length != 0)
			SetErrorInfo(e);
		return e.number
	}
}

function CreateCustomProject(strProjectName, strProjectPath)
{
	try
	{
		var strProjTemplatePath = wizard.FindSymbol('PROJECT_TEMPLATE_PATH');
		var strProjTemplate = '';
		strProjTemplate = strProjTemplatePath + '\\default.vcproj';

		var Solution = dte.Solution;
		var strSolutionName = "";
		if (wizard.FindSymbol("CLOSE_SOLUTION"))
		{
			Solution.Close();
			strSolutionName = wizard.FindSymbol("VS_SOLUTION_NAME");
			if (strSolutionName.length)
			{
				var strSolutionPath = strProjectPath.substr(0, strProjectPath.length - strProjectName.length);
				Solution.Create(strSolutionPath, strSolutionName);
			}
		}

		var strProjectNameWithExt = '';
		strProjectNameWithExt = strProjectName + '.vcproj';

		var oTarget = wizard.FindSymbol("TARGET");
		var prj;
		if (wizard.FindSymbol("WIZARD_TYPE") == vsWizardAddSubProject)  // vsWizardAddSubProject
		{
			var prjItem = oTarget.AddFromTemplate(strProjTemplate, strProjectNameWithExt);
			prj = prjItem.SubProject;
		}
		else
		{
			prj = oTarget.AddFromTemplate(strProjTemplate, strProjectPath, strProjectNameWithExt);
		}
		return prj;
	}
	catch(e)
	{
		throw e;
	}
}

function AddFilters(proj)
{
	try
	{
		SetupFilters(proj);
	}
	catch(e)
	{
		throw e;
	}
}

function AddConfig(proj, strProjectName)
{
	try
	{
        AddCommonConfig(proj, strProjectName, true ,false);

        /* Customise debug configuration. */
		var config = proj.Object.Configurations('Debug');
		config.IntermediateDirectory = '$(ConfigurationName)';
		config.OutputDirectory = '$(ConfigurationName)';
        config.InheritedPropertySheets = ".\\" + wizard.FindSymbol("PROPS_FILENAME") + ";"
            + wizard.FindSymbol("VISLIB_DIR") + "\\VSProps\\TargetDebug32.vsprops";

		var CLTool = config.Tools('VCCLCompilerTool');
		CLTool.PreprocessorDefinitions = 'WIN32;_DEBUG;';
		CLTool.UsePrecompiledHeader = pchOption.pchNone;
		// TODO: Add compiler settings

		var LinkTool = config.Tools('VCLinkerTool');
		// TODO: Add linker settings
		
        /* Customise release configuration. */
		config = proj.Object.Configurations('Release');
		config.IntermediateDirectory = '$(ConfigurationName)';
		config.OutputDirectory = '$(ConfigurationName)';
		config.InheritedPropertySheets = ".\\" + wizard.FindSymbol('PROPS_FILENAME') + ";"
            + wizard.FindSymbol("VISLIB_DIR") + "\\VSProps\\TargetRelease32.vsprops";

		var CLTool = config.Tools('VCCLCompilerTool');
		CLTool.PreprocessorDefinitions = 'WIN32;NDEBUG;';
		CLTool.UsePrecompiledHeader = pchOption.pchNone;
		// TODO: Add compiler settings

		var LinkTool = config.Tools('VCLinkerTool');
		// TODO: Add linker settings
		
	}
	catch(e)
	{
		throw e;
	}
}

function PchSettings(proj)
{
	// TODO: specify pch settings
}

function DelFile(fso, strWizTempFile)
{
	try
	{
		if (fso.FileExists(strWizTempFile))
		{
			var tmpFile = fso.GetFile(strWizTempFile);
			tmpFile.Delete();
		}
	}
	catch(e)
	{
		throw e;
	}
}

function CreateCustomInfFile()
{
	try
	{
		var fso, TemplatesFolder, TemplateFiles, strTemplate;
		fso = new ActiveXObject('Scripting.FileSystemObject');

		var TemporaryFolder = 2;
		var tfolder = fso.GetSpecialFolder(TemporaryFolder);
		var strTempFolder = tfolder.Drive + '\\' + tfolder.Name;

		var strWizTempFile = strTempFolder + "\\" + fso.GetTempName();

		var strTemplatePath = wizard.FindSymbol('TEMPLATES_PATH');
		var strInfFile = strTemplatePath + '\\Templates.inf';
		wizard.RenderTemplate(strInfFile, strWizTempFile);

		var WizTempFile = fso.GetFile(strWizTempFile);
		return WizTempFile;
	}
	catch(e)
	{
		throw e;
	}
}

function GetTargetName(strName, strProjectName)
{
	try
	{
		// TODO: set the name of the rendered file based on the template filename
		var strTarget = strName;

		if (strName == 'readme.txt')
			strTarget = 'readme.txt';
		if (strName == 'main.cpp') 
		    strTarget = strProjectName + ".cpp";
		if (strName == 'extlibs.vsprops') 
		    strTarget = wizard.FindSymbol('PROPS_FILENAME');

		return strTarget; 
	}
	catch(e)
	{
		throw e;
	}
}

function AddFilesToCustomProj(proj, strProjectName, strProjectPath, InfFile)
{
	try
	{
		var projItems = proj.ProjectItems

		var strTemplatePath = wizard.FindSymbol('TEMPLATES_PATH');

		var strTpl = '';
		var strName = '';

		var strTextStream = InfFile.OpenAsTextStream(1, -2);
		while (!strTextStream.AtEndOfStream)
		{
			strTpl = strTextStream.ReadLine();
			if (strTpl != '')
			{
				strName = strTpl;
				var strTarget = GetTargetName(strName, strProjectName);
				var strTemplate = strTemplatePath + '\\' + strTpl;
				var strFile = strProjectPath + '\\' + strTarget;

				var bCopyOnly = false;  //"true" will only copy the file from strTemplate to strTarget without rendering/adding to the project
				var strExt = strName.substr(strName.lastIndexOf("."));
				if(strExt==".bmp" || strExt==".ico" || strExt==".gif" || strExt==".rtf" || strExt==".css")
					bCopyOnly = true;
				wizard.RenderTemplate(strTemplate, strFile, bCopyOnly);
				if (strExt != '.vsprops') {
				    proj.Object.AddFile(strFile);
				}
			}
		}
		strTextStream.Close();
	}
	catch(e)
	{
		throw e;
	}
}
