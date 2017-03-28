[CmdletBinding(SupportsShouldProcess = $true)]
param(#[string] $Project = '.\cinematiccam02-2views-lasercross.mmprj',
    [Parameter(Mandatory=$true)] [string] $Project,
    [int] $From,
    [int] $To,
    [float] $TimeStep)


# Gets a parameter tag or creates a new one with the specified value.
function Get-Param($module, $name, $value) {
    $retval = $module.SelectSingleNode("param[@name=`"$name`"]")

    if (!$retval) {
        Write-Verbose "Appending mandatory parameter `"$name`" ..."
        $p = $module.OwnerDocument.CreateElement('param')
        $module.AppendChild($p) | Out-Null

        $a = $module.OwnerDocument.CreateAttribute('name')
        $a.Value = $name
        $p.Attributes.Append($a) | Out-Null

        $a = $module.OwnerDocument.CreateAttribute('value')
        $a.Value = $value
        $p.Attributes.Append($a) | Out-Null

        $retval = $p
    }

    return $retval
}


# Possible values for 'skyboxSide'.
$sides = @('Top', 'Front', 'Left', 'Back', 'Right', 'Bottom')

# Read the original project file.
[xml] $xml = gc $Project

# Get the view module we will edit and its containing view.
$viewMod = $xml.SelectSingleNode('//module[@class="CinematicView"]')
if (!$viewMod) {
    throw 'The project must contain a CinematicView used for rendering the scene.'
}

$view = $viewMod.ParentNode

# Get the screenshooter element to edit.
$shootMod = $view.SelectSingleNode('module[@class="ScreenShooter"]')
if (!$shootMod) {
    throw 'The project must contain a ScreenShooter which saves the output to disk.'
}

# Get the parameters we need to modify.
$paramSide = Get-Param $viewMod 'cinematicCam::skyboxSide' $sides[0]
$paramFile = Get-Param $shootMod 'filename' ''
$paramInst = Get-Param $shootMod 'view' 

$baseName = Split-Path -Leaf $Project
$ext = [System.IO.Path]::GetExtension($Project)
for ($i = 0; $i -lt $sides.Length; ++$i) {
    # Note: The identifier of the side in the project name is a numeric value,
    # which allows us to create parametric sweep on the cluster.
    $fileName = [System.IO.Path]::ChangeExtension($Project, "$i$ext")
    # Determine side human-readable name of the side for easier handling of the
    # output in post-processing of the video material.
    $sideName = $sides[$i].ToLower()
    
    $paramSide.value = $sides[$i]
    $paramFile.value = [System.IO.Path]::ChangeExtension($baseName, "$sideName.png")
    if ($PSCmdlet.ShouldProcess($filePath, 'Save')) {
        New-Item $fileName -Force -ItemType File
        $xml.Save((Resolve-Path $fileName))
    }
}

$hpcSweep = [System.IO.Path]::ChangeExtension($baseName, "*$ext")
$hpcCmd = @"
param([string] `$WorkingDirectory = '\\vesta1\Entwicklung\sfbdome',
      [string] `$HeadNode = 'vesta1',
      [string] `$MegaMol = 'bin\x64\Release\MegaMolCon.exe')
if (Test-Path env:CCP_SCHEDULER) { 
    `$HeadNode = (gc env:CCP_SCHEDULER) 
}
`$job = New-HpcJob -Scheduler `$HeadNode -Name '$baseName' -JobEnv 'HPC_ATTACHTOCONSOLE=True' -Exclusive `$true -NumNodes '*-*'
Add-HpcTask -Scheduler `$HeadNode -Job `$job -Type ParametricSweep -Start 0 -End $($sides.Length - 1) -NumNodes '1-$($sides.Length)' -WorkDir `$WorkingDirectory -Command `"`$MegaMol -p $hpcSweep -i $($view.name) $($paramInst.value)`"
Submit-HpcJob -Scheduler `$HeadNode -Job `$job
"@

$hpcCmd | Out-File -FilePath $([System.IO.Path]::ChangeExtension($Project, "ps1"))