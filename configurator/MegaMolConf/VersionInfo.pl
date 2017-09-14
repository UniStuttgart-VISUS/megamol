#
# versioninfo.pl
#
# Copyright (C) 2016 by MegaMol Team
# Alle Rechte vorbehalten.
#

my $haveRegistry = 0;

if (eval {require Win32::TieRegistry;}) {
    $haveRegistry = 1;
}

if ($haveRegistry) {
    Win32::TieRegistry->import(TiedRef => \$Registry,  Delimiter => "/",  ArrayValues => 1,
        SplitMultis => 1,  AllowLoad => 1,
        qw( REG_SZ REG_EXPAND_SZ REG_DWORD REG_BINARY REG_MULTI_SZ
            KEY_READ KEY_WRITE KEY_ALL_ACCESS ));
}

use File::Find;
use File::stat;
use Time::localtime;

my @installFolders;
my $foundRC = 0;
my $RCPath = "";

# http://stackoverflow.com/questions/2363142/how-to-iterate-through-hash-of-hashes-in-perl
sub hash_walk {
    my ($hash, $key_list, $callback) = @_;
    while (my ($k, $v) = each %$hash) {
        # Keep track of the hierarchy of keys, in case
        # our callback needs it.
        push @$key_list, $k;

        if (ref($v) eq 'Win32::TieRegistry') {
            # Recurse.
            hash_walk($v, $key_list, $callback);
        }
        else {
            # Otherwise, invoke our callback, passing it
            # the current key and value, along with the
            # full parentage of that key.
            $callback->($k, $v, $key_list);
        }

        pop @$key_list;
    }
}

sub wantedKey {
    my $k = shift;
    my $v = shift;
    my $keys = shift;
    if ($k =~ /InstallationFolder/) {
        #print "found folder " . @{$v}[0] . "\n";
        push @installFolders, @{$v}[0];
    }
    
}

sub wanted {
    if (/rc\.exe$/i) {
        #print "found rc.exe in " . $File::Find::name . "\n";
        $foundRC = 1;
        $RCPath = $File::Find::name;
    }
}

my $basepath = shift;
push @INC, "rev2ver";
push @INC, "$basepath/rev2ver";

require "rev2ver.inc";

my %hash;

my $proj = getRevisionInfo($basepath . '/.');

$hash{'$PROJ_NAME$'} = 'MegaMol';
$hash{'$PROJ_TEAM$'} = 'MegaMol Team: Visualization Research Center, University of Stuttgart; TU Dresden';
$hash{'$PROJ_DESC$'} = 'MegaMol Project File Configuration Utility. http://megamol.org';

$hash{'$PROJ_MAJVER$'} = 1;
$hash{'$PROJ_MINVER$'} = 2;
$hash{'$PROJ_BUILDVER$'} = 3;

$hash{'$PROJ_REVVER$'} = $proj->rev;
$hash{'$PROJ_YEAR$'} = substr($proj->date, 0, 4);
$hash{'$PROJ_DIRTY$'} = $proj->dirty;

my $oldMTime = (-e ($basepath . "/MegaMolConf/Resources/mmconfig.rc")) ? (stat($basepath . "/MegaMolConf/Resources/mmconfig.rc"))->mtime : 0;

processFile($basepath . "/MegaMolConf/Resources/mmconfig.rc", $basepath . "/MegaMolConf/Resources/mmconfig.rc.input", \%hash);

my $newMTime = (stat($basepath . "/MegaMolConf/Resources/mmconfig.rc"))->mtime;

if (! $haveRegistry) {
    print "Cannot access registry";
    exit 1;
}

if ($oldMTime != $newMTime || ! (-f "/MegaMolConf/Resources/mmconfig.res")) {
    my $machKey = $Registry->Open( "LMachine", {Access=>KEY_READ(),Delimiter=>"/"} ) or  die "Can't open HKEY_LOCAL_MACHINE key: $^E\n";
    my $sdks = $machKey->{"SOFTWARE\\Microsoft\\Microsoft SDKs\\Windows"};
    hash_walk($sdks, [], \&wantedKey);
    
    my $dir = $sdks->{"CurrentInstallFolder"}[0];
    
    find(\&wanted, @installFolders);
    
    if ($foundRC) {
        system(($RCPath, "/r", $basepath . "/MegaMolConf/Resources/mmconfig.rc"));
    } else {
        print "Cannot find resource compiler";
        exit 2;
    }
}

processFile($basepath . "/MegaMolConf/Properties/AssemblyInfo.cs", $basepath . "/MegaMolConf/Properties/AssemblyInfo.cs.input", \%hash);
