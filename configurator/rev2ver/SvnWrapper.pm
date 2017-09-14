package SvnWrapper;
use strict;
use File::Spec;
use File::stat;
use Cwd qw{realpath};
use RelevanceChecker;

sub new {
    my $type = shift;
    my $self = {};
    
    $self->{filterRef} = shift;
    my %localDefaultFilter = ( default => "consider" );
    if (! defined $self->{filterRef}) {
        $self->{filterRef} = \%localDefaultFilter;
    }
    
    my $oleModule = "Win32::OLE";
    eval "use $oleModule";
    if (!$@ && defined($oleModule) && defined(my $svnOLE = $oleModule->new('SubWCRev.object', ''))) {
        $self->{_ole} = $svnOLE;   
    }
    #$self->{_ole} = 0;
    return bless $self, $type;
}

sub DateReformat {
    my $d = shift;
    $d =~ s/\//-/g;
    $d =~ s/ /T/;
    $d .= 'Z';
    return $d;
}

sub findChanges {
    my $self = shift;
    my $dir = shift;
    my $d = new IO::Dir($dir);
    $self->{_ole}->GetWCInfo($dir, 1, 0);
    my $baseURI = $self->{_ole}->Url;
    # really?
    $baseURI =~ s/\/$//;
    #my $modDate = 0;
    #my $referenceDate = "1970-01-01T00:00:00Z";
    #my $referenceVersion = 0;

    while(defined(my $x = $d->read)) {
        if ($x ne "." && $x ne ".." && $x ne ".svn") {
            $self->{_ole}->GetWCInfo($dir."\\".$x, 1, 0);
            my $childURI = $self->{_ole}->Url;
            # really?
            $childURI =~ s/\/$//;
            my $relURI = $childURI;
            $relURI =~ s/^$self->{Url}\///;
            if (-f $dir."\\".$x) {
                if ($self->{_ole}->Revision == 0) {
                    #this is not under version control
                    next;
                }
                push @{$self->{RevPerFile}}, [$relURI, $self->{_ole}->Revision];
                my $tempDate = DateReformat($self->{_ole}->Date);
                if (RelevanceChecker::checkPathRelevance($relURI, $self->{filterRef})) {
                    my $statDate = (stat $dir."\\".$x)->[9];
                    if (($self->{_ole}->HasModifications == 1)) {
                        if ($statDate > $self->{NewestModification}) {
                            #print "svnwrapper: setting date from " . ($dir."\\".$x) . "\n";
                            $self->{NewestModification} = $statDate;
                        }
                    }
                    if ($tempDate gt $self->{ReferenceDate}) {
                        #print "svnwrapper: setting revision from " . ($dir."\\".$x) . ": " . $self->{_ole}->Revision . " (" . $tempDate . " > " . $self->{ReferenceDate} . ")\n";
                        $self->{ReferenceDate} = $tempDate;
                        $self->{RelevantRev} = $self->{_ole}->Revision;
                    }
                }
                $self->{RevInfo}->{$self->{_ole}->Revision} = [ $self->{_ole}->Author, $tempDate ];
            }
            if (($childURI eq "$baseURI/$x")) { # && ($self->{_ole}->HasModifications == 1)) {
                if (-d $dir."\\".$x) {
                    if ($self->{_ole}->HasModifications == 1) {
                        push @{$self->{ChangedDirs}}, $relURI;
                    }
                    $self->findChanges($dir."\\".$x);
                } else {
                    if ($self->{_ole}->HasModifications == 1) {
                        push @{$self->{ChangedFiles}}, $relURI;
                        $self->{HasModifications} = 1;
                    }
                }
            }
        }
    }
    $d->close;
}

sub getValue($$$) {
    my $text = shift;
    my $value = shift;
    my $regex = shift;
    $text =~ /$value="($regex)"/s;
    defined $1 and return $1;
    $text =~ /<$value[^>]*?>($regex)<\/$value>/s;
    return $1;
}

sub getCommits($) {
    my $entry = shift;
    my @ret = $entry =~ /(<commit.*?<\/commit>)/sg;
    return @ret;
}

sub getEntries($) {
    my $result = shift;
    my @ret = $result =~ /(<entry.*?<\/entry>)/sg;
    return @ret;
}

sub getEntryWcStatus($) {
    my $entry = shift;
    $entry =~ /(<wc-status.*?<\/wc-status>)/sg;
    return $1;
}

sub getCommitAuthor($) {
    my $commit = shift;
    return getValue($commit, "author", '\w+');
}

sub getCommitDate($) {
    my $commit = shift;
    return getValue($commit, "date", '.*?');
}

sub getCommitRevision($) {
    my $commit = shift;
    return getValue($commit, "revision", '\d+');
}

sub getEntryRevision($) {
    my $entry = shift;
    return getValue($entry, "revision", '\d+');
}

sub getEntryKind($) {
    my $entry = shift;
    return getValue($entry, "kind", '\w+');
}

sub getEntryPath($) {
    my $entry = shift;
    return getValue($entry, "path", '.*?');
}

sub getEntryUrl($) {
    my $entry = shift;
    return getValue($entry, "url", '.*?');
}

sub getWcStatusItem($) {
    my $wcs = shift;
    return getValue($wcs, "item", '\w+');
}

sub SetPath {
    my ($self, $path) = @_;
    $self->{_path} = $path if defined $path;

    # paranoia: remove leftover state!
    delete $self->{Author};
    delete $self->{Date};
    delete $self->{HasModifications};
    delete $self->{MaxRev};
    delete $self->{MinRev};
    delete $self->{Revision};
    delete $self->{Url};
    delete $self->{ModDate};
    delete $self->{NewestModification};
    delete $self->{RelevantRev};
    delete $self->{ReferenceDate};
    
    $self->{ChangedFiles} = [];
    $self->{ChangedDirs} = [];
    $self->{RevPerFile} = [];
    $self->{RevInfo} = {};

    if ($self->{_ole}) {
        $self->{_ole}->GetWCInfo($self->{_path}, 1, 0);
        $self->{Author} = $self->{_ole}->Author;
        $self->{Date} = DateReformat($self->{_ole}->Date);
        $self->{Revision} = $self->{_ole}->Revision;
        #$self->{HasModifications} = $self->{_ole}->HasModifications;
        $self->{MaxRev} = $self->{_ole}->MaxRev;
        $self->{MinRev} = $self->{_ole}->MinRev;
        $self->{Url} = $self->{_ole}->Url;
        
        #if (-f $path && $self->{HasModifications}) {
        #    $self->{ModDate} = (stat $1)[9];
        #}
        $self->findChanges($path);
        
    } else {

        open(HURZ, "svn info -R --xml \"$path\" 2>&1 |");
        my $result = do { local $/; <HURZ> };   # local slurp!
        close(HURZ);
        
        # collect all existing entry revisions
        my @entries = getEntries($result);
        # merge into hash
        my %rank = map{getEntryRevision($entries[$_]), 1} 0..$#entries;
        
        if ((scalar keys %rank) != 1) {
            # entries are not all the same -> dirty
            $self->{HasModifications} = 1;
        }
        # else {
        #    $self->{MaxRev} = (sort {$b <=> $a} keys %rank)[0];
        #}
        
        my %otherrank;
        # this is semi-magic and will fail if the first entry is not the root dir
        $self->{Url} = getEntryUrl($entries[0]);
        foreach my $e (@entries) {
            # but this does not work for subfolders of a repo
            #if (getEntryKind($e) eq "dir") {
            #    if (getEntryPath($e) eq ".") {
            #        #my @urls = grep (/SvnWrapper::url/, @{$e->{Kids}});
            #        #$self->{Url} = $urls[0]->{Kids}->[0]->{Text};
            #        $self->{Url} = getEntryUrl($e);
            #    }
            #    next;
            #}
            my @commits = getCommits($e);
            my $p = getEntryPath($e);
            $p =~ s|\\|/|g;
            my $rev = getCommitRevision($commits[0]);
            push @{$self->{RevPerFile}}, [$p, $rev];
            my $author = getCommitAuthor($commits[0]);
            my $date = getCommitDate($commits[0]);
            $self->{RevInfo}->{getCommitRevision($commits[0])} = [ $author, $date ];
            
            my $tempDate = $date;
            if (RelevanceChecker::checkPathRelevance($p, $self->{filterRef})) {
                if ($tempDate gt $self->{ReferenceDate}) {
                    #print "svnwrapper: setting revision from " . ($p) . ": " . $rev . " (" . $tempDate . " > " . $self->{ReferenceDate} . ")\n";
                    $self->{ReferenceDate} = $tempDate;
                    $self->{RelevantRev} = $rev;
                }
            }
        }
        # sort descending
        my @therevs = sort {$b <=> $a} keys %{$self->{RevInfo}};
        $self->{Revision} = $therevs[0];
        $self->{Author} = $self->{RevInfo}->{$self->{Revision}}[0];
        $self->{Date} = $self->{RevInfo}->{$self->{Revision}}[1];

        open(HURZ, "svn status --ignore-externals --xml \"$path\" 2>&1 |");
        $result = do { local $/; <HURZ> };   # local slurp!
        close(HURZ);
        
        @entries = getEntries($result);
        
        my $newest = 0;
        foreach my $hit (@entries) {
            my $wcstat = getEntryWcStatus($hit);
            if (getWcStatusItem($wcstat) eq "modified") {
                #print $1 . "\n";
                $self->{HasModifications} = 1;
                #my $p = $hit->{path};
                my $p = getEntryPath($hit);
                my $pp = Cwd::abs_path($path);
                #$pp =~ s|\\|/|g;
                #$p =~ s/^$pp\///;
                if (-f $p) {
                    $p = File::Spec->abs2rel($p, $pp);
                    $p =~ s|\\|/|g;
                    push @{$self->{ChangedFiles}}, $p;
                    if (RelevanceChecker::checkPathRelevance($p, $self->{filterRef})) {
                        my $statDate = (stat $p)->[9];
                        if ($statDate > $self->{NewestModification}) {
                            #print "svnwrapper: setting date from " . ($p) . "\n";
                            $self->{NewestModification} = $statDate;
                        }
                    }
                } else {
                    $p = File::Spec->abs2rel($p, $pp);
                    $p =~ s|\\|/|g;
                    push @{$self->{ChangedDirs}}, $p;
                }
            }
        }
        #if (-f $path) {
        #$self->{ModDate} = $newest;
        #}
    }
}

sub Author {
    my $self = shift;
    return $self->{Author};
}

sub ChangedDirs {
    my $self = shift;
    return $self->{ChangedDirs};
}

sub ChangedFiles {
    my $self = shift;
    return $self->{ChangedFiles};
}

sub Date {
    my $self = shift;
    return $self->{Date};
}

sub HasModifications {
    my $self = shift;
    return $self->{HasModifications};
}

sub MaxRev {
    my $self = shift;
    return $self->{MaxRev};
}

sub MinRev {
    my $self = shift;
    return $self->{MinRev};
}

sub NewestModification {
    my $self = shift;
    return $self->{NewestModification};
}

#sub ModDate {
#    my $self = shift;
#    return $self->{ModDate};
#}
sub RelevantRev {
    my $self = shift;
    return $self->{RelevantRev};
}

sub Revision {
    my $self = shift;
    return $self->{Revision};
}

sub RevisionInfo {
    my $self = shift;
    return $self->{RevInfo};
}

sub RevPerFile {
    my $self = shift;
    return $self->{RevPerFile};
}

sub Url {
    my $self = shift;
    return $self->{Url};
}

return 1;
