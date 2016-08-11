import gzip
import hashlib
import os
import sys
import itertools
import functools

hasher = hashlib.sha256

hh = hasher()
hh.update(b'')
ZERO_HASH = hh.hexdigest()

sortedkeys = functools.partial(sorted, key=lambda a: a[0])

debug_print = print
debug_print = lambda *_, **__: None
print_warning = functools.partial(print, 'WARNING:', file=sys.stderr)

def escape_string(s):
    import binascii
    return binascii.b2a_qp(s.encode('utf-8'), True, False, False).decode('ascii')

def escape_string_or_none(s):
    return escape_string(s or '')

def unescape_string(s):
    import binascii
    return binascii.a2b_qp(s.encode('ascii'), False).decode('utf-8')

def unescape_string_or_none(s):
    return unescape_string(s) or None

def path_split_all(path):
    parts = []
    while True:
        path, f = os.path.split(path)
        if f != '':
            parts.append(f)
        else:
            if path != '':
                parts.append(path)
            break
    parts.reverse()
    return parts

def ensure_directories(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

class TreeEntry:
    def __init__(self, mode, fn, blob):
        self.mode = mode
        self.fn = fn
        self.blob = blob
        self.path = None
    @classmethod
    def fromString(cls, line):
        l = line.rstrip('\n')
        mode, rfn, blob = l.split('\t')
        fn = unescape_string(rfn)
        return cls(mode, fn, blob)
    def __str__(self):
        return '\t'.join((self.mode, escape_string(self.fn), self.blob))
    def __eq__(s, o):
        return isinstance(o, TreeEntry) and s.mode == o.mode and s.fn == o.fn and s.blob == o.blob
    def __hash__(self):
        return hash((self.mode, self.fn, self.blob))
    @classmethod
    def fromDirEntry(cls, de):
        if de.is_dir():
            return cls('dir', de.name, None)
        return cls("{0:o}".format(de.stat().st_mode), de.name, hash_file(de.path))
    @classmethod
    def fromFileName(cls, fn):
        if os.path.isdir(fn):
            return cls('dir', os.path.basename(fn), None)
        return cls("{0:o}".format(os.stat(fn).st_mode), os.path.basename(fn), hash_file(fn))
    def prepend(self, path):
        # creates an attribute called 'path'
        import copy
        debug_print('prepending', path, 'to', self.fn)
        n = copy.copy(self)
        n.path = os.path.join(path, n.fn)
        return n
    def write(self, path, fpath):
        debug_print('writing', self.fn)
        assert self.path is not None
        store_file(self.path, self.blob, fpath)
    def download(self, path, dn):
        import shutil
        fn = lookup_path(self.blob, dn)
        debug_print("downloading", fn, "to", path)
        ensure_directories(path)
        with gzip.open(fn, 'rb') as rf:
            with open(path, 'wb') as wf:
                shutil.copyfileobj(rf, wf)

class VirtualDir:
    def __init__(self, h, d, name='.', parent=None):
        # h and d only used in lookup
        self.h = h
        self.d = d
        self.parent = parent
        self.name = name
        self.items = dict()
        self.cached = (h == None)
        self.updated = set()
        self.found = '' # directory
        self.realpath = ''
        self.toprepend = None
    def __getitem__(self, key):
        self.cache()
        if key == '.' or key == '':
            return self
        if key == '..':
            return self.parent
        return self.items[key]
    def __setitem__(self, key, value):
        self.cache()
        if key == '.' or key == '':
            self.parent[self.name] = value
            return
        if key == '..':
            self.parent['.'] = value
            return
        self.updated.add(key)
        self.items[key] = value
        if self.parent:
            self.parent[self.name] = self
    def __delitem__(self, key):
        self.cache()
        if key == '.' or key == '':
            del self.parent[self.name]
            return
        if key == '..':
            del self.parent['.']
            return
        self.updated.add(key)
        del self.items[key]
        if self.parent:
            if len(self.items) == 0:
                del self.parent[self.name]
            else:
                self.parent[self.name] = self
    def __contains__(self, key):
        self.cache()
        return key in self.items
    def __iter__(self):
        self.cache()
        return iter(self.items)
    def cache(self):
        if self.cached:
            return
        debug_print('uncaching', self.name)
        self.found, tt = lookup_tree(self.h, self.d)
        ftree, dirs = split_tree(tt)
        self.items.update(tree_to_dict(ftree))
        self.items.update((e.fn, VirtualDir(e.blob, self.d, e.fn, self)) for e in dirs)
        self.cached = True
        self.doprepend()
    def doprepend(self):
        if self.cached:
            path = self.toprepend
            if not path:
                return
            debug_print('prepending', path, 'to', self.name)
            np = os.path.join(path, self.name)
            if self.realpath:
                assert np == self.realpath
            self.realpath = np
            for k, v in self.items.items():
                self.items[k] = v.prepend(np)
            self.toprepend = None
    def prepend(self, path):
        import copy
        vd = copy.copy(self)
        vd.toprepend = path
        vd.doprepend()
        return vd
    def write(self, path, fpath):
        # returns hash
        if not self.updated:
            return self.h
        # write children first to update their hash
        for k, c in self.items.items():
            if k in self.updated:
                c.write(path, fpath)
        contents = '\n'.join(itertools.chain((str(v) for _, v in sortedkeys(self.items.items())), ('',)))
        fn = hash_str(contents)
        assert not_collision_str(os.path.join(path, fn), contents)
        debug_print('actually writing', self.name)
        fpname = os.path.join(path, fn)
        ensure_directories(fpname)
        with open(fpname, 'w') as f:
            f.write(contents)
        self.h = fn
        return fn
    def path(self, path):
        pc = path_split_all(path)
        cur = self
        for comp in pc:
            cur = cur[comp]
        return cur
    def create_path(self, path):
        # note: make sure to add something immediately
        # after calling this.
        pc = path_split_all(path)
        cur = self
        for comp in pc:
            if comp not in cur:
                cur[comp] = VirtualDir(None, self.d, comp, cur)
            cur = cur[comp]
        return cur
    def __str__(self):
        return 'dir\t%s\t%s' % (self.name, self.h)
    @classmethod
    def makeEntry(cls, fn, name='.', parent=None):
        # makes either a VirtualDir or TreeEntry
        if not valid_file(fn):
            return None
        te = TreeEntry.fromFileName(fn)
        if te.mode == 'dir':
            vd = cls(None, None, name, parent)
            vd.items.update((de.name, cls.makeEntry(de.path, de.name, vd)) for de in scandir_exclude(fn))
            vd.updated.update(vd.items.keys())
            vd.path = fn
            return vd
        else:
            return te.prepend(os.path.dirname(fn))
    @property
    def blob(self):
        return self.h
    def download(self, path, dn):
        for i in self:
            self[i].download(os.path.join(path, i), dn)

def not_collision_str(fn, contents):
    try:
        with open(fn, 'r') as f:
            return f.read() == contents
    except FileNotFoundError:
        return True
    return False

def not_collision_gzip(gfn1, fn2):
    try:
        with gzip.open(gfn1, 'rb') as f:
            with open(fn2, 'rb') as f2:
                return f.read() == f2.read()
    except FileNotFoundError:
        return True
    return False

def not_collision(fn1, fn2):
    import filecmp
    filecmp.clear_cache()
    return filecmp.cmp(fn1, fn2, False)

def read_config(fn):
    with open(fn) as f:
        ret = dict()
        for line_ in f:
            line = line_.strip()
            if line.startswith('#') or line == '':
                continue
            try:
                k, v = map(str.strip, line.split(':'))
                if k.endswith('@'):
                    ret[k] = list(map(unescape_string_or_none, v.split('\t')))
                else:
                    ret[k] = unescape_string_or_none(v)
            except Exception as e:
                continue
        return ret

def do_write_config(f, c):
    for k, v in sortedkeys(c.items()):
        if k.endswith('@'):
            f.write('%s:\t%s\n' % (k, '\t'.join(map(escape_string_or_none, v))))
        else:
            f.write('%s:\t%s\n' % (k, escape_string_or_none(v)))

def write_config(fn, c):
    ensure_directories(fn)
    with open(fn, 'w') as f:
        do_write_config(f, c)

def write_string(fn, s):
    ensure_directories(fn)
    with open(fn, 'w') as f:
        f.write(s)

def lookup_path(h, d):
    for dd in d:
        try:
            jp = os.path.join(dd, h)
            if os.path.exists(jp):
                return jp
        except FileNotFoundError:
            pass
    raise FileNotFoundError(h)

def lookup_tree(h, d=('.vcs/tree',)):
    for dd in d:
        try:
            with open(os.path.join(dd, h)) as f:
                return dd, list(map(TreeEntry.fromString, f))
        except FileNotFoundError:
            pass
    raise FileNotFoundError(h)

def get_tree(h, d=('.vcs/tree',)):
    return lookup_tree(h, d)[1]

def split_tree(tree):
    dirs = set(filter(lambda e: e.mode == 'dir', tree))
    return (set(tree) - dirs, dirs)

def hash_file(fn, bs=65536):
    with open(fn, 'rb', buffering=0) as f:
        hh = hasher()
        while True:
            buf = f.read(bs)
            if len(buf) == 0:
                break
            hh.update(buf)
        return hh.hexdigest()

def store_file(fn, h=None, path='.vcs/blob'):
    newfn = h or hash_file(fn)
    newpath = os.path.join(path, newfn)
    assert not_collision_gzip(newpath, fn)
    try:
        ensure_directories(newpath)
        with open(newpath, 'xb') as rf:
            with gzip.GzipFile(fileobj=rf, mode='xb', filename='', mtime=0) as f:
                with open(fn, 'rb') as f2:
                    f.write(f2.read())
    except FileExistsError:
        pass

def hash_str(s):
    hh = hasher()
    hh.update(s.encode('utf8'))
    return hh.hexdigest()

def valid_name(name):
    return name != '.vcs' and name != '__pycache__' and not name.endswith('.swp')

def valid_file(fn):
    name = os.path.basename(fn)
    return valid_name(name) and not os.path.islink(fn) and os.path.exists(fn)

def scandir_exclude(path='.'):
    return filter(lambda t: valid_name(t.name) and not t.is_symlink(), os.scandir(path))

def get_fs_tree(path='.'):
    ls = list(scandir_exclude(path))
    return list(map(TreeEntry.fromDirEntry, ls))

def tree_to_dict(ftree):
    return {x.fn: x for x in ftree}

def get_ftree_changes(fstree, ftree):
    # returns (added, removed, modified)
    # returns fstree's data on conflict
    ls = tree_to_dict(fstree)
    tl = tree_to_dict(ftree)
    lsf = set(ls.keys())
    tlf = set(tl.keys())
    added = list(map(ls.__getitem__, lsf - tlf))
    removed = list(map(tl.__getitem__, tlf - lsf))
    same = lsf & tlf
    changed = set(ls[x] for x in same if ls[x] != tl[x])
    return (added, removed, changed)

def get_ftree_dir_changes(fstree, fsdirs, ftree, dirs):
    # returns ((dirsadd, dirsrem, dismod), added, removed, modified)
    # used to return dirssame
    # returns fstree's data on conflict
    return (get_ftree_changes(fsdirs, dirs),
            get_ftree_changes(fstree, ftree))
    return tuple(ret)

def get_tree_changes(tree1, tree2):
    # returns ((dirsadd, dirsrem, dirsmod), added, removed, modified)
    # prefers tree2
    a1, b1 = split_tree(tree1)
    a2, b2 = split_tree(tree2)
    return get_ftree_dir_changes(a2, b2, a1, b1)

def get_dir_changes(tree, path='.'):
    return get_tree_changes(tree, get_fs_tree(path))

def rec_ls_fs(path='.'):
    t, d = split_tree(get_fs_tree(path))
    for f in t:
        yield f.prepend(path)
    for dd in d:
        for f in rec_ls_fs(os.path.join(path, dd.fn)):
            yield f

def rec_ls_tr(h, fd=('.vcs/tree',), path='.'):
    debug_print('lsing with fd', fd)
    t, d = split_tree(get_tree(h, fd))
    for f in t:
        yield f.prepend(path)
    for dd in d:
        for f in rec_ls_tr(dd.blob, fd, os.path.join(path, dd.fn)):
            yield f

def rec_ls(h, fd=('.vcs/tree',), path='.'):
    if h == None:
        for f in rec_ls_fs(path):
            yield f
    else:
        for f in rec_ls_tr(h, fd, path):
            yield f

def tree_lookup(tree, fn):
    for i in tree:
        if i.fn == fn:
            return i
    raise KeyError(fn)

def tree_fs_lookup(tree, fn, dn):
    # returns a tree
    if tree == None:
        return None
    return get_tree(tree_lookup(tree, fn).blob, dn)

def get_changes(src, sdn, dst, ddn, path='.'):
    # returns (added, removed, modified)
    # use None for fs
    srct = src
    if srct is None:
        srct = get_fs_tree(path)
    dstt = dst
    if dstt is None:
        dstt = get_fs_tree(path)
    ((da, dr, dm), (a, r, m)) = get_tree_changes(srct, dstt)
    pp = functools.partial(TreeEntry.prepend, path=path)
    add = list(map(pp, a))
    rem = list(map(pp, r))
    mod = list(map(pp, m))
    for d in da:
        debug_print('get_changes of', dst, ddn)
        add.extend(rec_ls(d.blob, ddn, os.path.join(path, d.fn)))
    for d in dr:
        rem.extend(rec_ls(d.blob, sdn, os.path.join(path, d.fn)))
    for d in dm:
        st = tree_fs_lookup(src, d.fn, sdn)
        dt = tree_fs_lookup(dst, d.fn, ddn)
        aa, rr, mm = get_changes(st, sdn, dt, ddn, os.path.join(path, d.fn))
        add.extend(aa)
        rem.extend(rr)
        mod.extend(mm)
    return add, rem, mod

def get_head():
    # ['branch' or 'commit', 'reference']
    try:
        return read_config(vcs('HEAD'))['ref@']
    except FileNotFoundError:
        return ['commit', None]

def set_head(ct, c):
    # ct is type
    # c is reference
    ensure_directories(vcs('HEAD'))
    write_config(vcs('HEAD'), {'ref@': [ct, c]})

def get_branch(b):
    return read_config(os.path.join(vcs('branch'), b))['commit']

def set_branch(b, commit):
    fn = os.path.join(vcs('branch'), b)
    if not os.path.exists(fn):
        raise FileNotFoundError(fn)
    write_config(fn, {'commit': commit})

def new_branch(b, commit=None):
    fn = os.path.join(vcs('branch'), b)
    if os.path.exists(fn):
        raise FileExistsError(fn)
    if commit == None:
        commit = get_head_commit(warn=False)
    write_config(fn, {'commit': commit})

def get_head_commit(warn=True):
    h = get_head()
    if h[0] == 'commit':
        if warn:
            print_warning('you are on detached HEAD')
        return h[1] if len(h) > 1 else None
    else:
        return get_branch(h[1])

def get_current_branch():
    h = get_head()
    if h[0] == 'commit':
        return None
    else:
        return h[1]

def get_staging_head():
    h = get_commit(get_head_commit()) or {'tree': None}
    return {'tree': h['tree']}

def get_staging_commit():
    try:
        return read_config(vcs('staging/commit'))
    except FileNotFoundError:
        return None

def get_staging_tree():
    try:
        return get_tree(get_staging_commit()['tree'], vcs('staging/tree', 'tree'))
    except TypeError as e:
        return []

def write_staging_commit(conf):
    write_config(vcs('staging/commit'), conf)

def summarize_changes(arm, pref=''):
    import operator
    ret = []
    a, r, m = arm
    kf = operator.attrgetter('path')
    for f in sorted(a, key=kf):
        ret.append('%sadded:    %s %s' % (pref, f.mode, f.path))
    for f in sorted(r, key=kf):
        ret.append('%sremoved:  %s %s' % (pref, f.mode, f.path))
    for f in sorted(m, key=kf):
        ret.append('%smodified: %s %s' % (pref, f.mode, f.path))
    return ret

def print_changes(arm, pref=''):
    changes = summarize_changes(arm, pref)
    if not changes:
        changes = [pref + 'no changes']
    for l in changes:
        print(l)

def status():
    stage_tree = None
    found = True
    try:
        stage_tree = get_tree(get_staging_commit()['tree'], vcs('staging/tree', 'tree'))
    except TypeError as e:
        found = False
    except Exception as e:
        import traceback
        traceback.print_exc()
    print('changes not staged:')
    if found:
        print_changes(get_changes(stage_tree, vcs('staging/tree', 'tree'), None, (), VCS_ROOT), '  ')
    else:
        print_changes(get_changes(get_commit_tree(get_head_commit()), (vcs('tree'),), None, (), VCS_ROOT), '  ')
    print()
    if found:
        # if nothing staged, nothing staged
        print('changes staged:')
        print_changes(get_changes(get_commit_tree(get_head_commit()), (vcs('tree'),), stage_tree, vcs('staging/tree', 'tree'), VCS_ROOT), '  ')
    else:
        print('nothing staged')

def stage(fn):
    # this function can either:
    # add a file/folder
    # remove a file/folder
    # modify a file/folder
    # it's pretty useful
    # get path relative to VCS_ROOT
    # returns 'added', 'removed', 'modified'
    rpath = os.path.relpath(fn, VCS_ROOT)
    rdir, rf = os.path.split(rpath)
    e = VirtualDir.makeEntry(fn, os.path.basename(fn))
    sc = get_staging_commit() or get_staging_head()
    root = VirtualDir(sc['tree'], vcs('staging/tree', 'tree')).prepend(VCS_ROOT)
    if e == None:
        if not valid_name(os.path.basename(fn)):
            raise Exception('ignored ' + fn)
        else:
            # try to remove
            del root.path(rdir)[rf]
            sc['tree'] = root.write(vcs('staging/tree'), vcs('staging/blob'))
            write_staging_commit(sc)
            return 'removed'
    else:
        df = root.create_path(rdir)
        op = 'added'
        if rf in df:
            op = 'modified'
        df[rf] = e
        sc['tree'] = root.write(vcs('staging/tree'), vcs('staging/blob'))
        write_staging_commit(sc)
        return op

def download(ch, path, dn, db):
    # ch is a commit-like object
    # works for staging commit (get_staging_commit())
    # as well as a commit
    # should be a dict
    # path is relative to current directory
    # db is path to blobs
    rpath = os.path.relpath(path, VCS_ROOT)
    root = VirtualDir(ch['tree'], dn)
    root.path(rpath).download(path, db)

def remove_except(files, path):
    # should be a set for performance
    removed = []
    for f in os.listdir(path):
        if f not in files:
            removed.append(f)
            os.remove(os.path.join(path, f))
    return removed

def cleanup():
    # cleans up the staging directory by removing unreachable files
    # returns (removed files, removed dirs)
    cur = [VirtualDir(get_staging_commit()['tree'], vcs('staging/tree', 'tree'))]
    stdir = vcs('staging/tree')
    seen = set()
    seendir = set()
    cur[0].cache()
    seendir.add(cur[0].h)
    while cur:
        new = []
        for i in cur:
            for k, v in i.items.items():
                if isinstance(v, VirtualDir):
                    if v.h in seendir:
                        continue
                    v.cache()
                    if v.found != stdir:
                        # don't process things already committed
                        continue
                    seendir.add(v.h)
                    new.append(v)
                else:
                    seen.add(v.blob)
        cur.clear()
        cur = new
    return remove_except(seen, vcs('staging/blob')), \
           remove_except(seendir, vcs('staging/tree'))

def get_date():
    # WARNING: MAY HAVE Y2038 BUG
    import time
    return int(time.time())

def get_config():
    if get_config._config is not None:
        return get_config._config
    get_config._config = dict()
    home = os.path.expanduser("~")
    searchpath = [vcs('config'), os.path.join(home, '.vcsconfig'), '/etc/.vcsconfig']
    searchpath.reverse()
    for path in searchpath:
        try:
            get_config._config.update(read_config(path))
        except FileNotFoundError:
            continue
    return get_config()
get_config._config = None

def commit(message, author=None, email=None, date=None):
    import io
    sc = get_staging_commit()
    sc['message'] = message
    sc['author'] = author or get_config()['author']
    sc['email'] = email or get_config()['email']
    sc['date'] = str(date if date or (isinstance(date, int)) else get_date())
    sc['branch'] = get_current_branch()
    sc['parents@'] = (get_head_commit(),)
    conts = ''
    with io.StringIO() as sio:
        do_write_config(sio, sc)
        conts = sio.getvalue()
    h = hash_str(conts)
    fn = os.path.join(vcs('commit'), h)
    ensure_directories(fn)
    write_string(fn, conts)
    update_head(h)
    # move trees and blobs
    cleanup()
    stg = vcs('staging')
    for p in ('tree', 'blob'):
        srcd = os.path.join(stg, p)
        dstd = vcs(p)
        for f in os.listdir(srcd):
            sf = os.path.join(srcd, f)
            df = os.path.join(dstd, f)
            if os.path.exists(df):
                assert not_collision(sf, df)
            else:
                import shutil
                shutil.move(sf, df)
    # delete staging commit
    os.remove(vcs('staging/commit'))
    return h
def get_empty_commit():
    ########################################################
    #    WHEN ADDING NEW FIELDS TO COMMITS MAKE SURE TO    #
    #                 CHANGE THIS FUNCTION                 #
    ########################################################
    return {'tree': None}


def get_commit(commit):
    if commit == None:
        return None
    fn = os.path.join(vcs('commit'), commit)
    return read_config(fn)

def get_commit_tree(commit):
    if commit == None:
        return []
    c = get_commit(commit)
    return get_tree(c['tree'], (vcs('tree'),))

def update_head(commit):
    # TODO: add_to_reflog(get_head())
    hc, h = get_head()
    if hc == 'commit':
        set_head('commit', commit)
    else:
        set_branch(h, commit)

def do_checkout(commit):
    # does not update HEAD
    assert not get_staging_commit() or get_staging_commit()['tree'] == get_staging_head()['tree']
    src = get_staging_head()
    dst = get_commit(commit)
    dn = (vcs('tree'),)
    a,r,m = get_changes(src, dn, dst, dn, VCS_ROOT)
    tocopy = itertools.chain(a, m)
    for f in tocopy:
        f.download(f.path, dn)
    for f in r:
        os.remove(f.path)

def checkout_branch(branch):
    do_checkout(get_branch(branch))
    set_head('branch', branch)

def checkout_commit(commit):
    do_checkout(commit)
    set_head('commit', commit)

def is_a_real_directory(x):
    return not os.path.islink(x) and os.path.isdir(x)

def find_root():
    p = '.'
    while not is_a_real_directory(os.path.join(p, '.vcs')):
        po = os.path.abspath(p)
        p = os.path.join(p, '..')
        pn = os.path.abspath(p)
        if po == pn:
            raise Exception('not a vcs directory')
    return p

VCS_ROOT = find_root()
def vcs(path, *args):
    if isinstance(path, str):
        if len(args) == 0:
            return os.path.join(VCS_ROOT, '.vcs', path)
        else:
            import itertools
            return tuple(map(vcs, itertools.chain((path,), args)))
    else:
        return tuple(map(vcs, path))
