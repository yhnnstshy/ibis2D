#! /usr/bin/env python
"""Tables with keys and columns"""
"""Keys are tuples, even if PK is a single key"""

# from x import y
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(module)s.%(funcName)s: %(message)s")
logger = logging.getLogger('tablelogger')
logger.setLevel(logging.INFO)

__author__ = "Joel Bader"
__copyright__ = "Copyright 2009, Bader Lab"
__credits__ = ["Joel Bader"]
__license__ = "BSD"
__version__ = "0.2"
__maintainer__ = "Joel Bader"
__email__ = "joel.bader@jhu.edu"
__status__ = "Development"

TABLESEP = "\t"

def fill(tbl, default=""):
    colnames = getCols(tbl)
    cnt = 0
    for key in tbl:
        for c in colnames:
            if c not in tbl[key]:
                tbl[key][c] = default
                cnt += 1
    logger.info("%d missing values with %s" %
                (cnt, str(default)))
    return(cnt)

def getCols(tbl):
    seencols = dict()
    for key in tbl.keys():
        for c in tbl[key].keys():
            if c not in seencols:
                seencols[c] = 0
            seencols[c] += 1
    colnames = sorted( seencols.keys() )
    for c in colnames:
        logger.debug("%s %d" % (c, seencols[c] ))
    return(colnames)

def write(tbl, filename, key_names, col_names=None, default=None, sep=TABLESEP, digits=5):
    """
    keynames can be a string, tuple, or list
    if a tuple or a list, the length of keynames provides the length of the primary key
    """
    keyname_list = None
    key_type = None
    if type(key_names) == type(""):
        keyname_list = [ key_names ]
        key_type = 'single'
    elif type(key_names) == type([ ]):
        keyname_list = key_names
        key_type = 'list'
    elif type(key_names) == type( () ):
        keyname_list = list(key_names)
        key_type = 'tuple'
    else:
        print "bad key_names: " + str(key_names)
    assert keyname_list is not None, "error, bad key_list"
    nkey = len(keyname_list)    
    
    if col_names is None:
        col_names = getCols(tbl)
        
    float_format = '%.' + str(digits) + 'g'
        
    fp = open(filename, "w")
    header = keyname_list + col_names
    fp.write( sep.join(header) + "\n")
    
    for key in sorted(tbl.keys()):
        key_list = None
        if (key_type == 'single'):
            key_list = [ key ]
        else:
            key_list = list(key)
        assert(len(key_list) == nkey), "bad key: " + str(key)
        val_list = [ ]
        for c in col_names:
            assert c in tbl[key], 'bad key/col: ' + key + ' ' + c
            val = tbl[key][c]
            val_str = None
            if type(val) == type(''):
                val_str = val
            elif type(val) == type(1):
                val_str = str(val)
            elif type(val) == type(1.0):
                val_str = float_format % val
            else:
                val_str = str(val)
            val_list.append(val_str)
        row_list = key_list + val_list
        fp.write(sep.join(row_list) + "\n")
    fp.close()
    pkstr = " ".join(keyname_list)
    logger.info("%d rows to %s PK %s" % (len(tbl), filename, pkstr))

NULL_CHAR = '\x00'
# printable ascii codes are 32 ... 127 inclusive
PRINTABLE_RANGE = range(32,128)
def printable(my_str):
    ret = ''.join( [c for c in my_str if ord(c) in PRINTABLE_RANGE]  )
    return ret

def stripquotes(my_str):
    ret = ''.join([ c for c in my_str if c != '"'])
    return ret

def get_first_tok(my_str):
    my_toks = my_str.strip().split()
    first_tok = my_toks[0]
    return(first_tok)

def read(filename, nkey = 1, sep=TABLESEP, printable_only=True, short_colnames=False, strip_quotes=True):
    assert(nkey > 0)
    fp = open(filename, "r")
    ret = dict()
    nheader = None
    keynames = None
    colnames = [ ]
    for line in fp:
        toks = line.strip().split(sep)
        if (len(toks) <= 1):
            print 'toks: ', str(toks)
        if printable_only:
            toks = [ printable(x) for x in toks]
        if strip_quotes:
            toks = [ stripquotes(x) for x in toks]
        ntok = len(toks)
        if (not nheader):
            nheader = ntok
            assert(nheader >= nkey)
            keynames = toks[0] if nkey == 1 else tuple(toks[:nkey])
            colnames = toks[:]
            if (short_colnames):
                new_colnames = [ get_first_tok(x) for x in colnames ]
                colnames = new_colnames
        else:
            if (ntok != nheader):
                logger.warn('skipping line ntok %d != nheader %d line: %s', ntok, nheader, line)
                continue
            for i in range(ntok, nheader):
                toks = toks + ['']
            assert(len(toks) == nheader)
            key = toks[0] if nkey == 1 else tuple(toks[:nkey])
            assert key not in ret, "repeated key: " + str(key)
            ret[key] = dict()
            for i in range(nkey, nheader):
                ret[key][colnames[i]] = toks[i]
    fp.close()
    pkstr = keynames if type(keynames) == type("") else " ".join(keynames)
    logger.info("%d rows from %s PK %s" % ( len(ret) , filename, pkstr ))
    return(ret, keynames, colnames[nkey:])

def addCols(table1, table2, cols, default="NA"):
    cnt = 0
    for key in table1:
        for c in cols:
            if c not in table1[key]:
                table1[key][c] = default
    for key in table2:
        if key in table1:
            for c in cols:
                if (c in table2[key]):
                    table1[key][c] = table2[key][c]
                    cnt = cnt + 1
    logger.info("%d values added for cols %s" % ( cnt, " ".join(cols)))
    
def addDict(tbl, new_dict, col_name):
    for key in tbl:
        assert(key in new_dict), 'missing key: ' + key
        assert(col_name not in tbl[key]), 'repeated column name: ' + col_name
        tbl[key][col_name] = new_dict[key]
    return(tbl)

def newKey(tbl, cols):
    newtbl = dict()
    for key in tbl:
        newkey = [ ]
        for c in cols:
            newkey.append( tbl[key][c] )
        newkey = tuple(newkey)
        newtbl[newkey] = dict()
        for c in tbl[key]:
            if c not in cols:
                newtbl[newkey][c] = tbl[key][c]
    return(newtbl)
    
def getCrossTabs(tbl, cols, output = False):
    cnt = dict()
    for key in tbl:
        rec = [ ]
        for c in cols:
            v = tbl[key][c]
            rec.append(v)
        index = tuple(rec)
        cnt[index] = cnt.get(index, 0) + 1
    if (output):
        print '\n*** crosstabs ***\n' + '\t'.join(cols) + '\tcnt'
        for i in sorted(cnt.keys()):
            print '\t'.join(list(i)) + '\t' + str(cnt[i])
        print '***\n'
    return(cnt)

def tableTests():
    pass

if __name__ == '__main__':    #code to execute if called from command-line
    tableTests()
