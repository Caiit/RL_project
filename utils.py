def boolean_string(s):
    '''Parse a string as boolean value '''
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'