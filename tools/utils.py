def sanity_check(check_debug = False, debug = None, 
                 check_eval = False, eval = None, 
                 check_horovod = False, use_horovod = None, rank = None, 
                 check_load_checkpoinit = False, load_checkpoint = None):
    '''
    debug: if debug, deny
    eval: if eval, deny
    horovod: if the rank 0, admit
    load_checkpoint: if load_checkpoint, admit
    '''

    if check_debug:
        if debug:
            return False

    if check_eval:
        if eval:
            return False
        
    if check_horovod:
        if not ((use_horovod and rank == 0) or not use_horovod):
            return False
        
    if check_load_checkpoinit:
        if not load_checkpoint:
            return False
        
    return True
