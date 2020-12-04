# This subroutine print out the RCAS expansion both as a text file and as a PDF
def xrange(a,b=None):
    if(b is None): return list(range(a))
    else:          return list(range(a,b))

def print_RCAS_wavefunction(calc,norb,neleca,nelecb,threshold,f,frozen=0,path='./molecule'):


    import numpy     as np
    from   pyscf.fci import cistring
    from   numpy     import abs, argsort, shape

    ci_vec        = calc.ci
    ci_dim1       = ci_vec.shape[1]
    na,nb         = ci_vec.shape

    c_matrix = np.zeros((na*nb,2*(norb+frozen)))
    c_labels = []

    def excitation_str(strng):
      strng_r = bin(strng)[-1:1:-1]
      liszt_r = [ (i-frozen) for (x,i) in zip(strng_r,xrange(frozen+1,frozen+len(strng_r)+1)) if(x=="1" and i>max(frozen,frozen)) ]
      liszt_r = list(range(frozen)) + [ x + frozen-1 for x in liszt_r]
      return "".join( " %3d" % j for j in liszt_r)

    fmt  = "%s           %s  # %18.14f\n"
    fmt2 = " %18.14f 0.0  # %5d %18.14f\n"

    def ampl_and_totwt(ampl):
      totwt = 0.0
      for (i,a) in enumerate(ampl):
        totwt += a*a
        yield (a, i+1, totwt)

    full_join = (lambda list_cfg_str, list_ampl: \
                   "multidet_cfg\n" \
                   + "".join(list_cfg_str))

    cfg_list   = []
    ampl_list  = []
    tot_weight = 0.0
    ampl       = 0
    ci_vec_abs = -abs(ci_vec.flatten())
    ci_order   = ci_vec_abs.argsort()
    stop_dump  = lambda ndets : tot_weight > threshold #or ndets > 20

    for ii_ab in xrange(na*nb):
      i_ab = ci_order[ii_ab]
      ia   = i_ab // ci_dim1
      ib   = i_ab - ia * ci_dim1

      ampl = ci_vec[ia,ib]
      tot_weight += ampl**2
      ampl_list.append(ampl)
      cfg_list.append(fmt % (excitation_str(cistring.addr2str(norb,neleca,ia)),excitation_str(cistring.addr2str(norb,nelecb,ib)),ampl))

      sa  = ''.join(excitation_str(cistring.addr2str(norb,neleca,ia)))
      sb  = ''.join(excitation_str(cistring.addr2str(norb,nelecb,ib)))
      cab = ampl
      for x in sa.split():
          c_matrix[ii_ab,int(x)]=1.0
      for x in sb.split():
          c_matrix[ii_ab,int(x)+norb+frozen]=-1.0 
      c_labels.append(str(round(ampl,3)))
      if stop_dump(ii_ab+1): break
      if len(ampl_list)>20: break

    c_matrix = c_matrix[:ii_ab,:]
    c_labels = c_labels[:ii_ab]

    import matplotlib
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    extent = (0,c_matrix.shape[1],c_matrix.shape[0],0)
    im     = ax.imshow(c_matrix,vmin=-1,vmax=1,cmap='coolwarm')
    aolab  = c_labels
    molab  = list(range(norb+frozen))+list(range(norb+frozen))
    ax.set_xticks(np.arange(len(molab)))
    ax.set_yticks(np.arange(len(aolab)))
    ax.set_xticklabels(molab)
    ax.set_yticklabels(aolab)
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
    xposition = [0.5+x for x in range(len(molab))]
    for xc in xposition:
        ax.axvline(x=xc,color='k',linestyle='-')
        yposition = [0.5+x for x in range(len(aolab))]
    for yc in yposition:
        ax.axhline(y=yc,color='k',linestyle='-')
    ax.set_title("CI wavefunction")
    ax.set_xlabel("orbitals")
    ax.set_ylabel("configurations")
    plt.savefig(path+'_FCI.pdf', bbox_inches='tight')
    

    f.write("\n")
    f.write(full_join(cfg_list, ampl_list))
    f.close()

