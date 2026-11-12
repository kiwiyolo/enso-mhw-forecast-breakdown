# Academic Figure Skill Asset Confirmation
# S1-S17: existing source-verified SI renderer -> param inherit; geometry only.
# No new data transformations, normalization, smoothing or axis-range changes.
# Typography, palette and export baselines are injected by base_si.py.
"""Physical-size layout contracts: dense primary panels, compact supporting panels."""
HEIGHTS={1:136,2:92,3:105,4:92,5:141,6:114,7:141,8:100,9:126,
         10:122,11:124,12:108,13:136,14:127,15:112,16:136,17:132}

def apply_layout(fig,n):
    fig.set_size_inches(180/25.4,HEIGHTS[n]/25.4,forward=True)
    a=fig.axes
    def place(i,rect):a[i].set_position(rect)
    if n in (2,4):
        place(0,[.095,.19,.375,.70]);place(1,[.59,.19,.39,.70])
    elif n==3:
        place(0,[.09,.17,.23,.72]);place(1,[.42,.17,.23,.72]);place(2,[.79,.17,.19,.72])
    elif n==5:
        for row,y in enumerate([.73,.445,.16]):
            place(2*row,[.115,y,.37,.19]);place(2*row+1,[.625,y,.285,.19])
        place(6,[.955,.16,.012,.76])
    elif n==6:
        for row,y in enumerate([.70,.435,.17]):
            for col in range(3):place(3*row+col,[.045+col*.315,y,.295,.22])
        place(9,[.075,.09,.55,.016]);place(10,[.71,.09,.25,.016])
    elif n==7:
        for i,(x,y) in enumerate([(.07,.56),(.49,.56),(.07,.12),(.49,.12)]):place(i,[x,y,.37,.34])
        place(4,[.90,.56,.018,.34]);place(5,[.90,.12,.018,.34])
    elif n==8:
        place(0,[.25,.20,.32,.68]);place(1,[.66,.20,.32,.68])
    elif n==9:
        for i,(x,y) in enumerate([(.115,.59),(.63,.59),(.115,.16),(.63,.16)]):place(i,[x,y,.35,.30])
    elif n==10:
        place(0,[.105,.60,.875,.31]);place(1,[.105,.14,.37,.29]);place(2,[.61,.14,.37,.29])
    elif n==11:
        for i,(x,y) in enumerate([(.10,.60),(.59,.60),(.10,.21),(.59,.21)]):place(i,[x,y,.39,.28])
    elif n==12:
        place(0,[.08,.23,.51,.66]);place(1,[.69,.23,.29,.66])
    elif n==13:
        # Two-row summary above; five-row removal test gets the larger main panel.
        place(0,[.29,.67,.56,.23]);place(1,[.29,.12,.56,.41])
        place(2,[.90,.67,.015,.23]);place(3,[.90,.12,.015,.41])
    elif n==14:
        place(0,[.11,.56,.38,.33]);place(1,[.64,.56,.25,.33])
        place(2,[.29,.15,.60,.24]);place(3,[.93,.56,.015,.33])
    elif n==15:
        place(0,[.115,.24,.32,.64]);place(1,[.69,.24,.29,.64])
    elif n==16:
        for i,(x,y,w) in enumerate([(.09,.59,.47),(.68,.59,.30),(.09,.15,.47),(.68,.15,.30)]):place(i,[x,y,w,.30])
    elif n==17:
        for i,(x,y,w) in enumerate([(.09,.54,.47),(.68,.54,.30),(.09,.13,.47),(.68,.13,.30)]):place(i,[x,y,w,.29])
    for ax in a:
        ax.xaxis.labelpad=3;ax.yaxis.labelpad=3
        ax.tick_params(axis='both',which='major',pad=2.5)
        for t in ax.texts:
            if len(t.get_text())==1 and t.get_text().islower():
                t.set_position((-.075,1.035));t.set_va('bottom');t.set_fontsize(10)
