#%% main
"""
-*- HERE -*-
"""


class binomialTrees:
    def __init__(self, params = None, **kwargs):
        import numpy as np
        import pathlib
        import os

        ################## unpacking of params dict and/or keyword arguments ######################
        copy_params = params
        paramsUnion = params
        kwunion = kwargs
        if isinstance(params, dict):
            copy_params = params.copy()
            kwunion = {**params, **kwargs}
            paramsUnion = {**kwargs, **params}
        if isinstance(params, type(None)) and len(kwargs) == 0 or \
                not isinstance(params, (type(None), dict)) and len(kwargs) == 0:
            self.help(['params', 'paramsexamples'])
            raise ValueError('See specifications above')


        ########################## directory, filename specification ##############################
        direc = kwunion.get('direc', None)  # if None -> current working directory
        folname = kwunion.get('folname', None)  # 'OutputFolder' if direc is None
        fname = kwunion.get('fname', 'binotree')

        self.copy_params = copy_params
        self.paramsUnion = paramsUnion
        self.kwunion = kwunion

        def findmakedirec(direc = None, folname = None, fname = 'binotree', filetype = '.xlsx'):
            foldir = str()
            dirfile = str()
            if direc is None:
                curwd = os.path.abspath(os.getcwd())
                if folname is None:
                    foldir = os.path.join(curwd, 'OutputFolder')
                else:
                    foldir = os.path.join(curwd, str(folname))

                pdirec = pathlib.Path(foldir).resolve()

                if pdirec.is_file():
                    pdirec = pdirec.parent

                if pdirec.suffix !=  '':
                    pdirec = pdirec.parent

                foldir = str(pdirec)

                dirfile = os.path.join(foldir, str(fname) + filetype)

            elif direc is not None:
                if folname is not None:
                    pdirec = pathlib.Path(os.path.join(str(direc), str(folname))).resolve()
                else:
                    pdirec = pathlib.Path(str(direc)).resolve()

                if pdirec.is_file():
                    pdirec = pdirec.parent

                if pdirec.suffix !=  '':
                    pdirec = pdirec.parent

                foldir = str(pdirec)
                dirfile = os.path.join(foldir, str(fname) + filetype)


            filename = fname + filetype

            return foldir, dirfile, filename

        foldir, dirfile, filename = findmakedirec(direc = direc, folname = folname, fname = fname)
        self.foldir = foldir
        self.dirfile = dirfile
        self.filename = filename


        ############################## spot and strike specification ##############################
        spot = kwunion.get('spot', False)
        if spot is False:
            self.help(['params'])
            raise ValueError("'spot' parameter must be specified")
        strike = kwunion.get('strike', False)
        if strike is False:
            self.help(['params'])
            raise ValueError("'strike' parameter must be specified")

        if not isinstance(spot, (float, int)):
            self.help(['paramsexamples'])
            raise TypeError('spot must be a number (float or int)')
        elif not isinstance(strike, (float, int)):
            self.help(['paramsexamples'])
            raise TypeError('strike must be a number (float or int)')

        self.spot = spot
        self.strike = strike


        ############################## time parameters specification ##############################
        T = kwunion.get('T', None)
        dt = kwunion.get('dt', None)
        periods = kwunion.get('periods', None)
        if isinstance(periods, float):
            periods = int(periods)
        dtfreq = kwunion.get('dtfreq', None)

        if T is not None and dt is not None and periods is None:
            periods = int(T / dt)
        elif T is not None and periods is not None and dt is None:
            dt = T / periods
        elif dt is not None and periods is not None and T is None:
            T = dt * periods
        elif dt is not None and periods is not None and T is not None:
            dt = T / periods
        elif dt is None and periods is None and T is None:
            self.help(['paramsexamples'])

        self.T = T
        self.dt = dt
        self.periods = int(periods)
        self.dtfreq = dtfreq
        if dtfreq == 'd' or dtfreq == 'w' or dtfreq == 'm':
            self.headerformat = kwunion.get('headerformat', 'dt')
        else:
            self.headerformat = kwunion.get('headerformat', 'Periods')

        # header
        treeheader = ['Periods ->'] + np.arange(self.periods + 1).tolist()
        if self.headerformat == 'dt' and self.dtfreq is None:
            treeheader = ['dt ->'] + np.arange(0,
                                               self.T + self.dt, self.dt).tolist()
        elif self.headerformat == 'dt' and self.dtfreq == 'd':
            dtstep = int(self.dt * 365)
            treeheader = ['dt ->'] + np.char.add((np.arange(self.periods + 1)*dtstep).astype(str),
                                                 np.array(['/365'] * (self.periods + 1))).tolist()
        elif self.headerformat == 'dt' and self.dtfreq == 'w':
            dtstep = int(self.dt * 52)
            treeheader = ['dt ->'] + np.char.add((np.arange(self.periods + 1)*dtstep).astype(str),
                                                 np.array(['/52'] * (self.periods + 1))).tolist()
        elif self.headerformat == 'dt' and self.dtfreq == 'm':
            dtstep = int(self.dt * 12)
            treeheader = ['dt ->'] + np.char.add((np.arange(self.periods + 1)*dtstep).astype(str),
                                                 np.array(['/12'] * (self.periods + 1))).tolist()

        self.treeheader = treeheader


        ############################## interest rate++ specification ##############################
        r = kwunion.get('r', 0.0)
        rcont = kwunion.get('rcont', True)
        divyield = kwunion.get('divyield', 0)

        discountRate = np.exp(-r * dt)
        discountDiv = np.exp(-divyield * dt)
        discountRateMinusDiv = np.exp((r - divyield) * dt)
        if rcont is False:
            discountDiv = 1 / ((1 + divyield)**dt)
            discountRate = 1 / ((1 + r)**dt)
            discountRateMinusDiv = 1 / ((1 + (r - divyield))**dt)

        discdiv = kwunion.get('discdiv', None)
        nonrec = kwunion.get('nonrec', False)
        preex = kwunion.get('preex', True)
        treetype = 'normal'
        if discdiv == 0 or discdiv == float(0):
            discdiv = None
        if discdiv is not None and nonrec is True:
            treetype = 'nonrecombining'
        elif discdiv is not None and nonrec is False:
            treetype = 'fsolution'

        self.r = r
        self.rcont = rcont
        self.divyield = divyield
        self.discountRate = discountRate
        self.discountDiv = discountDiv
        self.discountRateMinusDiv = discountRateMinusDiv

        self.discdiv = discdiv
        self.nonrec = nonrec
        self.preex = preex
        self.treetype = treetype


        ############################## up, down & vola specification ##############################
        def udfunc_default(vola = None, T = None, dt = None, periods = None,
                           r = None, divyield = None, discountRate = None,
                           discountDiv = None, discountRateMinusDiv = None,
                           spot = None, strike = None):
            u = np.exp(vola * np.sqrt(dt))
            d = 1 / u
            return u, d

        self.udfunc = kwunion.get('udfunc', udfunc_default)


        vola = kwunion.get('vola', False)
        u = kwunion.get('u', False)
        d = kwunion.get('d', False)

        if vola is not False and u is False and d is False:
            if callable(self.udfunc) is False:
                self.udfunc = udfunc_default
            u, d = self.udfunc(vola = vola, T = T, dt = dt, periods = periods,
                          r = r, divyield = divyield, discountRate = discountRate,
                          discountDiv = discountDiv, discountRateMinusDiv = discountRateMinusDiv,
                          spot = spot, strike = strike)
        elif u is not False and vola is False and d is False:
            d = 1 / u
            vola = (np.log(u) - np.log(d)) / (2 * np.sqrt(dt))
        elif d is not False and vola is False and u is False:
            u = 1 / d
            vola = (np.log(u) - np.log(d)) / (2 * np.sqrt(dt))
        elif vola is False and u is not False and d is not False:
            vola = (np.log(u) -
                    np.log(d)) / (2 * np.sqrt(dt))
        elif d is False and vola is not False and u is not False:
            d = u / (np.exp(2 * vola *
                            np.sqrt(dt)))
        elif u is False and vola is not False and d is not False:
            u = d * (np.exp(2 * vola *
                            np.sqrt(dt)))
        elif all([u, d, vola]):
            vola = (np.log(u) - np.log(d)) / (2 *
                                              np.sqrt(dt))
            if self.udfunc == udfunc_default:
                print(f"Since 'u', 'd', and 'vola' were passed explicitly"
                      f"\n-> generated new vola: {round(vola * 100, 2)}%"
                      f"\nfrom formula: vola = (np.log(u) - np.log(d)) / (2 * np.sqrt(dt))\n")
            else:
                print(f"Since 'u', 'd', and 'vola' were passed explicitly"
                      f"\n-> generated new vola: {round(vola * 100, 2)}%"
                      f"\nfrom formula: {self.udfunc}\n")
        else:
            self.help(['params', 'paramsexamples'])
            raise KeyError("Neither 'vola', 'u', or 'd' were found in passed parameters. \n"
                           "At least one of 'vola', 'u', or 'd' must be passed\n"
                           "See specification above")

        self.vola = vola
        self.u = u
        self.d = d
        self.collapsed = kwunion.get('collapsed', False)


        ######################### risk-neutral probability specification ##########################
        q = (np.exp((r - divyield) * dt) - d) / (u - d)
        # q = (discountRateMinusDiv - d) / (u - d)
        if rcont is False:
            q = ((1 + (r-divyield)) - d) / (u - d)

        self.q = q


        ################################## class wide functions ###################################
        def makenl(arr):
            ind = np.arange(1, len(arr)).cumsum()

            nl = arr[np.tril_indices_from(arr)]
            nl = np.split(nl, ind)
            nl = list(map(lambda x: x.tolist(), nl))
            return nl
        self.makenl = makenl

        def updownzip(periods):
            columnsnr = int(periods + 1)
            ind = np.arange(1, columnsnr).cumsum()

            # up indices
            upind = np.ones((columnsnr, columnsnr))
            upind[np.triu_indices_from(upind, 0)] = 0
            upind = upind.cumsum(0)
            upind = upind[np.tril_indices_from(upind)].astype(int)
            upind = np.split(upind, ind)

            # down indices
            downind = np.arange(columnsnr) * np.ones((columnsnr, columnsnr))
            downind[np.triu_indices_from(downind, 1)] = 0
            downind = downind[np.tril_indices_from(downind)].astype(int)
            downind = np.split(downind, ind)

            updownzipped = list(zip(upind, downind))

            return updownzipped
        self.updownzip = updownzip
        self.updownzipped = updownzip(self.periods)

        def updoNonrec(ups = 0, downs = 0, periods = 1):
            upArange = np.arange(ups, ups + periods + 1)
            up = np.linspace(upArange, upArange - len(upArange) + 1, len(upArange)).astype(int).T
            up[np.triu_indices_from(up, 1)] = 0

            do = np.arange(downs, downs + periods + 1) * np.ones_like(up)
            do[np.triu_indices_from(do, 1)] = 0

            return np.array([up, do])
        self.updoNonrec = updoNonrec


        ################################### tree specification ####################################
        self.showIntrinsic = kwunion.get('showIntrinsic', True)
        if self.showIntrinsic is True:
            self.rowPad = 3
        elif self.showIntrinsic is False:
            self.rowPad = 2

        self.rounding = int(kwunion.get('rounding', 2))

        ################################ which trees to calculate #################################
        self.maketrees = kwunion.get('maketrees', ['ec', 'ep', 'ac', 'ap'])
        self.makedfs = kwunion.get('makedfs', True)
        self.trees = dict()

        ########################### run proper tree construction method ###########################
        self.dfcalled = kwunion.get('dfcalled', False)
        self.called = kwunion.get('called', False)

        if self.called is True and self.dfcalled is True:
            self.makedfs = True

        if kwunion.get('test', False) is True:
            pass
        else:
            self.calculate()
            if kwunion.get('write', False) is True:
                self.write()
        ###########################################################################################


    def spotsUpDownInd(self, spot, periods, archive = True, spotname ='spotarr', ftree = False):
        import numpy as np

        colnum = int(periods + 1)

        u = self.u
        d = self.d
        if ftree == True:
            volaF = (self.spot / spot) * self.vola
            u, d = self.udfunc(vola = volaF, T = self.T, dt = self.dt, periods = self.periods,
                               r = self.r, divyield = self.divyield, discountRate = self.discountRate,
                               discountDiv = self.discountDiv, discountRateMinusDiv = self.discountRateMinusDiv,
                               spot = self.spot, strike = self.strike)
            self.volaF = volaF

        up = np.ones((colnum, colnum))
        up[np.triu_indices_from(up, 0)] = 0
        up = up.cumsum(0)
        ua = u**up
        ua[np.triu_indices_from(ua, 1)] = 0

        do = np.arange(colnum) * np.ones_like(up)
        do[np.triu_indices_from(do, 1)] = 0
        daa = d**do
        daa[np.triu_indices_from(daa, 1)] = 0

        updownArr = ua * daa
        updoind = [up.astype(int), do.astype(int)]
        spotarr = (updownArr * spot).round(self.rounding)

        self.updoind = updoind

        if archive is True:
            setattr(self, spotname, spotarr)

        return spotarr, updoind


    def normaltrees(self, optType, spots, manualOpt = None,
                    manualDeltas = None, manualBonds = None, manualIntr = None):
        import numpy as np

        # intrinsic values
        intrinsic = np.maximum(spots - self.strike, np.zeros_like(spots)).round(self.rounding)
        if optType[1] == 'p':
            intrinsic = np.maximum(self.strike - spots, np.zeros_like(spots)).round(self.rounding)
            intrinsic[np.triu_indices_from(intrinsic, 1)] = 0

        if manualIntr is not None:
            intrinsic = manualIntr

        # premiums
        options = np.zeros_like(spots)
        options[-1] = intrinsic[-1]
        if manualOpt is not None:
            options[-1] = manualOpt

        # check against based on type
        ind = np.arange(1, len(spots)-1).cumsum()

        checkagainst = np.zeros_like(spots[:-1, :-1])
        checkagainst = np.split(checkagainst[np.tril_indices_from(checkagainst)], ind)
        checkagainst = checkagainst[::-1]

        if optType[0] == 'a':
            checkagainst = intrinsic[:-1, :-1].copy()
            checkagainst = np.split(checkagainst[np.tril_indices_from(checkagainst)], ind)
            checkagainst = checkagainst[::-1]


        for col in enumerate(self.updownzip(len(spots)-1)[::-1][1:]):
            up = (col[1][0].astype(int) + col[1][1].astype(int) + 1, col[1][1].astype(int))
            down = (col[1][0].astype(int) + col[1][1].astype(int) + 1, col[1][1].astype(int) + 1)

            optnew = np.maximum(self.discountRate * (self.q * options[up] + (1 - self.q) * options[down]),
                                checkagainst[col[0]]).round(self.rounding)
            options[(up[0] - 1, up[1])] = optnew

        # portfolios
        optu = options[np.tril_indices_from(options, -1)]
        optd = options[1:, 1:][np.tril_indices_from(options[1:, 1:])]

        spotu = spots[np.tril_indices_from(spots, -1)]
        spotd = spots[1:, 1:][np.tril_indices_from(spots[1:, 1:])]

        d = (self.discountDiv * ((optu - optd) / (spotu - spotd))).round(self.rounding)
        deltas = np.zeros_like(spots)
        deltas[np.tril_indices_from(spots[:-1, :-1])] = d

        if manualDeltas is not None:
            deltas[-1] = manualDeltas

        b = (self.discountRate * ((self.u * optd - self.d * optu) / (self.u - self.d))).round(self.rounding)
        bonds = np.zeros_like(spots)
        bonds[np.tril_indices_from(spots[:-1, :-1])] = b

        if manualBonds is not None:
            bonds[-1] = manualBonds

        return intrinsic, options, deltas, bonds


    def getOptionsNormal(self, optType):
        import numpy as np


        spotarr = getattr(self, 'spotarr', self.spotsUpDownInd(self.spot, self.periods)[0])
        updoind = getattr(self, 'updoind', self.spotsUpDownInd(self.spot, self.periods)[1])

        intrinsic, options, deltas, bonds = self.normaltrees(optType, spotarr)

        # flat tree indices for array
        upflat = updoind[0][np.tril_indices_from(updoind[0])]
        downflat = updoind[1][np.tril_indices_from(updoind[1])]

        treecols = upflat + downflat
        treerows = (self.periods * self.rowPad) - (upflat * self.rowPad) + (downflat * self.rowPad)
        rows = int(2 * (self.periods * self.rowPad) + self.rowPad)
        if self.collapsed is True:
            treerows = downflat * self.rowPad
            rows = int((self.periods + 1) * self.rowPad)

        # tree construction
        if self.makedfs is True:
            class treeWithDF:
                def __init__(self, spots, intrinsic, options, deltas, bonds, ups, downs, colIndFlat, rowIndFlat,
                             header, mainobject, rows):
                    import pandas as pd

                    # nested list trees
                    self.spots = mainobject.makenl(spots)
                    self.intrinsic = mainobject.makenl(intrinsic)
                    self.options = mainobject.makenl(options)
                    self.deltas = mainobject.makenl(deltas)
                    self.bonds = mainobject.makenl(bonds)
                    self.ups = mainobject.makenl(ups)
                    self.downs = mainobject.makenl(downs)

                    # flat trees
                    self.spotsflat = spots[np.tril_indices_from(spots)]
                    self.intrinsicflat = intrinsic[np.tril_indices_from(intrinsic)]
                    self.optionsflat = options[np.tril_indices_from(options)]
                    self.deltasflat = deltas[np.tril_indices_from(deltas)]
                    self.bondsflat = bonds[np.tril_indices_from(bonds)]
                    self.upflat = ups[np.tril_indices_from(ups)]
                    self.downflat = downs[np.tril_indices_from(downs)]

                    # dataframe array
                    NoneType = None

                    dfarr = np.full((rows, mainobject.periods + 1), None)
                    dfarr[rowIndFlat, colIndFlat] = self.spotsflat.round(mainobject.rounding)
                    if mainobject.showIntrinsic is True:
                        intrinsicString = np.char.add(np.array(['['] * len(self.intrinsicflat)),
                                                      self.intrinsicflat.round(mainobject.rounding).astype(str))
                        intrinsicString2 = np.char.add(intrinsicString, np.array([']'] * len(self.intrinsicflat)))
                        dfarr[rowIndFlat + 1, colIndFlat] = intrinsicString2

                        optionsString = np.char.add(np.array(['('] * len(self.optionsflat)),
                                                    self.optionsflat.round(mainobject.rounding).astype(str))
                        optionsString2 = np.char.add(optionsString, np.array([')'] * len(self.optionsflat)))
                        dfarr[rowIndFlat + 2, colIndFlat] = optionsString2

                        dfarr[np.where(dfarr == NoneType)] = ''
                        fc = np.array(['Spot', '[Intrinsic]', '(Premium)'] + [''] * (rows - 3))
                        dfarr = np.hstack((fc.reshape(fc.shape[0], 1), dfarr))
                    else:
                        optionsString = np.char.add(np.array(['('] * len(self.optionsflat)),
                                                    self.optionsflat.round(mainobject.rounding).astype(str))
                        optionsString2 = np.char.add(optionsString, np.array([')'] * len(self.optionsflat)))
                        dfarr[rowIndFlat + 1, colIndFlat] = optionsString2
                        dfarr[np.where(dfarr == NoneType)] = ''
                        fc = np.array(['Spot', '(Premium)'] + [''] * (rows - 2))
                        dfarr = np.hstack((fc.reshape(fc.shape[0], 1), dfarr))

                    self.colIndFlat = colIndFlat
                    self.rowIndFlat = rowIndFlat
                    self.treeheader = header

                    self.df = pd.DataFrame(dfarr, index = [''] * len(dfarr), columns = header)


                def getnode(self, up, down):
                    spot = self.spots[up + down][down]
                    intrinsic = self.intrinsic[up + down][down]
                    opt = self.options[up + down][down]
                    delta = self.deltas[up + down][down]
                    bond = self.bonds[up + down][down]
                    return dict(Spot = spot, Intrinsic = intrinsic, Premium = opt, Delta = delta, Bond = bond)


                def __call__(self, up, down):
                    spot = self.spots[up + down][down]
                    intrinsic = self.intrinsic[up + down][down]
                    opt = self.options[up + down][down]
                    delta = self.deltas[up + down][down]
                    bond = self.bonds[up + down][down]
                    return dict(Spot = spot, Intrinsic = intrinsic, Premium = opt, Delta = delta, Bond = bond)

                def __repr__(self):
                    return self.df.__repr__()


            mytree = treeWithDF(spotarr, intrinsic, options, deltas, bonds,
                                updoind[0], updoind[1], treecols, treerows, self.treeheader, self, rows)
        else:
            class treeWithoutDF:
                def __init__(self, spots, intrinsic, options, deltas, bonds,
                             ups, downs, colIndFlat, rowIndFlat, header, mainobject):

                    # nested list trees
                    self.spots = mainobject.makenl(spots)
                    self.intrinsic = mainobject.makenl(intrinsic)
                    self.options = mainobject.makenl(options)
                    self.deltas = mainobject.makenl(deltas)
                    self.bonds = mainobject.makenl(bonds)
                    self.ups = mainobject.makenl(ups)
                    self.downs = mainobject.makenl(downs)

                    # flat trees
                    self.spotsflat = spots[np.tril_indices_from(spots)]
                    self.intrinsicflat = intrinsic[np.tril_indices_from(intrinsic)]
                    self.optionsflat = options[np.tril_indices_from(options)]
                    self.deltasflat = deltas[np.tril_indices_from(deltas)]
                    self.bondsflat = bonds[np.tril_indices_from(bonds)]
                    self.upflat = ups[np.tril_indices_from(ups)]
                    self.downflat = downs[np.tril_indices_from(downs)]

                    self.colIndFlat = colIndFlat
                    self.rowIndFlat = rowIndFlat
                    self.treeheader = header


                def getnode(self, up, down):
                    spot = self.spots[up + down][down]
                    intrinsic = self.intrinsic[up + down][down]
                    opt = self.options[up + down][down]
                    delta = self.deltas[up + down][down]
                    bond = self.bonds[up + down][down]
                    return dict(Spot = spot, Intrinsic = intrinsic, Premium = opt, Delta = delta, Bond = bond)


                def __call__(self, up, down):
                    spot = self.spots[up + down][down]
                    intrinsic = self.intrinsic[up + down][down]
                    opt = self.options[up + down][down]
                    delta = self.deltas[up + down][down]
                    bond = self.bonds[up + down][down]
                    return dict(Spot = spot, Intrinsic = intrinsic, Premium = opt, Delta = delta, Bond = bond)


            mytree = treeWithoutDF(spotarr, intrinsic, options, deltas,
                                   bonds, updoind[0], updoind[1], treecols, treerows, self.treeheader, self)

        # setting tree object as attribute and return
        setattr(self, optType+'Tree', mytree)
        setattr(self, optType+'OptionPrice', mytree.optionsflat[0])
        self.trees.update({optType + 'Tree': mytree})

        if optType == 'ec':
            self.BScall()
        elif optType == 'ep':
            self.BSput()

        return mytree


    def getOptionsFsol(self, optType):
        import numpy as np

        # indices, etc.
        dt_all = np.arange(0, self.T + self.dt, self.dt)
        divdt = np.array(self.discdiv)[:, 0]
        divs = np.array(self.discdiv)[:, 1]
        divind = np.abs(np.subtract.outer(dt_all, divdt)).argmin(0)

        # present value of all dividends
        pvdiv = (divs * (self.discountRate**divind)).sum()
        F0 = (self.spot - pvdiv).round(self.rounding)

        # F tree
        Ftree = getattr(self, 'Ftree', self.spotsUpDownInd(F0, self.periods, True, 'Ftree', ftree = True)[0])
        FtreeShaved = Ftree[:divind.max() + 1, :divind.max() + 1]
        updoind = getattr(self, 'updoind', self.spotsUpDownInd(F0, self.periods, True, 'Ftree')[1])

        # S tree
        divpowpv = np.linspace(divind, divind - len(Ftree) + 1, len(Ftree)).astype(int).T
        divpowpv[divpowpv < 0] = 0
        antidivind = np.where(divpowpv == 0)
        divpowpv = self.discountRate**divpowpv
        divpowpv[antidivind] = 0
        divpowpv = (divpowpv.T[:] * divs).T
        divpowpv = divpowpv.sum(0)
        divpowpv[divind] +=  divs

        spotarr = (Ftree.T + divpowpv).T.round(self.rounding)
        spotarr[np.triu_indices_from(spotarr, 1)] = 0
        self.spotarr = spotarr

        # options, intrinsics, etc.
        intrinsic, options, deltas, bonds = self.normaltrees(optType, spotarr)

        # flat tree indices for arrays - spotarr and FtreeShaved
        upflat = updoind[0][np.tril_indices_from(updoind[0])]
        downflat = updoind[1][np.tril_indices_from(updoind[1])]
        treecols = upflat + downflat
        treerows = (self.periods * self.rowPad) - (upflat * self.rowPad) + (downflat * self.rowPad)

        upindF = updoind[0][:divind.max() + 1, :divind.max() + 1]
        doindF = updoind[1][:divind.max() + 1, :divind.max() + 1]
        upflatF = upindF[np.tril_indices_from(upindF)]
        downflatF = doindF[np.tril_indices_from(upindF)]
        treecolsF = upflatF + downflatF
        treerowsF = treerows[:len(treecolsF)]

        treerowsS = treerows.copy()
        treerowsS[:len(treerowsF)] -=  1

        # rows ++
        rows = int(2 * (self.periods * self.rowPad) + self.rowPad)
        if self.collapsed is True:
            rows = int((self.periods + 1) * self.rowPad)

            treerowsF = (downflatF * (self.rowPad + 1)) + 1
            treerows = np.hstack((treerowsF, downflat[len(treerowsF):] * self.rowPad))
            treerowsS = np.hstack((treerowsF - 1, downflat[len(treerowsF):] * self.rowPad))

        # check if div is in last period
        if dt_all[-1] in divdt:
            treerows +=  1
            treerowsS +=  1
            rows +=  1

        # tree construction
        if self.makedfs is True:
            class FtreeWithDF:
                def __init__(self, spots, intrinsic, options, deltas, bonds, ups, downs, colIndFlat, rowIndFlat,
                             header, mainobject, rows, Ftree, colIndFlatF, rowIndFlatF, rowIndFlatS):
                    import pandas as pd

                    # nested list trees
                    self.spots = mainobject.makenl(spots)
                    self.Ftree = mainobject.makenl(Ftree)
                    self.intrinsic = mainobject.makenl(intrinsic)
                    self.options = mainobject.makenl(options)
                    self.deltas = mainobject.makenl(deltas)
                    self.bonds = mainobject.makenl(bonds)
                    self.ups = mainobject.makenl(ups)
                    self.downs = mainobject.makenl(downs)

                    # flat trees
                    self.spotsflat = spots[np.tril_indices_from(spots)]
                    self.Ftreeflat = Ftree[np.tril_indices_from(Ftree)]
                    self.intrinsicflat = intrinsic[np.tril_indices_from(intrinsic)]
                    self.optionsflat = options[np.tril_indices_from(options)]
                    self.deltasflat = deltas[np.tril_indices_from(deltas)]
                    self.bondsflat = bonds[np.tril_indices_from(bonds)]
                    self.upflat = ups[np.tril_indices_from(ups)]
                    self.downflat = downs[np.tril_indices_from(downs)]

                    # dataframe array
                    NoneType = None

                    dfarr = np.full((rows, mainobject.periods + 1), None)
                    dfarr[rowIndFlatS, colIndFlat] = self.spotsflat.round(mainobject.rounding)

                    FspotString = np.char.add(np.array(['{'] * len(self.Ftreeflat)),
                                              self.Ftreeflat.round(mainobject.rounding).astype(str))
                    FspotString2 = np.char.add(FspotString, np.array(['}'] * len(self.Ftreeflat)))
                    dfarr[rowIndFlatF, colIndFlatF] = FspotString2

                    if mainobject.showIntrinsic is True:
                        intrinsicString = np.char.add(np.array(['['] * len(self.intrinsicflat)),
                                                      self.intrinsicflat.round(mainobject.rounding).astype(str))
                        intrinsicString2 = np.char.add(intrinsicString, np.array([']'] * len(self.intrinsicflat)))
                        dfarr[rowIndFlat + 1, colIndFlat] = intrinsicString2

                        optionsString = np.char.add(np.array(['('] * len(self.optionsflat)),
                                                    self.optionsflat.round(mainobject.rounding).astype(str))
                        optionsString2 = np.char.add(optionsString, np.array([')'] * len(self.optionsflat)))
                        dfarr[rowIndFlat + 2, colIndFlat] = optionsString2

                        dfarr[np.where(dfarr == NoneType)] = ''
                        fc = np.array(['Spot', '{F-Spot}', '[Intrinsic]', '(Premium)'] + [''] * (rows - 4))
                        dfarr = np.hstack((fc.reshape(fc.shape[0], 1), dfarr))
                    else:
                        optionsString = np.char.add(np.array(['('] * len(self.optionsflat)),
                                                    self.optionsflat.round(mainobject.rounding).astype(str))
                        optionsString2 = np.char.add(optionsString, np.array([')'] * len(self.optionsflat)))
                        dfarr[rowIndFlat + 1, colIndFlat] = optionsString2
                        dfarr[np.where(dfarr == NoneType)] = ''
                        fc = np.array(['Spot', '{F-Spot}', '(Premium)'] + [''] * (rows - 3))
                        dfarr = np.hstack((fc.reshape(fc.shape[0], 1), dfarr))

                    self.colIndFlat = colIndFlat
                    self.colIndFlatF = colIndFlatF
                    self.rowIndFlat = rowIndFlat
                    self.rowIndFlatS = rowIndFlatS
                    self.rowIndFlatF = rowIndFlatF
                    self.treeheader = header

                    self.df = pd.DataFrame(dfarr, index = [''] * len(dfarr), columns = header)


                def getnode(self, up, down):
                    fdict = dict()
                    if up + down <=  self.colIndFlatF[-1]:
                        fdict = {'F-Spot': self.Ftree[up + down][down]}
                    spot = self.spots[up + down][down]
                    intrinsic = self.intrinsic[up + down][down]
                    opt = self.options[up + down][down]
                    delta = self.deltas[up + down][down]
                    bond = self.bonds[up + down][down]
                    return dict(Spot = spot, **fdict, Intrinsic = intrinsic, Premium = opt, Delta = delta, Bond = bond)


                def __call__(self, up, down):
                    fdict = dict()
                    if up + down <=  self.colIndFlatF[-1]:
                        fdict = {'F-Spot': self.Ftree[up + down][down]}
                    spot = self.spots[up + down][down]
                    intrinsic = self.intrinsic[up + down][down]
                    opt = self.options[up + down][down]
                    delta = self.deltas[up + down][down]
                    bond = self.bonds[up + down][down]
                    return dict(Spot = spot, **fdict, Intrinsic = intrinsic, Premium = opt, Delta = delta, Bond = bond)


                def __repr__(self):
                    return self.df.__repr__()


            mytree = FtreeWithDF(spotarr, intrinsic, options, deltas, bonds, updoind[0], updoind[1],
                                 treecols, treerows, self.treeheader, self, rows,
                                 FtreeShaved, treecolsF, treerowsF, treerowsS)
        else:
            class FtreeWithoutDF:
                def __init__(self, spots, intrinsic, options, deltas, bonds, ups, downs, colIndFlat, rowIndFlat,
                             header, mainobject, Ftree, colIndFlatF, rowIndFlatF, rowIndFlatS):
                    # nested list trees
                    self.spots = mainobject.makenl(spots)
                    self.Ftree = mainobject.makenl(Ftree)
                    self.intrinsic = mainobject.makenl(intrinsic)
                    self.options = mainobject.makenl(options)
                    self.deltas = mainobject.makenl(deltas)
                    self.bonds = mainobject.makenl(bonds)
                    self.ups = mainobject.makenl(ups)
                    self.downs = mainobject.makenl(downs)

                    # flat trees
                    self.spotsflat = spots[np.tril_indices_from(spots)]
                    self.Ftreeflat = Ftree[np.tril_indices_from(Ftree)]
                    self.intrinsicflat = intrinsic[np.tril_indices_from(intrinsic)]
                    self.optionsflat = options[np.tril_indices_from(options)]
                    self.deltasflat = deltas[np.tril_indices_from(deltas)]
                    self.bondsflat = bonds[np.tril_indices_from(bonds)]
                    self.upflat = ups[np.tril_indices_from(ups)]
                    self.downflat = downs[np.tril_indices_from(downs)]

                    self.colIndFlat = colIndFlat
                    self.colIndFlatF = colIndFlatF
                    self.rowIndFlat = rowIndFlat
                    self.rowIndFlatS = rowIndFlatS
                    self.rowIndFlatF = rowIndFlatF
                    self.treeheader = header


                def getnode(self, up, down):
                    fdict = dict()
                    if up + down <=  self.colIndFlatF[-1]:
                        fdict = {'F-Spot': self.Ftree[up + down][down]}
                    spot = self.spots[up + down][down]
                    intrinsic = self.intrinsic[up + down][down]
                    opt = self.options[up + down][down]
                    delta = self.deltas[up + down][down]
                    bond = self.bonds[up + down][down]
                    return dict(Spot = spot, **fdict, Intrinsic = intrinsic, Premium = opt, Delta = delta, Bond = bond)


                def __call__(self, up, down):
                    fdict = dict()
                    if up + down <=  self.colIndFlatF[-1]:
                        fdict = {'F-Spot': self.Ftree[up + down][down]}
                    spot = self.spots[up + down][down]
                    intrinsic = self.intrinsic[up + down][down]
                    opt = self.options[up + down][down]
                    delta = self.deltas[up + down][down]
                    bond = self.bonds[up + down][down]
                    return dict(Spot = spot, **fdict, Intrinsic = intrinsic, Premium = opt, Delta = delta, Bond = bond)


            mytree = FtreeWithoutDF(spotarr, intrinsic, options, deltas, bonds, updoind[0], updoind[1],
                                    treecols, treerows, self.treeheader, self, FtreeShaved, treecolsF, treerowsF,
                                    treerowsS)


        # setting tree object as attribute and return
        setattr(self, optType+'Tree', mytree)
        setattr(self, optType+'OptionPrice', mytree.optionsflat[0])
        self.trees.update({optType + 'Tree': mytree})

        return mytree


    def getOptionsNonrec(self, optType):
        import numpy as np
        import itertools


        # indices, etc.
        dt_all = np.arange(0, self.T + self.dt, self.dt)
        divdt = np.array(self.discdiv)[:, 0]
        divs = np.array(self.discdiv)[:, 1]
        divind = np.abs(np.subtract.outer(dt_all, divdt)).argmin(0)
        startindPadded = np.hstack(([0], divind, [self.periods]))
        startperiods = startindPadded[1:] - startindPadded[:-1]
        startind = startindPadded[:-1]
        ind = np.arange(1, self.periods + 1).cumsum()


        # calculating all spot trees
        def makeSpotarrUpindDoind(mainobj, startperiods, divs):
            spotarr = [[self.spotsUpDownInd(self.spot, startperiods[0], False)[0].round(self.rounding)]]
            upind = [[np.array(self.spotsUpDownInd(self.spot, startperiods[0], False)[1][0])]]
            doind = [[np.array(self.spotsUpDownInd(self.spot, startperiods[0], False)[1][1])]]

            def sortNLarrays(nlarrs, flip=True):
                arr = np.array(list(itertools.chain(*nlarrs)))
                if flip is True:
                    arraySorted = np.sort(arr, 0)[::-1]
                else:
                    arraySorted = np.sort(arr, 0)
                arraySortedSplit = np.split(arraySorted, arraySorted.shape[0])
                sortedlist = list(map(lambda x: np.squeeze(x), arraySortedSplit))

                return sortedlist

            for indper in enumerate(startperiods[1:]):
                periodind = indper[0]
                periodsLoop = indper[1]
                previousTrees = spotarr[-1]  # list of arrays/trees
                spotexdiv = list(map(lambda x: x[-1] - divs[periodind], previousTrees))  # list of arrays

                ups = list(map(lambda x: x[-1], upind[-1]))
                downs = list(map(lambda x: x[-1], doind[-1]))

                tempS = []
                tempU = []
                tempD = []
                for i in enumerate(spotexdiv):
                    spotsLoop = list(map(lambda x:
                                         np.array(self.spotsUpDownInd(x, periodsLoop, False)[0]).round(self.rounding),
                                         i[1]))
                    upLoop = list(map(lambda u, d:
                                      self.updoNonrec(u, d, periodsLoop)[0], ups[i[0]],
                                      downs[i[0]]))
                    doLoop = list(map(lambda u, d:
                                      self.updoNonrec(u, d, periodsLoop)[1], ups[i[0]],
                                      downs[i[0]]))

                    tempS.append(spotsLoop)
                    tempU.append(upLoop)
                    tempD.append(doLoop)

                spotarr.append(sortNLarrays(tempS))
                upind.append(sortNLarrays(tempU))
                doind.append(sortNLarrays(tempD, flip=False))

            return spotarr, upind, doind
        spotarr, upind, doind = makeSpotarrUpindDoind(self, startperiods, divs)


        # calculating intrinsics
        def addbackDiv(sa, d):
            spots = sa[np.tril_indices_from(sa)]
            spots[0] += d
            mal = np.zeros_like(sa)
            mal[np.tril_indices_from(mal)] = spots
            return mal
        def intrinsicsCalc(s):
            intrinsic = np.maximum(s - self.strike, np.zeros_like(s)).round(self.rounding)
            if optType[1] == 'p':
                intrinsic = np.maximum(self.strike - s, np.zeros_like(s)).round(self.rounding)
                intrinsic[np.triu_indices_from(intrinsic, 1)] = 0
            return intrinsic
        def manualIntrinsics(spotarr, divs):
            spotspre = []
            dd = np.hsplit(divs, len(divs))
            dd.insert(0, [0])
            for sl, div in zip(spotarr, dd):
                spotspre.append(list(map(addbackDiv, sl, [div] * len(sl))))

            intrinsics = []
            for spots in spotspre:
                intrinsics.append(list(map(intrinsicsCalc, spots)))
            return intrinsics
        intrinsics = manualIntrinsics(spotarr, divs)


        # options, etc
        def makeOptionsDeltasEtc(spotarr, mainobj, optType, intrinsics):
            options, deltas, bonds = [], [], []

            manualOpt = None
            for arrs, intr in zip(spotarr[::-1], intrinsics[::-1]):
                if len(options) != 0:
                    manualOpt = np.array(options[-1])[:, 0, 0]
                    manualOpt = np.split(manualOpt, len(arrs))
                    iodb_collection = list(map(lambda arr, manO, manI:
                                               mainobj.normaltrees(optType, arr, manualOpt=manO, manualIntr=manI),
                                               arrs, manualOpt, intr))
                else:
                    iodb_collection = list(map(lambda arr, manI:
                                               mainobj.normaltrees(optType, arr, manualIntr=manI),
                                               arrs, intr))

                opts = list(map(lambda coll: coll[1].round(self.rounding), iodb_collection))
                delts = list(map(lambda coll: coll[2].round(self.rounding), iodb_collection))
                bnds = list(map(lambda coll: coll[3].round(self.rounding), iodb_collection))

                # intrinsics.append(intr)
                options.append(opts)
                deltas.append(delts)
                bonds.append(bnds)

            options, deltas, bonds = options[::-1], deltas[::-1], bonds[::-1]

            return options, deltas, bonds
        options, deltas, bonds = makeOptionsDeltasEtc(spotarr, self, optType, intrinsics)


        # rows
        rows = int(2 * (self.periods * self.rowPad) + self.rowPad)
        if self.collapsed is True:
            rows = int((self.periods + 1) * self.rowPad)


        # row indices
        def makeTreerows(upind, doind, rowpad, periods):
            treerows = []
            for u1, d1 in zip(upind, doind):
                rowsTemp = []
                for u, d in zip(u1, d1):
                    treerowsTemp = (periods * rowpad) - (u * rowpad) + (d * rowpad)
                    treerowsTemp[np.triu_indices_from(treerowsTemp, 1)] = 0
                    rowsTemp.append(treerowsTemp)

                treerows.append(rowsTemp)

            return treerows
        treerows = makeTreerows(upind, doind, self.rowPad, self.periods)


        # column indices
        def makeTreesInNode(startind, startindPadded, periods):

            def treeones(totPeriods, start, end, startAdj):
                treelist = []
                for i, a in zip(range(start + 1), startAdj):
                    mal = np.zeros((int(totPeriods + 1), int(totPeriods + 1))).astype(int)
                    mal[start:end + 1, i:end + 1 - start + i][
                        np.tril_indices_from(mal[start:end + 1, i:end + 1 - start + i])] = 1
                    treelist.append(mal * a)

                treearr = np.array(treelist).sum(0)

                return treearr

            mal = np.zeros((int(periods + 1), int(periods + 1))).astype(int)
            mal[startind[0]:startindPadded[1:][0] + 1,
            :startindPadded[1:][0] + 1 - startind[0]][np.tril_indices_from(mal[startind[0]:startindPadded[1:][0] + 1,
                                                                           :startindPadded[1:][0] + 1 - startind[
                                                                               0]])] = 1
            treesInNode = [mal]

            for s, e in zip(startind[1:], startindPadded[2:]):
                treesInNode.append(treeones(periods, s, e, treesInNode[-1][s, :s + 1]))

            return treesInNode
        treesInNode = makeTreesInNode(startind, startindPadded, self.periods)

        def makeTreecols(treesInNode, startind, startperiods, startindPadded):
            treecols = [[(treesInNode[0] * np.arange(len(treesInNode[0])).reshape(len(treesInNode[0]), 1))
                         [:startind[1] + 1, :startind[1] + 1]]]
            for treeset in zip(treesInNode[1:], startperiods[1:], startind[1:], startindPadded[2:]):
                startcol = np.array(treecols[-1])[:, -1].flatten()
                start = np.min(startcol)
                points = np.max(treeset[0][treeset[2]:treeset[3] + 1][0])

                colcluster = [np.linspace([start] * treeset[0][treeset[2]:treeset[3] + 1][0].shape[0],
                                          np.array([start] * treeset[0][treeset[2]:treeset[3] + 1][0].shape[0]) +
                                          treeset[0][treeset[2]:treeset[3] + 1][0].max() - 1,
                                          points).astype(int).T.tolist()]

                for row in treeset[0][treeset[2]:treeset[3] + 1][1:]:
                    start = np.max(np.array(colcluster[-1])) + 1
                    lin = np.linspace([start] * row.shape[0], np.array([start] * row.shape[0]) + row.max() - 1,
                                      row.max()).astype(int)
                    colcluster.append(lin.T.tolist())

                trilind = np.tril_indices(treeset[1] + 1)
                trilrows = trilind[0]
                trilcols = trilind[1]
                tempcol = []
                for i in range(treeset[2] + 1):
                    for _ in range(treeset[0][treeset[2]][i]):
                        columnspop = list(map(lambda x: x.pop(0), np.array(colcluster)[(trilrows, trilcols + i)]))
                        tempmal = np.zeros((treeset[1] + 1, treeset[1] + 1)).astype(int)
                        tempmal[trilrows, trilcols] = columnspop
                        tempcol.append(tempmal)

                treecols.append(tempcol)

            return treecols
        treecols = makeTreecols(treesInNode, startind, startperiods, startindPadded)


        # enumerated prediv spots and indices
        def lastnodes(Nlistarrays):
            retlist = []
            for arrlist in enumerate(itertools.chain(*Nlistarrays[:-1])):
                retlist.append((arrlist[0], arrlist[1][-1]))

            return retlist
        spotsPrediv = lastnodes(spotarr)

        def lastnodesRows(Nlistarrays):
            retlist = []
            for arrlist in enumerate(itertools.chain(*Nlistarrays[:-1])):
                retlist.append((arrlist[0], arrlist[1][-1] - 1))

            return retlist
        rowsPreddiv = lastnodesRows(treerows)
        colsPrediv = lastnodes(treecols)

        # cleanup in indices and value arrays, enumerated
        def nlCleanup(Nlistarrays):
            retlist = []
            toloop = Nlistarrays[:-1]
            for tree in enumerate(itertools.chain(*toloop)):
                retlist.append((tree[0], tree[1][:-1, :-1][np.tril_indices_from(tree[1][:-1, :-1])]))

            for tree in enumerate(Nlistarrays[-1], start=retlist[-1][0] + 1):
                retlist.append((tree[0], tree[1][np.tril_indices_from(tree[1])]))

            return retlist
        spotsToWrite = nlCleanup(spotarr)
        optsToWrite = nlCleanup(options)
        intrToWrite = nlCleanup(intrinsics)
        deltasToWrite = nlCleanup(deltas)
        bondsToWrite = nlCleanup(bonds)
        rowsToWrite = nlCleanup(treerows)
        colsToWrite = nlCleanup(treecols)

        # tree construction
        class treeWithoutDF:
            def __init__(self, spots, intrinsics, options, deltas, bonds,
                         colIndices, rowIndices, predivspots, predivrows, predivcols):

                # enumerated flat trees
                self.spots = spots
                self.intrinsics = intrinsics
                self.options = options
                self.deltas = deltas
                self.bonds = bonds
                self.colIndFlat = colIndices
                self.rowIndFlat = rowIndices

                self.predivspots = predivspots
                self.predivrows = predivrows
                self.predivcols = predivcols

        mytree = treeWithoutDF(spotsToWrite, intrToWrite, optsToWrite, deltasToWrite, bondsToWrite,
                               colsToWrite, rowsToWrite, spotsPrediv, rowsPreddiv, colsPrediv)


        # setting tree object as attribute and return
        setattr(self, optType + 'Tree', mytree)
        setattr(self, optType + 'OptionPrice', mytree.options[0][1][0])
        self.trees.update({optType + 'Tree': mytree})


    def makeTreeEC(self):
        treelist = self.maketrees
        treelist.append('ec')
        self.maketrees = list(set(treelist))
        self.calculate()
        return self.trees['ecTree']


    def makeTreeEP(self):
        treelist = self.maketrees
        treelist.append('ep')
        self.maketrees = list(set(treelist))
        self.calculate()
        return self.trees['epTree']


    def makeTreeAC(self):
        treelist = self.maketrees
        treelist.append('ac')
        self.maketrees = list(set(treelist))
        self.calculate()
        return self.trees['acTree']


    def makeTreeAP(self):
        treelist = self.maketrees
        treelist.append('ap')
        self.maketrees = list(set(treelist))
        self.calculate()
        return self.trees['apTree']


    def BScall(self):
        from scipy.stats import norm
        import numpy as np

        d1 = (np.log(self.spot/self.strike) + ((self.r - self.divyield) + (0.5*(self.vola**2))) * self.T) \
             /(self.vola * np.sqrt(self.T))
        d2 = d1 - (self.vola * np.sqrt(self.T))

        BSpremium = self.spot * np.exp(-self.divyield * self.T) * norm.cdf(d1) - \
                    self.strike * np.exp(-self.r * self.T) * norm.cdf(d2)

        self.ecOptionPriceBS = BSpremium

        return BSpremium


    def BSput(self):
        from scipy.stats import norm
        import numpy as np

        d1 = (np.log(self.spot / self.strike) + ((self.r - self.divyield) + (0.5 * (self.vola**2))) * self.T) \
             / (self.vola * np.sqrt(self.T))
        d2 = d1 - (self.vola * np.sqrt(self.T))

        BSpremium = self.strike * np.exp(-self.r * self.T) * norm.cdf(-d2) - \
                    self.spot * np.exp(-self.divyield * self.T) * norm.cdf(-d1)

        self.epOptionPriceBS = BSpremium

        return BSpremium


    def removediv(self):
        self.discdiv = None
        self.treetype = 'normal'


    def strings(self):
        """
        See portfolioStrings.py


        def sostr(x):
            return f'S∗u^{int(x[0])}∗d^{int(x[1])}'

        def opstr(x):
            return f'C_u^{int(x[0])} d^{int(x[1])}'

        def oppstr(x):
            return f'C_u^{int(x[0])} d^{int(x[1])} = ∆S∗u^{int(x[0])}∗d^{int(x[1])} ± B'

        pl = list(map(lambda x: x[0] + ' = ' + x[1] + '*' + x[2] + ' + ' + x[3], p))

        >> so
        [array(['S∗u^3∗d^0', 'S∗u^2∗d^1', 'S∗u^1∗d^2', 'S∗u^0∗d^3'], dtype = '<U9'),
         array(['S∗u^2∗d^0', 'S∗u^1∗d^1', 'S∗u^0∗d^2'], dtype = '<U9'),
         array(['S∗u^1∗d^0', 'S∗u^0∗d^1'], dtype = '<U9'),
         array(['S'], dtype = '<U1')]

        >> opt
        [array(['C_u^3 d^0', 'C_u^2 d^1', 'C_u^1 d^2', 'C_u^0 d^3'], dtype = '<U9'),
         array(['C_u^2 d^0', 'C_u^1 d^1', 'C_u^0 d^2'], dtype = '<U9'),
         array(['C_u^1 d^0', 'C_u^0 d^1'], dtype = '<U9'),
         array(['C'], dtype = '<U1')]

        >> opp
        [array(['C_u^3 d^0', 'C_u^2 d^1', 'C_u^1 d^2', 'C_u^0 d^3'], dtype = '<U9'),
         array(['C_u^2 d^0 = ∆S∗u^2∗d^0 + B', 'C_u^1 d^1 = ∆S∗u^1∗d^1 + B', 'C_u^0 d^2 = ∆S∗u^0∗d^2 + B'], dtype = '<U26'),
         array(['C_u^1 d^0 = ∆S∗u^1∗d^0 + B', 'C_u^0 d^1 = ∆S∗u^0∗d^1 + B'], dtype = '<U26'),
         array(['C_u^0 d^0 = ∆S∗u^0∗d^0 + B'], dtype = '<U26')]
        """
        pass


    def calculate(self):
        if self.treetype == 'normal':
            for i in self.maketrees:
                self.getOptionsNormal(i)
        elif self.treetype == 'fsolution':
            for i in self.maketrees:
                self.getOptionsFsol(i)
        elif self.treetype == 'nonrecombining':
            for i in self.maketrees:
                self.getOptionsNonrec(i)


    def write(self, fname_override = None):
        import xlsxwriter
        import numpy as np
        import os

        # make directory/folders if they don't exist
        if fname_override is None:
            filepath = self.dirfile
        else:
            filepath = os.path.join(self.foldir, str(fname_override) + '.xlsx')

        if not os.path.exists(self.foldir):
            os.makedirs(self.foldir)

        # universal variables
        if self.treetype == 'fsolution':
            startingRow = self.trees[list(self.trees.keys())[0]].rowIndFlat[0] + 1
        elif self.treetype == 'nonrecombining':
            startingRow = self.trees[list(self.trees.keys())[0]].rowIndFlat[0][1][0] + 2
        else:
            startingRow = self.trees[list(self.trees.keys())[0]].rowIndFlat[0] + 2

        timeheader = self.treeheader
        paramsLeft = ['S',
                      'K',
                      'vola',
                      'u',
                      'd',
                      'r',
                      'T',
                      'dt',
                      'periods',
                      '∂',
                      'Discrete div.',
                      'Continous r']
        paramsRightDef = ['Underlying spot now',
                          'Strike',
                          'Volatility',
                          'Up movement',
                          'Down movement',
                          'p.a. Risk-free rate',
                          'Years to maturity',
                          'Period length',
                          'Periods',
                          'Dividend yield',
                          'Discrete dividends',
                          'False -> discrete r']
        greenRow = len(paramsLeft) + 4

        if self.discdiv is None:
            exceldiscdiv = 'None'
        else:
            exceldiscdiv = 'True'

        paramsRight1 = [self.spot,
                        self.strike,
                        self.vola,
                        self.u,
                        self.d,
                        self.r]
        if self.dtfreq is not None and self.headerformat == 'dt':
            paramT = timeheader[-1]
            paramdt = timeheader[2]
        else:
            paramT = self.T
            paramdt = self.dt
        paramsRight2 = [paramT,
                        paramdt]
        paramsRight3 = [self.periods,
                        self.divyield,
                        exceldiscdiv,
                        self.rcont]

        workbook = xlsxwriter.Workbook(filepath, {'strings_to_numbers': True})

        # format objects
        formatstring = '#,##0.' + '0'*self.rounding

        headerformat = workbook.add_format({'bold': True,
                                            'align': 'center',
                                            'valign': 'vcenter',
                                            'bottom': 2,
                                            'font_size': 12})

        paramh = workbook.add_format({'bold': True,
                                      'align': 'center',
                                      'valign': 'vcenter',
                                      'fg_color': '#F3B084',
                                      'border': 2,
                                      'font_size': 12})
        paraml = workbook.add_format({'align': 'right',
                                      'valign': 'vcenter',
                                      'fg_color': '#F3B084',
                                      'font_size': 12,
                                      'left': 2,
                                      'bottom': 1})
        paramr = workbook.add_format({'align': 'center',
                                      'valign': 'vcenter',
                                      'fg_color': '#F3B084',
                                      'num_format': formatstring,
                                      'font_size': 12,
                                      'right': 2,
                                      'bottom': 1})
        paramr2 = workbook.add_format({'align': 'center',
                                      'valign': 'vcenter',
                                      'fg_color': '#F3B084',
                                      'font_size': 12,
                                      'right': 2,
                                      'bottom': 1})
        bp = workbook.add_format({'top':
                                      2})

        srowl = workbook.add_format({'align': 'right',
                                     'valign': 'vcenter',
                                     'fg_color': '#E2EFDA',
                                     'top': 1,
                                     'left': 1,
                                     'bottom': 1,
                                     'font_size': 12})
        srowr = workbook.add_format({'align': 'left',
                                     'valign': 'vcenter',
                                     'fg_color': '#E2EFDA',
                                     'top': 1,
                                     'right': 1,
                                     'bottom': 1,
                                     'font_size': 12})

        spce = workbook.add_format({'align': 'center',
                                    'valign': 'vcenter',
                                    'fg_color': '#FFFF00',
                                    'num_format': formatstring,
                                    'left': 1,
                                    'top': 1,
                                    'right': 1,
                                    'bottom': 4,
                                    'font_size': 12})
        spcePrediv = workbook.add_format({'align': 'center',
                                          'valign': 'vcenter',
                                          'fg_color': '#CC5555',
                                          'num_format': formatstring,
                                          'left': 1,
                                          'top': 1,
                                          'right': 1,
                                          'bottom': 4,
                                          'font_size': 12})
        ince = workbook.add_format({'align': 'center',
                                    'valign': 'vcenter',
                                    'fg_color': '#00FF85',
                                    'num_format': formatstring,
                                    'left': 1,
                                    'top': 4,
                                    'right': 1,
                                    'bottom': 4,
                                    'font_size': 12})
        opce = workbook.add_format({'align': 'center',
                                    'valign': 'vcenter',
                                    'fg_color': '#BDD7EE',
                                    'num_format': formatstring,
                                    'left': 1,
                                    'bottom': 1,
                                    'right': 1,
                                    'top': 4,
                                    'font_size': 12})

        fsce = workbook.add_format({'align': 'center',
                                    'valign': 'vcenter',
                                    'fg_color': '#C3FF85',
                                    'num_format': formatstring,
                                    'left': 1,
                                    'top': 4,
                                    'right': 1,
                                    'bottom': 4,
                                    'font_size': 12})

        discheaderL = workbook.add_format({'bold': True,
                                           'align': 'center',
                                           'valign': 'vcenter',
                                           'fg_color': '#D6B4FF',
                                           'left': 2,
                                           'top': 2,
                                           'bottom': 2,
                                           'font_size': 12})
        discheaderR = workbook.add_format({'bold': True,
                                           'align': 'center',
                                           'valign': 'vcenter',
                                           'fg_color': '#D6B4FF',
                                           'right': 2,
                                           'top': 2,
                                           'bottom': 2,
                                           'font_size': 12})
        discL = workbook.add_format({'align': 'center',
                                     'valign': 'vcenter',
                                     'fg_color': '#D6B4FF',
                                     'left': 2,
                                     'top': 1,
                                     'bottom': 1,
                                     'font_size': 12})
        discR = workbook.add_format({'align': 'center',
                                     'valign': 'vcenter',
                                     'fg_color': '#D6B4FF',
                                     'num_format': formatstring,
                                     'right': 2,
                                     'top': 1,
                                     'bottom': 1,
                                     'font_size': 12})


        with workbook as workbook:
            workbook.set_size(2000, 1200)

            # writing overview page
            def overviewPage():
                ov = workbook.add_worksheet('Def. and overview')

                ov.set_column(0, 0, 2.18579234972678)
                ov.set_column(3, 3, 2.18579234972678)
                ov.set_zoom(140)
                ov.set_default_row(16)
                ov.set_column(4, (4 + len(timeheader) - 1), 10.15625)
                ov.freeze_panes(1, 0)
                ov.write_row(0, 4, timeheader, headerformat)
                ov.merge_range(2, 1, 2, 2, 'Parameters', paramh)
                ov.write_column(3, 1, paramsLeft, paraml)
                ov.write_row(greenRow - 1, 1, ["", ""], bp)
                ov.write(greenRow, 1, 'S start row:', srowl)
                ov.write(1, 4, 'Spot', spce)
                ov.write(greenRow, 2, startingRow, srowr)
                if self.treetype == 'normal':
                    if self.showIntrinsic is False:
                        ov.write(2, 4, 'Premium', opce)
                    elif self.showIntrinsic is True:
                        ov.write(2, 4, 'Intrinsic', ince)
                        ov.write(3, 4, 'Premium', opce)
                elif self.treetype == 'fsolution':
                    dt_all = np.arange(0, self.T + self.dt, self.dt)
                    divdt = np.array(self.discdiv)[:, 0]
                    divind = np.abs(np.subtract.outer(dt_all, divdt)).argmin(0)

                    divs = np.array(self.discdiv)[:, 1]
                    divtimes = np.array(self.treeheader)[divind + 1]

                    ov.write(18, 1, 't', discheaderL)
                    ov.write(18, 2, 'Dividends', discheaderR)
                    ov.write_column(19, 1, divtimes, discL)
                    ov.write_column(19, 2, divs, discR)


                    if self.showIntrinsic is False:
                        ov.write(2, 4, 'F-Spot', fsce)
                        ov.write(3, 4, 'Premium', opce)
                    elif self.showIntrinsic is True:
                        ov.write(2, 4, 'F-Spot', fsce)
                        ov.write(3, 4, 'Intrinsic', ince)
                        ov.write(4, 4, 'Premium', opce)
                elif self.treetype == 'nonrecombining':
                    ov.write(2, 4, 'Spot', spce)
                    ov.write(1, 4, 'Pre-Div Spot', spcePrediv)

                    dt_all = np.arange(0, self.T + self.dt, self.dt)
                    divdt = np.array(self.discdiv)[:, 0]
                    divind = np.abs(np.subtract.outer(dt_all, divdt)).argmin(0)

                    divs = np.array(self.discdiv)[:, 1]
                    divtimes = np.array(self.treeheader)[divind + 1]

                    ov.write(18, 1, 't', discheaderL)
                    ov.write(18, 2, 'Dividends', discheaderR)
                    ov.write_column(19, 1, divtimes, discL)
                    ov.write_column(19, 2, divs, discR)

                    if self.showIntrinsic is False:
                        ov.write(3, 4, 'Premium', opce)
                    elif self.showIntrinsic is True:
                        ov.write(3, 4, 'Intrinsic', ince)
                        ov.write(4, 4, 'Premium', opce)
                ov.set_column(1, 1, 10.15625)
                ov.set_column(2, 2, 10.15625)
                ov.set_column(2, 2, 17.5)
                ov.write_column(3, 2, paramsRightDef, paramr)

                return None
            overviewPage()


            def sheetsfundamentalLayout(sheet):
                sheet.set_column(0, 0, 2.18579234972678)
                sheet.set_column(3, 3, 2.18579234972678)
                sheet.set_zoom(140)
                sheet.set_default_row(16)
                sheet.set_column(4, (4 + len(timeheader) - 1), 10.15625)
                sheet.freeze_panes(1, 0)
                sheet.write_row(0, 4, timeheader, headerformat)
                sheet.merge_range(2, 1, 2, 2, 'Parameters', paramh)
                sheet.write_column(3, 1, paramsLeft, paraml)
                sheet.write_row(greenRow - 1, 1, ["", ""], bp)
                sheet.write(greenRow, 1, 'S start row:', srowl)
                sheet.write(1, 4, 'Spot', spce)
                if self.showIntrinsic is False:
                    sheet.write(2, 4, 'Premium', opce)
                elif self.showIntrinsic is True:
                    sheet.write(2, 4, 'Intrinsic', ince)
                    sheet.write(3, 4, 'Premium', opce)
                sheet.set_column(1, 1, 10.15625)
                sheet.set_column(2, 2, 10.15625)
                sheet.write_column(3, 2, paramsRight1, paramr)
                sheet.write_column(9, 2, paramsRight2, paramr2)
                sheet.write_column(11, 2, paramsRight3, paramr2)
                sheet.write(greenRow, 2, startingRow, srowr)

            def writecells(tree, sheet):
                spots = tree.spotsflat
                options = tree.optionsflat
                colind = tree.colIndFlat + 5
                rowind = tree.rowIndFlat + 1

                for collection in list(zip(spots, rowind, colind)):
                    spot = collection[0]
                    row = collection[1]
                    column = collection[2]
                    sheet.write(row, column, spot, spce)

                if self.showIntrinsic is False:
                    for collection in list(zip(options, rowind, colind)):
                        option = collection[0]
                        row = collection[1]
                        column = collection[2]
                        sheet.write(row + 1, column, option, opce)
                elif self.showIntrinsic is True:
                    intrinsics = tree.intrinsicflat
                    for collection in list(zip(options, intrinsics, rowind, colind)):
                        option = collection[0]
                        intrinsic = collection[1]
                        row = collection[2]
                        column = collection[3]
                        sheet.write(row + 1, column, intrinsic, ince)
                        sheet.write(row + 2, column, option, opce)


            def sheetsfundamentalLayoutF(sheet):
                sheet.set_column(0, 0, 2.18579234972678)
                sheet.set_column(3, 3, 2.18579234972678)
                sheet.set_zoom(140)
                sheet.set_default_row(16)
                sheet.set_column(4, (4 + len(timeheader) - 1), 10.15625)
                sheet.freeze_panes(1, 0)
                sheet.write_row(0, 4, timeheader, headerformat)
                sheet.merge_range(2, 1, 2, 2, 'Parameters', paramh)
                sheet.write_column(3, 1, paramsLeft, paraml)
                sheet.write_row(greenRow - 1, 1, ["", ""], bp)
                sheet.write(greenRow, 1, 'S start row:', srowl)
                sheet.write(1, 4, 'Spot', spce)
                dt_all = np.arange(0, self.T + self.dt, self.dt)
                divdt = np.array(self.discdiv)[:, 0]
                divind = np.abs(np.subtract.outer(dt_all, divdt)).argmin(0)

                divs = np.array(self.discdiv)[:, 1]
                divtimes = np.array(self.treeheader)[divind + 1]

                sheet.write(18, 1, 't', discheaderL)
                sheet.write(18, 2, 'Dividends', discheaderR)
                sheet.write_column(19, 1, divtimes, discL)
                sheet.write_column(19, 2, divs, discR)
                if self.showIntrinsic is False:
                    sheet.write(2, 4, 'F-Spot', fsce)
                    sheet.write(3, 4, 'Premium', opce)
                elif self.showIntrinsic is True:
                    sheet.write(2, 4, 'F-Spot', fsce)
                    sheet.write(3, 4, 'Intrinsic', ince)
                    sheet.write(4, 4, 'Premium', opce)
                sheet.set_column(1, 1, 10.15625)
                sheet.set_column(2, 2, 10.15625)
                sheet.write_column(3, 2, paramsRight1, paramr)
                sheet.write_column(9, 2, paramsRight2, paramr2)
                sheet.write_column(11, 2, paramsRight3, paramr2)
                sheet.write(greenRow, 2, startingRow, srowr)

            def writecellsF(tree, sheet):
                spots = tree.spotsflat
                Ftree = tree.Ftreeflat
                options = tree.optionsflat

                colind = tree.colIndFlat + 5
                colindF = tree.colIndFlatF + 5
                rowind = tree.rowIndFlat + 1
                rowindS = tree.rowIndFlatS + 1
                rowindF = tree.rowIndFlatF + 1

                for collection in list(zip(spots, rowind, colind)):
                    spot = collection[0]
                    row = collection[1]
                    column = collection[2]
                    sheet.write(row, column, spot, spce)

                if self.showIntrinsic is False:
                    for collection in list(zip(spots, options, colind, rowind, rowindS)):
                        spot, option, col, row, rowS = collection
                        sheet.write(rowS, col, spot, spce)
                        sheet.write(row+1, col, option, opce)

                    for collectionF in list(zip(Ftree, colindF, rowindF)):
                        Fspot, col, row = collectionF
                        sheet.write(row, col, Fspot, fsce)
                elif self.showIntrinsic is True:
                    intrinsics = tree.intrinsicflat
                    for collection in list(zip(spots, intrinsics, options, colind, rowind, rowindS)):
                        spot, intrinsic, option, col, row, rowS = collection
                        sheet.write(rowS, col, spot, spce)
                        sheet.write(row+1, col, intrinsic, ince)
                        sheet.write(row+2, col, option, opce)

                    for collectionF in list(zip(Ftree, colindF, rowindF)):
                        Fspot, col, row = collectionF
                        sheet.write(row, col, Fspot, fsce)


            def sheetsfundamentalLayoutNonrec(sheet):
                sheet.set_column(0, 0, 2.18579234972678)
                sheet.set_column(3, 3, 2.18579234972678)
                sheet.set_zoom(140)
                sheet.set_default_row(16)
                sheet.set_column(4, (4 + len(timeheader) - 1), 10.15625)
                sheet.freeze_panes(1, 0)

                # TODO - merge header cells in nonRec

                sheet.write_row(0, 4, timeheader, headerformat)
                sheet.merge_range(2, 1, 2, 2, 'Parameters', paramh)
                sheet.write_column(3, 1, paramsLeft, paraml)
                sheet.write_row(greenRow - 1, 1, ["", ""], bp)
                sheet.write(greenRow, 1, 'S start row:', srowl)
                sheet.write(1, 4, 'Pre-Div Spot', spcePrediv)
                sheet.write(2, 4, 'Spot', spce)
                dt_all = np.arange(0, self.T + self.dt, self.dt)
                divdt = np.array(self.discdiv)[:, 0]
                divind = np.abs(np.subtract.outer(dt_all, divdt)).argmin(0)

                divs = np.array(self.discdiv)[:, 1]
                divtimes = np.array(self.treeheader)[divind + 1]

                sheet.write(18, 1, 't', discheaderL)
                sheet.write(18, 2, 'Dividends', discheaderR)
                sheet.write_column(19, 1, divtimes, discL)
                sheet.write_column(19, 2, divs, discR)
                if self.showIntrinsic is False:
                    sheet.write(3, 4, 'Premium', opce)
                elif self.showIntrinsic is True:
                    sheet.write(3, 4, 'Intrinsic', ince)
                    sheet.write(4, 4, 'Premium', opce)
                sheet.set_column(1, 1, 10.15625)
                sheet.set_column(2, 2, 10.15625)
                sheet.write_column(3, 2, paramsRight1, paramr)
                sheet.write_column(9, 2, paramsRight2, paramr2)
                sheet.write_column(11, 2, paramsRight3, paramr2)
                sheet.write(greenRow, 2, startingRow, srowr)

            def writecellsNonrec(tree, sheet):
                import itertools


                colours = ['#000000',
                           '#0000FF',
                           '#FF0000',
                           '#00E100',
                           '#FF40FF',
                           '#FF8C00',
                           '#535093',
                           '#00B2FF',
                           '#885B1F']
                spotformats, spotPredivformats, intrinsicformats, optionformats = [], [], [], []
                for col in colours:
                    spotformats.append(workbook.add_format({'align': 'center',
                                                            'valign': 'vcenter',
                                                            'fg_color': '#FFFF00',
                                                            'border_color': col,
                                                            'num_format': formatstring,
                                                            'left': 2,
                                                            'top': 2,
                                                            'right': 2,
                                                            'bottom': 4,
                                                            'font_size': 12}))
                    spotPredivformats.append(workbook.add_format({'align': 'center',
                                                                  'valign': 'vcenter',
                                                                  'fg_color': '#CC5555',
                                                                  'border_color': col,
                                                                  'num_format': formatstring,
                                                                  'left': 2,
                                                                  'top': 2,
                                                                  'right': 2,
                                                                  'bottom': 4,
                                                                  'font_size': 12}))
                    intrinsicformats.append(workbook.add_format({'align': 'center',
                                                                 'valign': 'vcenter',
                                                                 'fg_color': '#00FF85',
                                                                 'border_color': col,
                                                                 'num_format': formatstring,
                                                                 'left': 2,
                                                                 'top': 4,
                                                                 'right': 2,
                                                                 'bottom': 4,
                                                                 'font_size': 12}))
                    optionformats.append(workbook.add_format({'align': 'center',
                                                              'valign': 'vcenter',
                                                              'fg_color': '#BDD7EE',
                                                              'border_color': col,
                                                              'num_format': formatstring,
                                                              'left': 2,
                                                              'bottom': 2,
                                                              'right': 2,
                                                              'top': 4,
                                                              'font_size': 12}))

                spotPredivformats.append(spotPredivformats[0])
                spotPredivformats.remove(spotPredivformats[0])

                spotIter = itertools.cycle(spotformats)
                spotPredivIter = itertools.cycle(spotPredivformats)
                intrinsicIter = itertools.cycle(intrinsicformats)
                optionIter = itertools.cycle(optionformats)

                spots = tree.spots
                predivspots = tree.predivspots
                options = tree.options

                rowind = tree.rowIndFlat
                colind = tree.colIndFlat

                rowindPrediv = tree.predivrows
                colindPrediv = tree.predivcols

                # write spots
                if self.showIntrinsic is False:
                    for spot, option, row, column in zip(spots, options, rowind, colind):
                        spotformat = next(spotIter)
                        optionformat = next(optionIter)

                        for s, o, r, c in zip(spot[1], option[1], row[1], column[1]):
                            sheet.write(r + 1, c + 5, s, spotformat)
                            sheet.write(r + 2, c + 5, o, optionformat)

                    for prespot, prow, pcol in zip(predivspots, rowindPrediv, colindPrediv):
                        spotpredivsformat = next(spotPredivIter)

                        for ps, pr, pc in zip(prespot[1], prow[1], pcol[1]):
                            sheet.write(pr + 1, pc + 5, ps, spotpredivsformat)
                elif self.showIntrinsic is True:
                    intrinsics = tree.intrinsics
                    for spot, option, intrinsic, row, column in zip(spots, options, intrinsics, rowind, colind):
                        spotformat = next(spotIter)
                        intrinsicformat = next(intrinsicIter)
                        optionformat = next(optionIter)

                        for s, o, i, r, c in zip(spot[1], option[1], intrinsic[1], row[1], column[1]):
                            sheet.write(r + 1, c + 5, s, spotformat)
                            sheet.write(r + 2, c + 5, i, intrinsicformat)
                            sheet.write(r + 3, c + 5, o, optionformat)

                    for prespot, prow, pcol in zip(predivspots, rowindPrediv, colindPrediv):
                        spotpredivsformat = next(spotPredivIter)

                        for ps, pr, pc in zip(prespot[1], prow[1], pcol[1]):
                            sheet.write(pr + 1, pc + 5, ps, spotpredivsformat)


            if self.treetype == 'normal':
                for treeName, treeObj in self.trees.items():
                    sheet = workbook.add_worksheet(treeName)

                    sheetsfundamentalLayout(sheet)
                    writecells(treeObj, sheet)
            elif self.treetype == 'fsolution':
                for treeName, treeObj in self.trees.items():
                    sheet = workbook.add_worksheet(treeName)

                    sheetsfundamentalLayoutF(sheet)
                    writecellsF(treeObj, sheet)
            elif self.treetype == 'nonrecombining':
                for treeName, treeObj in self.trees.items():
                    sheet = workbook.add_worksheet(treeName)

                    sheetsfundamentalLayoutNonrec(sheet)
                    writecellsNonrec(treeObj, sheet)

        print(f'File was made at: {filepath}')


    def __call__(self, toreturn, **kwargs):
        """
        Purpose:
            -> override parameters
            -> choose parameters to return

        Time call:
            new T       -> keep dt -> new periods
                        -> keep dtfreq

            new dt      -> keep T -> new periods
                -> discard dtfrq
            new periods -> keep T -> new dt
                -> discard dtfrq

            2 or 3 new  -> act as __init__

            -> if for example 'T' is passed in call (kwargs)
                -> fetch dt from self.kwunion and pass those two

        Make Dataframes:
            -> if Dataframes are to be made from __call__, you must parse 'dfs' in toreturn parameter
        """
        toreturnParams = ['direc',
                          'dirfile',
                          'fname',
                          'spot',
                          'spotarr',
                          'strike',
                          'vola',
                          'r',
                          'discountRate',
                          'discountDiv',
                          'discountRateMinusDiv',
                          'u',
                          'd',
                          'q',
                          'T',
                          'dt',
                          'periods',
                          'dtfreq',
                          'headerformat',
                          'treeheader',
                          'divyield',
                          'discdiv',
                          'treetype',
                          'trees',
                          'ecTree',
                          'epTree',
                          'acTree',
                          'apTree',
                          'ecOptionPrice',
                          'epOptionPrice',
                          'acOptionPrice',
                          'apOptionPrice',
                          'showIntrinsic',
                          'ecOptionPriceBS',
                          'epOptionPriceBS',
                          'dfs',
                          'nonrec',
                          'makedfs',
                          'rcont',
                          'preex',
                          'collapsed',
                          'makedfs',
                          'udfunc',
                          'paramsUnion',
                          'kwunion',
                          'rounding',
                          'updoind',
                          'maketrees']

        if toreturn is None and kwargs is None:
            self.help(['callable'])
            print('\n\n\n')
            print("Possible parameters for 'toreturn' input (either in a list or alone):")
            for i in toreturnParams:
                print(i)
            print("\nPossible parameters for 'toreturn' input (either in a list or alone) ↑")
            return None
        else:
            T = kwargs.get('T', False)
            dt = kwargs.get('dt', False)
            periods = kwargs.get('periods', False)
            dtfreq = kwargs.get('dtfreq', None)

            dfcalled = False
            if toreturn == 'dfs' or 'dfs' in toreturn:
                dfcalled = True

            dummyobject = None
            if T is not False and not any([dt, periods]):
                dt = self.dt
                periods = None
                dtfreq = self.dtfreq
                timeparams = dict(T = T, dt = dt, periods = periods, dtfreq = dtfreq)
                newkwargs = {**kwargs, **timeparams}
                dummyobject = binomialTrees(self.kwunion, **newkwargs, called = True, dfcalled = dfcalled)
            elif dt is not False and not any([T, periods]):
                T = self.T
                periods = None
                timeparams = dict(T = T, dt = dt, periods = periods, dtfreq = dtfreq)
                newkwargs = {**kwargs, **timeparams}
                dummyobject = binomialTrees(self.kwunion, **newkwargs, called = True, dfcalled = dfcalled)
            elif periods is not False and not any([T, dt]):
                T = self.T
                dt = None
                timeparams = dict(T = T, dt = dt, periods = periods, dtfreq = dtfreq)
                newkwargs = {**kwargs, **timeparams}
                dummyobject = binomialTrees(self.kwunion, **newkwargs, called = True, dfcalled = dfcalled)
            elif not any([T, dt, periods]):
                dummyobject = binomialTrees(self.kwunion, **kwargs, called = True, dfcalled = dfcalled)


            if isinstance(toreturn, (list, tuple)):
                if 'dfs' in toreturn:
                    toreturn.remove('dfs')
                    stuff = dict()

                    for name, df in dummyobject.trees.items():
                        stuff.setdefault(name, df)

                    def retfunc(attr):
                        stuff.setdefault(attr, getattr(dummyobject, attr))
                    list(map(retfunc, toreturn))
                else:
                    attrsdict = dict()
                    def retfunc(attr):
                        attrsdict.setdefault(attr, getattr(dummyobject, attr))
                    list(map(retfunc, toreturn))
                    stuff = attrsdict
            else:
                if toreturn == 'dfs':
                    stuff = dict()
                    for name, df in dummyobject.trees.items():
                        stuff.setdefault(name, df)
                else:
                    stuff = getattr(dummyobject, toreturn)
            return stuff


    @staticmethod
    def help(tohelp = None):
        parameters = """
        ################################    Possible parameters    #################################
        direc:          Default: None       -> string
        folname:        Default: None       -> string
        fname:          Default: 'binotree' -> string

        spot:           Default: None       -> numerical
        strike:         Default: None       -> numerical

        T:              Default: None       -> numerical
        dt:             Default: None       -> numerical
        periods:        Default: None       -> integer
        dtfreq:         Default: None       -> string

        r:              Default: 0.00       -> numerical
        rcont:          Default: True       -> boolean
        divyield:       Default: 0.00       -> numerical

        vola:           Default: None       -> numerical
        u:              Default: None       -> numerical
        d:              Default: None       -> numerical
        udfunc:         Default: udfunc     -> callable function
        
        riskNeutral:    Default: False      -> boolean

        discdiv:        Default: None       -> dict or paired tuple/list
        nonrec:         Default: False      -> boolean
        preex:          Default: True       -> boolean

        collapsed:      Default: False      -> boolean
        write:          Default: False      -> boolean
        
        maketrees:      Default: None       -> list
        headerformat:   Default: None       -> string
        rounding:       Default: 2          -> integer
        makedfs:        Default: True       -> boolean
        """

        parametersExamples = """
        ##############################    Parameter specification    ###############################
        direc:          Default: None
            -> chosen directory
        folname:        Default: None
            -> optional folder for output in chosen directory
        fname:          Default: 'binotree'
            -> filename (not including filetype suffix)
        
        
        spot:
            -> Numerical value for current spot, e.g. 100
        strike:
            -> Numerical value for strike on options, e.g. 95
        
        
        T:
            -> Time to maturity in terms of years, e.g. 
                                                    -> 30 days: 30/365
                                                    -> 3 weeks: 3/52
                                                    -> 3 months: 3/12
                                                    -> 2 years: 2
        dt:
            -> Length of each period in terms of years, e.g.
                                                    -> 1 day: 1/365
                                                    -> 1 week: 1/52
                                                    -> 1 month: 1/12
                                                    -> 1 year: 1
        periods:
            -> Number (integer) of periods in binomial tree, e.g. 3
        dtfreq:
            -> Needed for formatting of header if wanted
                -> one of 'd', 'w', 'm'
        
        
        r:
            -> Yearly risk-free interest ratein decimal percentage, e.g. 
                                                                For 4% you would parse -> 0.04
        rcont:
            -> Continous or discrete compunding interest rate and dividend yield, e.g.
                                                                True -> Continous
                                                                False -> Discrete
        divyield:
            -> Yearly dividend yield in decimal percentage, e.g. for 4% you would parse -> 0.04
        
        
        vola:
            -> Volatility in terms of yearly standard deviation (sigma), e.g. 0.20
        u:
            -> Up factor for each node movement up, e.g. 1.10
        d:
            -> Down factor for each node movement down, e.g. 0.90
        udfunc:
            IMPORTANT:
            -> The function must include **kwargs argument
            -> All parameters must/should be parsed as keywords
            Can take any of these parameters:
            -> vola, T, dt, periods, r, divyield, discountRate, discountDiv, discountRateMinusDiv, spot, strike
            Must return the two parameters:
            -> u, d
        
        
        discdiv:
            -> paired tuple/lists, e.g.
                -> [(1/12, 2), (4/12, 5)]
                -> [[1/12, 2], [4/12, 5]]
        nonrec:
            -> determines if tree is non-recombining, if False(default) -> F-solution
        preex:
            -> if american options can be executed before ex-div spot, e.g.
                                                                -> True -> use pre ex-div spot
        
        
        collapsed:
            -> if True -> no empty cells between nodes
        write:
            -> if True -> write excel output directly from construction
        
        
        maketrees:
            -> list of options to calculate e.g. -> maketrees = ['ec', 'ep', 'ac', 'ap']
        headerformat:  
            -> Determines if header is formated in terms of periods or 'actual' time, e.g.
                                                                        -> string -> 'periods' 
                                                                         or
                                                                        -> string -> 'dt' 
        rounding:
            -> integer specifying rounding for decimals, e.g. 2
        makedfs:
            -> True or False -> determining if dataframes are to be constructed (speeds up code when False)
        """

        updownpriohelp = """
        ############################ up, down, volatility specification ############################
        vola        -> u, d = udfunc(vola, dt)
        u           -> d = 1/u
                    -> vola = (np.log(u) - np.log(d)) / (2 * np.sqrt(dt))
        d           -> u = 1/d
                    -> vola = (np.log(u) - np.log(d)) / (2 * np.sqrt(dt))
        u, d        -> vola = (np.log(u) - np.log(d)) / (2 * np.sqrt(dt))
        vola, u     -> d = u / (np.exp(2 * vola * np.sqrt(dt)))
        vola, d     -> u = d * (np.exp(2 * vola * np.sqrt(dt)))
        
        # u and d dominates if all 3 are passed
        all 3       -> vola = (np.log(u) - np.log(d)) / (2 * np.sqrt(dt))
        """

        timehelp = """
        ###################################  time specification  ###################################
        T, dt              -> periods = T/dt
        T, periods         -> dt = T/periods
        dt, periods        -> T = dt * periods
        T, dt, periods     -> dt = T/periods
        """

        udfunchelp = """
        udfunc:
            IMPORTANT:
                -> The function must include **kwargs argument
                -> All parameters must/should be parsed as keywords
            Can take any of these parameters:
                -> vola, T, dt, periods, r, divyield, discountRate, discountDiv, discountRateMinusDiv, spot, strike
            Must return the two parameters:
                -> u, d
        """

        nonepassed = """
        Parameters must be passed as a dictionary and/or keywords 
        (keywords override passed params dictionary)
        """

        callablehelp = """
        Purpose:
            -> override parameters
            -> choose parameters to return

        Time call:
            new T       -> keep dt -> new periods
                        -> keep dtfreq

            new dt      -> keep T -> new periods
                -> discard dtfrq
            new periods -> keep T -> new dt
                -> discard dtfrq

            2 or 3 new  -> act as __init__

            -> if for example 'T' is passed in call (kwargs)
                -> fetch dt from self.kwunion and pass those two

        Make Dataframes:
            -> if Dataframes are to be made from __call__, you must parse 'dfs' in toreturn parameter
        """

        helpstatement = {'params': parameters,
                         'paramsexamples': parametersExamples,
                         'updown': updownpriohelp,
                         'time': timehelp,
                         'udfunc': udfunchelp,
                         'noinput': nonepassed,
                         'callable': callablehelp}

        if tohelp is None:
            for i in helpstatement.values():
                print(i)
                print('\n\n§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§'
                      '§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§')
        else:
            for i in tohelp:
                print(helpstatement[i])



# TODO list:
#  - make factorized method for calculating deltas for different S at t=0
#  - plot method for intrinsic/B-S/bino price + deltas/gammas/etc.















































































