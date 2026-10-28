import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from os import path
from pathlib import Path
from copy import deepcopy as copy
from matplotlib import rc # , checkdep_usetex

usetex = False # checkdep_usetex(True)
# rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=usetex)

scatt_keys = [*list(map(lambda z: "infS"+str(z), range(2))),
            *list(map(lambda z: "infSp"+str(z), range(2)))]
xsdf_keys = ['infTot', 'infAbs', 'infDiffcoef', 'infTranspxs', 'infCapt',
            'infRemxs', 'infFiss', 'infNsf']
ene_keys = ['infNubar', 'infInvv', 'infKappa', 'infInvv',  'infChit',
            'infChip', 'infChid']

serp_keys = [*scatt_keys, *xsdf_keys, *ene_keys, 'infFlx']

sumxs = ['Tot', 'Abs', 'Remxs']
indepdata = ['Capt', 'Fiss', 'S0', 'Nubar', 'Chid', 'Chip']
basicdata = ['Fiss', 'Nubar', 'S0', 'Chit']
kinetics = ['lambda', 'beta']
alldata = list(set([*sumxs, *indepdata, *basicdata, *kinetics]))

collapse_xs = ['Fiss', 'Capt', *list(map(lambda z: "S"+str(z), range(0, 2))),
               *list(map(lambda z: "Sp"+str(z), range(0, 2))), 'Invv', 'Diffcoef']
collapse_xsf = ['Nubar', 'Chid', 'Chit', 'Chip', 'FissEn']

units = {'Chid': '-', 'Chit': '-', 'Chip': '-', 'Tot': 'cm^{-1}',
        'Capt': 'cm^{-1}', 'Abs': 'cm^{-1}', 'Fiss': 'cm^{-1}',
        'Nsf': 'cm^{-1}', 'Remxs': 'cm^{-1}', 'Transpxs': 'cm^{-1}',
        'FissEn': 'MeV', 'S': 'cm^{-1}', 'Nubar': '-', 'Invv': 's/cm',
        'Difflenght': 'cm', 'Diffcoef': 'cm', 'Flux': 'a.u.'}

xslabels = {'Chid': 'delayed fiss. emission spectrum', 'Chit': 'total fiss. emission spectrum',
            'Chip': 'prompt fiss. emission spectrum', 'Tot': 'Total xs',
            'Capt': 'Capture xs', 'Abs': 'Absorption xs', 'Fiss': 'Fission xs',
            'Nsf': 'Fiss. production xs', 'Remxs': 'Removal xs', 'Transpxs': 'Transport xs',
            'FissEn': 'Fiss. energy', 'S': 'Scattering xs', 'Nubar': 'neutrons by fission',
            'Invv': 'Inverse velocity', 'Difflenght': 'Diff. length', 'Diffcoef': 'Diff. coeff.',
            'Flux': 'Flux spectrum'}

class MGC():
    """Create material regions with multi-group constants.

    Parameters
    ----------
    uniName: str
        Universe name.
    energygrid: iterable
        Energy group structure containing nE+1 group boundaries where nE is the
        number of energy groups.
    datapath: str, optional
        Path to the file containing the data, by default ``None``. If ``None``,
        data are taken from the local database.
    egridname : str, optional
        Name of the energy group structure, by default ``None``.
    h5file: object
        h5 group from .h5 files.
    reader: str``
        Type or reader. It can be ``'serpent'``, ``'json'`` or ``'txt'``.
    serpres: :class:`serpentTools.ResultsReader`
        Object created parsing the _res.m Serpent file with ``serpentTools``
    basename : bool or str
        if not ``False``, base name is used to compose the filenames, which 
        needs to be in the form <basename>_Tf_XXX_Tc_XXX. 
    temp: tuple
        if ``basename`` is not None, directories in the form 
        "Tf_{}_Tc_{}" are searched  and "Tf_{}_Tc_{}" suffix 
        is attached to the file name
    datacheck: bool, optional
        Flag to check and ensure data consistency, by default ``True``.
    init: bool, optional
        Flag to initialise the object as empty, by default ``False``

    Attributes
    ----------
    nE: int
        Number of energy groups.
    egridname: str
        Name of the energy grid.
    energygrid: list
        List of energy group boundaries.
    UniName: str
        Name of the material.
    NPF: int
        Number of neutron precursors families.
    L: int
        Scattering anisotropy order
    Tot: np.array
        1D array of length ``nE`` with the total cross section in cm^-1.
    Abs: np.array
        1D array of length ``nE`` with the absorption cross section in cm^-1.
    Capt: np.array
        1D array of length ``nE`` with the capture cross section in cm^-1.
    Fiss: np.array
        1D array of length ``nE`` with the fission cross section in cm^-1.
    Remxs: np.array
        1D array of length ``nE`` with the removal cross section in cm^-1.
    Transpxs: np.array
        1D array of length ``nE`` with the transport cross section in cm^-1.
     Nsf: np.array
        1D array of length ``nE`` with the fission production cross section in cm^-1.
    Diffcoef: np.array
        1D array of length ``nE`` with the diffusion coefficient in cm.
    Difflength: np.array
        1D array of length ``nE`` with the diffusion length in cm.
    S0: np.array
        2D array of size ``(nE, nE)`` with the scattering matrix cross section in cm^-1.
    Chit: np.array
        1D array of size ``nE`` with the total fission emission spectrum.
    Chip: np.array
        1D array of size ``nE`` with the prompt fission emission spectrum.
    Chid: np.array
        1D array of size ``nE`` with the delayed fission emission spectrum.
    Nubar: np.array
        1D array of size ``nE`` with the number of neutrons emitted by fission.
    Invv: np.array
        1D array of size ``nE`` with the inverse of the neutron velocity in s/cm.
    lambda: np.array
        1D array of size ``NPF`` with the decay constants of the precursors.
    beta: np.array
        1D array of size ``NPF`` with the delayed neutron fracitons.
    Flx: np.array
        1D array of size ``nE`` with the flux energy spectrum in arbitrary units.

    
    """

    def __init__(self, uniName=None, energygrid=None, datapath=None,
                 egridname=None, h5file=None, reader='json', serpres=None,
                 basename=False, temp=False, datacheck=True, init=False, 
                 ERANOS=False, P1consistent=False):
        if h5file:
            if isinstance(h5file, dict):
                for k, v in h5file.items():
                    if type(v) is bytes:
                        v = v.decode()
                    self.__dict__[k] = v
            elif isinstance(h5file, str):
                raise OSError('To do')
            else:
                msg = f"h5file must be dict or str, not {type(h5file)}"
                raise TypeError(msg)
        else:
            if init:
                return

            if temp:
                Tf, Tc = temp
            nE = len(energygrid)-1
            egridname = egridname if egridname else f"{nE}G"
            pwd = Path(__file__).parent

            if serpres is None:
                if datapath is None:
                    pwd = Path(__file__).parent.parent.parent
                    if 'coreutils' not in str(pwd):
                        raise OSError(f'Check coreutils tree for NEdata: {pwd}')
                    # look into default NEdata dir
                    datapath = str(pwd.joinpath('NEdata', f'{egridname}'))
                    filename = uniName
                elif not path.isdir(datapath):
                    pwd = Path(__file__).parent.parent.parent
                    if 'coreutils' not in str(pwd):
                        raise OSError(f'Check coreutils tree for NEdata: {pwd}')
                    filename = copy(datapath)
                    datapath = str(pwd.joinpath('NEdata', f'{egridname}'))
                else:
                    if basename:
                        filename = basename
                    else:
                        filename = uniName
                if reader == 'json':
                    # look into json folder
                    if Path(path.join(datapath, "json")).exists():
                        jpath = path.join(datapath, "json")
                    else:
                        jpath = datapath
                    # check for temperatures
                    if temp:
                        dirTfTc = f"Tf_{Tf}_Tc_{Tc}"
                        dirTcTf = f"Tc_{Tc}_Tf_{Tf}"
                        if Path(path.join(jpath, dirTfTc)).exists():
                            jpath = path.join(jpath, dirTfTc, filename)
                        elif Path(path.join(jpath, dirTcTf)).exists():
                            path.join(jpath, dirTcTf, filename)

                        # if dirTcTf not in filename and dirTfTc not in filename:
                    else:
                        fname = path.join(jpath, filename)

                    if '.json' not in str(filename):
                        fname = f'{str(filename)}.{reader}'
                    else:
                        fname = filename

                    fname = path.join(jpath, fname)
                    if Path(fname).exists():
                        self._readjson(fname)
                    else:
                        logging.debug(f'{fname} not found!')
                        reader = 'txt'

                if reader == 'txt':
                    # look into txt folder
                    if Path(path.join(datapath, "txt")).exists():
                        tpath = path.join(datapath, "txt")
                    else:
                        tpath = datapath

                    if temp:
                        dirTfTc = f"Tf_{Tf}_Tc_{Tc}"
                        dirTcTf = f"Tc_{Tc}_Tf_{Tf}"
                        if Path(path.join(tpath, dirTfTc)).exists():
                            spath = path.join(tpath, dirTfTc, filename)
                        elif Path(path.join(tpath, dirTcTf)).exists():
                            path.join(tpath, dirTcTf, filename)

                    else:
                        fname = path.join(tpath, filename)    
                    
                    if '.txt' not in str(fname):
                        fname = f'{str(fname)}.{reader}'
                    else:
                        fname = filename

                    fname = path.join(tpath, fname)
                    if Path(fname).exists():
                        self._readtxt(fname, nE)
                    else:
                        raise OSError(f'{fname} not found!')
            else:
                self._readserpentres(serpres, uniName, nE, egridname)
            
            self.nE = nE
            self.egridname = egridname
            self.energygrid = energygrid
            self.UniName = uniName
            self.P1consistent = P1consistent
            self.ERANOS = ERANOS

            try:
                self.NPF = (self.beta).size
            except AttributeError:
                logging.info('Kinetic parameters not available!')
                self.NPF = 1

            # --- complete data and perform sanity check
            L = 0
            datastr = list(self.__dict__.keys())
            # //2 since there are 'S' and 'Sp'
            S = sum('S' in s for s in datastr)//2
            self.L = S if S > L else L  # get maximum scattering order
            if datacheck:
                self.datacheck()

    def _readjson(self, path):
        """
        Read data from json file.

        Parameters
        ----------
        path: str
            Path to json file.

        Returns
        -------
        None.

        """
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            if isinstance(v, list):
                self.__dict__[k] = np.asarray(v)
            else:
                self.__dict__[k] = v

    def _readserpentres(self, serpres, uniName, nE, egridname):
        """Transform :class:`serpentTools.ResultsReader` object 
            into :class:``coreutils.NEMaterial`` object.

        Parameters
        ----------
        serpres : dict
            Dictionary of :class:`serpentTools.ResultsReader` objects.
        uniName : str
            Name of the material.
        nE: int
            Number of energy groups.
        egridname: str
            Name of the energy grid.

        Raises
        ------
        OSError
            If the material indicated by ``UniName`` is not available.
        OSError
            If the number of energy groups indicated by ``nE`` is not available.
        """        
        data = None
        for res in serpres.values():
            try:
                data = res.getUniv(uniName, 0, 0, 0)
            except KeyError:
                continue

        if data is None:
            raise OSError(f'{uniName} data not available in Serpent files!')

        if len(data.infExp['infAbs']) != nE:
            raise OSError(f'{uniName} energy groups do not match with \
                          input grid!')

        selfdic = self.__dict__
        for my_key in serp_keys:
            if my_key.startswith('infS') or my_key.startswith('infSp'):
                vals = np.reshape(data.infExp[my_key], (nE, nE), order='F')
            else:
                vals = data.infExp[my_key]

            if 'Kappa' in my_key:
                selfdic['FissEn'] = vals
            else:
                selfdic[my_key.split('inf')[1]] = vals

        # kinetics parameters
        selfdic['beta'] = res.resdata['fwdAnaBetaZero'][::2]
        selfdic['beta_tot'] = selfdic['beta'][0]
        selfdic['beta'] = selfdic['beta'][1:]
        # this to avoid confusion with python lambda function
        selfdic['lambda'] = res.resdata['fwdAnaLambda'][::2]
        selfdic['lambda_tot'] = selfdic['lambda'][0]
        selfdic['lambda'] = selfdic['lambda'][1:]

    def _readtxt(self, fname, nE):
        """
        Parse the material data from a .txt file.

        Macro-group constants are parsed from a formatted file with column-wise
        data separated by headers beginning with "#" and the name of the data:
            * Tot: total cross section [cm^-1]
            * Transpxs: transport cross section [cm^-1]
                        It is defined as total_xs-avg_direction*scattering_xs
                        according to P1 approximation.
            * Diffcoef: diffusion coefficient [cm]
                        It is defined as 1/(3*Transpxs).
            * Abs: absorption cross section [cm^-1]
                   It is the sum of Capt and Fiss cross sections.
            * Capt: capture cross section [cm^-1]
            * Fiss: fission cross section [cm^-1]
            * Remxs: removal cross section [cm^-1]
                    It is the sum of Abs and group-removal.
            * Chit: total emission spectrum [-]
            * Chip: prompt emission spectrum [-]
            * Chid: delayed emission spectrum [-]
            * Nsf: fission production cross section [cm^-1]
            * Nubar: neutron multiplicities [-]
            * FissEn: average fission deposited heat [MeV]
            * Invv: particle inverse velocity [s/cm]
            * S0, S1, S2,... : scattering matrix cross section [cm^-1]
            * Sp0, Sp1, Sp2,... : scattering production matrix cross section
                                [cm^-1]
            * beta: delayed neutron fractions [-]
            * lambda: precursors families decay constant [-]

        Parameters
        ----------
        fname : string
            Material data file name.
        nE : int
            Number of energy groups.

        Returns
        -------
        None.

        """
        selfdic = self.__dict__
        G = None

        lines = open(fname).read().split('\n')

        for il, line in enumerate(lines):

            if line.startswith('#'):
                key = (line.split('#')[1]).strip()
                matrix = None

            elif line == '':
                continue

            else:

                data = np.asarray([float(val) for val in line.split()])
                if G is None:
                    G = len(data)

                if G != nE:
                    raise OSError('Number of groups in line %g is not \
                                  consistent!', il)

                if key.startswith('S') or key.startswith('Sp'):
                    # multi-line data (scattering matrix)
                    if matrix is None:
                        matrix = np.asarray(data)
                    else:
                        matrix = np.c_[matrix, data]

                    if matrix.shape == (G, G):
                        selfdic[key] = matrix.T
                    elif matrix.shape == (G, ):
                        selfdic[key] = matrix
                else:
                    # single-line data (scattering matrix)
                    selfdic[key] = np.asarray(data)

    def getxs(self, key, pos1=None, pos2=None):
        """Get material data (for a certain energy group, if needed).

        Parameters
        ----------
        key : str
            User selected nuclear data.
        pos1 : int, optional
            Departure energy group for scattering matrix. If not provided,
            data over all the energy groups are returned.
            The default is ``None``.
        pos2 : int, optional
            Arrival energy group for scattering matrix. If not provided,
            data over all the energy groups are returned.
            The default is ``None``.

        Returns
        -------
        vals : numpy.ndarray
            1-D ``numpy.ndarray`` with G/NPF (groups) rows.

        """
        if pos1 is None and pos2 is None:
            try:
                vals = self.__dict__[key]
            except KeyError:
                if key.startswith('S') or key.startswith('Sp'):
                    # set higher moments to zero if not available
                    vals = self.__dict__['S0']*0
                else:
                    raise OSError(f'{key} data not available!')
        else:
            if key.startswith('S') or key.startswith('Sp'):
                if pos2 is None:
                    raise OSError('Two coordinates needed for %s data' % key)
                else:
                    vals = self.__dict__[key][pos1, pos2]
            else:
                vals = self.__dict__[key][pos1]

        return vals

    def plot(self, what, depgro=False, family=1, ax=None, figname=None,
             normalise=True, **kwargs):
        """Plot multi-group data from the object.

        Parameters
        ----------
        what : str
            Data to be plotted.
        depgro : int, optional
            Departure energy group, by default ``False``. This argument is needed to plot
            the scattering cross section.
        family : int, optional
            Number of neutron precursor family, by default 1
        ax : `matplotlib.axes.Axes`, optional
            Ax on which to plot the data, by default `None`. If not provided,
            a new figure is created.
        figname : str, optional
            Figure name with its extension, by default ``None``
        normalise : bool, optional
            Normalisation flag, by default ``True``

        Raises
        ------
        OSError
            If the ``depgro`` argument is not provided when the data to be plotted
            is the scattering matrix.
        """        
        E = self.energygrid
        ax = ax or plt.gca()
        xs = self.__dict__[what]
        whatlabel = xslabels[what]
        if 'S' in what:
            if depgro:
                xs = xs[depgro, :]
                whatlabel = f'{xslabels[what]} from g={depgro}'
            else:
                raise OSError('Material.plot: depgro variable needed!')
        elif what == 'Chid':
            xs = xs[family-1, :]
        elif what == 'Flx':
            if normalise:
                u = np.log(self.energygrid/self.energygrid[0])
                xs = xs/np.diff(-u)


        if 'Chi' in what:
            xs = xs/xs.dot(-np.diff(E))

        if 'S' in what:
            uom = units['S']
        else:
            uom = units[what]

        if 'Flx' in what and normalise:
            whatlabel = 'Flux per unit lethargy'

        if usetex:
            uom = f'$\\rm {uom}$'

        if 'label' not in kwargs.keys():
            kwargs['label'] = what

        ax.stairs(xs, edges=E, baseline=None, **kwargs)
        ax.set_xlabel('E [MeV]')
        ax.set_ylabel(f'{whatlabel} [{uom}]')
        ax.set_xscale('log')
        if what not in ['Nubar', 'Chid', 'Chip', 'Chit']:
            ax.set_yscale('log')

        plt.grid(which='both', alpha=0.2)
        if figname:
            plt.tight_layout()
            plt.savefig(f"{figname}.png")

    def perturb(self, what, howmuch, depgro=None, sanitycheck=True):
        """Perturb material composition.

        Parameters
        ----------
        what : str
            Type of perturbation. If ``what="density"``, the density of the 
            material is perturbed, otherwise the other data can be perturbed by
            indicating the data. For instance, ``what="Fiss"`` or ``what="Nubar"`.
        howmuch : list or float
            Magnitude of the perturbation. If list, its length must be equal to
            ``nE``, and the perturbation is applied to each group. If it is a float,
            the perturbation is applied to the material density. 
        depgro : int, optional
            Departure energy group, by default ``False``. This argument is needed to perturb
            the scattering cross section.
        sanitycheck: bool, optional
            Flag to check and ensure data consistency, by default ``True``.

        Returns
        -------
        None.

        """
        if what == 'density':
            densdata = ['Capt', 'Fiss', *list(map(lambda z: "S"+str(z), range(self.L))),
                        *list(map(lambda z: "Sp"+str(z), range(self.L)))]
            if howmuch < 0:
                raise OSError('Cannot apply negative density perturbations!')
            for xs in densdata:
                self.__dict__[xs][:] = self.__dict__[xs][:]*howmuch
        else:
            depgro = depgro-1 if depgro is not None else depgro
            for g in range(self.nE):
                # no perturbation
                if howmuch[g] == 0:
                    continue

                mydic = self.__dict__
                if what in indepdata:
                    # update perturbed parameter
                    if depgro is None:
                        delta = mydic[what][g]*howmuch[g]
                        mydic[what][g] = mydic[what][g]+delta
                    else:  # select departure group for scattering matrix
                        delta = mydic[what][depgro]*howmuch[depgro]
                        mydic[what][depgro] = mydic[what][depgro]+delta

                    # select case to ensure data consistency
                    if what == 'Fiss':
                        self.Nsf[g] = self.Nubar[g]*mydic[what][g]
                    elif what == 'Nubar':
                        self.Nsf[g] = self.Fiss[g]*mydic[what][g]
                        # computesumxs = False
                    elif what.startswith('Chi'):
                        if what in ['Chit']:
                            mydic[what] = mydic[what]*(1+delta)
                        else:
                            raise OSError('Delayed/prompt spectra \
                                           perturbation still missing!')
                    elif what == 'Diffcoef':
                        # Hp: change in diffcoef implies change in capture
                        delta = 1/(3*mydic[what][g])-self.Transpxs[g]
                    elif what == 'S0':
                        # change higher moments, if any
                        for ll in range(self.L): # FIXME
                            R = (mydic[what][g]/mydic[what][g]-delta)
                            key = 'S%d' % ll
                            mydic[key][depgro][g] = mydic[key][depgro][g]*R

                else:
                    if sanitycheck:
                        raise OSError(f'{what} cannot be perturbed \
                                      directly!')
                    else:
                        # update perturbed parameter
                        if depgro is None:
                            delta = mydic[what][g]*howmuch[g]
                            mydic[what][g] = mydic[what][g]+delta
                        else:  # select departure group for scattering matrix
                            delta = mydic[what][depgro]*howmuch[g]
                            mydic[what][depgro] = mydic[what][depgro]+delta

        if sanitycheck:
            # force normalisation
            if abs(sum(self.Chit) - 1) > 1E-4:
                if np.any(self.Chit == 0) :
                    pass
                else:
                    self.Chit = self.Chit/sum(self.Chit)

            self.datacheck()

    def datacheck(self):
        """Check data consistency and add missing data.

        Parameters
        ----------
        ``None``.

        Returns
        -------
        ``None``.

        """
        E = self.energygrid
        datadic = self.__dict__
        datavail = copy(list(datadic.keys()))
        # check basic reactions existence
        for s in basicdata:
            if s not in datavail:
                raise OSError(f'{s} is missing in {self.UniName} data!')
        # --- compute in-group scattering
        InScatt = np.diag(self.S0)
        sTOT = self.S0.sum(axis=0) if len(self.S0.shape) > 1 else self.S0
        # --- compute fission production cross section
        self.Nsf = self.Fiss*self.Nubar
        # --- compute missing sum reactions
        if 'Capt' in datavail:
            self.Abs = self.Fiss+self.Capt
        elif 'Abs' in datavail:
            self.Capt = self.Abs-self.Fiss
        elif 'Tot' in datavail:
            self.Capt = self.Tot-sTOT-self.Fiss
            self.Abs = self.Fiss+self.Capt

        self.Remxs = self.Abs+sTOT-InScatt
        self.Tot = self.Remxs+InScatt
        # ensure non-zero total XS
        self.bad_data = False
        if np.count_nonzero(self.Tot) != self.Tot.shape[0]:
            self.bad_data = True
        self.Tot[self.Tot <= 0] = 1E-8

        # ensure velocity consistency only if data are not collapsed
        if not hasattr(self, "fine_energygrid"):
            if not hasattr(self, "Invv"):
                avgE = 1/2*(E[:-1]+E[1:])*1.602176634E-13  # J
                v = np.sqrt(2*avgE/1.674927351e-27)
                self.Invv = 1/(v*100)  # s/cm

        if self.ERANOS: # ensure velocity is recomputed as in ERANOS calculation
            avgE = 1/2*(E[:-1]+E[1:])*1.602176634E-13  # J
            v = np.sqrt(2*avgE/1.674927351e-27)
            self.Invv = 1/(v*100)  # s/cm
            # The factor 9.8E6 used by ERANOS is obtained with the formula reported below
            # print(f"{np.sqrt(1.602176634E-13/1.674927351e-27)*100:1.1e}")

        # --- compute diffusion coefficient and transport xs
        if self.P1consistent:
            if 'S1' in datavail:
                # --- compute transport xs (derivation from P1)
                self.Transpxs = self.Tot-self.S1.sum(axis=0)
                # mu0 = self.S1/self.S0
                self.Diffcoef = 1/(3*self.Transpxs)
            elif 'Diffcoef' in datavail:
                self.Transpxs = 1/(3*self.Diffcoef)
            elif 'Transpxs' in datavail:
                self.Transpxs[self.Transpxs <= 0] = 1E-8
                self.Diffcoef = 1/(3*self.Transpxs)
            else:
                self.Transpxs = self.Tot
                self.Diffcoef = 1/(3*self.Transpxs)
        else:
            self.Diffcoef = 1/(3*self.Tot)
            if 'S1' in datavail:
                # --- compute transport xs (derivation from P1)
                self.Transpxs = self.Tot-self.S1.sum(axis=0)
            elif 'Transpxs' in datavail:
                self.Transpxs[self.Transpxs <= 0] = 1E-8
            else:
                self.Transpxs = self.Tot

        # --- compute diffusion length
        self.Remxs[self.Remxs <= 0] = 1E-8 # avoid huge diff. coeff. and length
        self.DiffLength = np.sqrt(self.Diffcoef/self.Remxs)
        # --- compute mean free path
        self.MeanFreePath = 1/self.Tot.max()
        # --- ensure consistency kinetic parameters (if fissile medium)
        self.Fiss[self.Fiss <= 5E-7] = 0
        isFiss = self.Fiss.max() > 0
        if isFiss:
            # FIXME FIXME check Serpent RSD and do correction action
            # self.Chit[self.Chit <= 1E-4] = 0
            if abs(sum(self.Chit) - 1) > 1E-4:
                logging.warning(f'Total fission spectra in {self.UniName} not normalised!'
                                'Forcing normalisation...')
            # ensure pdf normalisation
            self.Chit /= sum(self.Chit)
            if "FissEn" not in self.__dict__.keys():
                self.FissEn = np.array([200]*self.nE)
        else:
            self.FissEn = np.array([0]*self.nE)

        kincons = True
        for s in kinetics:
            if s not in datavail:
                kincons = False
                self.__dict__[s] = [0]

        if kincons:
            try:
                self.beta_tot = sum(self.beta)

                if isFiss:
                    if len(self.Chid.shape) == 1:
                        # each family has same emission spectrum
                        # FIXME FIXME check Serpent RSD and do correction action
                        self.Chid[self.Chid <= 1E-4] = 0
                        self.Chid /= sum(self.Chid)
                        self.Chid = np.asarray([self.Chid]*self.NPF)
                    elif self.Chid.shape != (self.NPF, self.nE):
                        raise NEMaterialError(f'Delayed fiss. spectrum should be \
                                        ({self.NPF}, {self.nE})')

                    # FIXME FIXME check Serpent RSD and do correction action
                    self.Chip[self.Chip <= 1E-4] = 0

                    try:
                        for g in range(0, self.nE):
                            chit = (1-sum(self.beta))*self.Chip[g] + \
                                    np.dot(self.beta, self.Chid[:, g])
                            if abs(self.Chit[g]-chit) > 1E-4:
                                raise NEMaterialError()
                    except NEMaterialError:
                        logging.warning(f'Fission spectra or delayed fractions'
                                        f' in {self.UniName} not consistent! '
                                        'Forcing consistency acting on chi-prompt...')
                    else:
                        self.Chip = (self.Chit-np.dot(self.beta, self.Chid))/(1-sum(self.beta))
                        for g in range(0, self.nE):
                            chit = (1-sum(self.beta))*self.Chip[g] + \
                                    np.dot(self.beta, self.Chid[:, g])
                            if abs(self.Chit[g]-chit) > 1E-4:
                                raise NEMaterialError("Normalisation failed!")
                else:
                    self.Chit = np.zeros((self.nE, ))
                    self.Chip = np.zeros((self.nE, ))
                    self.Chid = np.zeros((self.NPF, self.nE))

            except AttributeError as err:
                if 'Chid' in str(err) or 'Chip' in str(err):
                    self.Chid = np.asarray([self.Chit]*self.NPF)
                    self.Chip = self.Chit
                else:
                    logging.warning(err)

            # ensure pdf normalisation
            if isFiss:
                self.Chip /= sum(self.Chip)
                for p in range(self.NPF):
                    self.Chid[p, :] /= sum(self.Chid[p, :])
        else:
            if isFiss:
                self.Chip = self.Chit
                self.Chid = np.tile(self.Chit, (self.NPF, self.nE))
            else:
                self.Chit = np.zeros((self.nE, ))
                self.Chip = np.zeros((self.nE, ))
                self.Chid = np.zeros((self.NPF, self.nE))

    def to_json(self, fname=None):
        """Dump object to json file.

        Parameters
        ----------
        fname: str, optional
            Filename, by default ``None``.

        Returns
        -------
        None.

        """
        if fname is None:
            f'{self.UniName}_{self.egridname}.json'
        tmp = {}
        with open(fname, 'w') as f:

            for k, v in self.__dict__.items():
                if isinstance(v, (np.ndarray)):
                    tmp[k] = v.tolist()
                else:
                    tmp[k] = v

            json.dump(tmp, f, sort_keys=True, indent=10)

    def collapse(self, fewgrp, spectrum=None, egridname=None, datacheck=False,
                 infspectrum=None, adjoint=None, adjointInvv=None):
        """Collapse in energy the multi-group data.

        Parameters
        ----------
        fewgrp : iterable
            Few-group structure to perform the collapsing.
        spectrum: array, optional
            Spectrum to perform the energy collapsing, by default ``None``. If ``None``,
            the ``Flx`` attribute is used as a weighting spectrum.
        egridname: str, optional
            Name of the energy grid, by default ``None``.
        adjoint: array, optional
            Spectrum of the adjoint flux to perform the energy collapsing preserving the
            total importance, by default ``None``.
        adjointInvv: array, optional
            Spectrum of the adjoint flux to perform the energy collapsing of the inverse
            of the neutron velocity preserving the total importance, by default ``None``.

        Raises
        ------
        OSError
            Collapsing failed: weighting flux missing in {}.

        Returns
        -------
        None.

        """
        if spectrum is not None:
            flx = spectrum
            if infspectrum is not None:
                flxinf = infspectrum
                self.FlxInf = infspectrum
            else:
                flxinf = spectrum
                self.FlxInf = flxinf
        else:
            if 'Flx' not in self.__dict__.keys():
                raise OSError('Collapsing failed: weighting flux missing in'
                              f'{self.UniName}')
            else:
                flx = self.Flx
                if not hasattr(self, "FlxInf"):
                    flxinf = self.Flx

        multigrp = self.energygrid
        if isinstance(fewgrp, list):
            fewgrp = np.asarray(fewgrp)
        # ensure descending order
        fewgrp = fewgrp[np.argsort(-fewgrp)]
        H = len(multigrp)-1
        G = len(fewgrp)-1
        # sanity check
        if G > H:
            raise OSError(f'Collapsing failed: few-group structure should \
                          have less than {H} group')
        if multigrp[0] != fewgrp[0] or multigrp[0] != fewgrp[0]:
            raise OSError('Collapsing failed: few-group structure'
                          'boundaries do not match with multi-group'
                          'one')
        for ig, g in enumerate(fewgrp):
            if g not in multigrp:
                raise OSError(f'Group boundary n.{ig}, {g} MeV not present in fine grid!')

        iS = 0
        collapsed = {}
        if adjoint is None:
            collapsed['Flx'] = np.zeros((G, ))
        else:
            collapsed['TotImportance'] = np.zeros((G, ))

        collapsed['FlxInf'] = np.zeros((G, ))
        for g in range(G):
            # select fine groups in g
            G1, G2 = fewgrp[g], fewgrp[g+1]
            iE = np.argwhere(np.logical_and(multigrp[iS:] < G1,
                                            multigrp[iS:] >= G2))[-1][0]+iS
            # compute flux in g
            if adjoint is None:
                NC = sum(flx[iS:iE])
                collapsed['Flx'][g] = NC
            else:
                NC = np.dot(adjoint[iS:iE], flx[iS:iE])
                collapsed['TotImportance'][g] = NC

            collapsed['FlxInf'][g] = sum(flxinf[iS:iE])
            # --- collapse
            for key, v in self.__dict__.items():
                # --- cross section and inverse of velocity
                if key in collapse_xs:
                    # --- preallocation
                    dims = (G, G) if 'S' in key else (G, )
                    if g == 0:
                        collapsed[key] = np.zeros(dims)

                    if len(dims) == 1:
                        if key == 'Diffcoef':
                            if self.P1consistent:
                                v = self.Transpxs
                            else:
                                v = self.Tot
                            v = 1/3/v
                        if adjoint is None: # preserve reaction rates and neutron density
                            if adjointInvv is not None and key == 'Invv':
                                NTI = np.dot(adjointInvv[iS:iE], flx[iS:iE])
                                collapsed[key][g] = np.divide(adjointInvv[iS:iE].dot(flx[iS:iE]*v[iS:iE]), NTI, where=NC!=0)
                            else:
                                collapsed[key][g] = np.divide(flx[iS:iE].dot(v[iS:iE]), NC, where=NC!=0)
                        else: # preserve adjoint weighted reaction rates and total importance
                            collapsed[key][g] = np.divide(adjoint[iS:iE].dot(flx[iS:iE]*v[iS:iE]), NC, where=NC!=0)
                    else:
                        # --- scattering
                        iS2 = 0
                        for g2 in range(G):  # arrival group
                            I1, I2 = fewgrp[g2], fewgrp[g2+1]
                            iE2 = np.argwhere(np.logical_and
                                              (multigrp[iS2:] < I1,
                                               multigrp[iS2:] >= I2))
                            iE2 = iE2[-1][0]+iS2
                            s = v[iS:iE, iS2:iE2].sum(axis=0)
                            if adjoint is None:
                                NCS = sum(flx[iS2:iE2])
                                collapsed[key][g][g2] = np.divide(flx[iS2:iE2].dot(s), NCS, where=NCS!=0)
                            else:
                                NCS = np.dot(adjoint[iS2:iE2], flx[iS2:iE2])
                                collapsed[key][g][g2] = np.divide(adjoint[iS2:iE2].dot(flx[iS2:iE2]*s), NCS, where=NCS!=0)

                            iS2 = iE2
                # --- fission-related data
                elif key in collapse_xsf:
                    if not self.isfiss():
                        if key == 'Chid':
                            collapsed[key] = np.zeros((self.NPF, G))
                        else:
                            collapsed[key] = np.zeros((G, ))
                        continue
                    if adjoint is None:
                        fissrate = flx[iS:iE]*self.Fiss[iS:iE]
                        FRC = sum(fissrate)
                    else:
                        fissrate = adjoint[iS:iE]*flx[iS:iE]*self.Fiss[iS:iE]
                        FRC = sum(fissrate)
                    if key == 'Chid':
                        if g == 0:
                            collapsed[key] = np.zeros((self.NPF, G))
                        for p in range(self.NPF):
                            collapsed[key][p, g] = sum(v[p, iS:iE])
                    else:
                        if g == 0:
                            collapsed[key] = np.zeros((G, ))

                        if key in ['Chit', 'Chip']:
                            collapsed[key][g] = sum(v[iS:iE])
                        else:
                            collapsed[key][g] = np.divide(fissrate.dot(v[iS:iE]), FRC, where=FRC!=0)
                else:
                    continue
            iS = iE

        collapsed['Transpxs'] = 1/(3*collapsed['Diffcoef'])
        # overwrite data
        self.fine_energygrid = self.energygrid+0
        self.energygrid = fewgrp
        self.nE = G
        self.egridname = egridname if egridname else f'{G}G'
        for key in self.__dict__.keys():
            if key in collapsed.keys():
                self.__dict__[key] = collapsed[key]

        # ensure data consistency
        if datacheck:
            self.datacheck()

    def isfiss(self):
        """Assess whether the material is fissile"""
        return self.Fiss.max() > 0


class NEMaterialError(Exception):
    pass
