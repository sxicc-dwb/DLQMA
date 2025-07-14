
from airPLS import airPLS

from tqdm import tqdm

import os
import numpy as np
import nmrglue as ng
def read_bruker_h_base(nmr_path, bRaw=False, bMinMaxScale=False):
    nmr_path = os.path.normpath(nmr_path)


    if bRaw:
        dic, fid = ng.fileio.bruker.read(f'{nmr_path}/1')
        zero_fill_size = dic['acqus']['TD']
        fid = ng.bruker.remove_digital_filter(dic, fid)
        fid = ng.proc_base.zf_size(fid, zero_fill_size)
        fid = ng.proc_base.fft(fid)
    else:

        path_2 = os.path.join(nmr_path, '2', 'pdata', '1')
        path_1 = os.path.join(nmr_path, '1', 'pdata', '1')

        if os.path.exists(path_2):
            dic, fid = ng.fileio.bruker.read_pdata(path_2)
        elif os.path.exists(path_1):
            dic, fid = ng.fileio.bruker.read_pdata(path_1)
        else:
            raise FileNotFoundError(f"No valid data directory found in {path_1} or {path_2}")

        zero_fill_size = dic['acqus']['TD']
        offset = (float(dic['acqus']['SW']) / 2) - (float(dic['acqus']['O1']) / float(dic['acqus']['BF1']))
        start = float(dic['acqus']['SW']) - offset
        end = -offset
        step = float(dic['acqus']['SW']) / zero_fill_size
        ppms = np.arange(start, end, -step)[:zero_fill_size]


        baseline = airPLS(fid, lambda_=100, porder=1, itermax=15)
        fid = fid - baseline




        if os.path.exists(path_2):

            print("Shape of fid before signal processing:", fid.shape)

            o = np.min(np.where(np.round(ppms, 3) == 4.910))
            p = np.min(np.where(np.round(ppms, 3) == 4.845))
            a = np.min(np.where(np.round(ppms, 3) == 3.340))
            s = np.min(np.where(np.round(ppms, 3) == 3.260))
            fid[o:p] = 0
            fid[a:s] = 0
            print("Shape of fid after signal processing:", fid.shape)
        else:

            print("Shape of fid before signal processing:", fid.shape)


            q = np.min(np.where(np.round(ppms, 3) == 3.375))
            w = np.min(np.where(np.round(ppms, 3) == 3.300))
            t = np.min(np.where(np.round(ppms, 3) == 2.520))
            y = np.min(np.where(np.round(ppms, 3) == 2.475))
            fid[q:w] = 0
            fid[t:y] = 0  # dmso
            print("Shape of fid after signal processing:", fid.shape)


        if bMinMaxScale:
            fid = fid / np.max(fid)
        v = np.max(np.where(np.round(ppms, 3) == 10.700))
        b = v + 32724

        return {'name': nmr_path.split(os.sep)[-1], 'ppm': ppms[v:b], 'fid': fid[v:b], 'bRaw': bRaw}



def read_bruker_hs_base(data_folder, bRaw, bMinMaxScale, bDict):
    if bDict:
        spectra = {}
    else:
        spectra = []
    for name in tqdm(os.listdir(data_folder), desc="Read Bruker H-NMR files"):
        nmr_path = os.path.normpath(os.path.join(data_folder, name))
        s = read_bruker_h_base(nmr_path, bRaw, bMinMaxScale)
        if bDict:
            spectra[s['name']] = s
        else:
            spectra.append(s)
    return spectra
