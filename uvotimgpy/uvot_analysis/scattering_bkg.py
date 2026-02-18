from __future__ import annotations

import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.time import Time
from astropy.stats import sigma_clip

from uvotimgpy.config import paths
from uvotimgpy.utils.image_operation import upscale_mean_fill, stack_images, crop_image, upscale_mean_fill, shrink_valid_image
from uvotimgpy.base.region import RegionConverter, select_mask_regions, get_exclude_region, RegionSelector, RegionCombiner
from uvotimgpy.base.instruments import normalize_filter_name
from uvotimgpy.uvot_image.motion_smear_reducer import for_sk_like



PathLike = Union[str, os.PathLike]

class UVOTXformError(RuntimeError):
    pass


def _p(x: PathLike) -> Path:
    return x if isinstance(x, Path) else Path(x)



def clear_tmpdir(tmpdir: PathLike) -> None:
    """
    Remove all files/subdirectories under tmpdir, but keep the directory itself.
    Intended for HEADAS_TMPDIR cleanup between runs.
    """
    tmp = _p(tmpdir)
    tmp.mkdir(parents=True, exist_ok=True)
    for p in tmp.iterdir():
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
        except Exception:
            pass

def find_attfile(obsdir: PathLike) -> Path:
    """
    Find a Swift attitude history file under obsdir/auxil.
    Heuristic: prefer *pat*.fits, then *att*.fits, then *.att, then any *att* under auxil.
    """
    obsdir = _p(obsdir)
    aux = obsdir / "auxil"
    if not aux.exists():
        raise FileNotFoundError(f"Cannot find auxil directory: {aux}")

    cands = []
    cands += sorted(aux.glob("*pat*.fits"))
    cands += sorted(aux.glob("*att*.fits"))
    cands += sorted(aux.glob("*.att"))
    cands += sorted(aux.glob("*pat*.fits.gz"))
    cands += sorted(aux.glob("*att*.fits.gz"))
    if not cands:
        cands += sorted(aux.glob("*att*"))
    if not cands:
        raise FileNotFoundError(f"No attitude-like file found under {aux}")

    return cands[0]


def make_two_hdu_raw_like(
    template_fits: PathLike,
    replacement_data: np.ndarray,
    out_fits: PathLike,
    *,
    ext_index: int = 1,
    overwrite: bool = True,
    sanity_check: bool = True,
    ) -> Path:
    template_fits = _p(template_fits)
    out_fits = _p(out_fits)
    img = np.asarray(replacement_data)
    if img.ndim != 2:
        raise ValueError(f"replacement_data must be 2D, got {img.shape}")

    with fits.open(template_fits, memmap=False) as hdul:
        h0 = hdul[0]
        h = hdul[ext_index]
        if h.data is None:
            raise ValueError(f"Template HDU[{ext_index}] has no data")
        if h.data.shape != img.shape:
            raise ValueError(f"Shape mismatch: template ext={ext_index} {h.data.shape} vs replacement {img.shape}")

        new0 = fits.PrimaryHDU(data=h0.data, header=h0.header.copy())
        new1 = fits.ImageHDU(data=np.array(img), header=h.header.copy(), name=h.name)
        fits.HDUList([new0, new1]).writeto(out_fits, overwrite=overwrite)

    if sanity_check:
        with fits.open(out_fits, memmap=False) as h:
            # 用 allclose 而不是全等，避免 float 写入/读取的细微差
            if not np.allclose(h[1].data, img, equal_nan=True):
                raise RuntimeError(
                    "Sanity check failed: ext=1 in written FITS != provided replacement array.\n"
                    "This means you did not actually write the intended scattering image."
                )

    return out_fits

def get_pointing_from_template(template_fits: str) -> tuple[float, float, float]:
    with fits.open(template_fits, memmap=False) as h:
        # 常见在主HDU或图像HDU；两边都试一下
        for hdr in (h[1].header, h[0].header):
            if "RA_OBJ" in hdr and "DEC_OBJ" in hdr and "PA_PNT" in hdr:
                return float(hdr["RA_OBJ"]), float(hdr["DEC_OBJ"]), float(hdr["PA_PNT"])
    raise KeyError("Cannot find RA_OBJ/DEC_OBJ/PA_PNT in template FITS headers")


def run_swiftxform_noninteractive_stream(
    infile_twohdu: PathLike,
    outfile: PathLike,
    *,
    obsdir: PathLike,                 # 用作 cwd，解析相对路径（不写入原始目录）
    attfile: PathLike,                # 必须给，避免交互
    template_fits: PathLike,          # 用于从 header 读取 RA_PNT/DEC_PNT/PA_PNT，避免交互输入 RA/DEC
    tempdir: PathLike,                # HEASoft 中间文件目录（HEADAS_TMPDIR）
    method: str = "DEFAULT",
    to: str = "SKY",
    teldeffile: str = "CALDB",
    chatter: int = 5,
    clobber: bool = True,
    silent: bool = True,              # True: 不输出任何中间日志
    ) -> Path:
    infile_twohdu = _p(infile_twohdu)
    outfile = _p(outfile)
    obsdir = _p(obsdir)
    attfile = _p(attfile)
    tempdir = _p(tempdir)
    tempdir.mkdir(parents=True, exist_ok=True)

    ra_pnt, dec_pnt, roll_pnt = get_pointing_from_template(str(template_fits))

    cmd = [
        "swiftxform",
        f"infile={infile_twohdu}+1",
        f"outfile={outfile}",
        f"attfile={attfile}",
        f"method={method}",
        f"to={to}",
        f"teldeffile={teldeffile}",
        f"ra={ra_pnt}",
        f"dec={dec_pnt}",
        f"roll={roll_pnt}",
        f"chatter={int(chatter)}",
        f"clobber={'yes' if clobber else 'no'}",
        "mode=h",  # 关键：避免交互提示
    ]

    env = os.environ.copy()
    env["HEADAS_TMPDIR"] = str(tempdir)
    # 避免交互提示，方便jupyter notebook运行
    # 关键：让 HEASoft 在无TTY环境也不去 /dev/tty
    env["HEADASNOQUERY"] = "1"
    env["HEADASPROMPT"] = "/dev/null"
    # 建议：隔离本地 PFILES，避免 learned 参数/权限问题
    pfiles = tempdir / "pfiles"
    pfiles.mkdir(parents=True, exist_ok=True)
    syspfiles = Path(env["HEADAS"]) / "syspfiles"
    env["PFILES"] = f"{pfiles};{syspfiles}"

    if silent:
        proc = subprocess.Popen(
            cmd,
            cwd=str(obsdir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=False,
        )
        ret = proc.wait()
    else:
        print("\nRUN:", " ".join(cmd), flush=True)
        print("CWD:", obsdir, flush=True)
        print("HEADAS_TMPDIR:", tempdir, "\n", flush=True)

        proc = subprocess.Popen(
            cmd,
            cwd=str(obsdir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
        ret = proc.wait()

    if not outfile.exists():
        raise UVOTXformError(
            "swiftxform did not create output file.\n"
            f"Return code: {ret}\n"
            f"Expected output: {outfile}"
        )

    return outfile


def read_fits_image_data(fits_path: PathLike, ext: Union[int, str] = 1) -> np.ndarray:
    fits_path = _p(fits_path)
    with fits.open(fits_path, memmap=False) as h:
        if h[ext].data is None:
            raise ValueError(f"{fits_path} ext={ext} has no data")
        return np.array(h[ext].data)



def scattering_to_sky_array(
    *,
    raw_fits: PathLike,      # 建议用 *_rw.img 作为模板
    scattering_data: np.ndarray,
    workdir: PathLike,                # 必须在原始数据树之外
    obsdir: PathLike,                 # 原始观测目录根（包含 auxil/）
    label: Optional[str],      # 输出文件名前缀，例如 "05000951002"
    ext: int = 1,
    delete_raw_like: bool = False,    # 是否删除 <label>_raw_like.fits
    move_sk_to_archive: bool = False, # 是否把 <label>_sk_like.fits 移动到 archive_dir
    archive_dir: Optional[PathLike] = None,
    attfile: Optional[PathLike] = None,
    heasoft_tmpdir: Optional[PathLike] = None,  # HEADAS_TMPDIR 固定目录；不填默认 '/Users/zexixing/Documents/heasoft_tmp'
    silent: bool = True,             # True: 不打印 swiftxform 输出；False: 打印（调试用）
    ) -> Tuple[np.ndarray, Dict[str, Path]]:
    """
    Return (sky_array, paths_dict)

    paths_dict contains:
      - raw_like_fits: the generated input FITS path (may be deleted if delete_raw_like=True)
      - sk_like_fits:  the generated SKY FITS path (may be moved if move_sk_to_archive=True)
      - sk_like_final: final location of SKY FITS (workdir or archive)
      - attfile: attitude file used
      - heasoft_tmpdir: tmpdir used for HEADAS_TMPDIR (kept, but contents cleared)
    """
    workdir_p = _p(workdir)
    workdir_p.mkdir(parents=True, exist_ok=True)

    obsdir_p = _p(obsdir)

    #if label is None:
    #    # Default label: use obsdir basename if it looks like an obsid
    #    cand = obsdir_p.name
    #    label = cand if cand else "uvot"

    raw_like_name = f"{label}_raw_like.fits"
    sk_like_name = f"{label}_sk_like.fits"

    raw_like_fits = workdir_p / raw_like_name
    sk_like_fits = workdir_p / sk_like_name

    if attfile is None:
        attfile_p = find_attfile(obsdir_p)
    else:
        attfile_p = _p(attfile)

    # Fixed temp directory for HEASoft intermediates; keep directory, clear contents per run.
    tmpdir = _p(heasoft_tmpdir) if heasoft_tmpdir is not None else '/Users/zexixing/Documents/heasoft_tmp'
    clear_tmpdir(tmpdir)

    # 1) Write raw-like FITS with ext=1 replaced by scattering
    make_two_hdu_raw_like(raw_fits, scattering_data, raw_like_fits, ext_index=ext, sanity_check=True)

    # 2) swiftxform (non-interactive): write intermediates under HEADAS_TMPDIR=tmpdir
    run_swiftxform_noninteractive_stream(
        raw_like_fits,
        sk_like_fits,
        obsdir=obsdir_p,
        attfile=attfile_p,
        template_fits=raw_fits,
        tempdir=tmpdir,
        method="DEFAULT",
        to="SKY",
        teldeffile="CALDB",
        chatter=5,
        clobber=True,
        silent=silent,
    )

    # 3) Clean intermediates but keep tmpdir
    clear_tmpdir(tmpdir)

    # 4) Read output SKY array
    #sky = read_fits_image_data(sk_like_fits, ext=1)

    # 5) Optionally move SKY FITS to archive
    #sk_like_final = sk_like_fits
    if move_sk_to_archive:
        if archive_dir is None:
            raise ValueError("move_sk_to_archive=True requires archive_dir")
        arch = _p(archive_dir)
        arch.mkdir(parents=True, exist_ok=True)
        target = arch / sk_like_fits.name
        # overwrite target if exists
        if target.exists():
            target.unlink()
        shutil.move(str(sk_like_fits), str(target))
        #sk_like_final = target

    # 6) Optionally delete raw-like input FITS
    if delete_raw_like:
        try:
            raw_like_fits.unlink(missing_ok=True)
        except Exception:
            pass

    #return sky, {
    #    "raw_like_fits": raw_like_fits,
    #    "sk_like_fits": sk_like_fits,
    #    "sk_like_final": sk_like_final,
    #    "attfile": attfile_p,
    #    "heasoft_tmpdir": _p(tmpdir),
    #}

def get_scattering_sk(*, data_dir, obsid, filt, temporary_dir,
                      scattering_dir=None,
                      delete_raw_like=True):
    """
    data_dir_path: e.g. /Users/zexixing/Library/CloudStorage/OneDrive-Personal/ZexiWork/data/Swift/C_2025N1
    """
    #scattering = fits.getdata(scattering_path, ext=0)
    filt_filename = normalize_filter_name(filt, output_format='filename')
    data_dir = Path(data_dir)
    data_path = paths.get_subpath(data_dir, f'{obsid}', 'uvot', 'image', f'sw{obsid}{filt_filename}_rw.img.gz')
    obs_dir = paths.get_subpath(data_dir, f'{obsid}')
    if scattering_dir is None:
        package_path = paths.package_uvotimgpy
        scattering_dir = paths.get_subpath(package_path, 'auxil', 'bkg_valid')
    scattering_path = paths.get_subpath(scattering_dir, f'{filt_filename}.fits')
    sk_like_path = paths.get_subpath(temporary_dir, f'{obsid}_{filt}_sk_like.fits')
    rw_image_shape = fits.getdata(data_path, ext=1).shape
    scattering_data = fits.getdata(scattering_path, ext=0)
    scattering_data[scattering_data == 0] = np.nan
    if rw_image_shape == (1024, 1024):
        datatype = 'image'
    elif rw_image_shape == (2048, 2048):
        datatype = 'event'
        scattering_data = upscale_mean_fill(scattering_data, n=2, conserve_flux=True)
    else:
        raise ValueError(f"Unsupported raw image shape: {rw_image_shape}")
    scattering_to_sky_array(
        raw_fits=data_path,
        scattering_data=scattering_data,
        workdir=temporary_dir,
        obsdir=obs_dir,
        ext=1,
        label=f'{obsid}_{filt}',
        delete_raw_like=delete_raw_like,
        move_sk_to_archive=False,
    )

    with fits.open(sk_like_path, mode='update') as hdul:
        hdul[1].data[hdul[1].data == 0] = np.nan
        hdul[0].header['EXT1NAME'] = ('BKG', 'Scattering background image')
        hdul.flush()
    
    print(f"scattering_sk saved for {obsid} {filt}")

def align_scattering(temporary_path, archive_dir, data_dir, obsid, filt, ext, sk_coord, target_coord):
    with fits.open(temporary_path, mode='readonly') as hdul:
        temporary_data = hdul[1].data.copy()
    hdr = fits.Header()
    filt_filename = normalize_filter_name(filt, output_format='filename')
    data_path = paths.get_subpath(data_dir, f'{obsid}', 'uvot', 'image', f'sw{obsid}{filt_filename}_rw.img.gz')
    raw_hdr = fits.getheader(data_path, ext=1)
    date_start = Time(raw_hdr['DATE-OBS'])
    date_end = Time(raw_hdr['DATE-END'])
    mid_time = date_start + 0.5 * (date_end - date_start)
    hdr['MID_TIME'] = str(mid_time.isot)
    hdr['PLATESCL'] = (1.004, 'Platescale in arcsec/pixel')
    hdr['COLPIXEL'] = (target_coord[0], 'Target X position in Python coordinates')
    hdr['ROWPIXEL'] = (target_coord[1], 'Target Y position in Python coordinates')
    hdr['DS9XPIX'] = (target_coord[0] + 1, 'Target X position in DS9 coordinates')
    hdr['DS9YPIX'] = (target_coord[1] + 1, 'Target Y position in DS9 coordinates')
    hdr['HISTORY'] = f'Created by Zexi Xing'
    hdr['REDUCER'] = 'scattering_bkg.py'
    primary_hdu = fits.PrimaryHDU(header=hdr)
    #image_hdu = fits.ImageHDU(data=temporary_data)
    temporary_data = crop_image(image=temporary_data, old_target_coord=sk_coord, new_target_coord=target_coord, fill_value=np.nan)
    image_hdu = fits.ImageHDU(data=temporary_data, name='BKG')
    new_hdul = fits.HDUList([primary_hdu, image_hdu])
    archive_path = paths.get_subpath(archive_dir, f'{obsid}_{ext}_{filt_filename}.fits')
    new_hdul.writeto(archive_path, overwrite=True, output_verify="fix")
    print(f"scattering_sk saved for {obsid} {ext} {filt_filename} (aligned)")


def _scattering_sk_event_hdr(processing_info):
    hdr = fits.Header()
    hdr['MID_TIME'] = processing_info['mid_time']
    hdr['NSEGMENT'] = (processing_info['num_segments'], 'Total number of time segments')
    hdr['STACKMTH'] = (processing_info['stack_method'], 'Stacking method')
    if processing_info['binby2']:
        hdr['BINNED'] = ('True', 'Image is binned by 2x2')
    else:
        hdr['BINNED'] = 'False'
    hdr['PLATESCL'] = (processing_info['platescale'], 'Platescale in arcsec/pixel')
    hdr['COLPIXEL'] = (processing_info['target_coord'][0], 'Target X position in Python coordinates')
    hdr['ROWPIXEL'] = (processing_info['target_coord'][1], 'Target Y position in Python coordinates')
    hdr['DS9XPIX'] = (processing_info['target_coord'][0] + 1, 'Target X position in DS9 coordinates')
    hdr['DS9YPIX'] = (processing_info['target_coord'][1] + 1, 'Target Y position in DS9 coordinates')
    hdr.add_comment('Event number ratio in each segment: '+', '.join(f'{x:.3f}' for x in processing_info['event_ratio']))
    hdr['EXT1NAME'] = ('BKG', 'Scattering background image')
    hdr['HISTORY'] = f'Created by Zexi Xing'
    hdr['REDUCER'] = 'scattering_bkg.py'
    return hdr

def get_scattering_sk_event(*, group_number, target_id, target_coord,
                            data_dir, obsid, filt, temporary_dir, archive_dir, 
                            stack_method='sum', binby2=True,
                            scattering_dir=None, delete_raw_like=True):
    # generate sk_like image 
    get_scattering_sk(data_dir=data_dir, obsid=obsid, filt=filt, temporary_dir=temporary_dir, 
                      scattering_dir=scattering_dir, delete_raw_like=delete_raw_like)
    temporary_path = paths.get_subpath(temporary_dir, f'{obsid}_{filt}_sk_like.fits')
    
    # read sk_like image
    temporary_data = fits.getdata(temporary_path, ext=1)
    if binby2:
        temporary_data = block_reduce(temporary_data, block_size=2, func=np.nanmean)
    # read event file
    filt_filename = normalize_filter_name(filt, output_format='filename')
    event_path = paths.get_subpath(data_dir, f'{obsid}', 'uvot', 'event', f'sw{obsid}{filt_filename}w1po_uf.evt.gz')
    sk_file_path = paths.get_subpath(data_dir, f'{obsid}', 'uvot', 'image', f'sw{obsid}{filt_filename}_sk.img.gz')
    if not os.path.exists(event_path):
        raise FileNotFoundError(f"Event file not found: {event_path}")
    target_col_list, target_row_list, event_ratio = for_sk_like(evt_file_path = event_path, sk_file_path = sk_file_path, group_number = group_number, 
                                                                target_id = target_id, binby2 = binby2)
    aligned_images = []
    for i in range(len(event_ratio)):
        sk_coord = (target_col_list[i], target_row_list[i])
        temporary_data_i = crop_image(temporary_data, sk_coord, target_coord, fill_value=np.nan)
        ratio = event_ratio[i]
        aligned_images.append(temporary_data_i*ratio)
    stacked_image = stack_images(aligned_images, method=stack_method, verbose=False)
    evt_hdr = fits.getheader(event_path, ext=1)
    date_start = Time(evt_hdr['DATE-OBS'])
    date_end = Time(evt_hdr['DATE-END'])
    mid_time = date_start + 0.5 * (date_end - date_start)
    processing_info = {
        'mid_time': str(mid_time.isot),
        'num_segments': group_number,
        'stack_method': stack_method,
        'binby2': binby2,
        'platescale': 0.502*2 if binby2 else 0.502,
        'target_coord': target_coord,
        'event_ratio': event_ratio,
    }
    # update header and data
    primary_hdr = _scattering_sk_event_hdr(processing_info)
    primary_hdu = fits.PrimaryHDU(header=primary_hdr)
    image_hdu = fits.ImageHDU(data=stacked_image, name='BKG')
    new_hdul = fits.HDUList([primary_hdu, image_hdu])
    archive_path = paths.get_subpath(archive_dir, f'{obsid}_{1}_{filt_filename}.fits')
    new_hdul.writeto(archive_path, overwrite=True, output_verify="fix")
    #temporary_path.unlink(missing_ok=True)
    print(f"scattering_sk_event saved for {obsid} {filt}")

import numpy as np
from scipy.optimize import minimize

def fit_poisson_a_b(img, flat, maxiter=200):
    """
    Fit k ~ Poisson(a*F + b) on 1D arrays k, F (both finite, F>0).
    Returns (a, b).
    """
    img = np.asarray(img, dtype=np.float64)
    flat = np.asarray(flat, dtype=np.float64)

    # 初值：先用无常数项的 MLE
    a0 = np.nansum(img) / np.nansum(flat)
    # 用残差的中位数给 c0（取非负）
    b0 = np.nanmedian(img - a0 * flat)
    b0 = float(max(0.0, b0))

    # Poisson NLL: sum(mu - k*log(mu)) up to constant
    eps = 1e-12
    def nll(theta):
        a, b = theta
        mu = a * flat + b
        if np.any(mu <= 0):
            return np.inf
        return np.sum(mu - img * np.log(mu + eps))

    # 约束：a>=0, c>=0
    res = minimize(
        nll,
        x0=np.array([a0, b0], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(0.0, None), (0.0, None)],
        options={"maxiter": maxiter},
    )
    if not res.success:
        raise RuntimeError(f"Poisson fit failed: {res.message}")

    a_hat, b_hat = res.x
    return float(a_hat), float(b_hat)

def get_scattering_factor(img_path, bkg_path, target_region=None, exclude_region=None, focus_region=None,
                          img_ext=1, bkg_ext=1, shrink_pixels=5, if_sigma_clip=False, plot_save=False, plot_show=False):

    with fits.open(img_path, mode='readonly', memmap=True) as hdul:
        img = hdul[img_ext].data  # 不 copy

        mask = np.zeros(img.shape, dtype=bool)  # 直接用 img.shape（别用 _img）
        if 'STARMASK' in hdul:
            mask |= hdul['STARMASK'].data.astype(bool)

        if if_sigma_clip:
            mask_copy = np.zeros(img.shape, dtype=bool)
            clipped = sigma_clip(img[np.isfinite(img)], sigma=3, maxiters=3, masked=True)
            mask_copy[np.isfinite(img)] = clipped.mask
            mask |= select_mask_regions(mask_copy, min_area=4, max_area=None)

        if 'EXPOSURE' in hdul:
            exp = hdul['EXPOSURE'].data
            exp_mask = (exp != np.nanmax(exp))

    with fits.open(bkg_path, mode='readonly', memmap=True) as hdul:
        bkg = hdul[bkg_ext].data  # 不 copy
        bkg[exp_mask] = np.nan
        bkg = shrink_valid_image(bkg, shrink_pixels=shrink_pixels) if (shrink_pixels and shrink_pixels > 0) else bkg
    # regions -> mask（这块可能最慢；如果循环调用，强烈建议外部预先算好bool mask传进来）
    if target_region is not None:
        mask |= RegionConverter.to_bool_array(target_region, image_shape=img.shape)
    if exclude_region is not None or focus_region is not None:
        #mask |= RegionConverter.to_bool_array_general(exclude_region, combine_regions=True, shape=img.shape)[0]
        mask |= get_exclude_region(img.shape, focus_region=focus_region, exclude_region=exclude_region)

    # 有效像素：未mask、img/bkg有限、且bkg!=0
    valid_map = (~mask) & np.isfinite(img) & np.isfinite(bkg)
    k_bg = img[valid_map].astype(np.float64, copy=False)
    F_bg = bkg[valid_map].astype(np.float64, copy=False)

    # 只对有效像素做 1D 比值（避免全图除法）
    ratio = (k_bg / F_bg).astype(np.float64, copy=False)

    factor_median = np.nanmedian(ratio)
    factor_mean = np.nanmean(ratio)
    factor_sum = np.nansum(k_bg)/np.nansum(F_bg)
    std = ratio.std(ddof=0)
    factor_a, factor_b = fit_poisson_a_b(k_bg, F_bg)

    #sky_model = a_hat * bkg + b_hat
    result = {
        'factor_median': factor_median,
        'factor_mean': factor_mean,
        'factor_sum': factor_sum,
        'std': std,
        'factor_a': factor_a,
        'factor_b': factor_b,
    }


    # plot
    if plot_save or plot_show:
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        axes[0].imshow(img, origin='lower', vmin=0, vmax=(factor_mean+2*std)*np.nanmedian(bkg))
        axes[0].set_title('Original image')
        axes[0].set_xticklabels([])
        axes[0].set_yticklabels([])
        # 做个可视化用的masked版本（只在plot时创建）
        #img_vis = np.array(img/bkg, copy=True)
        img_vis = np.array((img- factor_b)/bkg, copy=True)
        img_vis[~valid_map] = np.nan
        axes[2].imshow(img_vis, origin='lower', vmin=factor_mean-2*std, vmax=factor_mean+2*std)
        axes[2].set_title('Masked factor image')
        axes[2].set_xticklabels([])
        axes[2].set_yticklabels([])
        axes[3].hist(ratio, bins=100, range=(factor_mean-5*std, factor_mean+5*std)); axes[2].set_title('Factor')
        axes[3].axvline(factor_mean, color='green', linestyle='-', lw=0.5, label='factor (mean)')
        axes[3].axvline(factor_median, color='blue', linestyle='-', lw=0.5, label='factor (median)')
        axes[3].axvline(factor_sum, color='red', linestyle='-', lw=0.5, label='factor (sum)')
        axes[3].axvline(factor_a + factor_b/np.nanmean(F_bg), color='purple', linestyle='-', lw=0.5, label='factor (a + b/mean(BKG))')
        axes[3].axvline(factor_sum+std, color='red', linestyle='--', lw=0.5, label='f (sum) + std')
        axes[3].axvline(factor_sum-std, color='red', linestyle='--', lw=0.5, label='f (sum) - std')
        axes[3].legend(loc='lower left', fontsize=8)
        axes[1].imshow(bkg, origin='lower')
        axes[1].set_title('Bkg (sk like)')
        axes[1].set_xticklabels([])
        axes[1].set_yticklabels([])
        axes[4].imshow(img - (factor_a*bkg + factor_b), origin='lower', vmin=0, vmax=2*std*np.nanmedian(bkg))
        #axes[4].imshow(img - factor_sum*bkg, origin='lower', vmin=0, vmax=2*std*np.nanmedian(bkg))
        axes[4].set_title('Residual image')
        axes[4].set_xticklabels([])
        axes[4].set_yticklabels([])
        plt.tight_layout()

        # ---------- output path ----------
        if plot_save:
            bkg_path_p = Path(bkg_path)
            fig_dir = bkg_path_p.parent / "figure"
            fig_dir.mkdir(parents=True, exist_ok=True)    
            plot_save_path = fig_dir / f"{bkg_path_p.stem}.jpeg"
            fig.savefig(plot_save_path, dpi=300, bbox_inches='tight')
        if plot_show:
            plt.show(block=True)
        plt.close(fig)

    return result, valid_map


def save_scattering_bkg(
    bkg_path,
    result,
    valid_map,
    img_path,
    bkg_ext=1,
    ):
    bkg_path = Path(bkg_path)
    img_path = Path(img_path)

    # 读单位（你原来用 ext=0，我保持一致）
    hdr_img0 = fits.getheader(img_path, ext=0)
    unit = hdr_img0.get('BUNIT', 'UNKNOWN')

    #if not (np.isfinite(factor_median) and factor_median != 0):
    #    raise ValueError(f"Factor is not finite or zero: {factor}")
    #err_rel = float(std / factor_median)

    mask_i8 = np.asarray(~valid_map, dtype=np.int8)

    # 直接在原 bkg_path 上 update
    with fits.open(bkg_path, mode='update', memmap=True) as hdul:

        #data = hdul[bkg_ext].data

        # 确保浮点，避免整数截断
        #if not np.issubdtype(data.dtype, np.floating):
        #    hdul[bkg_ext].data = data.astype(np.float32)
        #    data = hdul[bkg_ext].data

        # 缩放背景（原地写回）
        #data *= factor

        # 单位与元数据
        hdul[bkg_ext].header['BUNIT'] = unit
        hdul[0].header['BUNIT']   = unit
        hdul[0].header['SCALED']  = (True, 'Background is scaled by factor')
        hdul[0].header['FACT_MED'] = (float(result['factor_median']), 'Factor applied to background (median of img/bkg)')
        hdul[0].header['FACT_AVE'] = (float(result['factor_mean']), 'Factor applied to background (mean of img/bkg)')
        hdul[0].header['FACT_SUM'] = (float(result['factor_sum']), 'Factor applied to background (sum of img/bkg)')
        hdul[0].header['SCLSTD']  = (float(result['std']), 'Std of (img/bkg) used for scaling')
        hdul[0].header['FACT_A']  = (float(result['factor_a']), 'a_hat of the sky model')
        hdul[0].header['FACT_B']  = (float(result['factor_b']), 'b_hat of the sky model')
        #hdul[0].header['ERR_REL'] = (err_rel, 'Relative error of the background (std/factor)')

        # 写/更新 MASK（压缩）
        if 'MASK' in hdul:
            hdul['MASK'].data = mask_i8
            ext_number = hdul.index_of('MASK')
        else:
            hdu = fits.CompImageHDU(data=mask_i8, name='MASK', compression_type='RICE_1')
            hdul.append(hdu)
            ext_number = len(hdul) - 1

        hdul[0].header[f'EXT{ext_number}NAME'] = ('MASK', 'Compressed mask extension')

        hdul.flush()

    return str(bkg_path)

def get_bkg_sk_api(*, data_dir, obsid, filt, ext,
                   temporary_dir, archive_dir, datatype, img_path,
                   target_region=None, exclude_region=None, focus_region=None, shrink_pixels=30, if_sigma_clip=None,
                   bkg_ext=1, img_ext=1, scattering_dir=None, delete_raw_like=True,
                   group_number=None, target_id=None, target_coord=None, sk_coord=None,
                   stack_method=None, binby2=None, plot_save=False, plot_show=False,
                   ):
    """
    Get the background sky image for a given observation and filter.
    E.g.,
    data_dir = paths.get_subpath(paths.data, 'Swift', 'C_2025N1')
    obsid = '05000930001'
    filt = 'uuu'
    datatype = 'image'
    temporary_dir = paths.get_subpath(paths.projects, 'C_2025N1', 'temporary')
    archive_dir = paths.get_subpath(paths.projects, 'C_2025N1', 'scattering_bkg')
    ext = 1 # extension to save the scattering sky image
    scattering_dir = None
    label = None
    """
    filt_filename = normalize_filter_name(filt, output_format='filename')
    if datatype == 'image':
        with fits.open(img_path, mode='readonly') as hdul:
            target_coord = (hdul[0].header['COLPIXEL'], hdul[0].header['ROWPIXEL']) if target_coord is None else target_coord
        get_scattering_sk(data_dir=data_dir, obsid=obsid, filt=filt, temporary_dir=temporary_dir,
                          scattering_dir=scattering_dir, delete_raw_like=delete_raw_like)
        temporary_path = paths.get_subpath(temporary_dir, f'{obsid}_{filt}_sk_like.fits')
        align_scattering(temporary_path=temporary_path, archive_dir=archive_dir, data_dir=data_dir, obsid=obsid, filt=filt, ext=ext, sk_coord=sk_coord, target_coord=target_coord)
    elif datatype == 'event':
        #if group_number is None or target_id is None or target_coord is None:
        #    raise ValueError(f"group_number, target_id, and target_coord must be provided for event mode")
        with fits.open(img_path, mode='readonly') as hdul:
            group_number = hdul[0].header['NSEGMENT'] if group_number is None else group_number
            target_id = hdul[0].header['TARGETID'] if target_id is None else target_id
            target_coord = (hdul[0].header['COLPIXEL'], hdul[0].header['ROWPIXEL']) if target_coord is None else target_coord
            stack_method = hdul[0].header['STACKMTH'] if stack_method is None else stack_method
            binby2 = hdul[0].header['BINNED'] if binby2 is None else binby2
        get_scattering_sk_event(group_number=group_number, target_id=target_id, target_coord=target_coord,
                                data_dir=data_dir, obsid=obsid, filt=filt, temporary_dir=temporary_dir, archive_dir=archive_dir, 
                                stack_method=stack_method, binby2=binby2,
                                scattering_dir=scattering_dir, delete_raw_like=delete_raw_like)
    else:
        raise ValueError(f"Invalid datatype: {datatype}")
    with fits.open(img_path, memmap=True) as hdul:
        has_starmask = 'STARMASK' in hdul
    if not has_starmask and if_sigma_clip is None:
        if_sigma_clip = True
    elif if_sigma_clip is None:
        if_sigma_clip = False
    bkg_path = paths.get_subpath(archive_dir, f'{obsid}_{ext}_{filt_filename}.fits')
    result, valid_map = get_scattering_factor(img_path=img_path, bkg_path=bkg_path, target_region=target_region, 
                                        exclude_region=exclude_region, focus_region=focus_region, 
                                        img_ext=img_ext, bkg_ext=bkg_ext, shrink_pixels=shrink_pixels, if_sigma_clip=if_sigma_clip, 
                                        plot_save=plot_save, plot_show=plot_show)
    
    save_scattering_bkg(bkg_path=bkg_path, result=result, valid_map=valid_map, img_path=img_path, bkg_ext=bkg_ext)
    #p = Path(bkg_path)
    #if p.exists():
    #    p.unlink()

# -------------------------
# Example usage (edit paths)
# -------------------------
import matplotlib.pyplot as plt
if __name__ == "__main__":

    #filt = 'uvv'
    #obsid = '05000951002'
    #temporary_dir = "/Users/zexixing/Downloads/scattering_bkg_test/temp"
    #archive_dir = "/Users/zexixing/Downloads/scattering_bkg_test/archive"
    #data_dir = "/Users/zexixing/Library/CloudStorage/OneDrive-Personal/ZexiWork/data/Swift/C_2025N1"
    #get_scattering_sk(data_dir=data_dir, obsid=obsid, filt=filt, temporary_dir=temporary_dir, archive_dir=archive_dir, ext=1, scattering_dir=None, label=None)
    #obsid = '05000931002'
    #raw_fits = f'/Users/zexixing/Library/CloudStorage/OneDrive-Personal/ZexiWork/data/Swift/C_2025N1/{obsid}/uvot/image/sw{obsid}uvv_rw.img.gz'
    #scattering_data = fits.getdata('/Users/zexixing/Library/CloudStorage/OneDrive-Personal/ZexiWork/projects/C_2025N1/bkg/uvv.fits', ext=0)
    #workdir = "/Users/zexixing/Downloads/scattering_bkg_test/temp"
    #obsdir = f'/Users/zexixing/Library/CloudStorage/OneDrive-Personal/ZexiWork/data/Swift/C_2025N1/{obsid}'
    #label = f'{obsid}_1_uvv'
    #delete_raw_like = False
    #move_sk_to_archive = True
    #archive_dir = '/Users/zexixing/Library/CloudStorage/OneDrive-Personal/ZexiWork/projects/C_2025N1/scattering_bkg'
    #attfile = f'/Users/zexixing/Library/CloudStorage/OneDrive-Personal/ZexiWork/data/Swift/C_2025N1/{obsid}/auxil/sw{obsid}pat.fits.gz'
    #heasoft_tmpdir = None
    #silent = True
    #scattering_to_sky_array(raw_fits=raw_fits, scattering_data=scattering_data, workdir=workdir, obsdir=obsdir, label=label,
    #                       delete_raw_like=delete_raw_like, move_sk_to_archive=move_sk_to_archive, archive_dir=archive_dir,
    #                       attfile=attfile, heasoft_tmpdir=heasoft_tmpdir, silent=silent)


    #raw_fits = '/Users/zexixing/Library/CloudStorage/OneDrive-Personal/ZexiWork/data/Swift/C_2025N1/05000931002/uvot/image/sw05000931002uvv_rw.img.gz'
    #workdir = '/Users/zexixing/Library/CloudStorage/OneDrive-Personal/ZexiWork/projects/C_2025N1/temporary'
    #obsdir = '/Users/zexixing/Library/CloudStorage/OneDrive-Personal/ZexiWork/data/Swift/C_2025N1/05000931002'
    #label = '05000931002_1_uvv'
    #delete_raw_like = False
    #move_sk_to_archive = True
    #archive_dir = '/Users/zexixing/Library/CloudStorage/OneDrive-Personal/ZexiWork/projects/C_2025N1/scattering_bkg'
    #attfile = '/Users/zexixing/Library/CloudStorage/OneDrive-Personal/ZexiWork/data/Swift/C_2025N1/05000931002/auxil/sw05000931002pat.fits.gz'
    #heasoft_tmpdir = None
    #silent = True
    #
    #scattering_data = fits.getdata('/Users/zexixing/Library/CloudStorage/OneDrive-Personal/ZexiWork/projects/C_2025N1/bkg/uvv.fits', ext=0)
    #
    #scattering_to_sky_array(raw_fits=raw_fits, scattering_data=scattering_data, workdir=workdir, obsdir=obsdir, label=label,
    #                       delete_raw_like=delete_raw_like, move_sk_to_archive=move_sk_to_archive, archive_dir=archive_dir,
    #                       attfile=attfile, heasoft_tmpdir=heasoft_tmpdir, silent=silent)

    data_dir = paths.get_subpath(paths.data, 'Swift', 'C_2025N1')
    obsid = '05000931002'
    filt_v = 'uvv'
    temporary_dir = paths.get_subpath(paths.projects, 'C_2025N1', 'temporary')
    archive_dir = paths.get_subpath(paths.projects, 'C_2025N1', 'scattering_bkg')
    get_scattering_sk(data_dir = data_dir, obsid=obsid, filt=filt_v, temporary_dir=temporary_dir, archive_dir=archive_dir,
                     ext=1, scattering_dir=None, label=None, delete_raw_like=True)