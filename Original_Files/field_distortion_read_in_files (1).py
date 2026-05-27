import os
import numpy as np
from astropy.io import fits


def files_in(directory, brights_reg_exp, darks_reg_exp, m, buff1=0, buff2=None, ceil=2**16-1, flats_reg_exp=None, fdarks_reg_exp=None, remake=True):
    '''For a given directory containing hologram data and darks with
    filenames with regular expressions, this function extracts the data from
    the .fits files, finds the nominal center of the saturated 0th-order PSF,
    and gets the average dark-subtracted bright frame.
    The returns get saved as FITS file if that FITS file doesn't already
    exist.  If it does, then that file is simply loaded via this function.

    Parameters
    ----------
    directory : str
    Absolute path of the directory containing the brights and darks.

    brights_reg_exp : str
    A string that uniquely identifies bright frames, a string that all
    filenames for brights contain that the darks do not contain.  This should
    contain one instance of 'bright' and none of 'dark'.

    darks_reg_exp : str
    A string that uniquely identifies dark frames, a string that all
    filenames for darks contain that the brights do not contain.  If there are
    no darks to input, set this variable to None.  Must contain one instance
    of 'dark' and none of 'bright'.

    m : int
    Distance allocated around each PSF peak so that another peak is not
    identified within this square of side length 2m+1 pixels (integer).  Needed
    so that the 0th-order brightest PSF can be taken out so that the
    centroiding algorithm does not try and fail to centroid it.

    buff1 : int
    Frame number index to start at for reading in frames.  Enables user to skip
    over any frames that are not good due to the transition of the detector to
    the next observation set.  Defaults to 0.

    buff2 : int
    Frame number index to end at for reading in frames.  Enables user to skip
    over any frames that are not good due to the transition of the detector to
    the next observation set.  If None, the last frame is included.
    Defaults to None.  If not None, then buff2 must be > buff1.

    ceil : int
    Number of counts that saturates the detector.  Defaults to the number
    for a 16-bit detector.  If None, then nom_center_row and nom_center_col
    outputs are 0, which means that later in the pipeline, no PSF will be
    blocked out in the function nPeaks2().

    flats_reg_exp : str, optional
    A string that uniquely identifies flat frames, a string that all
    filenames for flats contain that the darks and brights do not contain.
    This should contain one instance of 'flat' and none of 'dark' or 'bright'.
    Defaults to None, in which case no flat divison is performed.

    fdarks_reg_exp : str, optional
    A string that uniquely identifies dark frames intended for flats,
    a string that all filenames for "flat darks" contain that the regular
    darks and brights do not contain.
    This should contain one instance of 'fdark' and none of 'dark' or 'bright'
    or 'flat'.
    Defaults to None.

    remake : bool, optional
    If True, the data is processed even it has been before as indicated by
    a particular filename in the folder the user specifies for this function.
    If False, the data is only processed if that filename is not present in
    the folder.  Defaults to True.

    Returns
    -------
    avg_lights_sub : array
    The average dark-subtracted bright frame, with 'buff' frames skipped over

    nom_center_row : float
    Nominal row coordinate of the center of brightest PSF.

    nom_center_col : float
    Nominal col coordinate of the center of brightest PSF.
    '''
    if buff2 is not None:
        if buff2 <= buff1:
            raise Exception('buff2 must be > buff1.')
    if 'dark' in brights_reg_exp:
        raise Exception('brights_reg_exp must not contain \'dark\'')
    if 'bright' not in brights_reg_exp:
        raise Exception("brights_reg_exp must contain one instance of \'bright\'")
    br_start = brights_reg_exp.rfind('bright') #finds that last instance of it
    if 'bright' in brights_reg_exp[:br_start]:
        raise Exception('brights_reg_exp must contain only one instance of \'bright\'')
    if darks_reg_exp is not None:
        if 'bright' in darks_reg_exp:
            raise Exception('darks_reg_exp must not contain \'bright\'')
        if 'dark' not in darks_reg_exp:
            raise Exception("darks_reg_exp must contain one instance of \'dark\'")
        dark_start = brights_reg_exp.rfind('bright') #finds that last instance of it
        if 'bright' in darks_reg_exp[:dark_start]:
            raise Exception('darks_reg_exp must contain only one instance of \'dark\'')

    #XXX implement residual nonlinearity correction to frame first, then subtract nonlin-corrected dark subtraction, then divide by nonlin-corrected dark-subtracted flat division; inputs for res nonlin and flat needed

    brights_part = brights_reg_exp[:br_start]+brights_reg_exp[br_start+6:]
    if 'avg_lights_sub_'+brights_part+'.fits' not in os.listdir(directory) or remake: # so that new files are generated when, e.g., buffers changed; better to have processing every time be the default in general
        darks = []
        lights = []
        flats = []
        fdarks = []
        nom_center_row = []
        nom_center_col = []
        for file in os.listdir(directory):
            if not file.endswith('fits'):
                continue
            f = os.path.join(directory, file)

            if brights_reg_exp in f:
                df = (fits.getdata(f)).astype(float)#[0] #[0] if older CIS120 testbed
                if ceil is not None:
                    sat_rows, sat_cols = np.where(df>=ceil)
                    if sat_rows.size == 0: # then simulated
                        # nom_center_row.append(int(np.ceil(df.shape[0]/2)))
                        # nom_center_col.append(int(np.ceil(df.shape[1]/2)))
                        df_sub = df[int(np.ceil(df.shape[0]/2)-m):int(np.ceil(df.shape[0]/2)+m),int(np.ceil(df.shape[1]/2)-m):int(np.ceil(df.shape[1]/2)+m)]
                        pk = np.unravel_index(np.argmax(df_sub), df_sub.shape)
                        nom_center_row.append(pk[0]+int(np.ceil(df.shape[0]/2)-m))
                        nom_center_col.append(pk[1]+int(np.ceil(df.shape[1]/2)-m))
                    else:
                        # find where row and col indicies are most bunched together, for saturated spot:
                        D_sat_rows = np.abs(sat_rows - np.roll(sat_rows, 1))
                        row_min_inc = np.min(D_sat_rows)  # finding where derivative goes to 0, essentially
                        row_indices = np.where(D_sat_rows==row_min_inc)
                        nom_center_row.append(sat_rows[int(np.median(row_indices))])
                        D_sat_cols = np.abs(sat_cols - np.roll(sat_cols, 1))
                        col_min_inc = np.min(D_sat_cols)
                        col_indices = np.where(D_sat_cols==col_min_inc)
                        nom_center_col.append(sat_cols[int(np.median(col_indices))])
                    # nom_center_row.append(np.median(sat_rows))
                    # nom_center_col.append(np.median(sat_cols))
                    # mask = np.zeros_like(df)
                    # mask[int(np.round(nom_center_row-m/2)):int(np.round(nom_center_row+m/2)),int(np.round(nom_center_col-m/2)):int(np.round(nom_center_col+m/2))] = 1
                    # df = np.ma.masked_array(df, mask)
                lights.append(df)
            if darks_reg_exp is not None:
                if darks_reg_exp in f:
                    d = (fits.getdata(f)).astype(float)#[0] # [0] if older CIS120 testbed
                    darks.append(d) # to convert from unsigned integer uint16
            if flats_reg_exp is not None:
                if flats_reg_exp in f:
                    d = (fits.getdata(f)).astype(float)#[0] # [0] if older CIS120 testbed
                    flats.append(d) # to convert from unsigned integer uint16
            if fdarks_reg_exp is not None:
                if fdarks_reg_exp in f:
                    fdark = (fits.getdata(f)).astype(float)
                    fdarks.append(fdark)
        # mask = lights[3].mask
        lights = np.stack(lights)
        if darks_reg_exp is not None:
            darks = np.stack(darks)
            if buff2 is not None:
                avg_lights_sub = np.mean(lights[buff1:buff2], axis=0) - np.mean(darks[buff1:buff2], axis=0) #leaving off first few, which are "transition" frames
            else:
                avg_lights_sub = np.mean(lights[buff1:], axis=0) - np.mean(darks[buff1:], axis=0)
            if ceil is not None:
                nom_center_row = np.mean(nom_center_row)
                nom_center_col = np.mean(nom_center_col)
        if darks_reg_exp is None:
            # for i in range(len(lights_subtracted)):
                #lights_subtracted[i][int(np.round(nom_center_row[i] - m/2)):int(np.round(nom_center_row[i] + m/2)),int(np.round(nom_center_col[i] - m/2)):int(np.round(nom_center_col[i] + m/2))] = np.median(lights_subtracted[i])
            if buff2 is not None:
                avg_lights_sub = np.mean(lights[buff1:buff2], axis=0) #leaving off first few, which are "transition" frames
            else:
                avg_lights_sub = np.mean(lights[buff1:], axis=0)
            if ceil is not None:
                nom_center_row = np.mean(nom_center_row)
                nom_center_col = np.mean(nom_center_col)
        if ceil is None:
            nom_center_row = 0
            nom_center_col = 0
        if flats_reg_exp is not None:
            flats = np.stack(flats)
            if fdarks_reg_exp is not None:
                fdarks = np.stack(fdarks)
                if buff2 is not None:
                    avg_flat = np.mean(flats[buff1:buff2], axis=0) - np.mean(fdarks[buff1:buff2], axis=0)
                    norm_avg_flat = avg_flat*(avg_flat.size/np.sum(avg_flat))
                    avg_lights_sub = avg_lights_sub/(norm_avg_flat)
                else:
                    avg_flat = np.mean(flats[buff1:], axis=0) - np.mean(fdarks[buff1:], axis=0)
                    norm_avg_flat = avg_flat*(avg_flat.size/np.sum(avg_flat))
                    avg_lights_sub = avg_lights_sub/(norm_avg_flat)
            if fdarks_reg_exp is None:
                if buff2 is not None:
                    avg_flat = np.mean(flats[buff1:buff2], axis=0)
                    norm_avg_flat = avg_flat*(avg_flat.size/np.sum(avg_flat))
                    avg_lights_sub = avg_lights_sub/(norm_avg_flat)
                else:
                    avg_flat = np.mean(flats[buff1:], axis=0)
                    norm_avg_flat = avg_flat*(avg_flat.size/np.sum(avg_flat))
                    avg_lights_sub = avg_lights_sub/(norm_avg_flat)
        hdr = fits.Header()
        hdr['CEN_ROW'] = nom_center_row
        hdr['CEN_COL'] = nom_center_col
        prim = fits.PrimaryHDU(header=hdr)
        img = fits.ImageHDU(avg_lights_sub)
        hdul = fits.HDUList([prim, img])
        hdul.writeto(os.path.join(directory, 'avg_lights_sub_'+brights_part+'.fits'), overwrite=True)
    else:
        f = os.path.join(directory, 'avg_lights_sub_'+brights_part+'.fits')
        avg_lights_sub = fits.getdata(f)
        with fits.open(f) as hdul:
            hdr = hdul[0].header
            nom_center_row = float(hdr['CEN_ROW'])
            nom_center_col = float(hdr['CEN_COL'])

    return avg_lights_sub, nom_center_row, nom_center_col

def files_in_individual(directory, brights_reg_exp, darks_reg_exp, m, buff1=0, buff2=None, ceil=2**16-1, flats_reg_exp=None, fdarks_reg_exp=None):
    '''For a given directory containing hologram data and darks with
    filenames with regular expressions, this function extracts the data from
    the .fits files, finds the nominal center of the sa2turated 0th-order PSF,
    and gets the average dark-subtracted bright frame.

    The only difference betwween this function and files_in() is that this
    function does the dark subtraction and flat division individually for each
    frame rather than for average frames.

    The returns get saved as FITS file if that FITS file doesn't already
    exist.  If it does, then that file is simply loaded via this function.

    Parameters
    ----------
    directory : str
    Absolute path of the directory containing the brights and darks.

    brights_reg_exp : str
    A string that uniquely identifies bright frames, a string that all
    filenames for brights contain that the darks do not contain.  This should
    contain one instance of 'bright' and none of 'dark'.

    darks_reg_exp : str
    A string that uniquely identifies dark frames, a string that all
    filenames for darks contain that the brights do not contain.  If there are
    no darks to input, set this variable to None.  Must contain one instance
    of 'dark' and none of 'bright'.

    m : int
    Distance allocated around each PSF peak so that another peak is not
    identified within this square of side length 2m+1 pixels (integer).  Needed
    so that the 0th-order brightest PSF can be taken out so that the
    centroiding algorithm does not try and fail to centroid it.

    buff1 : int
    Frame number index to start at for reading in frames.  Enables user to skip
    over any frames that are not good due to the transition of the detector to
    the next observation set.  Defaults to 0.

    buff2 : int
    Frame number index to end at for reading in frames.  Enables user to skip
    over any frames that are not good due to the transition of the detector to
    the next observation set.  If None, the last frame is included.
    Defaults to None.  If not None, then buff2 must be > buff1.

    ceil : int
    Number of counts that saturates the detector.  Defaults to the number
    for a 16-bit detector.  If None, then nom_center_row and nom_center_col
    outputs are 0, which means that later in the pipeline, no PSF will be
    blocked out in the function nPeaks2().

    flats_reg_exp : str, optional
    A string that uniquely identifies flat frames, a string that all
    filenames for flats contain that the darks and brights do not contain.
    This should contain one instance of 'flat' and none of 'dark' or 'bright'.
    Defaults to None, in which case no flat divison is performed.

    fdarks_reg_exp : str, optional
    A string that uniquely identifies dark frames intended for flats,
    a string that all filenames for "flat darks" contain that the regular
    darks and brights do not contain.
    This should contain one instance of 'fdark' and none of 'dark' or 'bright'
    or 'flat'.
    Defaults to None.

    Returns
    -------
    avg_lights_sub : array
    The average dark-subtracted bright frame, with 'buff' frames skipped over

    nom_center_row : float
    Nominal row coordinate of the center of brightest PSF.

    nom_center_col : float
    Nominal col coordinate of the center of brightest PSF.
    '''
    if buff2 is not None:
        if buff2 <= buff1:
            raise Exception('buff2 must be > buff1.')
    if 'dark' in brights_reg_exp:
        raise Exception('brights_reg_exp must not contain \'dark\'')
    if 'bright' not in brights_reg_exp:
        raise Exception("brights_reg_exp must contain one instance of \'bright\'")
    br_start = brights_reg_exp.rfind('bright') #finds that last instance of it
    if 'bright' in brights_reg_exp[:br_start]:
        raise Exception('brights_reg_exp must contain only one instance of \'bright\'')
    if darks_reg_exp is not None:
        if 'bright' in darks_reg_exp:
            raise Exception('darks_reg_exp must not contain \'bright\'')
        if 'dark' not in darks_reg_exp:
            raise Exception("darks_reg_exp must contain one instance of \'dark\'")
        dark_start = brights_reg_exp.rfind('bright') #finds that last instance of it
        if 'bright' in darks_reg_exp[:dark_start]:
            raise Exception('darks_reg_exp must contain only one instance of \'dark\'')

    #XXX implement residual nonlinearity correction to frame first, then subtract nonlin-corrected dark subtraction, then divide by nonlin-corrected dark-subtracted flat division; inputs for res nonlin and flat needed

    brights_part = brights_reg_exp[:br_start]+brights_reg_exp[br_start+6:]
    if 'avg_lights_sub_'+brights_part+'.fits' not in os.listdir(directory) or True: # so that new files are generated when, e.g., buffers changed; better to have processing every time be the default in general
        darks = []
        lights = []
        flats = []
        fdarks = []
        nom_center_row = []
        nom_center_col = []
        for file in os.listdir(directory):
            if not file.endswith('fits'):
                continue
            f = os.path.join(directory, file)

            if brights_reg_exp in f:
                df = (fits.getdata(f)).astype(float)#[0] #[0] if older CIS120 testbed
                if ceil is not None:
                    sat_rows, sat_cols = np.where(df>=ceil)
                    if sat_rows.size == 0: # then simulated
                        # nom_center_row.append(int(np.ceil(df.shape[0]/2)))
                        # nom_center_col.append(int(np.ceil(df.shape[1]/2)))
                        df_sub = df[int(np.ceil(df.shape[0]/2)-m):int(np.ceil(df.shape[0]/2)+m),int(np.ceil(df.shape[1]/2)-m):int(np.ceil(df.shape[1]/2)+m)]
                        pk = np.unravel_index(np.argmax(df_sub), df_sub.shape)
                        nom_center_row.append(pk[0]+int(np.ceil(df.shape[0]/2)-m))
                        nom_center_col.append(pk[1]+int(np.ceil(df.shape[1]/2)-m))
                    else:
                        # find where row and col indicies are most bunched together, for saturated spot:
                        D_sat_rows = np.abs(sat_rows - np.roll(sat_rows, 1))
                        row_min_inc = np.min(D_sat_rows)  # finding where derivative goes to 0, essentially
                        row_indices = np.where(D_sat_rows==row_min_inc)
                        nom_center_row.append(sat_rows[int(np.median(row_indices))])
                        D_sat_cols = np.abs(sat_cols - np.roll(sat_cols, 1))
                        col_min_inc = np.min(D_sat_cols)
                        col_indices = np.where(D_sat_cols==col_min_inc)
                        nom_center_col.append(sat_cols[int(np.median(col_indices))])
                    # nom_center_row.append(np.median(sat_rows))
                    # nom_center_col.append(np.median(sat_cols))
                    # mask = np.zeros_like(df)
                    # mask[int(np.round(nom_center_row-m/2)):int(np.round(nom_center_row+m/2)),int(np.round(nom_center_col-m/2)):int(np.round(nom_center_col+m/2))] = 1
                    # df = np.ma.masked_array(df, mask)
                lights.append(df)
            if darks_reg_exp is not None:
                if darks_reg_exp in f:
                    d = (fits.getdata(f)).astype(float)#[0] # [0] if older CIS120 testbed
                    darks.append(d) # to convert from unsigned integer uint16
            if flats_reg_exp is not None:
                if flats_reg_exp in f:
                    d = (fits.getdata(f)).astype(float)#[0] # [0] if older CIS120 testbed
                    flats.append(d) # to convert from unsigned integer uint16
            if fdarks_reg_exp is not None:
                if fdarks_reg_exp in f:
                    fdark = (fits.getdata(f)).astype(float)
                    fdarks.append(fdark)
        # mask = lights[3].mask
        lights = np.stack(lights)
        if darks_reg_exp is not None:
            darks = np.stack(darks)
            if buff2 is not None:
                avg_lights_sub = lights[buff1:buff2] - darks[buff1:buff2] #leaving off first few, which are "transition" frames
            else:
                avg_lights_sub = lights[buff1:] - darks[buff1:]
            if ceil is not None:
                nom_center_row = np.mean(nom_center_row)
                nom_center_col = np.mean(nom_center_col)
        if darks_reg_exp is None:
            # for i in range(len(lights_subtracted)):
                #lights_subtracted[i][int(np.round(nom_center_row[i] - m/2)):int(np.round(nom_center_row[i] + m/2)),int(np.round(nom_center_col[i] - m/2)):int(np.round(nom_center_col[i] + m/2))] = np.median(lights_subtracted[i])
            if buff2 is not None:
                avg_lights_sub = lights[buff1:buff2] #leaving off first few, which are "transition" frames
            else:
                avg_lights_sub = lights[buff1:]
            if ceil is not None:
                nom_center_row = np.mean(nom_center_row)
                nom_center_col = np.mean(nom_center_col)
        if ceil is None:
            nom_center_row = 0
            nom_center_col = 0
        if flats_reg_exp is not None:
            flats = np.stack(flats)
            if fdarks_reg_exp is not None:
                fdarks = np.stack(fdarks)
                if buff2 is not None:
                    dsub_flats = flats[buff1:buff2] - fdarks[buff1:buff2]
                    for fr in dsub_flats:
                        fr = fr*(fr.size/np.sum(fr))
                    avg_lights_sub = np.mean(avg_lights_sub/(dsub_flats), axis=0)
                else:
                    dsub_flats = flats[buff1:] - fdarks[buff1:]
                    for fr in dsub_flats:
                        fr = fr*(fr.size/np.sum(fr))
                    avg_lights_sub = np.mean(avg_lights_sub/dsub_flats, axis=0)
            if fdarks_reg_exp is None: #no dark subtraction for flats; dangerous XXX
                if buff2 is not None:
                    dnosub_flats = flats[buff1:buff2]
                    for fr in dnosub_flats:
                        fr = fr*(fr.size/np.sum(fr))
                    avg_lights_sub = np.mean(avg_lights_sub/dnosub_flats, axis=0)
                else:
                    dnosub_flats = flats[buff1:]
                    for fr in dnosub_flats:
                        fr = fr*(fr.size/np.sum(fr))
                    avg_lights_sub = np.mean(avg_lights_sub/dnosub_flats, axis=0)
        hdr = fits.Header()
        hdr['CEN_ROW'] = nom_center_row
        hdr['CEN_COL'] = nom_center_col
        prim = fits.PrimaryHDU(header=hdr)
        img = fits.ImageHDU(avg_lights_sub)
        hdul = fits.HDUList([prim, img])
        hdul.writeto(os.path.join(directory, 'avg_lights_sub_'+brights_part+'.fits'), overwrite=True)
    else:
        f = os.path.join(directory, 'avg_lights_sub_'+brights_part+'.fits')
        avg_lights_sub = fits.getdata(f)
        with fits.open(f) as hdul:
            hdr = hdul[0].header
            nom_center_row = float(hdr['CEN_ROW'])
            nom_center_col = float(hdr['CEN_COL'])

    return avg_lights_sub, nom_center_row, nom_center_col

def files_in_old(directory, brights_reg_exp, darks_reg_exp, m, buff, ceil=2**16-1):
    '''For a given directory containing hologram data and darks with
    filenames with regular expressions, this function extracts the data from
    the .fits files, finds the nominal center of the saturated 0th-order PSF,
    and gets the average dark-subtracted bright frame.

    The returns get saved as FITS file if that FITS file doesn't already
    exist.  If it does, then that file is simply loaded via this function.

    Parameters
    ----------
    dir : str
    Absolute path of the directory containing the brights and darks.

    brights_reg_exp : str
    A string that uniquely identifies bright frames, a string that all
    filenames for brights contain that the darks do not contain.  This should
    contain one instance of 'bright' and none of 'dark'.

    darks_reg_exp : str
    A string that uniquely identifies dark frames, a string that all
    filenames for darks contain that the brights do not contain.  If there are
    no darks to input, set this variable to None.  Must contain one instance
    of 'dark' and none of 'bright'.

    m : int
    Distance allocated around each PSF peak so that another peak is not
    identified within this square of side length 2m+1 pixels (integer).  Needed
    so that the 0th-order brightest PSF can be taken out so that the
    centroiding algorithm does not try and fail to centroid it.

    buff : int
    Number of frames to ignore in order to skip over the frames that are not
    good due to the transition of the detector to the next observation set.
    Set to None if there are no buffer frames.

    ceil : int
    Number of counts that saturates the detector.  Defaults to the number
    for a 16-bit detector.  If None, then nom_center_row and nom_center_col
    outputs are 0, which means that later in the pipeline, no PSF will be
    blocked out in the function nPeaks2().

    Returns
    -------
    avg_lights_sub : array
    The average dark-subtracted bright frame, with 'buff' frames skipped over

    nom_center_row : float
    Nominal row coordinate of the center of brightest PSF.

    nom_center_col : float
    Nominal col coordinate of the center of brightest PSF.
    '''
    if 'dark' in brights_reg_exp:
        raise Exception('brights_reg_exp must not contain \'dark\'')
    if 'bright' not in brights_reg_exp:
        raise Exception("brights_reg_exp must contain one instance of \'bright\'")
    br_start = brights_reg_exp.rfind('bright') #finds that last instance of it
    if 'bright' in brights_reg_exp[:br_start]:
        raise Exception('brights_reg_exp must contain only one instance of \'bright\'')
    if darks_reg_exp is not None:
        if 'bright' in darks_reg_exp:
            raise Exception('darks_reg_exp must not contain \'bright\'')
        if 'dark' not in darks_reg_exp:
            raise Exception("darks_reg_exp must contain one instance of \'dark\'")
        dark_start = brights_reg_exp.rfind('bright') #finds that last instance of it
        if 'bright' in darks_reg_exp[:dark_start]:
            raise Exception('darks_reg_exp must contain only one instance of \'dark\'')

    brights_part = brights_reg_exp[:br_start]+brights_reg_exp[br_start+6:]
    if 'avg_lights_sub_'+brights_part+'.fits' not in os.listdir(directory):
        darks = []
        lights = []
        nom_center_row = []
        nom_center_col = []
        for file in os.listdir(directory):
            f = os.path.join(directory, file)
            if brights_reg_exp in f:
                df = (fits.getdata(f)).astype(float)
                if ceil is not None:
                    sat_rows, sat_cols = np.where(df>=ceil)
                    if sat_rows.size == 0: # then simulated
                        nom_center_row.append(int(np.ceil(df.shape[0]/2)))
                        nom_center_col.append(int(np.ceil(df.shape[1]/2)))
                    else:
                        # find where row and col indicies are most bunched together, for saturated spot:
                        D_sat_rows = np.abs(sat_rows - np.roll(sat_rows, 1))
                        row_min_inc = np.min(D_sat_rows)  # finding where derivative goes to 0, essentially
                        row_indices = np.where(D_sat_rows==row_min_inc)
                        nom_center_row.append(sat_rows[int(np.median(row_indices))])
                        D_sat_cols = np.abs(sat_cols - np.roll(sat_cols, 1))
                        col_min_inc = np.min(D_sat_cols)
                        col_indices = np.where(D_sat_cols==col_min_inc)
                        nom_center_col.append(sat_cols[int(np.median(col_indices))])
                    # nom_center_row.append(np.median(sat_rows))
                    # nom_center_col.append(np.median(sat_cols))
                    # mask = np.zeros_like(df)
                    # mask[int(np.round(nom_center_row-m/2)):int(np.round(nom_center_row+m/2)),int(np.round(nom_center_col-m/2)):int(np.round(nom_center_col+m/2))] = 1
                    # df = np.ma.masked_array(df, mask)
                lights.append(df)
            if darks_reg_exp is not None:
                if darks_reg_exp in f:
                    d = (fits.getdata(f)).astype(float)
                    darks.append(d) # to convert from unsigned integer uint16
        # mask = lights[3].mask
        lights = np.stack(lights)
        if darks_reg_exp is not None:
            darks = np.stack(darks)
            lights_subtracted = lights - darks
            # for i in range(len(lights_subtracted)):
                #lights_subtracted[i][int(np.round(nom_center_row[i] - m/2)):int(np.round(nom_center_row[i] + m/2)),int(np.round(nom_center_col[i] - m/2)):int(np.round(nom_center_col[i] + m/2))] = np.median(lights_subtracted[i])
            if buff is not None:
                avg_lights_sub = np.mean(lights_subtracted[buff:], axis=0) #leaving off first few, which are "transition" frames
            else:
                avg_lights_sub = np.mean(lights_subtracted, axis=0)
            if ceil is not None:
                nom_center_row = np.mean(nom_center_row)
                nom_center_col = np.mean(nom_center_col)
        if darks_reg_exp is None:
            lights_subtracted = lights
            # for i in range(len(lights_subtracted)):
                #lights_subtracted[i][int(np.round(nom_center_row[i] - m/2)):int(np.round(nom_center_row[i] + m/2)),int(np.round(nom_center_col[i] - m/2)):int(np.round(nom_center_col[i] + m/2))] = np.median(lights_subtracted[i])
            if buff is not None:
                avg_lights_sub = np.mean(lights_subtracted[buff:], axis=0) #leaving off first few, which are "transition" frames
            else:
                avg_lights_sub = np.mean(lights_subtracted, axis=0)
            if ceil is not None:
                nom_center_row = np.mean(nom_center_row)
                nom_center_col = np.mean(nom_center_col)
        if ceil is None:
            nom_center_row = 0
            nom_center_col = 0
        hdr = fits.Header()
        hdr['CEN_ROW'] = nom_center_row
        hdr['CEN_COL'] = nom_center_col
        prim = fits.PrimaryHDU(header=hdr)
        img = fits.ImageHDU(avg_lights_sub)
        hdul = fits.HDUList([prim, img])
        hdul.writeto(os.path.join(directory, 'avg_lights_sub_'+brights_part+'.fits'), overwrite=True)
    else:
        f = os.path.join(directory, 'avg_lights_sub_'+brights_part+'.fits')
        avg_lights_sub = fits.getdata(f)
        with fits.open(f) as hdul:
            hdr = hdul[0].header
            nom_center_row = float(hdr['CEN_ROW'])
            nom_center_col = float(hdr['CEN_COL'])

    return avg_lights_sub, nom_center_row, nom_center_col