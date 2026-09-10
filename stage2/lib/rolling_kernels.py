"""rolling_kernels.py -- VERBATIM numba kernels for rolling betas / momentum / skew, extracted
mechanically from stage2/process_bond_data.py lines 3488-4430 (golden tree, read-only). Do not edit the
kernel bodies; faithfulness is arbitrated by the G5 validator. Orchestrators live in steps/step4.

W5 (speed_up/01): `nogil=True` added to every decorator so lib/betas can fan the independent
per-model calls out on a thread pool. nogil changes GIL handling only -- the generated machine code
and numerics are untouched (validator-arbitrated)."""
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True, nogil=True)
def _panel_rolling_ols_k1(gid, y, x, window, min_obs):
    N = y.shape[0]

    alpha   = np.full(N, np.nan)
    beta    = np.full(N, np.nan)
    sig_tot = np.full(N, np.nan)
    sig_idi = np.full(N, np.nan)
    adj_r2  = np.full(N, np.nan)

    # within-group cumulative sums
    cx  = np.empty(N, dtype=np.float64)
    cy  = np.empty(N, dtype=np.float64)
    cxx = np.empty(N, dtype=np.float64)
    cyy = np.empty(N, dtype=np.float64)
    cxy = np.empty(N, dtype=np.float64)

    prev = -1
    sx = sy = sxx = syy = sxy = 0.0
    for i in range(N):
        g = gid[i]
        if g != prev:
            prev = g
            sx = sy = sxx = syy = sxy = 0.0

        xi = x[i]
        yi = y[i]

        sx  += xi
        sy  += yi
        sxx += xi * xi
        syy += yi * yi
        sxy += xi * yi

        cx[i]  = sx
        cy[i]  = sy
        cxx[i] = sxx
        cyy[i] = syy
        cxy[i] = sxy

    gstart = 0
    for i in range(N):
        if i == 0 or gid[i] != gid[i - 1]:
            gstart = i

        left = i - window + 1
        if left < gstart:
            left = gstart

        m = i - left + 1
        if m < min_obs:
            continue

        if left == gstart:
            Sx, Sy, Sxx, Syy, Sxy = cx[i], cy[i], cxx[i], cyy[i], cxy[i]
        else:
            j = left - 1
            Sx  = cx[i]  - cx[j]
            Sy  = cy[i]  - cy[j]
            Sxx = cxx[i] - cxx[j]
            Syy = cyy[i] - cyy[j]
            Sxy = cxy[i] - cxy[j]

        mf = float(m)
        xbar = Sx / mf
        ybar = Sy / mf

        Sxxc = Sxx - (Sx * Sx) / mf
        Sxyc = Sxy - (Sx * Sy) / mf
        Syyc = Syy - (Sy * Sy) / mf  # SST

        if Sxxc <= 0.0 or Syyc <= 0.0:
            continue

        b = Sxyc / Sxxc
        a = ybar - b * xbar

        # SSE = SST - b*Sxyc
        SSE = Syyc - b * Sxyc
        if SSE < 0.0:
            SSE = 0.0

        if m > 1:
            sig_tot[i] = np.sqrt(Syyc / (mf - 1.0))

        # dof = m - (K+1) = m-2
        if m > 2:
            sig_idi[i] = np.sqrt(SSE / (mf - 2.0))
            R2 = 1.0 - SSE / Syyc
            adj_r2[i] = 1.0 - (1.0 - R2) * (mf - 1.0) / (mf - 2.0)

        alpha[i] = a
        beta[i]  = b

    return alpha, beta, sig_tot, sig_idi, adj_r2


@njit(cache=True, fastmath=True, nogil=True)
def _panel_rolling_ols_k1_with_mom(gid, y, x, window, min_obs):
    """
    Rolling OLS for K=1 factor with momentum signals.

    At each position where we have valid alpha/beta:
    - Compute fitted values for ALL positions in the rolling window using CURRENT alpha/beta
    - Sum fitted values for systematic momentum (sysmom)
    - Sum residuals for idiosyncratic momentum (idimom)

    Momentum convention (same as build_mom_ltr_and_industry):
    - sysmom3_1: L=3, S=1 → sum 2 values (positions t-2 to t-1, skip current)
    - sysmom6_1: L=6, S=1 → sum 5 values (positions t-5 to t-1)
    - sysmom12_1: L=12, S=1 → sum 11 values (positions t-11 to t-1)

    Returns
    -------
    alpha, beta : float64 arrays of shape (N,)
    sysmom3_1, sysmom6_1, sysmom12_1 : float64 arrays of shape (N,)
    idimom3_1, idimom6_1, idimom12_1 : float64 arrays of shape (N,)
    """
    N = y.shape[0]

    alpha = np.full(N, np.nan)
    beta = np.full(N, np.nan)
    sysmom3_1 = np.full(N, np.nan)
    sysmom6_1 = np.full(N, np.nan)
    sysmom12_1 = np.full(N, np.nan)
    idimom3_1 = np.full(N, np.nan)
    idimom6_1 = np.full(N, np.nan)
    idimom12_1 = np.full(N, np.nan)

    # within-group cumulative sums for OLS
    cx = np.empty(N, dtype=np.float64)
    cy = np.empty(N, dtype=np.float64)
    cxx = np.empty(N, dtype=np.float64)
    cxy = np.empty(N, dtype=np.float64)

    prev = -1
    sx = sy = sxx = sxy = 0.0
    for i in range(N):
        g = gid[i]
        if g != prev:
            prev = g
            sx = sy = sxx = sxy = 0.0

        xi = x[i]
        yi = y[i]

        sx += xi
        sy += yi
        sxx += xi * xi
        sxy += xi * yi

        cx[i] = sx
        cy[i] = sy
        cxx[i] = sxx
        cxy[i] = sxy

    gstart = 0
    for i in range(N):
        if i == 0 or gid[i] != gid[i - 1]:
            gstart = i

        left = i - window + 1
        if left < gstart:
            left = gstart

        m = i - left + 1
        if m < min_obs:
            continue

        if left == gstart:
            Sx, Sy, Sxx, Sxy = cx[i], cy[i], cxx[i], cxy[i]
        else:
            j = left - 1
            Sx = cx[i] - cx[j]
            Sy = cy[i] - cy[j]
            Sxx = cxx[i] - cxx[j]
            Sxy = cxy[i] - cxy[j]

        mf = float(m)
        xbar = Sx / mf
        ybar = Sy / mf

        Sxxc = Sxx - (Sx * Sx) / mf
        Sxyc = Sxy - (Sx * Sy) / mf

        if Sxxc <= 0.0:
            continue

        b = Sxyc / Sxxc
        a = ybar - b * xbar

        alpha[i] = a
        beta[i] = b

        # Compute momentum signals using CURRENT alpha, beta
        # For each momentum spec (L, S), sum fitted/residual from position (i-L+1) to (i-S)
        # All positions are within the current window [left, i]

        # sysmom3_1 / idimom3_1: L=3, S=1 → positions [i-2, i-1], 2 values
        if i - 2 >= left:
            sys_sum = 0.0
            idi_sum = 0.0
            for j in range(i - 2, i):  # j = i-2, i-1
                fit_j = a + b * x[j]
                sys_sum += fit_j
                idi_sum += y[j] - fit_j
            sysmom3_1[i] = sys_sum
            idimom3_1[i] = idi_sum

        # sysmom6_1 / idimom6_1: L=6, S=1 → positions [i-5, i-1], 5 values
        if i - 5 >= left:
            sys_sum = 0.0
            idi_sum = 0.0
            for j in range(i - 5, i):  # j = i-5, i-4, i-3, i-2, i-1
                fit_j = a + b * x[j]
                sys_sum += fit_j
                idi_sum += y[j] - fit_j
            sysmom6_1[i] = sys_sum
            idimom6_1[i] = idi_sum

        # sysmom12_1 / idimom12_1: L=12, S=1 → positions [i-11, i-1], 11 values
        if i - 11 >= left:
            sys_sum = 0.0
            idi_sum = 0.0
            for j in range(i - 11, i):  # j = i-11, ..., i-1
                fit_j = a + b * x[j]
                sys_sum += fit_j
                idi_sum += y[j] - fit_j
            sysmom12_1[i] = sys_sum
            idimom12_1[i] = idi_sum

    return (alpha, beta,
            sysmom3_1, sysmom6_1, sysmom12_1,
            idimom3_1, idimom6_1, idimom12_1)


@njit(cache=True, fastmath=True, nogil=True)
def _panel_rolling_ols_k1_with_fitted(gid, y, x, window, min_obs):
    """
    Rolling OLS for K=1 factor, also returning fitted values and residuals.

    Same as _panel_rolling_ols_k1 but additionally computes:
    - fitted[i] = alpha[i] + beta[i] * x[i]
    - residual[i] = y[i] - fitted[i]

    These are computed using the rolling estimates available at each position.
    """
    N = y.shape[0]

    alpha   = np.full(N, np.nan)
    beta    = np.full(N, np.nan)
    fitted  = np.full(N, np.nan)
    residual = np.full(N, np.nan)

    # within-group cumulative sums
    cx  = np.empty(N, dtype=np.float64)
    cy  = np.empty(N, dtype=np.float64)
    cxx = np.empty(N, dtype=np.float64)
    cxy = np.empty(N, dtype=np.float64)

    prev = -1
    sx = sy = sxx = sxy = 0.0
    for i in range(N):
        g = gid[i]
        if g != prev:
            prev = g
            sx = sy = sxx = sxy = 0.0

        xi = x[i]
        yi = y[i]

        sx  += xi
        sy  += yi
        sxx += xi * xi
        sxy += xi * yi

        cx[i]  = sx
        cy[i]  = sy
        cxx[i] = sxx
        cxy[i] = sxy

    gstart = 0
    for i in range(N):
        if i == 0 or gid[i] != gid[i - 1]:
            gstart = i

        left = i - window + 1
        if left < gstart:
            left = gstart

        m = i - left + 1
        if m < min_obs:
            continue

        if left == gstart:
            Sx, Sy, Sxx, Sxy = cx[i], cy[i], cxx[i], cxy[i]
        else:
            j = left - 1
            Sx  = cx[i]  - cx[j]
            Sy  = cy[i]  - cy[j]
            Sxx = cxx[i] - cxx[j]
            Sxy = cxy[i] - cxy[j]

        mf = float(m)
        xbar = Sx / mf
        ybar = Sy / mf

        Sxxc = Sxx - (Sx * Sx) / mf
        Sxyc = Sxy - (Sx * Sy) / mf

        if Sxxc <= 0.0:
            continue

        b = Sxyc / Sxxc
        a = ybar - b * xbar

        alpha[i] = a
        beta[i]  = b

        # Compute fitted and residual for current observation
        fitted[i] = a + b * x[i]
        residual[i] = y[i] - fitted[i]

    return alpha, beta, fitted, residual


@njit(cache=True, fastmath=True, nogil=True)
def _panel_rolling_ols_kgt1_with_fitted(gid, y, X, window, min_obs, ridge):
    """
    Rolling OLS for K>=2 factors, also returning fitted values and residuals.

    Same as _panel_rolling_ols_kgt1 but additionally computes:
    - fitted[i] = alpha[i] + sum(beta_k[i] * X[i,k])
    - residual[i] = y[i] - fitted[i]
    """
    N, K = X.shape
    K1 = K + 1

    betas    = np.full((N, K1), np.nan)
    fitted   = np.full(N, np.nan)
    residual = np.full(N, np.nan)

    # ring buffers (store last W rows for subtracting)
    buf_y = np.empty(window, dtype=np.float64)
    buf_x = np.empty((window, K), dtype=np.float64)

    # rolling sums/moments
    ZZ = np.zeros((K1, K1), dtype=np.float64)
    Zy = np.zeros(K1, dtype=np.float64)
    n = 0

    # augmented regressor
    z = np.empty(K1, dtype=np.float64)

    prev = -1
    head = 0  # ring index

    for i in range(N):
        g = gid[i]
        if g != prev:
            # reset for new asset
            prev = g
            ZZ[:, :] = 0.0
            Zy[:] = 0.0
            n = 0
            head = 0

        yi = y[i]
        z[0] = 1.0
        for k in range(K):
            z[k + 1] = X[i, k]

        # if window is full, subtract oldest
        if n >= window:
            yo = buf_y[head]
            zo = np.empty(K1, dtype=np.float64)
            zo[0] = 1.0
            for k in range(K):
                zo[k + 1] = buf_x[head, k]
            _outer_sub(ZZ, zo)
            for k in range(K1):
                Zy[k] -= zo[k] * yo
            n -= 1

        # add current
        _outer_add(ZZ, z)
        for k in range(K1):
            Zy[k] += z[k] * yi

        buf_y[head] = yi
        for k in range(K):
            buf_x[head, k] = X[i, k]
        n += 1

        head += 1
        if head == window:
            head = 0

        if n < min_obs:
            continue

        # solve (ZZ + ridge*I) b = Zy
        A = ZZ.copy()
        for k in range(K1):
            A[k, k] += ridge

        b = np.linalg.solve(A, Zy)
        for k in range(K1):
            betas[i, k] = b[k]

        # Compute fitted and residual
        fit_val = b[0]  # alpha
        for k in range(K):
            fit_val += b[k + 1] * X[i, k]
        fitted[i] = fit_val
        residual[i] = y[i] - fit_val

    return betas, fitted, residual


@njit(cache=True, fastmath=True, nogil=True)
def _panel_rolling_ols_kgt1_with_mom(gid, y, X, window, min_obs, ridge):
    """
    Rolling OLS for K>=2 factors with momentum signals.

    At each position where we have valid betas:
    - Compute fitted values for ALL positions in the rolling window using CURRENT betas
    - Sum fitted values for systematic momentum (sysmom)
    - Sum residuals for idiosyncratic momentum (idimom)

    Momentum convention (same as build_mom_ltr_and_industry):
    - sysmom3_1: L=3, S=1 → sum 2 values (positions t-2 to t-1, skip current)
    - sysmom6_1: L=6, S=1 → sum 5 values (positions t-5 to t-1)
    - sysmom12_1: L=12, S=1 → sum 11 values (positions t-11 to t-1)

    Returns
    -------
    betas : float64 array of shape (N, K+1)
    sysmom3_1, sysmom6_1, sysmom12_1 : float64 arrays of shape (N,)
    idimom3_1, idimom6_1, idimom12_1 : float64 arrays of shape (N,)
    """
    N, K = X.shape
    K1 = K + 1

    betas = np.full((N, K1), np.nan)
    sysmom3_1 = np.full(N, np.nan)
    sysmom6_1 = np.full(N, np.nan)
    sysmom12_1 = np.full(N, np.nan)
    idimom3_1 = np.full(N, np.nan)
    idimom6_1 = np.full(N, np.nan)
    idimom12_1 = np.full(N, np.nan)

    # ring buffers (store last W rows for subtracting and momentum computation)
    buf_y = np.empty(window, dtype=np.float64)
    buf_x = np.empty((window, K), dtype=np.float64)
    buf_idx = np.empty(window, dtype=np.int64)  # original index in y/X arrays

    # rolling sums/moments
    ZZ = np.zeros((K1, K1), dtype=np.float64)
    Zy = np.zeros(K1, dtype=np.float64)
    n = 0

    # augmented regressor
    z = np.empty(K1, dtype=np.float64)

    prev = -1
    head = 0  # ring index

    for i in range(N):
        g = gid[i]
        if g != prev:
            # reset for new asset
            prev = g
            ZZ[:, :] = 0.0
            Zy[:] = 0.0
            n = 0
            head = 0

        yi = y[i]
        z[0] = 1.0
        for k in range(K):
            z[k + 1] = X[i, k]

        # if window is full, subtract oldest
        if n >= window:
            yo = buf_y[head]
            zo = np.empty(K1, dtype=np.float64)
            zo[0] = 1.0
            for k in range(K):
                zo[k + 1] = buf_x[head, k]
            _outer_sub(ZZ, zo)
            for k in range(K1):
                Zy[k] -= zo[k] * yo
            n -= 1

        # add current
        _outer_add(ZZ, z)
        for k in range(K1):
            Zy[k] += z[k] * yi

        buf_y[head] = yi
        for k in range(K):
            buf_x[head, k] = X[i, k]
        buf_idx[head] = i
        n += 1

        old_head = head
        head += 1
        if head == window:
            head = 0

        if n < min_obs:
            continue

        # solve (ZZ + ridge*I) b = Zy
        A = ZZ.copy()
        for k in range(K1):
            A[k, k] += ridge

        b = np.linalg.solve(A, Zy)
        for k in range(K1):
            betas[i, k] = b[k]

        # Compute momentum signals using CURRENT betas
        # The ring buffer contains n valid observations
        # We need to compute fitted values for past positions and sum them

        # Reconstruct positions in ring buffer from oldest to newest
        # The ring buffer is circular: oldest is at head, newest is at old_head
        # But we need positions relative to current i

        # For momentum, we need positions at lag L-1 to S (exclusive of current)
        # sysmom3_1: positions i-2, i-1 (2 values)
        # sysmom6_1: positions i-5, ..., i-1 (5 values)
        # sysmom12_1: positions i-11, ..., i-1 (11 values)

        # Check if we have enough observations in the window for each momentum spec
        # For L_val=3, S_val=1: need positions at i-2, i-1 → need n >= 3
        # For L_val=6, S_val=1: need positions at i-5..i-1 → need n >= 6
        # For L_val=12, S_val=1: need positions at i-11..i-1 → need n >= 12

        # sysmom3_1 / idimom3_1: L=3, S=1 → sum 2 values
        if n >= 3:
            sys_sum = 0.0
            idi_sum = 0.0
            for lag in range(2, 0, -1):  # lag=2, 1
                # ring index for position i-lag
                ridx = old_head - lag
                if ridx < 0:
                    ridx += window
                # compute fitted value using current betas
                fit_val = b[0]
                for k in range(K):
                    fit_val += b[k + 1] * buf_x[ridx, k]
                sys_sum += fit_val
                idi_sum += buf_y[ridx] - fit_val
            sysmom3_1[i] = sys_sum
            idimom3_1[i] = idi_sum

        # sysmom6_1 / idimom6_1: L=6, S=1 → sum 5 values
        if n >= 6:
            sys_sum = 0.0
            idi_sum = 0.0
            for lag in range(5, 0, -1):  # lag=5,4,3,2,1
                ridx = old_head - lag
                if ridx < 0:
                    ridx += window
                fit_val = b[0]
                for k in range(K):
                    fit_val += b[k + 1] * buf_x[ridx, k]
                sys_sum += fit_val
                idi_sum += buf_y[ridx] - fit_val
            sysmom6_1[i] = sys_sum
            idimom6_1[i] = idi_sum

        # sysmom12_1 / idimom12_1: L=12, S=1 → sum 11 values
        if n >= 12:
            sys_sum = 0.0
            idi_sum = 0.0
            for lag in range(11, 0, -1):  # lag=11,10,...,1
                ridx = old_head - lag
                if ridx < 0:
                    ridx += window
                fit_val = b[0]
                for k in range(K):
                    fit_val += b[k + 1] * buf_x[ridx, k]
                sys_sum += fit_val
                idi_sum += buf_y[ridx] - fit_val
            sysmom12_1[i] = sys_sum
            idimom12_1[i] = idi_sum

    return (betas, sysmom3_1, sysmom6_1, sysmom12_1,
            idimom3_1, idimom6_1, idimom12_1)


# ============================================================
# Numba helpers for K>1
# ============================================================
@njit(cache=True, fastmath=True, nogil=True)
def _outer_add(A, z):
    # A += z z'
    p = z.shape[0]
    for i in range(p):
        zi = z[i]
        for j in range(p):
            A[i, j] += zi * z[j]


@njit(cache=True, fastmath=True, nogil=True)
def _outer_sub(A, z):
    # A -= z z'
    p = z.shape[0]
    for i in range(p):
        zi = z[i]
        for j in range(p):
            A[i, j] -= zi * z[j]


@njit(cache=True, fastmath=True, nogil=True)
def _panel_rolling_ols_kgt1(gid, y, X, window, min_obs, ridge):
    """
    Rolling OLS for K>=2 using rolling moments + ring buffer.

    Maintains (within each asset):
      ZZ = sum z z'      where z=[1, x]
      Zy = sum z y
      Sy = sum y
      Syy= sum y^2

    Then per t:
      beta = solve(ZZ + ridge*I, Zy)
      SST  = Syy - Sy^2/n
      SSE  = Syy - 2 beta'Zy + beta'ZZ beta
    """
    N, K = X.shape
    K1 = K + 1

    betas   = np.full((N, K1), np.nan)
    sig_tot = np.full(N, np.nan)
    sig_idi = np.full(N, np.nan)
    adj_r2  = np.full(N, np.nan)

    # ring buffers (store last W rows for subtracting)
    buf_y = np.empty(window, dtype=np.float64)
    buf_x = np.empty((window, K), dtype=np.float64)

    # rolling sums/moments
    ZZ = np.zeros((K1, K1), dtype=np.float64)
    Zy = np.zeros(K1, dtype=np.float64)
    Sy = 0.0
    Syy = 0.0
    n = 0

    # augmented regressor
    z = np.empty(K1, dtype=np.float64)

    prev = -1
    head = 0  # ring index

    for i in range(N):
        g = gid[i]
        if g != prev:
            # reset for new asset
            prev = g
            ZZ[:, :] = 0.0
            Zy[:] = 0.0
            Sy = 0.0
            Syy = 0.0
            n = 0
            head = 0

        yi = y[i]

        # if window full, remove oldest
        if n >= window:
            yo = buf_y[head]

            z[0] = 1.0
            for k in range(K):
                z[k + 1] = buf_x[head, k]

            _outer_sub(ZZ, z)
            for k1 in range(K1):
                Zy[k1] -= z[k1] * yo
            Sy -= yo
            Syy -= yo * yo

            n -= 1  # will add new one below

        # add new obs into ring position
        buf_y[head] = yi
        for k in range(K):
            buf_x[head, k] = X[i, k]

        z[0] = 1.0
        for k in range(K):
            z[k + 1] = X[i, k]

        _outer_add(ZZ, z)
        for k1 in range(K1):
            Zy[k1] += z[k1] * yi
        Sy += yi
        Syy += yi * yi
        n += 1

        # advance ring head
        head += 1
        if head == window:
            head = 0

        if n < min_obs:
            continue

        mf = float(n)
        SST = Syy - (Sy * Sy) / mf
        if SST <= 0.0:
            continue

        # Solve (ZZ + ridge I) beta = Zy
        A = ZZ.copy()
        for d in range(K1):
            A[d, d] += ridge

        # if singular, solve may fail; ridge usually prevents that
        try:
            b = np.linalg.solve(A, Zy)
        except:
            continue

        betas[i, :] = b

        # total vol
        if n > 1:
            sig_tot[i] = np.sqrt(SST / (mf - 1.0))

        dof = n - K1
        if dof <= 0:
            continue

        # SSE via quadratic form using *unregularized* ZZ (standard OLS identity)
        # SSE = y'y - 2 b'Zy + b'ZZ b
        quad = 0.0
        for a in range(K1):
            tmp = 0.0
            for c in range(K1):
                tmp += ZZ[a, c] * b[c]
            quad += b[a] * tmp

        SSE = Syy - 2.0 * np.dot(b, Zy) + quad
        if SSE < 0.0:
            SSE = 0.0

        sig_idi[i] = np.sqrt(SSE / float(dof))

        R2 = 1.0 - SSE / SST
        adj_r2[i] = 1.0 - (1.0 - R2) * (mf - 1.0) / float(dof)

    return betas, sig_tot, sig_idi, adj_r2


@njit(cache=True, fastmath=True, nogil=True)
def _panel_rolling_skew(gid, resid, window, min_obs):
    """
    Compute rolling skewness of residuals per group.

    Uses the adjusted Fisher-Pearson standardized moment coefficient:
    skew = n / ((n-1)(n-2)) * sum((x-mean)^3) / std^3

    For computational efficiency, we use:
    m1 = sum(x)/n (mean)
    m2 = sum(x^2)/n
    m3 = sum(x^3)/n
    var = m2 - m1^2
    std = sqrt(var)
    central_m3 = m3 - 3*m1*m2 + 2*m1^3
    skew = central_m3 / std^3 (for population), adjusted for sample
    """
    N = len(resid)
    iskew = np.full(N, np.nan)

    # Ring buffers for rolling window
    buf = np.empty(window, dtype=np.float64)

    # Rolling sums
    S1 = 0.0  # sum of x
    S2 = 0.0  # sum of x^2
    S3 = 0.0  # sum of x^3
    n = 0

    prev = -1
    head = 0

    for i in range(N):
        g = gid[i]
        if g != prev:
            # Reset for new group
            prev = g
            S1 = 0.0
            S2 = 0.0
            S3 = 0.0
            n = 0
            head = 0

        ri = resid[i]

        # If window full, remove oldest
        if n >= window:
            ro = buf[head]
            S1 -= ro
            S2 -= ro * ro
            S3 -= ro * ro * ro
            n -= 1

        # Add new observation
        buf[head] = ri
        S1 += ri
        S2 += ri * ri
        S3 += ri * ri * ri
        n += 1

        # Advance ring head
        head += 1
        if head == window:
            head = 0

        if n < min_obs:
            continue

        nf = float(n)
        m1 = S1 / nf
        m2 = S2 / nf
        m3 = S3 / nf

        var = m2 - m1 * m1
        if var <= 1e-14:
            continue

        std = np.sqrt(var)

        # Central third moment: E[(x-μ)^3] = m3 - 3*m1*m2 + 2*m1^3
        central_m3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 * m1 * m1

        # Adjusted Fisher-Pearson (scipy.stats.skew default)
        # G1 = m3_central / std^3
        # For sample: adjust by sqrt(n*(n-1)) / (n-2)
        if n > 2:
            G1 = central_m3 / (std * std * std)
            adjust = np.sqrt(nf * (nf - 1.0)) / (nf - 2.0)
            iskew[i] = G1 * adjust

    return iskew


@njit(cache=True, fastmath=True, nogil=True)
def _panel_sum_signals(gid, vals, L_arr, S_arr):
    """
    Compute sum of values over rolling windows per group.

    For each (L, S) pair:
      - Window is from position (current - L + 1) to (current - S) inclusive
      - Length of window = L - S observations

    This follows the convention:
      "S excludes the most-recent S positions INCLUDING current"

    Handles NaN values: if any value in the window is NaN, output is NaN.

    Parameters
    ----------
    gid : int32 array
        Group IDs (sorted by group, then time)
    vals : float64 array
        Values to sum (e.g., fitted values or residuals)
    L_arr : int64 array
        Lookback lengths for each signal
    S_arr : int64 array
        Skip lengths for each signal

    Returns
    -------
    out : float64 array of shape (N, K)
        Sum signals for each observation and each (L, S) specification
    """
    N = len(vals)
    K = L_arr.shape[0]
    out = np.full((N, K), np.nan)

    # Track group boundaries
    prev = -1
    buf = np.empty(64, dtype=np.float64)  # buffer (max 64 obs lookback)
    count = 0

    for i in range(N):
        g = gid[i]
        if g != prev:
            # New group - reset
            prev = g
            count = 0

        # Add current value to buffer
        if count < 64:
            buf[count] = vals[i]
        count += 1

        # Compute each signal
        for k in range(K):
            L = L_arr[k]
            S = S_arr[k]

            # Position within group is (count - 1)
            pos = count - 1

            # end index (within buffer): pos - S
            end_i = pos - S
            if end_i < 0:
                continue

            # start index: pos - L + 1
            start_i = pos - L + 1
            if start_i < 0:
                continue

            # Need at least L observations
            if count < L:
                continue

            # Clamp to buffer size
            if start_i >= 64 or end_i >= 64:
                continue

            # Sum from start_i to end_i inclusive, checking for NaN
            total = 0.0
            valid = True
            for j in range(start_i, end_i + 1):
                v = buf[j]
                if np.isnan(v):
                    valid = False
                    break
                total += v

            if valid:
                out[i, k] = total

    return out
